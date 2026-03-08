// Copyright (c) 2026 Damien Boureille
// Licensed under the MIT License.

//! JIT auto-compilation: transparently compile pure functions on first call.
//!
//! When the interpreter calls a function that has no pre-compiled VMFB,
//! the JIT attempts to compile it on the fly via the same pipeline as
//! `sheaf build`: type inference → dict lowering → inlining → codegen → MLIR
//! → iree-compile → load VMFB. On success, subsequent calls dispatch via IREE.
//! On failure, the function is added to a blocklist and the interpreter
//! handles it normally.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::PathBuf;
use std::sync::Arc;

/// IREE compiler version — single source of truth in Cargo.toml [package.metadata]
const IREE_COMPILER_VERSION: &str = env!("IREE_VERSION");

use crate::compiler::codegen::CodeGenerator;
use crate::compiler::config::{layout_to_index_map, lower_get_calls};
use crate::compiler::effects::{collect_effects, collect_hof_calls};
use crate::compiler::stablehlo::StableHLOEmitter;
use crate::compiler::transforms::{
    extract_scalar_constants, lower_inlined_gets, propagate_let_layouts,
    resolve_static_constants,
};
use crate::core::compiler::{FunctionDef, VmfbSession};
use crate::core::inference::{infer_function_signature_with_known, FunctionSignature};
use crate::core::trace::{value_to_param_layout, value_to_stablehlo_type};
use crate::interpreter::value::Value;
use crate::StableHLOType;

pub struct JitCompiler {
    iree_compile_path: Option<String>,
    target_backend: String,
    failed_fns: HashSet<String>,
    verbose: bool,
}

impl JitCompiler {
    pub fn new() -> Self {
        let verbose = std::env::var("SHEAF_JIT_VERBOSE").is_ok();
        let target_backend = Self::detect_target_backend();
        let iree_compile_path = find_iree_compile().or_else(|| {
            match ensure_toolchain() {
                Ok(path) => Some(path),
                Err(e) => {
                    if verbose {
                        eprintln!("sheaf: toolchain download failed: {}", e);
                    }
                    None
                }
            }
        });
        Self {
            iree_compile_path,
            target_backend,
            failed_fns: HashSet::new(),
            verbose,
        }
    }

    fn detect_target_backend() -> String {
        match std::env::var("SHEAF_DEVICE") {
            Ok(ref d) if d == "cpu" => return "llvm-cpu".to_string(),
            Ok(ref d) => {
                return match d.as_str() {
                    "metal" => "metal-spirv",
                    "cuda" => "cuda",
                    "vulkan" => "vulkan-spirv",
                    _ => "llvm-cpu",
                }.to_string();
            }
            Err(_) => {}
        }
        // Try creating a Metal/CUDA session to see what's available.
        // If IreeSession::new() succeeds, use its detected backend.
        if let Ok(session) = crate::runtime::iree_session::IreeSession::new() {
            return session.target_backend().to_string();
        }
        "llvm-cpu".to_string()
    }

    pub fn try_jit_compile(
        &mut self,
        func_def: &FunctionDef,
        args: &[Value],
        registry: &HashMap<String, FunctionDef>,
        vmfb_sessions: &mut Vec<VmfbSession>,
    ) -> Option<(usize, FunctionSignature)> {
        let iree_compile = self.iree_compile_path.as_ref()?;
        let name = &func_def.name;

        if self.failed_fns.contains(name) {
            return None;
        }

        let body_compiled = func_def.body_compiled.as_ref()?;

        // Skip impure functions and functions using higher-order calls
        if !collect_effects(body_compiled).is_empty() {
            return None;
        }
        if !collect_hof_calls(body_compiled).is_empty() {
            return None;
        }

        // Skip scalar-only functions (no benefit from IREE)
        let has_tensor = args.iter().any(|a| {
            matches!(a, Value::Tensor { .. } | Value::Dict(_) | Value::Tuple(_))
        });
        if !has_tensor {
            return None;
        }

        let backend = self.target_backend.clone();
        self.compile_function(iree_compile.clone(), func_def, args, registry, vmfb_sessions, &backend)
    }

    fn compile_function(
        &mut self,
        iree_compile: String,
        func_def: &FunctionDef,
        args: &[Value],
        registry: &HashMap<String, FunctionDef>,
        vmfb_sessions: &mut Vec<VmfbSession>,
        target_backend: &str,
    ) -> Option<(usize, FunctionSignature)> {
        let name = &func_def.name;
        let mut body = func_def.body_compiled.clone()?;

        // Type inference from runtime args
        let mut known_types: Vec<(String, StableHLOType)> = Vec::new();
        let mut param_index_maps: Vec<(String, BTreeMap<Vec<String>, Vec<usize>>)> = Vec::new();
        let mut constants: HashMap<(String, Vec<usize>), f64> = HashMap::new();

        for (param_name, arg_val) in func_def.params.iter().zip(args) {
            let ty = match value_to_stablehlo_type(arg_val) {
                Ok(ty) => ty,
                Err(e) => {
                    self.jit_fail(name, &format!("type inference: {}", e));
                    return None;
                }
            };

            if let Some(layout) = value_to_param_layout(param_name, arg_val) {
                let imap = layout_to_index_map(&layout);
                extract_scalar_constants(arg_val, param_name, &imap, &mut constants);
                body = lower_get_calls(&body, param_name, &imap);
                param_index_maps.push((param_name.clone(), imap));
            }

            known_types.push((param_name.clone(), ty));
        }

        // Signature inference
        let dummy_compiler = crate::core::compiler::CompilerContext::new();
        let mut sig = match infer_function_signature_with_known(
            &dummy_compiler,
            &func_def.params,
            &body,
            &known_types,
        ) {
            Ok(s) => s,
            Err(e) => {
                self.jit_fail(name, &format!("signature inference: {}", e));
                return None;
            }
        };

        // Override param types for dict/tuple params
        for (param_name, ty) in &known_types {
            if let Some(idx) = func_def.params.iter().position(|p| p == param_name) {
                sig.param_types[idx] = ty.clone();
            }
        }

        // Inline user-defined function calls
        body = crate::autodiff::inline_function_calls(&body, registry);

        // Post-inline: re-lower dict access from inlined bodies
        for (param_name, index_map) in &param_index_maps {
            body = lower_get_calls(&body, param_name, index_map);
        }
        body = lower_inlined_gets(&body, &param_index_maps);

        // Resolve static constants
        let param_shapes: HashMap<String, Vec<i64>> = func_def
            .params
            .iter()
            .zip(sig.param_types.iter())
            .filter_map(|(p, ty)| {
                let shape = ty.shape();
                if shape.is_empty() {
                    None
                } else {
                    Some((p.clone(), shape.to_vec()))
                }
            })
            .collect();
        body = resolve_static_constants(&body, &constants, &param_shapes);

        // Build key layouts for codegen
        let mut tuple_key_layouts: HashMap<String, BTreeMap<String, usize>> = HashMap::new();
        let mut idx_to_key: HashMap<(String, usize), String> = HashMap::new();
        for (param_name, index_map) in &param_index_maps {
            for (key_path, indices) in index_map {
                if key_path.len() == 2 && indices.len() == 2 {
                    tuple_key_layouts
                        .entry(key_path[0].clone())
                        .or_default()
                        .insert(key_path[1].clone(), indices[1]);
                }
                if key_path.len() == 3 && indices.len() == 3 {
                    tuple_key_layouts
                        .entry(key_path[1].clone())
                        .or_default()
                        .insert(key_path[2].clone(), indices[2]);
                }
                if key_path.len() == 1 && indices.len() == 1 {
                    idx_to_key
                        .insert((param_name.clone(), indices[0]), key_path[0].clone());
                }
            }
        }
        propagate_let_layouts(&body, &idx_to_key, &mut tuple_key_layouts);

        // Codegen (catch panics gracefully)
        let codegen_result = {
            let registry_clone = registry.clone();
            let params_clone = func_def.params.clone();
            let param_types = sig.param_types.clone();
            let return_type = sig.return_type.clone();
            let body_clone = body.clone();
            let name_clone = name.clone();
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
                let mut codegen = CodeGenerator::with_function_params(
                    registry_clone,
                    &params_clone,
                    &param_types,
                );
                codegen.set_tuple_key_layouts(tuple_key_layouts);
                codegen.set_idx_to_key(idx_to_key);
                codegen.emit_func_declaration(&name_clone, &body_clone, &param_types, &return_type)
            }))
        };

        let (mlir_decl, actual_return_ty) = match codegen_result {
            Ok(Ok(result)) => result,
            Ok(Err(e)) => {
                self.jit_fail(name, &format!("codegen: {}", e));
                return None;
            }
            Err(_) => {
                self.jit_fail(name, "codegen: internal panic");
                return None;
            }
        };

        sig.return_type = actual_return_ty;

        // Emit MLIR module
        let mlir = StableHLOEmitter::emit_module(&[mlir_decl]);

        // Content hash for VMFB cache: hash(MLIR + backend + compiler version)
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        mlir.hash(&mut hasher);
        target_backend.hash(&mut hasher);
        IREE_COMPILER_VERSION.hash(&mut hasher);
        let cache_hash = hasher.finish();

        let cache_dir = PathBuf::from("__sheaf__");
        let cached_vmfb = cache_dir.join(format!("{:016x}.vmfb", cache_hash));

        // Try loading from cache
        let vmfb_data = if cached_vmfb.exists() {
            match std::fs::read(&cached_vmfb) {
                Ok(d) => {
                    if self.verbose {
                        eprintln!("jit: {} (cached)", name);
                    }
                    d
                }
                Err(_) => {
                    self.jit_fail(name, "failed to read cached VMFB");
                    return None;
                }
            }
        } else {
            eprintln!("jit: compiling {}...", name);

            let tmp_dir = std::env::temp_dir();
            let stamp = std::process::id();
            let mlir_path = tmp_dir.join(format!("sheaf-jit-{}-{}.mlir", name, stamp));
            let vmfb_path = tmp_dir.join(format!("sheaf-jit-{}-{}.vmfb", name, stamp));

            if std::fs::write(&mlir_path, &mlir).is_err() {
                self.jit_fail(name, "failed to write temp MLIR");
                return None;
            }

            let stderr_cfg = if self.verbose {
                std::process::Stdio::inherit()
            } else {
                std::process::Stdio::null()
            };

            let mut cmd = std::process::Command::new(&iree_compile);
            cmd.arg(&mlir_path)
                .arg(format!("--iree-hal-target-backends={}", target_backend))
                .arg("-o")
                .arg(&vmfb_path)
                .stderr(stderr_cfg);
            if target_backend == "metal-spirv" {
                cmd.arg("--iree-metal-compile-to-metallib=false");
            }
            let status = cmd.status();
            let _ = std::fs::remove_file(&mlir_path);

            let status = match status {
                Ok(s) => s,
                Err(e) => {
                    self.jit_fail(name, &format!("iree-compile exec: {}", e));
                    return None;
                }
            };

            if !status.success() {
                let _ = std::fs::remove_file(&vmfb_path);
                self.jit_fail(name, "iree-compile failed");
                return None;
            }

            let data = match std::fs::read(&vmfb_path) {
                Ok(d) => d,
                Err(_) => {
                    self.jit_fail(name, "failed to read compiled VMFB");
                    return None;
                }
            };
            let _ = std::fs::remove_file(&vmfb_path);

            // Cache for next run
            let _ = std::fs::create_dir_all(&cache_dir);
            let _ = std::fs::write(&cached_vmfb, &data);

            data
        };

        // Load into IREE session
        let mut session = match crate::runtime::iree_session::IreeSession::new() {
            Ok(s) => s,
            Err(e) => {
                self.jit_fail(name, &format!("IREE init: {}", e));
                return None;
            }
        };
        if let Err(e) = session.load_vmfb(vmfb_data) {
            self.jit_fail(name, &format!("VMFB load: {}", e));
            return None;
        }

        let session_idx = vmfb_sessions.len();
        vmfb_sessions.push(Arc::new(session));

        Some((session_idx, sig))
    }

    fn jit_fail(&mut self, name: &str, reason: &str) {
        self.failed_fns.insert(name.to_string());
        if self.verbose {
            eprintln!("jit: {} skipped ({})", name, reason);
        }
    }
}

/// Locate the `iree-compile` binary. Returns None if not found.
pub fn find_iree_compile() -> Option<String> {
    // Explicit env var
    if let Ok(path) = std::env::var("IREE_COMPILE") {
        if std::path::Path::new(&path).exists() {
            return Some(path);
        }
    }
    // Auto-downloaded toolchain cache
    if let Some(path) = find_cached_toolchain() {
        return Some(path);
    }
    // PATH lookup
    which("iree-compile")
}

fn toolchain_dir() -> Option<PathBuf> {
    std::env::var("HOME").ok().map(|h| PathBuf::from(h).join(".sheaf/toolchain"))
}

fn find_cached_toolchain() -> Option<String> {
    let dir = toolchain_dir()?;
    let binary = dir.join("iree-compile");
    if !binary.exists() {
        return None;
    }
    // Check version matches
    let version_file = dir.join("version");
    if let Ok(cached_version) = std::fs::read_to_string(&version_file) {
        if cached_version.trim() != IREE_COMPILER_VERSION {
            return None; // stale version, will trigger re-download
        }
    } else {
        return None;
    }
    Some(binary.to_string_lossy().to_string())
}

fn platform_wheel_tag() -> Option<&'static str> {
    match (std::env::consts::OS, std::env::consts::ARCH) {
        ("macos", _) => Some("macosx_13_0_universal2"),
        ("linux", "x86_64") => Some("manylinux_2_28_x86_64"),
        ("linux", "aarch64") => Some("manylinux_2_28_aarch64"),
        _ => None,
    }
}

fn compiler_lib_name() -> &'static str {
    if cfg!(target_os = "macos") { "libIREECompiler.dylib" }
    else { "libIREECompiler.so" }
}

/// Download and install the IREE compiler toolchain from PyPI.
pub fn ensure_toolchain() -> Result<String, Box<dyn std::error::Error>> {
    let platform_tag = platform_wheel_tag()
        .ok_or("unsupported platform for auto-download")?;
    let dir = toolchain_dir()
        .ok_or("cannot determine home directory")?;
    std::fs::create_dir_all(&dir)?;

    eprintln!("sheaf: downloading IREE compiler v{}...", IREE_COMPILER_VERSION);

    // Fetch PyPI JSON metadata to find the wheel URL
    let pypi_url = format!(
        "https://pypi.org/pypi/iree-base-compiler/{}/json",
        IREE_COMPILER_VERSION
    );
    let json_path = std::env::temp_dir().join("sheaf-pypi-metadata.json");
    let curl_status = std::process::Command::new("curl")
        .args(["-sSf", "-o"])
        .arg(&json_path)
        .arg(&pypi_url)
        .status()?;
    if !curl_status.success() {
        return Err("failed to fetch PyPI metadata (check network connection)".into());
    }

    let json_str = std::fs::read_to_string(&json_path)?;
    let _ = std::fs::remove_file(&json_path);

    // Parse JSON to find matching wheel URL
    let json: serde_json::Value = serde_json::from_str(&json_str)?;
    let urls = json["urls"].as_array()
        .ok_or("unexpected PyPI JSON format")?;

    let wheel_url = urls.iter()
        .filter_map(|entry| {
            let filename = entry["filename"].as_str()?;
            if filename.ends_with(".whl") && filename.contains(platform_tag) {
                entry["url"].as_str().map(|s| s.to_string())
            } else {
                None
            }
        })
        .next()
        .ok_or_else(|| format!(
            "no wheel found for platform '{}' at version {}",
            platform_tag, IREE_COMPILER_VERSION
        ))?;

    // Download the wheel
    let wheel_path = std::env::temp_dir().join("sheaf-iree-compiler.whl");
    let curl_status = std::process::Command::new("curl")
        .args(["-sSfL", "-o"])
        .arg(&wheel_path)
        .arg(&wheel_url)
        .status()?;
    if !curl_status.success() {
        return Err("failed to download IREE compiler wheel".into());
    }

    // Extract iree-compile and libIREECompiler from the wheel (ZIP file)
    let lib_name = compiler_lib_name();
    let unzip_status = std::process::Command::new("unzip")
        .args(["-j", "-o"])
        .arg(&wheel_path)
        .arg("iree/compiler/_mlir_libs/iree-compile")
        .arg("iree/compiler/_mlir_libs/iree-lld")
        .arg(format!("iree/compiler/_mlir_libs/{}", lib_name))
        .arg("-d")
        .arg(&dir)
        .stdout(std::process::Stdio::null())
        .status()?;
    let _ = std::fs::remove_file(&wheel_path);
    if !unzip_status.success() {
        return Err("failed to extract iree-compile from wheel".into());
    }

    // Ensure binaries are executable
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        for bin in &["iree-compile", "iree-lld"] {
            let path = dir.join(bin);
            if let Ok(meta) = std::fs::metadata(&path) {
                let mut perms = meta.permissions();
                perms.set_mode(0o755);
                let _ = std::fs::set_permissions(&path, perms);
            }
        }
    }

    // Write version file
    std::fs::write(dir.join("version"), IREE_COMPILER_VERSION)?;

    let binary = dir.join("iree-compile");
    eprintln!("sheaf: IREE compiler ready");
    Ok(binary.to_string_lossy().to_string())
}

fn which(name: &str) -> Option<String> {
    std::env::var("PATH").ok().and_then(|path_var| {
        path_var.split(':').find_map(|dir| {
            let candidate = format!("{}/{}", dir, name);
            if std::path::Path::new(&candidate).exists() {
                Some(candidate)
            } else {
                None
            }
        })
    })
}
