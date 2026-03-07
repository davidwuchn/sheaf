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
use std::sync::Arc;

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
use crate::forms::ml::param_layout_to_stablehlo_type;
use crate::interpreter::value::Value;
use crate::StableHLOType;

pub struct JitCompiler {
    iree_compile_path: Option<String>,
    failed_fns: HashSet<String>,
    verbose: bool,
}

impl JitCompiler {
    pub fn new() -> Self {
        Self {
            iree_compile_path: find_iree_compile(),
            failed_fns: HashSet::new(),
            verbose: std::env::var("SHEAF_JIT_VERBOSE").is_ok(),
        }
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

        self.compile_function(iree_compile.clone(), func_def, args, registry, vmfb_sessions)
    }

    fn compile_function(
        &mut self,
        iree_compile: String,
        func_def: &FunctionDef,
        args: &[Value],
        registry: &HashMap<String, FunctionDef>,
        vmfb_sessions: &mut Vec<VmfbSession>,
    ) -> Option<(usize, FunctionSignature)> {
        let name = &func_def.name;
        let mut body = func_def.body_compiled.clone()?;

        // Type inference from runtime args
        let mut known_types: Vec<(String, StableHLOType)> = Vec::new();
        let mut param_index_maps: Vec<(String, BTreeMap<Vec<String>, Vec<usize>>)> = Vec::new();
        let mut constants: HashMap<(String, Vec<usize>), f64> = HashMap::new();

        for (param_name, arg_val) in func_def.params.iter().zip(args) {
            if let Some(layout) = value_to_param_layout(param_name, arg_val) {
                let tuple_ty = param_layout_to_stablehlo_type(&layout);
                let imap = layout_to_index_map(&layout);
                extract_scalar_constants(arg_val, param_name, &imap, &mut constants);
                body = lower_get_calls(&body, param_name, &imap);
                param_index_maps.push((param_name.clone(), imap));
                known_types.push((param_name.clone(), tuple_ty));
            } else {
                match value_to_stablehlo_type(arg_val) {
                    Ok(ty) => known_types.push((param_name.clone(), ty)),
                    Err(e) => {
                        self.jit_fail(name, &format!("type inference: {}", e));
                        return None;
                    }
                }
            }
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

        // Compile with iree-compile
        let tmp_dir = std::env::temp_dir();
        let stamp = std::process::id();
        let mlir_path = tmp_dir.join(format!("sheaf-jit-{}-{}.mlir", name, stamp));
        let vmfb_path = tmp_dir.join(format!("sheaf-jit-{}-{}.vmfb", name, stamp));

        if std::fs::write(&mlir_path, &mlir).is_err() {
            self.jit_fail(name, "failed to write temp MLIR");
            return None;
        }

        eprintln!("jit: compiling {}...", name);

        let stderr_cfg = if self.verbose {
            std::process::Stdio::inherit()
        } else {
            std::process::Stdio::null()
        };

        let status = std::process::Command::new(&iree_compile)
            .arg(&mlir_path)
            .arg("--iree-hal-target-backends=llvm-cpu")
            .arg("-o")
            .arg(&vmfb_path)
            .stderr(stderr_cfg)
            .status();

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

        let vmfb_data = match std::fs::read(&vmfb_path) {
            Ok(d) => d,
            Err(_) => {
                self.jit_fail(name, "failed to read compiled VMFB");
                return None;
            }
        };
        let _ = std::fs::remove_file(&vmfb_path);

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
    // 1. Explicit env var
    if let Ok(path) = std::env::var("IREE_COMPILE") {
        if std::path::Path::new(&path).exists() {
            return Some(path);
        }
    }
    // 2. Standard SDK install location
    if let Ok(home) = std::env::var("HOME") {
        let candidate = format!("{}/bin/iree-build/tools/iree-compile", home);
        if std::path::Path::new(&candidate).exists() {
            return Some(candidate);
        }
    }
    // 3. PATH lookup
    which("iree-compile")
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
