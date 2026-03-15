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

/// IREE compiler version, single source of truth in Cargo.toml [package.metadata]
const IREE_COMPILER_VERSION: &str = env!("IREE_VERSION");

use crate::sheaf_msg;
use crate::autodiff::reverse::{to_anf, reverse_grad};
use crate::compiler::codegen::{
    collect_tuple_leaves, expand_tuple_to_symbols, CodeGenerator,
};
use crate::compiler::config::{layout_to_index_map, lower_get_calls};
use crate::compiler::effects::{collect_effects, collect_hof_calls};
use crate::compiler::stablehlo::{Register, StableHLOEmitter};
use crate::compiler::transforms::{
    extract_scalar_constants, lower_inlined_gets, propagate_let_layouts,
    resolve_static_constants, unroll_reduces,
};
use crate::core::compiler::{CompiledExpr, FunctionDef, VmfbSession};
use crate::core::inference::{infer_function_signature_with_known, FunctionSignature};
use crate::core::trace::{value_to_param_layout, value_to_stablehlo_type};
use crate::interpreter::value::Value;
use crate::StableHLOType;

pub struct JitCompiler {
    iree_compile_path: Option<String>,
    target_backend: String,
    failed_fns: HashSet<String>,
    /// Cache compiled VAG sessions: vag_key → (session_idx, signature, param_names)
    vag_cache: HashMap<String, (usize, FunctionSignature, Vec<String>)>,
}

impl JitCompiler {
    pub fn new() -> Self {
        let target_backend = Self::detect_target_backend();
        let iree_compile_path = find_iree_compile().or_else(|| {
            match ensure_toolchain() {
                Ok(path) => Some(path),
                Err(e) => {
                    sheaf_msg!("sheaf: JIT compilation unavailable — {}", e);
                    sheaf_msg!("sheaf: functions without cached .vmfb will run in interpreter (slow)");
                    None
                }
            }
        });
        Self {
            iree_compile_path,
            target_backend,
            failed_fns: HashSet::new(),
            vag_cache: HashMap::new(),
        }
    }

    fn detect_target_backend() -> String {
        if let Some(d) = crate::core::config::device_override() {
            return match d {
                "cpu" => "llvm-cpu",
                "metal" => "metal-spirv",
                "cuda" => "cuda",
                "vulkan" => "vulkan-spirv",
                _ => "llvm-cpu",
            }.to_string();
        }
        // Use cached backend from the first IreeSession::new() call
        if let Some(backend) = crate::runtime::iree_session::IreeSession::cached_target_backend() {
            return backend.to_string();
        }
        // Fallback: create a session to probe available drivers
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
        let iree_compile = self.iree_compile_path.clone()?;
        let name = &func_def.name;

        if self.failed_fns.contains(name) {
            return None;
        }

        let body_compiled = func_def.body_compiled.as_ref()?;

        // Skip impure functions and functions using higher-order calls
        if !collect_effects(body_compiled).is_empty() {
            self.jit_fail(name, "has effects");
            return None;
        }
        if !collect_hof_calls(body_compiled).is_empty() {
            self.jit_fail(name, "has HOF calls");
            return None;
        }

        // Skip scalar-only functions (no benefit from IREE)
        let has_tensor = args.iter().any(|a| {
            matches!(a, Value::Tensor { .. } | Value::DeviceBuffer(_) | Value::Dict(_) | Value::Tuple(_))
        });
        if !has_tensor {
            self.jit_fail(name, "scalar-only args");
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

        // -vv: log signature and lowered params
        if crate::core::config::verbosity() >= 2 {
            for (pname, pty) in func_def.params.iter().zip(sig.param_types.iter()) {
                if let Some((_, imap)) = param_index_maps.iter().find(|(n, _)| n == pname) {
                    // Collect top-level fields with their types from the tuple
                    let mut top_fields: BTreeMap<usize, (String, String)> = BTreeMap::new();
                    for (path, indices) in imap.iter() {
                        if path.len() == 1 {
                            // Leaf field: resolve type from tuple
                            let ty_str = if let StableHLOType::Tuple(elems, _) = pty {
                                if let Some(t) = elems.get(indices[0]) {
                                    t.to_mlir()
                                } else {
                                    "?".to_string()
                                }
                            } else {
                                "?".to_string()
                            };
                            top_fields.entry(indices[0])
                                .or_insert((path[0].clone(), ty_str));
                        } else if !top_fields.contains_key(&indices[0]) {
                            // Nested field: show as tuple<...>
                            let ty_str = if let StableHLOType::Tuple(elems, _) = pty {
                                if let Some(StableHLOType::Tuple(sub, _)) = elems.get(indices[0]) {
                                    format!("tuple<...> ({} fields)", sub.len())
                                } else {
                                    "tuple<...>".to_string()
                                }
                            } else {
                                "tuple<...>".to_string()
                            };
                            top_fields.insert(indices[0], (path[0].clone(), ty_str));
                        }
                    }
                    for (_, (key, ty_str)) in &top_fields {
                        sheaf_msg!("jit: {} | {}.{}: {}", name, pname, key, ty_str);
                    }
                } else {
                    sheaf_msg!("jit: {} | {}: {}", name, pname, pty.to_mlir());
                }
            }
            sheaf_msg!("jit: {} | return: {}", name, sig.return_type.to_mlir());
        }

        // Capture value layouts for dict/tuple args (for return value reconstruction)
        {
            use crate::core::inference::ValueLayout;
            let mut seen_types = std::collections::HashSet::new();
            for (arg_val, ty) in args.iter().zip(sig.param_types.iter()) {
                let layout = ValueLayout::from_value(arg_val);
                if !matches!(layout, ValueLayout::Leaf) {
                    let type_key = format!("{:?}", ty);
                    if seen_types.insert(type_key) {
                        sig.arg_type_layouts.push((ty.clone(), layout));
                    }
                }
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
                for depth in 0..key_path.len() {
                    let parent = if depth == 0 {
                        param_name.clone()
                    } else {
                        key_path[depth - 1].clone()
                    };
                    let child = &key_path[depth];
                    let idx = indices[depth];
                    tuple_key_layouts
                        .entry(parent.clone())
                        .or_default()
                        .entry(child.clone())
                        .or_insert(idx);
                    idx_to_key
                        .entry((parent, idx))
                        .or_insert_with(|| child.clone());
                }
            }
        }
        propagate_let_layouts(&body, &idx_to_key, &mut tuple_key_layouts);

        // Collect scalar param values for constant propagation in codegen.
        // Scalar f32 params (e.g. top-k=40, temp=0.8) get their values recorded
        // so shape-critical ops (top_k slice sizes) can resolve K at compile time.
        let scalar_param_values: Vec<(String, f64)> = func_def
            .params
            .iter()
            .zip(args.iter())
            .filter_map(|(name, val)| match val {
                Value::Float(f) => Some((name.clone(), *f as f64)),
                Value::Int(n) => Some((name.clone(), *n as f64)),
                _ => None,
            })
            .collect();

        // Codegen (catch panics gracefully)
        let codegen_result = {
            let registry_clone = registry.clone();
            let params_clone = func_def.params.clone();
            let param_types = sig.param_types.clone();
            let return_type = sig.return_type.clone();
            let body_clone = body.clone();
            let name_clone = name.clone();
            let constants_clone = constants.clone();
            let param_index_maps_clone = param_index_maps.clone();
            let scalar_values_clone = scalar_param_values.clone();
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
                let mut codegen = CodeGenerator::with_function_params(
                    registry_clone,
                    &params_clone,
                    &param_types,
                );
                codegen.set_tuple_key_layouts(tuple_key_layouts);
                codegen.set_idx_to_key(idx_to_key);
                codegen.set_scalar_constants(constants_clone);
                codegen.set_param_index_maps(param_index_maps_clone);
                codegen.set_scalar_param_values(&scalar_values_clone);
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

        if crate::core::config::verbosity() >= 2 {
            sheaf_msg!("jit: {} | codegen return: {}", name, sig.return_type.to_mlir());
            sheaf_msg!("jit: {} | return_dict_keys: {:?}", name, sig.return_dict_keys);
        }

        // Register layout for return type (may differ from param type due to
        // scalar promotion, e.g. ScalarI64→scalar_f32 after adam-step)
        if let StableHLOType::Tuple(ret_elems, _) = &sig.return_type {
            if !sig.arg_type_layouts.iter().any(|(t, _)| t == &sig.return_type) {
                for (t, layout) in sig.arg_type_layouts.clone() {
                    if let StableHLOType::Tuple(param_elems, _) = &t {
                        if param_elems.len() == ret_elems.len() {
                            sig.arg_type_layouts.push((sig.return_type.clone(), layout));
                            break;
                        }
                    }
                }
            }
        }

        // Emit MLIR module
        let mlir = StableHLOEmitter::emit_module(&[mlir_decl]);

        if crate::core::config::verbosity() >= 2 {
            sheaf_msg!("jit: {} | MLIR {} lines", name, mlir.lines().count());
        }

        // Content hash for staleness check
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        mlir.hash(&mut hasher);
        target_backend.hash(&mut hasher);
        IREE_COMPILER_VERSION.hash(&mut hasher);
        let content_hash = format!("{:016x}", hasher.finish());

        let cache_dir = PathBuf::from("__sheaf__");
        let cached_vmfb = cache_dir.join(format!("{}.vmfb", name));

        // Check manifest for staleness (-vv forces recompile for full debug output)
        let force_recompile = crate::core::config::verbosity() >= 2;
        let vmfb_data = if !force_recompile && cached_vmfb.exists() && manifest_hash_matches(&cache_dir, name, &content_hash) {
            match std::fs::read(&cached_vmfb) {
                Ok(d) => {
                    if crate::core::config::verbosity() >= 2 {
                        sheaf_msg!("jit: {} (cached, {}KB, {})", name, d.len() / 1024, target_backend);
                    } else if crate::core::config::verbosity() >= 1 {
                        sheaf_msg!("jit: {} (cached)", name);
                    }
                    d
                }
                Err(_) => {
                    self.jit_fail(name, "failed to read cached compilation");
                    return None;
                }
            }
        } else {
            let data = match self.run_iree_compile(&iree_compile, name, &mlir, target_backend) {
                Some(d) => d,
                None => {
                    self.jit_fail(name, "iree-compile failed on all backends");
                    return None;
                }
            };

            // Cache: named VMFB + manifest entry
            let _ = std::fs::create_dir_all(&cache_dir);
            let _ = std::fs::write(&cached_vmfb, &data);
            update_manifest(&cache_dir, name, &content_hash);

            data
        };

        // Load into IREE session
        let mut session = match crate::runtime::iree_session::IreeSession::new() {
            Ok(s) => s,
            Err(e) => {
                self.jit_fail(name, &format!("JIT engine init: {}", e));
                return None;
            }
        };
        if let Err(e) = session.load_vmfb(vmfb_data) {
            if target_backend != "llvm-cpu" {
                sheaf_msg!("sheaf: {} backend failed for '{}', falling back to cpu", target_backend, name);
                let _ = std::fs::remove_file(&cached_vmfb);
                self.target_backend = "llvm-cpu".to_string();
                let cpu_data = match self.run_iree_compile(&iree_compile, name, &mlir, "llvm-cpu") {
                    Some(d) => d,
                    None => {
                        self.jit_fail(name, "compilation failed on all backends");
                        return None;
                    }
                };
                session = match crate::runtime::iree_session::IreeSession::new() {
                    Ok(s) => s,
                    Err(_) => {
                        self.jit_fail(name, "JIT engine init failed on all backends");
                        return None;
                    }
                };
                if session.load_vmfb(cpu_data).is_err() {
                    self.jit_fail(name, "JIT load failed on all backends");
                    return None;
                }
            } else {
                self.jit_fail(name, &format!("JIT load: {}", e));
                return None;
            }
        }

        let session_idx = vmfb_sessions.len();
        vmfb_sessions.push(Arc::new(session));

        Some((session_idx, sig))
    }

    /// JIT-compile a value-and-grad closure into a single VMFB.
    ///
    /// The closure `func` is `(fn [p] loss-body)` with captured values in its closure.
    /// We promote tensor captures to MLIR parameters, resolve scalar captures as constants,
    /// then generate forward + backward passes in a single MLIR function.
    ///
    /// Returns `(session_idx, signature, param_order)` where `param_order` lists
    /// the combined parameter names (fn params first, then tensor captures) for dispatch.
    pub fn try_jit_value_and_grad(
        &mut self,
        func: &Value,
        wrt_arg: &Value,
        registry: &HashMap<String, FunctionDef>,
        vmfb_sessions: &mut Vec<VmfbSession>,
    ) -> Option<(usize, FunctionSignature, Vec<String>)> {
        let iree_compile = self.iree_compile_path.clone()?;

        let (fn_params, body, closure) = match func {
            Value::Function {
                params,
                body,
                closure,
            } => (params, body, closure),
            _ => return None,
        };

        // Derive a human-readable name from the outermost function call
        let vag_fn_name = outermost_call_name(body).unwrap_or("anonymous".to_string());

        // Build a stable key for the blocklist and cache.
        // Include wrt_arg type so shape changes (e.g. after grow-hydra) cause a cache miss.
        let wrt_type_str = value_to_stablehlo_type(wrt_arg)
            .map(|t| t.to_mlir())
            .unwrap_or_default();
        let vag_key = format!("__vag_{:?}_{}", body, wrt_type_str);
        if self.failed_fns.contains(&vag_key) {
            return None;
        }

        // Return cached session if already compiled
        if let Some(cached) = self.vag_cache.get(&vag_key) {
            return Some(cached.clone());
        }

        // Skip impure or HOF-containing bodies
        if !collect_effects(body).is_empty() {
            return None;
        }
        if !collect_hof_calls(body).is_empty() {
            return None;
        }

        // Build combined parameter list: fn params first, then tensor captures.
        // Scalar captures are substituted directly in the body.
        let mut all_param_names: Vec<String> = Vec::new();
        let mut all_arg_values: Vec<Value> = Vec::new();
        let mut scalar_substitutions: Vec<(String, f64)> = Vec::new();
        let wrt_indices: Vec<usize>;

        // fn params (the wrt parameters)
        for p in fn_params {
            all_param_names.push(p.clone());
            all_arg_values.push(wrt_arg.clone());
        }
        wrt_indices = (0..fn_params.len()).collect();

        // Classify and add captures
        for (cap_name, cap_val) in closure {
            // Skip the __vag_fn__ sentinel
            if cap_name.starts_with("__") {
                continue;
            }
            match cap_val {
                Value::Float(f) => {
                    scalar_substitutions.push((cap_name.clone(), *f as f64));
                }
                Value::Int(n) => {
                    scalar_substitutions.push((cap_name.clone(), *n as f64));
                }
                _ => {
                    // Tensor, Dict, Tuple etc. → promote to MLIR parameter
                    if value_to_stablehlo_type(cap_val).is_ok() {
                        all_param_names.push(cap_name.clone());
                        all_arg_values.push(cap_val.clone());
                    } else {
                        self.jit_fail(&vag_key, &format!("unsupported capture type for '{}'", cap_name));
                        return None;
                    }
                }
            }
        }

        // Substitute scalar captures in body
        let mut body = body.clone();
        for (name, val) in &scalar_substitutions {
            body = substitute_scalar(&body, name, *val);
        }

        // Type inference from runtime args
        let mut known_types: Vec<(String, StableHLOType)> = Vec::new();
        let mut param_index_maps: Vec<(String, BTreeMap<Vec<String>, Vec<usize>>)> = Vec::new();
        let mut constants: HashMap<(String, Vec<usize>), f64> = HashMap::new();

        for (param_name, arg_val) in all_param_names.iter().zip(all_arg_values.iter()) {
            let ty = match value_to_stablehlo_type(arg_val) {
                Ok(ty) => ty,
                Err(e) => {
                    self.jit_fail(&vag_key, &format!("type inference: {}", e));
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
            &all_param_names,
            &body,
            &known_types,
        ) {
            Ok(s) => s,
            Err(e) => {
                self.jit_fail(&vag_key, &format!("signature inference: {}", e));
                return None;
            }
        };

        // Override param types from runtime values
        for (param_name, ty) in &known_types {
            if let Some(idx) = all_param_names.iter().position(|p| p == param_name) {
                sig.param_types[idx] = ty.clone();
            }
        }

        // -vv: log VAG signature and captures
        if crate::core::config::verbosity() >= 2 {
            for (pname, pty) in all_param_names.iter().zip(sig.param_types.iter()) {
                if let Some((_, imap)) = param_index_maps.iter().find(|(n, _)| n == pname) {
                    let mut top_fields: BTreeMap<usize, (String, String)> = BTreeMap::new();
                    for (path, indices) in imap.iter() {
                        if path.len() == 1 {
                            let ty_str = if let StableHLOType::Tuple(elems, _) = pty {
                                if let Some(t) = elems.get(indices[0]) {
                                    t.to_mlir()
                                } else {
                                    "?".to_string()
                                }
                            } else {
                                "?".to_string()
                            };
                            top_fields.entry(indices[0])
                                .or_insert((path[0].clone(), ty_str));
                        } else if !top_fields.contains_key(&indices[0]) {
                            let ty_str = if let StableHLOType::Tuple(elems, _) = pty {
                                if let Some(StableHLOType::Tuple(sub, _)) = elems.get(indices[0]) {
                                    format!("tuple<...> ({} fields)", sub.len())
                                } else {
                                    "tuple<...>".to_string()
                                }
                            } else {
                                "tuple<...>".to_string()
                            };
                            top_fields.insert(indices[0], (path[0].clone(), ty_str));
                        }
                    }
                    for (_, (key, ty_str)) in &top_fields {
                        sheaf_msg!("jit: value-and-grad | {}.{}: {}", pname, key, ty_str);
                    }
                } else {
                    sheaf_msg!("jit: value-and-grad | {}: {}", pname, pty.to_mlir());
                }
            }
            if !scalar_substitutions.is_empty() {
                sheaf_msg!("jit: value-and-grad | {} scalar captures", scalar_substitutions.len());
            }
        }

        // Inline user-defined function calls
        body = crate::autodiff::inline_function_calls(&body, registry);

        // Post-inline: re-lower dict access from inlined bodies
        for (param_name, index_map) in &param_index_maps {
            body = lower_get_calls(&body, param_name, index_map);
        }
        body = lower_inlined_gets(&body, &param_index_maps);

        // Unroll reduces so grad_simplified can differentiate through them
        let known_types_vec: Vec<(String, StableHLOType)> = sig
            .param_types
            .iter()
            .enumerate()
            .map(|(i, ty)| (all_param_names[i].clone(), ty.clone()))
            .collect();
        // Debug: log remaining reduces before unrolling
        if crate::core::config::verbosity() >= 2 {
            sheaf_msg!("jit: [vag] param_index_maps keys: {:?}",
                param_index_maps.iter().map(|(n, _)| n.as_str()).collect::<Vec<_>>());
            log_remaining_reduces(&body, "before unroll");
        }
        body = unroll_reduces(&body, &known_types_vec);
        if crate::core::config::verbosity() >= 2 {
            log_remaining_reduces(&body, "after unroll");
        }

        // Re-lower dict access introduced by unrolling (get on GetTupleElement elements)
        for (param_name, index_map) in &param_index_maps {
            body = lower_get_calls(&body, param_name, index_map);
        }
        body = lower_inlined_gets(&body, &param_index_maps);

        // Resolve static constants
        let mut param_shapes: HashMap<String, Vec<i64>> = all_param_names
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
        // Inject shapes of GetTupleElement leaves for shape inference
        for (param_name, param_ty) in all_param_names.iter().zip(sig.param_types.iter()) {
            inject_tuple_shapes(param_name, param_ty, &[], &mut param_shapes);
        }
        body = resolve_static_constants(&body, &constants, &param_shapes);

        // Build key layouts for codegen
        let mut tuple_key_layouts: HashMap<String, BTreeMap<String, usize>> = HashMap::new();
        let mut idx_to_key: HashMap<(String, usize), String> = HashMap::new();
        for (param_name, index_map) in &param_index_maps {
            for (key_path, indices) in index_map {
                for depth in 0..key_path.len() {
                    let parent = if depth == 0 {
                        param_name.clone()
                    } else {
                        key_path[depth - 1].clone()
                    };
                    let child = &key_path[depth];
                    let idx = indices[depth];
                    tuple_key_layouts
                        .entry(parent.clone())
                        .or_default()
                        .entry(child.clone())
                        .or_insert(idx);
                    idx_to_key
                        .entry((parent, idx))
                        .or_insert_with(|| child.clone());
                }
            }
        }
        propagate_let_layouts(&body, &idx_to_key, &mut tuple_key_layouts);

        // Debug: log unresolved shapes
        if crate::core::config::verbosity() >= 2 {
            log_unresolved_shapes(&body, "before codegen");
        }

        // Value-and-grad codegen (catch panics gracefully)
        let backend = self.target_backend.clone();
        let codegen_result = {
            let registry_clone = registry.clone();
            let param_names = all_param_names.clone();
            let param_types = sig.param_types.clone();
            let body_clone = body.clone();
            let wrt_idx = wrt_indices.clone();
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
                let mut codegen = CodeGenerator::with_function_params(
                    registry_clone,
                    &param_names,
                    &param_types,
                );
                codegen.set_tuple_key_layouts(tuple_key_layouts);
                codegen.set_idx_to_key(idx_to_key);

                let inlined_body = body_clone;

                // 1. Expand tuple params to synthetic leaf symbols
                let mut expanded_body = inlined_body.clone();
                let mut all_leaves: Vec<(usize, Vec<crate::compiler::codegen::TupleLeaf>)> = Vec::new();
                let mut all_wrt_symbols: Vec<String> = Vec::new();

                for &idx in &wrt_idx {
                    let param_name = &param_names[idx];
                    let param_ty = &param_types[idx];
                    match param_ty {
                        StableHLOType::Tuple(..) => {
                            let leaves = collect_tuple_leaves(&expanded_body, param_name);
                            if crate::core::config::verbosity() >= 2 {
                                sheaf_msg!("jit: [vag] param '{}': {} tuple leaves", param_name, leaves.len());
                            }
                            expanded_body = expand_tuple_to_symbols(&expanded_body, param_name);
                            for leaf in &leaves {
                                all_wrt_symbols.push(leaf.symbol.clone());
                            }
                            all_leaves.push((idx, leaves));
                        }
                        _ => {
                            all_wrt_symbols.push(param_name.clone());
                        }
                    }
                }

                // 2. Convert to ANF
                let anf_expr = to_anf(&expanded_body);
                let (anf_bindings, anf_body) = match &anf_expr {
                    CompiledExpr::Let { bindings, body } => {
                        (bindings.clone(), body.as_ref().clone())
                    }
                    other => (vec![], other.clone()),
                };

                // 3. Bind tuple leaves for codegen (use generate to handle both
                //    leaf lookups and sub-tuple reconstruction from flat args)
                for &(idx, ref leaves) in &all_leaves {
                    let param_name = &param_names[idx];
                    for leaf in leaves {
                        let gte = CompiledExpr::GetTupleElement {
                            param: param_name.clone(),
                            indices: leaf.indices.clone(),
                        };
                        let (reg, ty) = codegen.generate(&gte)?;
                        codegen.bind_symbol(&leaf.symbol, reg, ty);
                    }
                }

                // 4. Generate forward bindings (flat scope, no Let scoping).
                //    Use generate_binding to handle Lambda, destructuring, layouts.
                for (name, value_expr) in &anf_bindings {
                    codegen.generate_binding(name, value_expr)?;
                }

                // Generate the ANF body (loss value)
                let (loss_reg, loss_ty) = codegen.generate(&anf_body)?;

                // 5. Build shape map from forward codegen for reverse-mode AD
                let shape_map: HashMap<String, Vec<i64>> = codegen.binding_shapes();

                // 6. Run reverse-mode AD on ANF with shape info
                let (backward_bindings, grad_sym_map) =
                    reverse_grad(&anf_bindings, &anf_body, &all_wrt_symbols, &shape_map);

                if crate::core::config::verbosity() >= 2 {
                    sheaf_msg!("jit: [vag] ANF: {} fwd bindings, {} bwd bindings, {} wrt symbols",
                        anf_bindings.len(), backward_bindings.len(), all_wrt_symbols.len());
                }

                if std::env::var("SHEAF_DEBUG_GRAD").is_ok() {
                    eprintln!("--- Forward ANF bindings ---");
                    for (name, val) in &anf_bindings {
                        eprintln!("  {} = {:?}  [shape: {:?}]", name, val, shape_map.get(name));
                    }
                    eprintln!("  body = {:?}", anf_body);
                    eprintln!("--- Backward bindings ---");
                    for (name, val) in &backward_bindings {
                        eprintln!("  {} = {:?}", name, val);
                    }
                    eprintln!("--- Grad map ---");
                    for (sym, grad_name) in &grad_sym_map {
                        eprintln!("  {} → {}", sym, grad_name);
                    }
                    eprintln!("--- wrt symbols ---");
                    for s in &all_wrt_symbols {
                        eprintln!("  {}", s);
                    }
                }

                // 7. Generate backward bindings (adjoint computations).
                //    Backward bindings are always simple name=expr (no Lambda/destructuring).
                for (name, value_expr) in &backward_bindings {
                    let (reg, ty) = codegen.generate(value_expr)?;
                    codegen.bind_symbol(name, reg, ty);
                }

                // 6. Collect gradient registers for each wrt param.
                let mut grad_regs: Vec<Register> = Vec::new();
                let mut grad_tys: Vec<StableHLOType> = Vec::new();

                for &idx in &wrt_idx {
                    let param_name = &param_names[idx];
                    let param_ty = &param_types[idx];
                    match param_ty {
                        StableHLOType::Tuple(..) => {
                            let leaves = all_leaves.iter()
                                .find(|(i, _)| *i == idx)
                                .map(|(_, l)| l)
                                .unwrap();

                            // Each leaf gradient is a Symbol name from grad_sym_map
                            let leaf_grad_map: std::collections::HashMap<String, CompiledExpr> =
                                leaves.iter().map(|leaf| {
                                    let grad_expr = grad_sym_map
                                        .get(&leaf.symbol)
                                        .map(|sym_name| CompiledExpr::Symbol(sym_name.clone()))
                                        .unwrap_or_else(|| {
                                            // Generate zeros of the correct shape for missing grads
                                            let leaf_ty = resolve_leaf_type(param_ty, &leaf.indices);
                                            let shape = leaf_ty.shape();
                                            if shape.is_empty() {
                                                CompiledExpr::Float(0.0)
                                            } else {
                                                CompiledExpr::FunctionCall {
                                                    name: "zeros".to_string(),
                                                    args: vec![CompiledExpr::Vector(
                                                        shape.iter().map(|&d| CompiledExpr::Integer(d)).collect()
                                                    )],
                                                    loc: None,
                                                }
                                            }
                                        });
                                    (leaf.symbol.clone(), grad_expr)
                                }).collect();

                            let (grad_reg, grad_ty) =
                                codegen.build_grad_tuple_from_map(leaves, param_ty, &leaf_grad_map)?;
                            grad_regs.push(grad_reg);
                            grad_tys.push(grad_ty);
                        }
                        _ => {
                            let grad_expr = grad_sym_map
                                .get(param_name)
                                .map(|sym_name| CompiledExpr::Symbol(sym_name.clone()))
                                .unwrap_or(CompiledExpr::Float(0.0));
                            let (grad_reg, grad_ty) = codegen.generate(&grad_expr)?;
                            let (grad_reg, grad_ty) =
                                codegen.reduce_broadcast_grad(grad_reg, &grad_ty, param_ty)?;
                            grad_regs.push(grad_reg);
                            grad_tys.push(grad_ty);
                        }
                    }
                }

                // Pack result: (loss, grad0, grad1, ...)
                let mut all_regs = vec![loss_reg];
                all_regs.extend(grad_regs);
                let mut all_tys = vec![loss_ty.clone()];
                all_tys.extend(grad_tys.clone());

                let decl = codegen.finish_multi(
                    "value_and_grad",
                    &param_types,
                    &all_regs,
                    &all_tys,
                );
                let return_type = StableHLOType::Tuple(all_tys, None);
                Ok::<_, crate::core::error::SheafError>((decl, return_type))
            }))
        };

        let (mlir_decl, return_type) = match codegen_result {
            Ok(Ok(result)) => result,
            Ok(Err(e)) => {
                self.jit_fail(&vag_key, &format!("codegen: {}", e));
                return None;
            }
            Err(_) => {
                self.jit_fail(&vag_key, "codegen: internal panic");
                return None;
            }
        };

        sig.return_type = return_type;
        sig.return_dict_keys = None;

        // Emit MLIR module
        let mlir = StableHLOEmitter::emit_module(&[mlir_decl]);

        if crate::core::config::verbosity() >= 2 {
            sheaf_msg!("jit: value-and-grad | MLIR {} lines", mlir.lines().count());
        }

        // Content hash for staleness check
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        mlir.hash(&mut hasher);
        backend.hash(&mut hasher);
        IREE_COMPILER_VERSION.hash(&mut hasher);
        let content_hash = format!("{:016x}", hasher.finish());

        let cache_dir = PathBuf::from("__sheaf__");
        let vag_cache_name = format!("{}-vag", vag_fn_name);
        let cached_vmfb = cache_dir.join(format!("{}.vmfb", vag_cache_name));

        let force_recompile = crate::core::config::verbosity() >= 2;
        let vmfb_data = if !force_recompile && cached_vmfb.exists() && manifest_hash_matches(&cache_dir, &vag_cache_name, &content_hash) {
            match std::fs::read(&cached_vmfb) {
                Ok(d) => {
                    if crate::core::config::verbosity() >= 2 {
                        sheaf_msg!("jit: value_and_grad (cached, {}KB, {})", d.len() / 1024, backend);
                    } else if crate::core::config::verbosity() >= 1 {
                        sheaf_msg!("jit: value_and_grad (cached)");
                    }
                    d
                }
                Err(_) => {
                    self.jit_fail(&vag_key, "failed to read cached compilation");
                    return None;
                }
            }
        } else {
            // Debug: save a copy for inspection
            if crate::core::config::verbosity() >= 2 {
                let debug_path = cache_dir.join(format!("{}-vag-debug.mlir", vag_fn_name));
                let _ = std::fs::write(&debug_path, &mlir);
                sheaf_msg!("jit: value-and-grad | saved {}", debug_path.display());
            }

            let data = match self.run_iree_compile(&iree_compile, "value_and_grad", &mlir, &backend) {
                Some(d) => d,
                None => {
                    self.jit_fail(&vag_key, "compilation failed on all backends");
                    return None;
                }
            };

            let _ = std::fs::create_dir_all(&cache_dir);
            let _ = std::fs::write(&cached_vmfb, &data);
            update_manifest(&cache_dir, &vag_cache_name, &content_hash);

            data
        };

        // Load into session
        let mut session = match crate::runtime::iree_session::IreeSession::new() {
            Ok(s) => s,
            Err(e) => {
                self.jit_fail(&vag_key, &format!("JIT engine init: {}", e));
                return None;
            }
        };
        if let Err(e) = session.load_vmfb(vmfb_data) {
            if backend != "llvm-cpu" {
                sheaf_msg!("sheaf: {} backend failed for value-and-grad, falling back to cpu", backend);
                let _ = std::fs::remove_file(&cached_vmfb);
                self.target_backend = "llvm-cpu".to_string();
                let cpu_data = match self.run_iree_compile(&iree_compile, "value_and_grad", &mlir, "llvm-cpu") {
                    Some(d) => d,
                    None => {
                        self.jit_fail(&vag_key, "compilation failed on all backends");
                        return None;
                    }
                };
                session = match crate::runtime::iree_session::IreeSession::new() {
                    Ok(s) => s,
                    Err(_) => {
                        self.jit_fail(&vag_key, "JIT engine init failed");
                        return None;
                    }
                };
                if session.load_vmfb(cpu_data).is_err() {
                    self.jit_fail(&vag_key, "JIT load failed on all backends");
                    return None;
                }
            } else {
                self.jit_fail(&vag_key, &format!("JIT load: {}", e));
                return None;
            }
        }

        let session_idx = vmfb_sessions.len();
        vmfb_sessions.push(Arc::new(session));

        let result = (session_idx, sig, all_param_names);
        self.vag_cache.insert(vag_key, result.clone());
        Some(result)
    }

    /// Run iree-compile on MLIR source, with automatic CPU fallback.
    /// Returns compiled VMFB bytes, or None if all backends fail.
    fn run_iree_compile(&mut self, iree_compile: &str, name: &str, mlir: &str, backend: &str) -> Option<Vec<u8>> {
        let backends_to_try = if backend != "llvm-cpu" {
            vec![backend, "llvm-cpu"]
        } else {
            vec![backend]
        };

        for (i, try_backend) in backends_to_try.iter().enumerate() {
            if i > 0 {
                sheaf_msg!("sheaf: {} backend failed for '{}', falling back to cpu", backend, name);
                // Remember the fallback so we don't retry for every function
                self.target_backend = "llvm-cpu".to_string();
            }

            if crate::core::config::verbosity() >= 1 {
                sheaf_msg!("jit: compiling {} [{}]...", name, try_backend);
            }

            let tmp_dir = std::env::temp_dir();
            let stamp = std::process::id();
            let mlir_path = tmp_dir.join(format!("sheaf-jit-{}-{}.mlir", name, stamp));
            let vmfb_path = tmp_dir.join(format!("sheaf-jit-{}-{}.vmfb", name, stamp));

            if std::fs::write(&mlir_path, mlir).is_err() {
                eprintln!("sheaf: failed to write temp MLIR, aborting");
                std::process::exit(1);
            }

            let stderr_cfg = if crate::core::config::verbosity() >= 2 {
                std::process::Stdio::inherit()
            } else {
                std::process::Stdio::null()
            };

            let mut cmd = std::process::Command::new(iree_compile);
            cmd.arg(&mlir_path)
                .arg(format!("--iree-hal-target-backends={}", try_backend))
                .arg("--iree-opt-const-eval=false")
                .arg("-o")
                .arg(&vmfb_path)
                .stderr(stderr_cfg);
            if *try_backend == "metal-spirv" {
                cmd.arg("--iree-metal-compile-to-metallib=false");
            }
            if *try_backend == "llvm-cpu" {
                cmd.arg("--iree-llvmcpu-target-cpu=host");
                cmd.arg("--iree-llvmcpu-enable-ukernels=all");
            }
            let status = cmd.status();

            if crate::core::config::verbosity() >= 2 {
                let debug_mlir = format!("__sheaf__/{}-debug.mlir", name);
                let _ = std::fs::rename(&mlir_path, &debug_mlir);
                sheaf_msg!("jit: {} | saved {}", name, debug_mlir);
            } else {
                let _ = std::fs::remove_file(&mlir_path);
            }

            let ok = match &status {
                Ok(s) => s.success(),
                Err(_) => false,
            };

            if ok {
                if let Ok(data) = std::fs::read(&vmfb_path) {
                    let _ = std::fs::remove_file(&vmfb_path);
                    return Some(data);
                }
            }
            let _ = std::fs::remove_file(&vmfb_path);
        }

        None
    }

    fn jit_fail(&mut self, name: &str, reason: &str) {
        self.failed_fns.insert(name.to_string());
        let display_name = if name.starts_with("__vag_") { "value-and-grad" } else { name };
        if crate::core::config::verbosity() >= 1 {
            sheaf_msg!("jit: {} skipped ({})", display_name, reason);
        }
    }
}

/// Navigate a tuple type tree using indices to find the leaf type.
fn resolve_leaf_type(ty: &StableHLOType, indices: &[usize]) -> StableHLOType {
    let mut current = ty.clone();
    for &idx in indices {
        if let StableHLOType::Tuple(elems, _) = &current {
            if idx < elems.len() {
                current = elems[idx].clone();
            } else {
                return StableHLOType::scalar_f32();
            }
        } else {
            return current;
        }
    }
    current
}

/// Substitute all occurrences of `Symbol(name)` with `Float(val)` in a compiled expression.
fn substitute_scalar(expr: &CompiledExpr, name: &str, val: f64) -> CompiledExpr {
    match expr {
        CompiledExpr::Symbol(s) if s == name => CompiledExpr::Float(val),
        CompiledExpr::FunctionCall {
            name: fn_name,
            args, .. } => CompiledExpr::FunctionCall {
            name: fn_name.clone(),
            args: args.iter().map(|a| substitute_scalar(a, name, val)).collect(),
            loc: None,
        },
        CompiledExpr::Let { bindings, body } => CompiledExpr::Let {
            bindings: bindings
                .iter()
                .map(|(k, v)| (k.clone(), substitute_scalar(v, name, val)))
                .collect(),
            body: Box::new(substitute_scalar(body, name, val)),
        },
        CompiledExpr::Lambda { params, body } => {
            if params.contains(&name.to_string()) {
                expr.clone() // shadowed
            } else {
                CompiledExpr::Lambda {
                    params: params.clone(),
                    body: Box::new(substitute_scalar(body, name, val)),
                }
            }
        }
        CompiledExpr::LambdaCall { callee, args } => CompiledExpr::LambdaCall {
            callee: Box::new(substitute_scalar(callee, name, val)),
            args: args.iter().map(|a| substitute_scalar(a, name, val)).collect(),
        },
        CompiledExpr::If {
            condition,
            then_branch,
            else_branch,
        } => CompiledExpr::If {
            condition: Box::new(substitute_scalar(condition, name, val)),
            then_branch: Box::new(substitute_scalar(then_branch, name, val)),
            else_branch: else_branch
                .as_ref()
                .map(|e| Box::new(substitute_scalar(e, name, val))),
        },
        CompiledExpr::Do(exprs) => CompiledExpr::Do(
            exprs.iter().map(|e| substitute_scalar(e, name, val)).collect(),
        ),
        other => other.clone(),
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

    for tool in &["curl", "unzip"] {
        if which(tool).is_none() {
            return Err(format!("'{}' is required to download the compiler toolchain", tool).into());
        }
    }

    sheaf_msg!("sheaf: downloading compiler toolchain...");

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
    sheaf_msg!("sheaf: compiler successfully installed in {}", dir.display());
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

#[cfg(iree_runtime)]
fn inject_tuple_shapes(
    param_name: &str,
    ty: &StableHLOType,
    indices: &[usize],
    shapes: &mut HashMap<String, Vec<i64>>,
) {
    match ty {
        StableHLOType::Tuple(elems, _) => {
            for (i, elem_ty) in elems.iter().enumerate() {
                let mut child_indices = indices.to_vec();
                child_indices.push(i);
                inject_tuple_shapes(param_name, elem_ty, &child_indices, shapes);
            }
        }
        _ => {
            let shape = ty.shape();
            if !shape.is_empty() && !indices.is_empty() {
                let key = format!("{}@{:?}", param_name, indices);
                shapes.insert(key, shape.to_vec());
            }
        }
    }
}

#[cfg(iree_runtime)]
fn log_remaining_reduces(expr: &CompiledExpr, label: &str) {
    match expr {
        CompiledExpr::FunctionCall { name, args, .. } if name == "reduce" => {
            let coll_desc = if let Some(coll) = args.get(2) {
                format!("{:?}", coll)
            } else {
                "missing".to_string()
            };
            sheaf_msg!("jit: [{}] reduce with coll={}", label, coll_desc);
        }
        CompiledExpr::FunctionCall { args, .. } => {
            for a in args { log_remaining_reduces(a, label); }
        }
        CompiledExpr::Let { bindings, body } => {
            for (_, v) in bindings { log_remaining_reduces(v, label); }
            log_remaining_reduces(body, label);
        }
        CompiledExpr::Lambda { body, .. } => log_remaining_reduces(body, label),
        CompiledExpr::Do(exprs) => { for e in exprs { log_remaining_reduces(e, label); } }
        _ => {}
    }
}

#[cfg(iree_runtime)]
fn log_unresolved_shapes(expr: &CompiledExpr, label: &str) {
    match expr {
        CompiledExpr::FunctionCall { name, args, .. } if name == "reshape" && args.len() == 2 => {
            match &args[1] {
                CompiledExpr::Vector(elems) => {
                    let unresolved: Vec<_> = elems.iter()
                        .filter(|e| !matches!(e, CompiledExpr::Integer(_)))
                        .map(|e| format!("{:?}", e))
                        .collect();
                    if !unresolved.is_empty() {
                        sheaf_msg!("jit: [{}] reshape with unresolved shape elements: {:?}", label, unresolved);
                    }
                }
                other => {
                    sheaf_msg!("jit: [{}] reshape with non-vector shape: {:?}", label, other);
                }
            }
        }
        CompiledExpr::FunctionCall { args, .. } => {
            for a in args { log_unresolved_shapes(a, label); }
        }
        CompiledExpr::Let { bindings, body } => {
            for (_, v) in bindings { log_unresolved_shapes(v, label); }
            log_unresolved_shapes(body, label);
        }
        CompiledExpr::Lambda { body, .. } => log_unresolved_shapes(body, label),
        CompiledExpr::Do(exprs) => { for e in exprs { log_unresolved_shapes(e, label); } }
        _ => {}
    }
}

/// Extract the outermost function call name from an expression (for cache naming).
fn outermost_call_name(expr: &CompiledExpr) -> Option<String> {
    match expr {
        CompiledExpr::FunctionCall { name, .. } => Some(name.clone()),
        CompiledExpr::Let { body, .. } => outermost_call_name(body),
        CompiledExpr::Do(exprs) => exprs.last().and_then(outermost_call_name),
        _ => None,
    }
}

/// Read the manifest and check if the stored hash matches.
fn manifest_hash_matches(cache_dir: &std::path::Path, name: &str, expected_hash: &str) -> bool {
    let manifest_path = cache_dir.join("manifest.json");
    let data = match std::fs::read_to_string(&manifest_path) {
        Ok(d) => d,
        Err(_) => return false,
    };
    let manifest: serde_json::Value = match serde_json::from_str(&data) {
        Ok(v) => v,
        Err(_) => return false,
    };
    manifest.get(name).and_then(|v| v.as_str()) == Some(expected_hash)
}

/// Update the manifest with a new hash entry.
fn update_manifest(cache_dir: &std::path::Path, name: &str, hash: &str) {
    let manifest_path = cache_dir.join("manifest.json");
    let mut manifest: serde_json::Map<String, serde_json::Value> =
        std::fs::read_to_string(&manifest_path)
            .ok()
            .and_then(|d| serde_json::from_str(&d).ok())
            .unwrap_or_default();
    manifest.insert(name.to_string(), serde_json::Value::String(hash.to_string()));
    let _ = std::fs::write(&manifest_path, serde_json::to_string_pretty(&manifest).unwrap());
}
