// Copyright (c) 2026 Damien Boureille
// Licensed under the MIT License.

//! Standard JIT compilation pipeline.

use super::support::{manifest_hash_matches, update_manifest};
use super::*;

impl JitCompiler {
    pub fn try_jit_compile(
        &mut self,
        func_def: &FunctionDef,
        args: &[Value],
        registry: &HashMap<String, FunctionDef>,
        shared_session: &std::sync::Arc<crate::runtime::iree_session::IreeSession>,
    ) -> Option<FunctionSignature> {
        let name = &func_def.name;
        let iree_compile = self.iree_compile_path.clone()?;

        // Cache-key construction performs shape analysis with call inlining.
        // Establish eligibility first, including the negative definition cache.
        if !self.preflight_jit_eligibility(func_def, registry) {
            return None;
        }

        // The session's active driver is the source of truth for compilation.
        if let Some(backend) = crate::runtime::iree_session::IreeSession::cached_target_backend() {
            self.target_backend = backend.to_string();
        }

        // Success and failure caches use the same variant identity.
        let cache_key = cache_key_for_eligible_function(func_def, args, registry)?;
        if self.failed_definitions.contains(&cache_key.definition_hash) {
            return None;
        }

        // (1) Success cache hit: the module is already loaded into the shared
        // session; return its signature without recompiling.
        if let Some(info) = self.module_for_key(&cache_key) {
            JIT_CATALOG_HITS.fetch_add(1, Ordering::Relaxed);
            return Some(info.sig);
        }
        JIT_CATALOG_MISSES.fetch_add(1, Ordering::Relaxed);

        // (2) Failure cache hit: this exact (fn, shape) was tried before and
        // the compile failed. Don't retry.
        if self.failed_keys.contains(&cache_key) {
            return None;
        }

        // Skip scalar-only functions (no benefit from IREE)
        let has_tensor = args.iter().any(|a| {
            matches!(
                a,
                Value::Tensor { .. } | Value::DeviceBuffer(_) | Value::Dict(_) | Value::Tuple(_)
            )
        });
        if !has_tensor {
            self.failed_definitions
                .insert(cache_key.definition_hash.clone());
            self.jit_fail(name, "scalar-only args");
            return None;
        }

        let module_name = module_name_for(name, &cache_key);
        if let Some(existing) = shared_module_catalog()
            .lock()
            .expect("JIT catalogue lock poisoned")
            .identities
            .get(&module_name)
            .cloned()
            && existing != cache_key {
                sheaf_msg!(
                    "ERROR jit: module fingerprint collision for {} (refusing to load)",
                    module_name
                );
                self.failed_keys.insert(cache_key);
                return None;
            }

        // Reserve the variant without holding the catalogue lock during compilation.
        {
            let mut catalogue = shared_module_catalog()
                .lock()
                .expect("JIT catalogue lock poisoned");
            if !catalogue.compiling.insert(cache_key.clone()) {
                return None;
            }
        }
        let mut reservation = CompilationReservation::new(cache_key.clone());

        let backend = self.target_backend.clone();
        let sig = match self.compile_function(
            iree_compile.clone(),
            func_def,
            args,
            registry,
            shared_session,
            &backend,
            &cache_key,
        ) {
            Some(x) => x,
            None => {
                // Compile failed for this variant. The reservation guard removes
                // the in-flight marker; other variants remain eligible.
                self.failed_keys.insert(cache_key);
                return None;
            }
        };
        let info = CompiledModuleInfo {
            function_name: name.to_string(),
            module_name: module_name.clone(),
            sig: sig.clone(),
        };
        {
            let mut catalogue = shared_module_catalog()
                .lock()
                .expect("JIT catalogue lock poisoned");
            catalogue.identities.insert(module_name, cache_key.clone());
            catalogue.modules.insert(cache_key.clone(), info);
            catalogue.compiling.remove(&cache_key);
        }
        reservation.finish();
        // Module unloading is unavailable, so report excessive catalogue growth.
        let threshold = crate::core::config::jit_module_warning_threshold();
        let warning = {
            let mut catalogue = shared_module_catalog()
                .lock()
                .expect("JIT catalogue lock poisoned");
            let variants = catalogue
                .modules
                .values()
                .fold(BTreeMap::new(), |mut counts, info| {
                    *counts.entry(info.function_name.as_str()).or_default() += 1;
                    counts
                });
            let message = module_growth_warning(
                catalogue.modules.len(),
                threshold,
                catalogue.growth_warned,
                &variants,
            );
            if message.is_some() {
                catalogue.growth_warned = true;
            }
            message
        };
        if let Some(message) = warning {
            sheaf_msg!("{}", message);
        }

        Some(sig)
    }

    fn compile_function(
        &mut self,
        iree_compile: String,
        func_def: &FunctionDef,
        args: &[Value],
        registry: &HashMap<String, FunctionDef>,
        shared_session: &std::sync::Arc<crate::runtime::iree_session::IreeSession>,
        target_backend: &str,
        cache_key: &JitCacheKey,
    ) -> Option<FunctionSignature> {
        let name = &func_def.name;
        let mut body = func_def.body_compiled.clone()?;

        // The caller computed this key before lookup. Reuse that exact value
        // so lookup, compilation and insertion cannot diverge.
        let module_name = module_name_for(name, cache_key);

        // Type inference from runtime args
        let mut known_types: Vec<(String, StableHLOType)> = Vec::new();
        let mut param_index_maps: Vec<(String, BTreeMap<Vec<String>, Vec<usize>>)> = Vec::new();
        let mut constants: HashMap<(String, Vec<usize>), f64> = HashMap::new();

        for (param_name, arg_val) in func_def.params.iter().zip(args) {
            let ty = match value_to_stablehlo_type(arg_val) {
                Ok(ty) => ty,
                Err(e) => {
                    self.jit_fail(
                        name,
                        &format!("arg '{}' has {}", param_name, e.short_message()),
                    );
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
        let dummy_compiler = crate::core::expr::CompilerContext::new();
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
                            top_fields
                                .entry(indices[0])
                                .or_insert((path[0].clone(), ty_str));
                        } else {
                            top_fields.entry(indices[0]).or_insert_with(|| {
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
                                (path[0].clone(), ty_str)
                            });
                        }
                    }
                    for (key, ty_str) in top_fields.values() {
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

        // Compute param_shapes (needed by both preprocess_vag_lambda and resolve_static_constants)
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

        // Only shape-dependent scalars become compile-time constants.
        let filtered_constants = filter_constants_for_shape_positions(&constants, &body);

        // Resolve nested VAG lambdas before the enclosing body.
        let arity_err: std::cell::Cell<Option<SheafError>> = std::cell::Cell::new(None);
        body = self.preprocess_vag_lambda(
            &body,
            registry,
            &filtered_constants,
            &param_shapes,
            &param_index_maps,
            &known_types,
            &arity_err,
        );
        if let Some(err) = arity_err.into_inner() {
            self.jit_fail(name, &err.short_message());
            return None;
        }

        body = resolve_static_constants(&body, &filtered_constants, &param_shapes, false);

        body = match crate::lowering::transforms::lower_tuples_and_destructuring(
            body,
            &param_shapes,
        ) {
            Ok(b) => b,
            Err(e) => {
                self.jit_fail(name, &e.short_message());
                return None;
            }
        };

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

        // Convert codegen panics into JIT compilation failures.
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
            sheaf_msg!(
                "jit: {} | codegen return: {}",
                name,
                sig.return_type.to_mlir()
            );
            sheaf_msg!(
                "jit: {} | return_dict_keys: {:?}",
                name,
                sig.return_dict_keys
            );
        }

        // Register layout for return type (may differ from param type due to
        // scalar promotion, e.g. ScalarI64->scalar_f32 after adam-step)
        if let StableHLOType::Tuple(ret_elems, ret_keys) = &sig.return_type {
            // Only match return layout to a param layout if the return type already
            // has dict keys AND those keys match a param layout's keys.
            // A plain tuple (ret_keys == None) or mismatched keys must NOT inherit
            // dict structure from an unrelated param.
            if ret_keys.is_some()
                && !sig
                    .arg_type_layouts
                    .iter()
                    .any(|(t, _)| t == &sig.return_type)
            {
                for (t, layout) in sig.arg_type_layouts.clone() {
                    if let StableHLOType::Tuple(param_elems, param_keys) = &t
                        && param_elems.len() == ret_elems.len() && param_keys == ret_keys {
                            sig.arg_type_layouts.push((sig.return_type.clone(), layout));
                            break;
                        }
                }
            }
        }

        // Emit MLIR module. The module is namespaced (`module @<module_name> { ... }`)
        // so multiple compilations of the same logical function for distinct
        // shapes can coexist in the single shared `IreeSession` (each loaded
        // under a distinct name).
        let mlir = StableHLOEmitter::emit_module_named(Some(&module_name), &[mlir_decl]);

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
        let safe_name = name.replace('?', "_q").replace('!', "_b");
        let backend_suffix = target_backend.replace('-', "_");
        let cached_vmfb = cache_dir.join(format!("{}.{}.vmfb", safe_name, backend_suffix));
        // Check manifest for staleness (-vv forces recompile for full debug output)
        let force_recompile = crate::core::config::verbosity() >= 2;
        let vmfb_data = if !force_recompile
            && cached_vmfb.exists()
            && manifest_hash_matches(&cache_dir, name, &content_hash)
        {
            match std::fs::read(&cached_vmfb) {
                Ok(d) => {
                    if crate::core::config::verbosity() >= 2 {
                        sheaf_msg!(
                            "jit: {} (cached, {}KB, {})",
                            name,
                            d.len() / 1024,
                            target_backend
                        );
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
            JIT_EXTERNAL_COMPILATIONS.fetch_add(1, Ordering::Relaxed);
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

        // Modules are appended to the process-wide session.
        JIT_MODULE_LOADS.fetch_add(1, Ordering::Relaxed);
        if let Err(e) = shared_session.load_vmfb(vmfb_data) {
            let _ = std::fs::remove_file(&cached_vmfb);
            self.jit_fail(name, &format!("JIT load: {}", e));
            return None;
        }

        // Populate captured_scalars so the dispatcher knows which scalars were baked.
        sig.captured_scalars = filtered_constants.clone();

        Some(sig)
    }
}
