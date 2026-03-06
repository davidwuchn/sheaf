// Copyright (c) 2025 Damien Boureille
// Licensed under the MIT License.

//! Shape discovery via interpreter tracing for the build pipeline.

use std::process::exit;

/// Interpret a runner file to discover concrete shapes for function parameters.
/// Per-function type configs from traced call records.
pub(super) type TracedConfigs = std::collections::HashMap<
    String,
    Vec<(String, sheaf_compiler::StableHLOType, std::collections::BTreeMap<Vec<String>, Vec<usize>>)>,
>;
/// Per-function scalar constants from config dicts: fn_name → { (param, indices) → f64 }
pub(super) type TracedConstants = std::collections::HashMap<
    String,
    std::collections::HashMap<(String, Vec<usize>), f64>,
>;

/// Trace the runner file to discover argument shapes and scalar config values.
pub(super) fn trace_with_runner(
    runner_path: &std::path::Path,
    compiler: &sheaf_compiler::core::compiler::CompilerContext,
    target_fns: &[String],
    verbosity: u8,
) -> (TracedConfigs, TracedConstants) {
    use std::collections::HashMap;
    use sheaf_compiler::compiler::layout_to_index_map;
    use sheaf_compiler::core::trace::{value_to_param_layout, value_to_stablehlo_type};
    use sheaf_compiler::interpreter::builtins::register_builtins;
    use sheaf_compiler::interpreter::env::Env;
    use sheaf_compiler::interpreter::value::Value;
    use sheaf_compiler::StableHLOType;

    let runner_abs = runner_path
        .canonicalize()
        .unwrap_or_else(|_| runner_path.to_path_buf());

    let source = std::fs::read_to_string(&runner_abs).unwrap_or_else(|e| {
        eprintln!("sheaf build: cannot read runner '{}': {}", runner_path.display(), e);
        exit(1);
    });
    sheaf_compiler::core::error_format::register_source(
        runner_abs.to_str().unwrap_or("<trace-with>"),
        &source,
    );

    if verbosity >= 1 {
        println!("Tracing with '{}'...", runner_path.display());
    }

    // Compile the runner file (which will (use ...) the target module)
    let mut trace_compiler = sheaf_compiler::core::compiler::CompilerContext::new();
    trace_compiler.disable_vmfb = true; // pure interpreter during tracing
    if let Some(dir) = runner_abs.parent() {
        trace_compiler.current_dir = Some(dir.to_path_buf());
    }
    // Copy load_path from the build compiler so (use ...) resolves identically
    trace_compiler.load_path = compiler.load_path.clone();

    let exprs = sheaf_compiler::parse(
        &source,
        runner_abs.to_str().unwrap_or("<trace-with>"),
    ).unwrap_or_else(|e| {
        eprint!("{}", sheaf_compiler::core::error_format::format_error(&e));
        exit(1);
    });

    let mut compiled = Vec::new();
    for expr in &exprs {
        match trace_compiler.compile(expr) {
            Ok(c) => compiled.push(c),
            Err(e) => {
                eprint!("{}", sheaf_compiler::core::error_format::format_error(&e));
                exit(1);
            }
        }
    }

    // Interpret with call recording enabled and print suppressed
    let mut env = Env::with_registry(trace_compiler.registry.clone());
    register_builtins(&mut env);
    env.call_records = Some(HashMap::new());
    // Suppress print output during tracing (io must remain for entropy/random)
    env.set_builtin("print", |_args, _kwargs| {
        Ok(sheaf_compiler::interpreter::value::Value::Nil)
    });

    for c in &compiled {
        if !matches!(c, sheaf_compiler::core::compiler::CompiledExpr::Nil) {
            if let Err(e) = sheaf_compiler::interpreter::eval(c, &mut env) {
                eprint!("{}", sheaf_compiler::core::error_format::format_error(&e));
                exit(1);
            }
        }
    }

    // Extract per-function, per-param configs from recorded calls
    let records = env.call_records.take().unwrap_or_default();
    let mut result: TracedConfigs = HashMap::new();
    let mut constants: TracedConstants = HashMap::new();

    for fn_name in target_fns {
        let record = match records.get(fn_name) {
            Some(r) => r,
            None => {
                if verbosity >= 1 {
                    eprintln!("  trace: no call recorded for '{}' — skipping", fn_name);
                }
                continue;
            }
        };

        let func_def = match trace_compiler.registry.get(fn_name) {
            Some(fd) => fd,
            None => continue,
        };

        let mut fn_configs = Vec::new();
        let mut fn_constants: std::collections::HashMap<(String, Vec<usize>), f64> = std::collections::HashMap::new();
        for (param_name, arg_val) in func_def.params.iter().zip(record.arg_values.iter()) {
            let ty = value_to_stablehlo_type(arg_val).unwrap_or(StableHLOType::scalar_f32());

            // If the arg is a Dict, generate a ParamLayout + index_map for lowering
            if let Some(layout) = value_to_param_layout(param_name, arg_val) {
                let tuple_ty = sheaf_compiler::forms::ml::param_layout_to_stablehlo_type(&layout);
                if verbosity >= 1 {
                    let ty_str = tuple_ty.to_mlir();
                    let display = if verbosity >= 2 { &ty_str } else { super::compact_type(&ty_str) };
                    println!("  trace: '{}' param '{}' → {}{} ({} fields)",
                        fn_name, param_name, display,
                        if verbosity < 2 && ty_str.len() > 80 { "…" } else { "" }, layout.fields.len());
                }
                let imap = layout_to_index_map(&layout);
                // Extract scalar constants from config-like dicts (all-scalar values)
                extract_scalar_constants(arg_val, param_name, &imap, &mut fn_constants);
                fn_configs.push((param_name.clone(), tuple_ty, imap));
                continue;
            }

            // Fallback for dicts containing lists (e.g. {"head": {...}, "layers": [...]}).
            // value_to_param_layout fails on lists, but we can still build a top-level
            // index map from the sorted dict keys so lower_get_calls can work.
            let index_map = if let Value::Dict(map) = arg_val {
                let mut imap: std::collections::BTreeMap<Vec<String>, Vec<usize>> = std::collections::BTreeMap::new();
                let keys: Vec<&String> = map.keys().collect(); // BTreeMap: already sorted
                for (idx, key) in keys.iter().enumerate() {
                    imap.insert(vec![key.to_string()], vec![idx]);
                    // Nested dict: add second-level entries for field access
                    if let Value::Dict(sub) = &map[*key] {
                        let sub_keys: Vec<&String> = sub.keys().collect();
                        for (sub_idx, sub_key) in sub_keys.iter().enumerate() {
                            imap.insert(
                                vec![key.to_string(), sub_key.to_string()],
                                vec![idx, sub_idx],
                            );
                        }
                    }
                    // List of dicts: add element-level key entries so tuple_key_layouts
                    // can propagate the layout into reduce/scan lambda bodies.
                    if let Value::List(items) = &map[*key] {
                        if let Some(Value::Dict(elem_dict)) = items.first() {
                            let elem_keys: Vec<&String> = elem_dict.keys().collect();
                            for (elem_idx, elem_key) in elem_keys.iter().enumerate() {
                                imap.insert(
                                    vec![key.to_string(), elem_key.to_string()],
                                    vec![idx, elem_idx],
                                );
                                // Recurse into nested dicts for 3-level paths
                                if let Some(Value::Dict(sub_dict)) = elem_dict.get(*elem_key) {
                                    let sub_keys: Vec<&String> = sub_dict.keys().collect();
                                    for (sub_idx, sub_key) in sub_keys.iter().enumerate() {
                                        imap.insert(
                                            vec![key.to_string(), elem_key.to_string(), sub_key.to_string()],
                                            vec![idx, elem_idx, sub_idx],
                                        );
                                    }
                                }
                            }
                        }
                    }
                }
                if verbosity >= 1 {
                    let ty_str = ty.to_mlir();
                    let display = if verbosity >= 2 { &ty_str } else { super::compact_type(&ty_str) };
                    println!("  trace: '{}' param '{}' → {}{} ({} keys)",
                        fn_name, param_name, display,
                        if verbosity < 2 && ty_str.len() > 80 { "…" } else { "" }, imap.len());
                }
                extract_scalar_constants(arg_val, param_name, &imap, &mut fn_constants);
                imap
            } else {
                std::collections::BTreeMap::new()
            };

            if verbosity >= 1 {
                let ty_str = ty.to_mlir();
                let display = if verbosity >= 2 { &ty_str } else { super::compact_type(&ty_str) };
                println!("  trace: '{}' param '{}' → {}{}", fn_name, param_name,
                    display, if verbosity < 2 && ty_str.len() > 80 { "…" } else { "" });
            }
            fn_configs.push((param_name.clone(), ty, index_map));
        }

        if !fn_constants.is_empty() {
            constants.insert(fn_name.clone(), fn_constants);
        }
        result.insert(fn_name.clone(), fn_configs);
    }

    (result, constants)
}

/// Extract scalar values from a dict Value for compile-time constant propagation.
/// Recurses into nested dicts, building key paths to match against the index map.
fn extract_scalar_constants(
    val: &sheaf_compiler::interpreter::value::Value,
    param_name: &str,
    index_map: &std::collections::BTreeMap<Vec<String>, Vec<usize>>,
    out: &mut std::collections::HashMap<(String, Vec<usize>), f64>,
) {
    extract_scalars_rec(val, param_name, &mut vec![], index_map, out);
}

fn extract_scalars_rec(
    val: &sheaf_compiler::interpreter::value::Value,
    param_name: &str,
    path: &mut Vec<String>,
    index_map: &std::collections::BTreeMap<Vec<String>, Vec<usize>>,
    out: &mut std::collections::HashMap<(String, Vec<usize>), f64>,
) {
    use sheaf_compiler::interpreter::value::Value;
    let scalar = match val {
        Value::Int(n) => Some(*n as f64),
        Value::Float(f) => Some(*f),
        Value::Dict(map) => {
            for (key, child) in map {
                path.push(key.clone());
                extract_scalars_rec(child, param_name, path, index_map, out);
                path.pop();
            }
            return;
        }
        _ => return,
    };
    if let (Some(v), Some(indices)) = (scalar, index_map.get(path)) {
        out.insert((param_name.to_string(), indices.clone()), v);
    }
}
