// Copyright (c) 2025 Damien Boureille
// Licensed under the MIT License.

//! Shape discovery via interpreter tracing for the build pipeline.

use std::collections::HashMap;
use std::process::exit;

use sheaf_compiler::compiler::layout_to_index_map;
use sheaf_compiler::core::compiler::CompilerContext;
use sheaf_compiler::core::trace::{value_to_param_layout, value_to_stablehlo_type};
use sheaf_compiler::interpreter::builtins::register_builtins;
use sheaf_compiler::interpreter::env::{CallRecord, Env};
use sheaf_compiler::interpreter::value::Value;
use sheaf_compiler::StableHLOType;

/// Per-function type configs from traced call records.
pub(super) type TracedConfigs = HashMap<
    String,
    Vec<(String, StableHLOType, std::collections::BTreeMap<Vec<String>, Vec<usize>>)>,
>;
/// Per-function scalar constants from config dicts: fn_name → { (param, indices) → f64 }
pub(super) type TracedConstants = HashMap<
    String,
    HashMap<(String, Vec<usize>), f64>,
>;

/// Trace the runner file to discover argument shapes and scalar config values.
pub(super) fn trace_with_runner(
    runner_path: &std::path::Path,
    compiler: &CompilerContext,
    target_fns: &[String],
    verbosity: u8,
) -> (TracedConfigs, TracedConstants) {
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

    let mut trace_compiler = CompilerContext::new();
    trace_compiler.disable_vmfb = true;
    if let Some(dir) = runner_abs.parent() {
        trace_compiler.current_dir = Some(dir.to_path_buf());
    }
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

    let mut env = Env::with_registry(trace_compiler.registry.clone());
    register_builtins(&mut env);
    env.call_records = Some(HashMap::new());
    env.set_builtin("print", |_args, _kwargs| Ok(Value::Nil));

    for c in &compiled {
        if !matches!(c, sheaf_compiler::core::compiler::CompiledExpr::Nil) {
            if let Err(e) = sheaf_compiler::interpreter::eval(c, &mut env) {
                eprint!("{}", sheaf_compiler::core::error_format::format_error(&e));
                exit(1);
            }
        }
    }

    let records = env.call_records.take().unwrap_or_default();
    extract_traced_configs(&records, &trace_compiler.registry, target_fns, verbosity)
}

/// Auto-trace: interpret source files to discover argument shapes automatically.
///
/// Runs each source file through the interpreter with call recording enabled.
/// Merges call records across files and stops early when all target functions
/// have been observed.
pub(super) fn auto_trace(
    source_files: &[std::path::PathBuf],
    compiler: &CompilerContext,
    target_fns: &[String],
    verbosity: u8,
) -> (TracedConfigs, TracedConstants) {
    let mut all_records: HashMap<String, CallRecord> = HashMap::new();

    eprintln!("sheaf build: discovering shapes...");

    for src_path in source_files {
        let abs_path = src_path
            .canonicalize()
            .unwrap_or_else(|_| src_path.to_path_buf());

        let source = match std::fs::read_to_string(&abs_path) {
            Ok(s) => s,
            Err(_) => continue,
        };
        sheaf_compiler::core::error_format::register_source(
            abs_path.to_str().unwrap_or("<auto-trace>"),
            &source,
        );

        let mut trace_compiler = CompilerContext::new();
        trace_compiler.disable_vmfb = true;
        if let Some(dir) = abs_path.parent() {
            trace_compiler.current_dir = Some(dir.to_path_buf());
        }
        trace_compiler.load_path = compiler.load_path.clone();

        let exprs = match sheaf_compiler::parse(
            &source,
            abs_path.to_str().unwrap_or("<auto-trace>"),
        ) {
            Ok(e) => e,
            Err(_) => continue,
        };

        let mut compiled = Vec::new();
        let mut compile_ok = true;
        for expr in &exprs {
            match trace_compiler.compile(expr) {
                Ok(c) => compiled.push(c),
                Err(_) => { compile_ok = false; break; }
            }
        }
        if !compile_ok { continue; }

        // Compute local targets: target functions defined in this file
        // that haven't been observed yet.
        let local_targets: std::collections::HashSet<String> = target_fns.iter()
            .filter(|f| trace_compiler.registry.contains_key(f.as_str()))
            .filter(|f| !all_records.contains_key(f.as_str()))
            .cloned()
            .collect();

        let mut env = Env::with_registry(trace_compiler.registry.clone());
        register_builtins(&mut env);
        env.call_records = Some(HashMap::new());
        env.eval_deadline = Some(std::time::Instant::now() + std::time::Duration::from_secs(30));
        if !local_targets.is_empty() {
            env.trace_targets = Some(local_targets);
        }
        env.set_builtin("print", |_args, _kwargs| Ok(Value::Nil));

        let fname = src_path.file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("?");

        let mut interpret_ok = true;
        for c in &compiled {
            if matches!(c, sheaf_compiler::core::compiler::CompiledExpr::Nil) {
                continue;
            }
            match sheaf_compiler::interpreter::eval(c, &mut env) {
                Ok(_) => {}
                Err(ref e) if e.to_string().contains("trace complete") => {
                    break;
                }
                Err(e) => {
                    if verbosity >= 1 {
                        eprintln!("  auto-trace: {} failed: {}", fname, e);
                    }
                    interpret_ok = false;
                    break;
                }
            }
        }

        // Merge call records even if interpretation failed partway —
        // we may have recorded some calls before the error.
        if let Some(records) = env.call_records.take() {
            let new_count = records.keys()
                .filter(|k| !all_records.contains_key(*k))
                .count();
            for (name, record) in records {
                all_records.entry(name).or_insert(record);
            }
            if new_count > 0 {
                if verbosity >= 1 {
                    let status = if interpret_ok { "ok" } else { "partial" };
                    eprintln!("  {} — {} new function(s) traced ({})", fname, new_count, status);
                }
            }
        }

        // Early exit: all target functions covered
        if target_fns.iter().all(|f| all_records.contains_key(f)) {
            break;
        }
    }

    let traced_count = target_fns.iter().filter(|f| all_records.contains_key(f.as_str())).count();
    if traced_count > 0 {
        eprintln!("  {} of {} function(s) traced", traced_count, target_fns.len());
    } else {
        eprintln!("  no function calls observed");
    }

    extract_traced_configs(&all_records, &compiler.registry, target_fns, verbosity)
}

/// Extract per-function, per-param type configs and scalar constants from call records.
fn extract_traced_configs(
    records: &HashMap<String, CallRecord>,
    registry: &HashMap<String, sheaf_compiler::core::compiler::FunctionDef>,
    target_fns: &[String],
    verbosity: u8,
) -> (TracedConfigs, TracedConstants) {
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

        let func_def = match registry.get(fn_name) {
            Some(fd) => fd,
            None => continue,
        };

        let mut fn_configs = Vec::new();
        let mut fn_constants: HashMap<(String, Vec<usize>), f64> = HashMap::new();
        for (param_name, arg_val) in func_def.params.iter().zip(record.arg_values.iter()) {
            let ty = value_to_stablehlo_type(arg_val).unwrap_or(StableHLOType::scalar_f32());

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
                extract_scalar_constants(arg_val, param_name, &imap, &mut fn_constants);
                fn_configs.push((param_name.clone(), tuple_ty, imap));
                continue;
            }

            // Fallback for dicts containing lists (e.g. {"head": {...}, "layers": [...]}).
            let index_map = if let Value::Dict(map) = arg_val {
                let mut imap: std::collections::BTreeMap<Vec<String>, Vec<usize>> = std::collections::BTreeMap::new();
                let keys: Vec<&String> = map.keys().collect();
                for (idx, key) in keys.iter().enumerate() {
                    imap.insert(vec![key.to_string()], vec![idx]);
                    if let Value::Dict(sub) = &map[*key] {
                        let sub_keys: Vec<&String> = sub.keys().collect();
                        for (sub_idx, sub_key) in sub_keys.iter().enumerate() {
                            imap.insert(
                                vec![key.to_string(), sub_key.to_string()],
                                vec![idx, sub_idx],
                            );
                        }
                    }
                    if let Value::List(items) = &map[*key] {
                        if let Some(Value::Dict(elem_dict)) = items.first() {
                            let elem_keys: Vec<&String> = elem_dict.keys().collect();
                            for (elem_idx, elem_key) in elem_keys.iter().enumerate() {
                                imap.insert(
                                    vec![key.to_string(), elem_key.to_string()],
                                    vec![idx, elem_idx],
                                );
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

use sheaf_compiler::compiler::transforms::extract_scalar_constants;
