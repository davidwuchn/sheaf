// Copyright (c) 2025 Damien Boureille
// Licensed under the MIT License.

//! Sheaf CLI
//!
//! Usage:
//!   sheaf                               Launch interactive REPL
//!   sheaf file.shf                      Interpret a Sheaf file
//!   sheaf -c '(+ 1 2)'                 Evaluate an expression
//!   sheaf build                         Compile all pure functions in current directory
//!   sheaf build dir/                    Compile all pure functions in directory
//!   sheaf build file.shf               Compile pure functions from a single file
//!   sheaf build -r                      Recursive directory scan
//!   sheaf build -o out.vmfb            Override output filename

mod repl;

use std::process::exit;
use sheaf_compiler::core::compiler::CompiledExpr;

fn main() {
    let args: Vec<String> = std::env::args().collect();

    match args.get(1).map(|s| s.as_str()) {
        Some("build") => run_build(&args[2..]),
        Some("init-ai") => run_init_ai(),

        Some("-c") => {
            let expr = args[2..].join(" ");
            if expr.is_empty() {
                eprintln!("sheaf: -c requires an expression");
                exit(1);
            }
            run_expr(&expr);
        }

        Some("--help") | Some("-h") => print_help(),

        Some("--version") => println!("Sheaf {}", env!("CARGO_PKG_VERSION")),

        Some(arg) if !arg.starts_with('-') => run_file(&args[1..]),

        None => run_repl(),

        Some(arg) => {
            eprintln!("sheaf: unknown command '{}'", arg);
            eprintln!("Run 'sheaf --help' for usage.");
            exit(1);
        }
    }
}

fn print_help() {
    println!(
        "Sheaf {} - A Functional Language for Differentiable Computation

Usage:
    sheaf                              Launch interactive REPL
    sheaf FILE [OPTIONS]               Interpret a Sheaf file
    sheaf -c EXPR [OPTIONS]            Evaluate an expression
    sheaf build [DIR|FILE] [OPTIONS]   Compile pure functions to VMFB
    sheaf init-ai                      Generate sheaf-context.md for AI assistants

Interpreter options:
    --trace [FUNCTIONS]    Trace execution (optionally scoped to functions)
    --guard SPEC           Runtime guard: [scope:]variable:check (repeatable)

Build options:
    -o OUTPUT              Output file (default: compiled-functions.vmfb)
    -r                     Scan directories recursively
    -S                     Emit MLIR only, do not invoke iree-compile
    --backend BACKEND      IREE target backend (default: llvm-cpu)
    -v, --verbose          Verbose output

Examples:
    sheaf script.shf
    sheaf -c '(+ 1 2)'
    sheaf build                             Compile all .shf in current dir
    sheaf build examples/hydra/             Compile all .shf in directory
    sheaf build model.shf                   Compile a single file
    sheaf build -r                          Recursive scan
    sheaf build -o model.vmfb              Override output name
    sheaf build model.shf -o model.mlir -S  Emit MLIR only

SDK:
    'sheaf build' without -S requires iree-compile from the Sheaf SDK.
    Set IREE_COMPILE=/path/to/iree-compile to override the default location.",
        env!("CARGO_PKG_VERSION")
    );
}

fn is_silent_result(val: &sheaf_compiler::interpreter::value::Value) -> bool {
    use sheaf_compiler::interpreter::value::Value;
    match val {
        Value::Nil => true,
        Value::List(items) => items.iter().all(|v| matches!(v, Value::Nil)),
        _ => false,
    }
}

fn run_expr(source: &str) {
    use sheaf_compiler::interpreter::eval::eval_source;
    match eval_source(source) {
        Ok(val) => println!("{}", val),
        Err(e) => {
            eprint!("{}", sheaf_compiler::core::error_format::format_error(&e));
            exit(1);
        }
    }
}

fn run_file(args: &[String]) {
    use std::path::PathBuf;
    use sheaf_compiler::interpreter::eval::{eval_source_with_path, eval_source_with_tracing};
    use sheaf_compiler::interpreter::tracer::{CliGuard, LogFormat, TraceLevel, TracerConfig};

    let path = &args[0];

    let mut trace_enabled = false;
    let mut trace_scope: Option<Vec<String>> = None;
    let mut trace_level = TraceLevel::Normal;
    let mut trace_format = LogFormat::Console;
    let mut cli_guards: Vec<CliGuard> = Vec::new();

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--trace" => {
                trace_enabled = true;
                // Optional: comma-separated function names
                if args.get(i + 1).map(|a| !a.starts_with('-')).unwrap_or(false) {
                    i += 1;
                    trace_scope = Some(args[i].split(',').map(|s| s.to_string()).collect());
                }
            }
            "--trace-level" => {
                i += 1;
                match args.get(i).map(|s| s.as_str()) {
                    Some("fast") => trace_level = TraceLevel::Fast,
                    Some("normal") => trace_level = TraceLevel::Normal,
                    Some("verbose") => trace_level = TraceLevel::Verbose,
                    _ => {
                        eprintln!("sheaf: --trace-level expects fast|normal|verbose");
                        exit(1);
                    }
                }
            }
            "--trace-out" => {
                i += 1;
                match args.get(i).map(|s| s.as_str()) {
                    Some("console") => trace_format = LogFormat::Console,
                    Some("json") => trace_format = LogFormat::Json,
                    _ => {
                        eprintln!("sheaf: --trace-out expects console|json");
                        exit(1);
                    }
                }
            }
            "--guard" => {
                i += 1;
                match args.get(i) {
                    Some(spec) => match parse_guard_spec(spec) {
                        Ok(guard) => cli_guards.push(guard),
                        Err(msg) => {
                            eprintln!("sheaf: invalid guard spec '{}': {}", spec, msg);
                            exit(1);
                        }
                    }
                    None => {
                        eprintln!("sheaf: --guard requires a SPEC argument");
                        exit(1);
                    }
                }
            }
            arg => {
                eprintln!("sheaf: unknown option '{}' for file mode", arg);
                eprintln!("Run 'sheaf --help' for usage.");
                exit(1);
            }
        }
        i += 1;
    }

    let abs_path = PathBuf::from(path)
        .canonicalize()
        .unwrap_or_else(|_| PathBuf::from(path));

    let source = match std::fs::read_to_string(&abs_path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("sheaf: cannot read '{}': {}", path, e);
            exit(1);
        }
    };
    sheaf_compiler::core::error_format::register_source(
        abs_path.to_str().unwrap_or(path),
        &source,
    );

    let needs_tracing = trace_enabled || !cli_guards.is_empty();
    let result = if needs_tracing {
        let config = TracerConfig {
            enabled: trace_enabled,
            scope_filter: trace_scope,
            level: trace_level,
            format: trace_format,
            cli_guards,
        };
        eval_source_with_tracing(&source, Some(&abs_path), config)
    } else {
        eval_source_with_path(&source, Some(&abs_path))
    };

    match result {
        Ok(val) => {
            if !is_silent_result(&val) {
                println!("{}", val);
            }
        }
        Err(e) => {
            eprint!("{}", sheaf_compiler::core::error_format::format_error(&e));
            exit(1);
        }
    }
}

/// Parse a guard spec: `[scope:]check[:args]`
/// Examples: `no-nan`, `loss:no-nan`, `range:0:10`, `forward:range:-1:1`
fn parse_guard_spec(spec: &str) -> Result<sheaf_compiler::interpreter::tracer::CliGuard, String> {
    use sheaf_compiler::core::compiler::GuardCheck;
    use sheaf_compiler::interpreter::tracer::CliGuard;

    let parts: Vec<&str> = spec.split(':').collect();
    match parts.as_slice() {
        ["no-nan"] => Ok(CliGuard { scope: None, check: GuardCheck::NoNan }),
        [scope, "no-nan"] => Ok(CliGuard { scope: Some(scope.to_string()), check: GuardCheck::NoNan }),
        ["range", lo, hi] => {
            let lo: f64 = lo.parse().map_err(|_| format!("invalid lo: {}", lo))?;
            let hi: f64 = hi.parse().map_err(|_| format!("invalid hi: {}", hi))?;
            Ok(CliGuard { scope: None, check: GuardCheck::Range { lo, hi } })
        }
        [scope, "range", lo, hi] => {
            let lo: f64 = lo.parse().map_err(|_| format!("invalid lo: {}", lo))?;
            let hi: f64 = hi.parse().map_err(|_| format!("invalid hi: {}", hi))?;
            Ok(CliGuard { scope: Some(scope.to_string()), check: GuardCheck::Range { lo, hi } })
        }
        _ => Err("expected [scope:]no-nan or [scope:]range:lo:hi".to_string()),
    }
}

fn run_repl() {
    repl::run();
}

fn compact_type(mlir: &str) -> &str {
    if mlir.len() <= 80 { mlir } else { &mlir[..80] }
}

fn run_build(args: &[String]) {
    use std::fs;
    use std::path::PathBuf;
    use std::process::Command;

    use sheaf_compiler::compiler::{
        build_index_map, collect_effects, format_effects, json_to_stablehlo_type, lower_get_calls,
    };
    use sheaf_compiler::core::compiler::CompilerContext;
    use sheaf_compiler::{CodeGenerator, StableHLOEmitter, parse};

    if args.first().map(|a| a == "--help" || a == "-h").unwrap_or(false) {
        println!(
            "Usage: sheaf build [DIR|FILE] [OPTIONS]

Compile all pure functions from .shf files into a single VMFB artifact.

    DIR                 Directory to scan for .shf files (default: .)
    FILE                Single .shf file to compile
    -o OUTPUT           Output file (default: compiled-functions.vmfb)
    -r                  Scan directories recursively
    -S                  Emit MLIR only; do not invoke iree-compile
    --config JSON       Shape config for dict params (manual)
    --trace-with FILE   Interpret FILE to discover shapes automatically
    --backend B         IREE target backend (default: llvm-cpu)
    -v, --verbose       Verbose output (full verbosity: -vv)

Examples:
    sheaf build                         Compile all .shf in current dir
    sheaf build examples/hydra/         Compile all .shf in directory
    sheaf build model.shf               Compile a single file
    sheaf build -r -o model.vmfb        Recursive scan, custom output

Without -S, requires iree-compile (Sheaf SDK).
Set IREE_COMPILE=/path/to/iree-compile to override."
        );
        return;
    }

    let mut input: Option<PathBuf> = None;
    let mut output: Option<PathBuf> = None;
    let mut emit_mlir_only = false;
    let mut recursive = false;
    let mut backend = "llvm-cpu".to_string();
    let mut verbosity: u8 = 0;
    let mut config_json: Option<serde_json::Value> = None;
    let mut trace_with: Option<PathBuf> = None;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "-o" => {
                i += 1;
                if i >= args.len() {
                    eprintln!("sheaf build: -o requires an argument");
                    exit(1);
                }
                output = Some(PathBuf::from(&args[i]));
            }
            "-S" => emit_mlir_only = true,
            "-r" => recursive = true,
            "--backend" => {
                i += 1;
                if i >= args.len() {
                    eprintln!("sheaf build: --backend requires an argument");
                    exit(1);
                }
                backend = args[i].clone();
            }
            "-v" | "--verbose" => verbosity += 1,
            "-vv" => verbosity = 2,
            "--config" => {
                i += 1;
                if i >= args.len() {
                    eprintln!("sheaf build: --config requires a JSON argument");
                    exit(1);
                }
                config_json = Some(serde_json::from_str(&args[i]).unwrap_or_else(|e| {
                    eprintln!("sheaf build: --config: invalid JSON: {}", e);
                    exit(1);
                }));
            }
            "--trace-with" => {
                i += 1;
                if i >= args.len() {
                    eprintln!("sheaf build: --trace-with requires a file argument");
                    exit(1);
                }
                trace_with = Some(PathBuf::from(&args[i]));
            }
            "--trace" | "--guard" => {
                eprintln!(
                    "sheaf build: '{}' is an interpreter option and cannot be used with 'build'",
                    args[i]
                );
                exit(1);
            }
            arg if !arg.starts_with('-') => {
                input = Some(PathBuf::from(arg));
            }
            arg => {
                eprintln!("sheaf build: unknown option '{}'", arg);
                eprintln!("Run 'sheaf build --help' for usage.");
                exit(1);
            }
        }
        i += 1;
    }

    // Resolve input: default to current directory
    let input = input.unwrap_or_else(|| PathBuf::from("."));

    // Collect .shf source files
    let source_files: Vec<PathBuf> = if input.is_dir() {
        collect_shf_files(&input, recursive)
    } else if input.is_file() {
        vec![input.clone()]
    } else {
        eprintln!("sheaf build: '{}' is not a file or directory", input.display());
        exit(1);
    };

    if source_files.is_empty() {
        eprintln!("sheaf build: no .shf files found in '{}'", input.display());
        exit(1);
    }

    // Resolve output directory (where compiled-functions.vmfb goes)
    let output_dir = if input.is_dir() {
        input.canonicalize().unwrap_or_else(|_| input.clone())
    } else {
        input.parent().map(|p| p.to_path_buf()).unwrap_or_else(|| PathBuf::from("."))
            .canonicalize().unwrap_or_else(|_| PathBuf::from("."))
    };

    let output = output.unwrap_or_else(|| {
        if emit_mlir_only {
            output_dir.join("compiled-functions.mlir")
        } else {
            output_dir.join("compiled-functions.vmfb")
        }
    });

    // Detect output format from extension when -S not set
    let emit_mlir_only = emit_mlir_only
        || output.extension().and_then(|e| e.to_str()) == Some("mlir");

    eprintln!("sheaf build: scanning {}", if input.is_dir() {
        input.display().to_string()
    } else {
        input.file_name().unwrap_or_default().to_string_lossy().to_string()
    });

    // Parse and compile all source files into a single compiler context
    let mut compiler = CompilerContext::new();
    if let Some(dir) = output.parent() {
        compiler.current_dir = Some(dir.to_path_buf());
    }
    let mut all_compiled_exprs = Vec::new();
    // Track which functions come from which file
    let mut file_functions: Vec<(PathBuf, Vec<String>)> = Vec::new();

    for src_path in &source_files {
        let source = fs::read_to_string(src_path).unwrap_or_else(|e| {
            eprintln!("sheaf build: cannot read '{}': {}", src_path.display(), e);
            exit(1);
        });
        sheaf_compiler::core::error_format::register_source(
            src_path.to_str().unwrap_or("<unknown>"),
            &source,
        );

        let exprs = parse(&source, src_path.to_str().unwrap_or("<unknown>")).unwrap_or_else(|e| {
            eprint!("{}", sheaf_compiler::core::error_format::format_error(&e));
            exit(1);
        });

        if exprs.is_empty() {
            continue;
        }

        // Set current_dir to the file's directory for (use ...) resolution
        if let Some(dir) = src_path.canonicalize().ok().and_then(|p| p.parent().map(|d| d.to_path_buf())) {
            compiler.current_dir = Some(dir);
        }

        let mut fn_names = Vec::new();
        for expr in &exprs {
            // Collect defn names from this file
            if let Some(name) = expr.as_list()
                .and_then(|l| l.first())
                .and_then(|h| h.as_symbol())
                .filter(|&s| s == "defn")
                .and_then(|_| expr.as_list().and_then(|l| l.get(1)).and_then(|n| n.as_symbol()))
            {
                fn_names.push(name.to_string());
            }

            match compiler.compile(expr) {
                Ok(c) => all_compiled_exprs.push(c),
                Err(e) => {
                    eprint!("{}", sheaf_compiler::core::error_format::format_error(&e));
                    exit(1);
                }
            }
        }

        if !fn_names.is_empty() {
            file_functions.push((src_path.clone(), fn_names));
        }
    }

    let all_fn_names: Vec<String> = {
        let mut names: Vec<String> = file_functions.iter()
            .flat_map(|(_, names)| names.clone())
            .collect();
        let mut seen = std::collections::HashSet::new();
        names.retain(|n| seen.insert(n.clone()));
        names
    };

    if all_fn_names.is_empty() {
        eprintln!("sheaf build: no functions defined in scanned files");
        exit(1);
    }

    let extra_decls = resolve_vag_decls(&compiler, &all_compiled_exprs, verbosity);

    let mut all_decls = extra_decls;

    // Trace-driven shape discovery: interpret the runner file to observe concrete arg shapes
    let (traced_configs, traced_constants) = match trace_with {
        Some(runner_path) => {
            let (configs, constants) = trace_with_runner(&runner_path, &compiler, &all_fn_names, verbosity);
            (Some(configs), constants)
        }
        None => (None, std::collections::HashMap::new()),
    };

    // Build per-param config from --config JSON:
    // config_json top level: {"param_name": {dict structure}, ...}
    // e.g. {"p": {"l1": {"W": [2,8], "b": [8]}, "l2": {"W": [8,1], "b": [1]}}}
    let param_configs: Vec<(String, serde_json::Value)> = match &config_json {
        Some(serde_json::Value::Object(map)) => map
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect(),
        Some(_) => {
            eprintln!("sheaf build: --config must be a JSON object {{\"param\": {{...}}}}");
            exit(1);
        }
        None => vec![],
    };

    // Track compiled functions for manifest generation + log output
    let mut compiled_functions: Vec<ManifestEntry> = Vec::new();
    let mut compiled_per_file: std::collections::HashMap<String, Vec<String>> = std::collections::HashMap::new();
    let mut skipped_fns: Vec<(String, String, String)> = Vec::new(); // (file, name, reason)

    for name in &all_fn_names {
        let func_def = match compiler.registry.get(name).cloned() {
            Some(f) => f,
            None => continue,
        };
        let body_hash = func_def.body_hash();
        let mut body = match func_def.body_compiled {
            Some(b) => b,
            None => continue, // internal functions (use-imported), not user-defined
        };
        let src_file_for_skip = || -> String {
            file_functions.iter()
                .find(|(_, names)| names.contains(name))
                .map(|(p, _)| p.display().to_string())
                .unwrap_or_else(|| "?".to_string())
        };
        let mut sig = match func_def.signature {
            Some(s) => s,
            None => {
                // Without annotations, try to build a signature from traced call records
                if let Some(ref configs) = traced_configs {
                    if let Some(fn_config) = configs.get(name.as_str()) {
                        use sheaf_compiler::core::inference::FunctionSignature;
                        use sheaf_compiler::StableHLOType;
                        // Build param types from traced args
                        let param_types: Vec<StableHLOType> = func_def.params.iter().map(|p| {
                            fn_config.iter()
                                .find(|(n, _, _)| n == p)
                                .map(|(_, ty, _)| ty.clone())
                                .unwrap_or(StableHLOType::scalar_f32())
                        }).collect();
                        FunctionSignature {
                            param_types,
                            return_type: StableHLOType::scalar_f32(), // will be refined
                            return_dict_keys: None,
                        }
                    } else {
                        skipped_fns.push((src_file_for_skip(), name.clone(), "no type info".to_string()));
                        continue;
                    }
                } else {
                    skipped_fns.push((src_file_for_skip(), name.clone(), "no type info".to_string()));
                    continue;
                }
            }
        };

        // Skip functions with side effects: they cannot be emitted as StableHLO.
        let effects = collect_effects(&body);
        if !effects.is_empty() {
            let src_file = file_functions.iter()
                .find(|(_, names)| names.contains(name))
                .map(|(p, _)| p.display().to_string())
                .unwrap_or_else(|| "?".to_string());
            skipped_fns.push((src_file, name.clone(), format!("side effects: {}", format_effects(&effects))));
            continue;
        }

        // Skip functions that use higher-order functions not supported by the compiler.
        let hof_calls = sheaf_compiler::collect_hof_calls(&body);
        if !hof_calls.is_empty() {
            let src_file = file_functions.iter()
                .find(|(_, names)| names.contains(name))
                .map(|(p, _)| p.display().to_string())
                .unwrap_or_else(|| "?".to_string());
            skipped_fns.push((src_file, name.clone(), format!("higher-order: {}", hof_calls.join(", "))));
            continue;
        }

        // Apply dict-to-tuple lowering for each configured param that appears
        // in this function's parameter list.
        let mut known_types: Vec<(String, sheaf_compiler::StableHLOType)> = Vec::new();
        // Collect index maps for post-inline aliased-get lowering
        let mut param_index_maps: Vec<(String, std::collections::BTreeMap<Vec<String>, Vec<usize>>)> = Vec::new();

        // Source 1: --config JSON (manual)
        for (param_name, param_config) in &param_configs {
            if !func_def.params.contains(param_name) {
                continue;
            }
            let tuple_ty = match json_to_stablehlo_type(param_config) {
                Ok(t) => t,
                Err(e) => {
                    eprintln!("sheaf build: --config error for '{}': {}", param_name, e);
                    exit(1);
                }
            };
            let index_map = build_index_map(param_config);
            if verbosity >= 1 {
                let ty_str = tuple_ty.to_mlir();
                let display = if verbosity >= 2 { &ty_str } else { compact_type(&ty_str) };
                println!("  Lowering '{}' param '{}' → {}{}", name, param_name,
                    display, if verbosity < 2 && ty_str.len() > 80 { "…" } else { "" });
            }
            body = lower_get_calls(&body, param_name, &index_map);
            param_index_maps.push((param_name.clone(), index_map));
            known_types.push((param_name.clone(), tuple_ty));
        }

        // Source 2: --trace-with (automatic shape discovery)
        if let Some(ref configs) = traced_configs {
            if let Some(fn_config) = configs.get(name.as_str()) {
                for (param_name, tuple_ty, index_map) in fn_config {
                    // Skip if already configured by --config JSON
                    if known_types.iter().any(|(n, _)| n == param_name) {
                        continue;
                    }
                    if verbosity >= 1 {
                        let ty_str = tuple_ty.to_mlir();
                        let display = if verbosity >= 2 { &ty_str } else { compact_type(&ty_str) };
                        println!("  Traced '{}' param '{}' → {}{}", name, param_name,
                            display, if verbosity < 2 && ty_str.len() > 80 { "…" } else { "" });
                    }
                    body = lower_get_calls(&body, param_name, index_map);
                    param_index_maps.push((param_name.clone(), index_map.clone()));
                    known_types.push((param_name.clone(), tuple_ty.clone()));
                }
            }
        }

        // Re-infer signature if we have new known types from the config
        if !known_types.is_empty() {
            use sheaf_compiler::core::inference::infer_function_signature_with_known;
            sig = infer_function_signature_with_known(
                &compiler,
                &func_def.params,
                &body,
                &known_types,
            ).unwrap_or_else(|e| {
                eprintln!("type inference error in '{}': {}", name, e);
                exit(1);
            });
            // Override param types for configured params (inference may default to scalar)
            for (param_name, tuple_ty) in &known_types {
                if let Some(idx) = func_def.params.iter().position(|p| p == param_name) {
                    sig.param_types[idx] = tuple_ty.clone();
                }
            }
        }

        // Update registry with lowered body + refined signature so that
        // other functions (e.g. train-step inlining forward) see the lowered version.
        if let Some(fd) = compiler.registry.get_mut(name) {
            fd.body_compiled = Some(body.clone());
            fd.signature = Some(sig.clone());
        }

        // Inline user-defined function calls (e.g. transformer-block → multi-head-attention)
        body = sheaf_compiler::autodiff::inline_function_calls(&body, &compiler.registry);

        // After inlining, re-lower (get param :key) calls from inlined function bodies.
        // These weren't present in the original body, so the first lower_get_calls missed them.
        for (param_name, index_map) in &param_index_maps {
            body = lower_get_calls(&body, param_name, index_map);
        }
        // Also re-lower alias-based gets (Let-bound GetTupleElement).
        body = lower_inlined_gets(&body, &param_index_maps);

        // Substitute config scalar constants (e.g. (static (get config :d_model)) → Integer(256))
        // Also folds (shape X) → [d0, d1, ...] and (get [d0, d1] -2) → Integer(d)
        let empty_consts = std::collections::HashMap::new();
        let fn_consts = traced_constants.get(name.as_str()).unwrap_or(&empty_consts);
        if verbosity >= 1 && !fn_consts.is_empty() {
            for ((p, idx), val) in fn_consts {
                println!("  const: '{}'.{}[{:?}] = {}", name, p, idx, val);
            }
        }
        let param_shapes: std::collections::HashMap<String, Vec<i64>> = func_def.params.iter()
            .zip(sig.param_types.iter())
            .filter_map(|(p, ty)| {
                let shape = ty.shape();
                if shape.is_empty() { None } else { Some((p.clone(), shape.to_vec())) }
            })
            .collect();
        body = resolve_static_constants(&body, fn_consts, &param_shapes);

        // Extract nested key layouts from param index maps so the codegen can resolve
        // (get sym "key") when sym is a Let-bound tuple (e.g. hidden = (get params "hidden")).
        // For each 2-level path [parent, child] → [_, rel_idx], build parent → {child: rel_idx}.
        let mut tuple_key_layouts: std::collections::HashMap<String, std::collections::BTreeMap<String, usize>> = std::collections::HashMap::new();
        // Build index→key reverse map for first-level entries
        let mut idx_to_key: std::collections::HashMap<(String, usize), String> = std::collections::HashMap::new();
        for (param_name, index_map) in &param_index_maps {
            for (key_path, indices) in index_map {
                if key_path.len() == 2 && indices.len() == 2 {
                    tuple_key_layouts
                        .entry(key_path[0].clone())
                        .or_default()
                        .insert(key_path[1].clone(), indices[1]);
                }
                // 3-level paths: build sub-layouts keyed by the 2nd-level key name
                // e.g. ["layers", "attn", "Wq"] → [2, 0, 0] creates layout "attn" → {Wq: 0, ...}
                if key_path.len() == 3 && indices.len() == 3 {
                    tuple_key_layouts
                        .entry(key_path[1].clone())
                        .or_default()
                        .insert(key_path[2].clone(), indices[2]);
                }
                if key_path.len() == 1 && indices.len() == 1 {
                    idx_to_key.insert((param_name.clone(), indices[0]), key_path[0].clone());
                }
            }
        }
        // Propagate layouts to Let-bound variable names.
        // e.g. `input-layer = GetTupleElement("params", [2])` where index 2 = key "input"
        // → copy tuple_key_layouts["input"] to tuple_key_layouts["input-layer"]
        propagate_let_layouts(&body, &idx_to_key, &mut tuple_key_layouts);

        let mut codegen = CodeGenerator::with_function_params(
            compiler.registry.clone(),
            &func_def.params,
            &sig.param_types,
        );
        codegen.set_tuple_key_layouts(tuple_key_layouts);
        codegen.set_idx_to_key(idx_to_key);
        match codegen.emit_func_declaration(name, &body, &sig.param_types, &sig.return_type) {
            Ok((decl, actual_return_ty)) => {
                // Update the signature with the real codegen return type
                if let Some(fd) = compiler.registry.get_mut(name) {
                    if let Some(ref mut sig) = fd.signature {
                        sig.return_type = actual_return_ty.clone();
                    }
                }
                all_decls.push(decl);
                compiled_functions.push(ManifestEntry {
                    name: name.clone(),
                    hash: body_hash.clone(),
                    params: func_def.params.clone(),
                    param_types: sig.param_types.iter().map(|t| t.to_mlir()).collect(),
                    return_type: actual_return_ty.to_mlir(),
                });
                let src_file = file_functions.iter()
                    .find(|(_, names)| names.contains(name))
                    .map(|(p, _)| p.display().to_string())
                    .unwrap_or_else(|| "?".to_string());
                compiled_per_file.entry(src_file).or_default().push(name.clone());
            }
            Err(e) => {
                let src_file = file_functions.iter()
                    .find(|(_, names)| names.contains(name))
                    .map(|(p, _)| p.display().to_string())
                    .unwrap_or_else(|| "?".to_string());
                let reason = if verbosity >= 1 {
                    format!("codegen: {}", e)
                } else {
                    "codegen error (use -v for details)".to_string()
                };
                skipped_fns.push((src_file, name.clone(), reason));
            }
        }
    }

    if all_decls.is_empty() {
        eprintln!("\nsheaf build: no compilable functions found\n");
        let max_name = skipped_fns.iter().map(|(_, n, _)| n.len()).max().unwrap_or(0);
        for (_, name, reason) in &skipped_fns {
            eprintln!("  {:<width$}  skipped  ({})", name, reason, width = max_name);
        }
        let has_no_types = skipped_fns.iter().any(|(_, _, r)| r == "no type info");
        let has_effects = skipped_fns.iter().any(|(_, _, r)| r.starts_with("side effects"));
        if has_no_types {
            eprintln!();
            eprintln!("hint: use --trace-with <runner.shf> or --config to provide tensor shapes");
        }
        if has_effects && !has_no_types {
            eprintln!();
            eprintln!("hint: all functions have side effects — only pure functions can be compiled");
        }
        exit(1);
    }

    // Print build summary
    let n_compiled = compiled_functions.len();
    let n_interpreted = skipped_fns.len();
    {
        // Collect all entries: (name, status, detail)
        let mut entries: Vec<(String, &str, String)> = Vec::new();
        for mf in &compiled_functions {
            entries.push((mf.name.clone(), "compiled", String::new()));
        }
        for (_, name, reason) in &skipped_fns {
            entries.push((name.clone(), "interpreted", format!("({})", reason)));
        }
        let max_name = entries.iter().map(|(n, _, _)| n.len()).max().unwrap_or(0);
        eprintln!();
        for (name, status, detail) in &entries {
            if detail.is_empty() {
                eprintln!("  {:<width$}  {}", name, status, width = max_name);
            } else {
                eprintln!("  {:<width$}  {}  {}", name, status, detail, width = max_name);
            }
        }
        eprintln!();
    }

    let mlir = StableHLOEmitter::emit_module(&all_decls);

    if emit_mlir_only {
        fs::write(&output, &mlir).unwrap_or_else(|e| {
            eprintln!("error writing '{}': {}", output.display(), e);
            exit(1);
        });
        eprintln!("{} compiled, {} interpreted → {}", n_compiled, n_interpreted, output.display());
        return;
    }

    // VMFB path — requires iree-compile
    let iree_compile = find_iree_compile();

    // Write intermediate MLIR to a temp file
    let mlir_path = output.with_extension("mlir");
    fs::write(&mlir_path, &mlir).unwrap_or_else(|e| {
        eprintln!("error writing '{}': {}", mlir_path.display(), e);
        exit(1);
    });

    if verbosity >= 1 {
        eprintln!("running iree-compile ({})...", iree_compile);
    }

    let status = Command::new(&iree_compile)
        .arg(&mlir_path)
        .arg(format!("--iree-hal-target-backends={}", backend))
        .arg("-o")
        .arg(&output)
        .status()
        .unwrap_or_else(|e| {
            eprintln!("error running iree-compile '{}': {}", iree_compile, e);
            eprintln!("Install the Sheaf SDK or set IREE_COMPILE=/path/to/iree-compile");
            exit(1);
        });

    // Clean up temp MLIR
    let _ = fs::remove_file(&mlir_path);

    if !status.success() {
        eprintln!("iree-compile failed");
        exit(1);
    }

    // Write manifest with function hashes
    write_manifest(&output, &compiled_functions, verbosity);

    eprintln!("{} compiled, {} interpreted → {}", n_compiled, n_interpreted, output.display());
}

/// Manifest entry for a compiled function.
struct ManifestEntry {
    name: String,
    hash: String,
    params: Vec<String>,       // param names, in order
    param_types: Vec<String>,  // MLIR type strings, matching params
    return_type: String,       // MLIR type string
}

/// Write manifest.json alongside the VMFB with hashes and inferred signatures.
/// Used by `sheaf run` and `(use)` to detect stale artifacts and load signatures.
fn write_manifest(vmfb_path: &std::path::Path, functions: &[ManifestEntry], verbosity: u8) {
    let dir = vmfb_path.parent().unwrap_or(std::path::Path::new("."));
    let manifest_path = dir.join("manifest.json");
    let vmfb_name = vmfb_path.file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("compiled-functions.vmfb");

    let mut map = serde_json::Map::new();
    map.insert("_comment".into(), serde_json::Value::String(
        "Sheaf build manifest — generated by `sheaf build`, do not edit".into()
    ));
    map.insert("version".into(), serde_json::Value::Number(1.into()));
    map.insert("vmfb".into(), serde_json::Value::String(vmfb_name.into()));

    let mut fns = serde_json::Map::new();
    for entry in functions {
        let mut obj = serde_json::Map::new();
        obj.insert("hash".into(), serde_json::Value::String(entry.hash.clone()));
        let mut params = serde_json::Map::new();
        for (name, ty) in entry.params.iter().zip(entry.param_types.iter()) {
            params.insert(name.clone(), serde_json::Value::String(ty.clone()));
        }
        obj.insert("params".into(), serde_json::Value::Object(params));
        obj.insert("returns".into(), serde_json::Value::String(entry.return_type.clone()));
        fns.insert(entry.name.clone(), serde_json::Value::Object(obj));
    }
    map.insert("functions".into(), serde_json::Value::Object(fns));

    let json = serde_json::to_string_pretty(&serde_json::Value::Object(map)).unwrap();
    if let Err(e) = std::fs::write(&manifest_path, &json) {
        eprintln!("warning: could not write manifest '{}': {}", manifest_path.display(), e);
    } else if verbosity >= 1 {
        eprintln!("Wrote {}", manifest_path.display());
    }
}

/// Interpret a runner file to discover concrete shapes for function parameters.
/// Per-function type configs from traced call records.
type TracedConfigs = std::collections::HashMap<
    String,
    Vec<(String, sheaf_compiler::StableHLOType, std::collections::BTreeMap<Vec<String>, Vec<usize>>)>,
>;
/// Per-function scalar constants from config dicts: fn_name → { (param, indices) → f64 }
type TracedConstants = std::collections::HashMap<
    String,
    std::collections::HashMap<(String, Vec<usize>), f64>,
>;

/// Trace the runner file to discover argument shapes and scalar config values.
fn trace_with_runner(
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
                    let display = if verbosity >= 2 { &ty_str } else { compact_type(&ty_str) };
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
                    let display = if verbosity >= 2 { &ty_str } else { compact_type(&ty_str) };
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
                let display = if verbosity >= 2 { &ty_str } else { compact_type(&ty_str) };
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

/// Substitute known scalar constants and propagate Let-bound constants.
/// Handles: GetTupleElement → Integer, (static expr) → evaluate, Symbol → local constant,
/// and constant folding of arithmetic on known values.
/// Lower (get alias :key) patterns introduced after inlining stdlib functions.
///
/// After inlining, a call like (layer-norm h (GetTupleElement layer-p [1]) 2)
/// becomes (let [p (GetTupleElement layer-p [1])] (let [gamma (get p :gamma)] ...)).
/// This pass tracks such aliases and resolves (get alias :key) to
/// GetTupleElement { param: root_param, indices: full_indices }.
fn lower_inlined_gets(
    body: &CompiledExpr,
    index_maps: &[(String, std::collections::BTreeMap<Vec<String>, Vec<usize>>)],
) -> CompiledExpr {
    // Build reverse map: (param, indices) → key_path for alias tracking
    let mut reverse: std::collections::HashMap<(String, Vec<usize>), Vec<String>> = std::collections::HashMap::new();
    for (param, imap) in index_maps {
        for (path, indices) in imap {
            reverse.insert((param.clone(), indices.clone()), path.clone());
        }
    }
    lower_inlined_gets_rec(body, index_maps, &mut std::collections::HashMap::new(), &reverse)
}

// aliases: symbol → (root_param, prefix_path)
fn lower_inlined_gets_rec(
    expr: &CompiledExpr,
    index_maps: &[(String, std::collections::BTreeMap<Vec<String>, Vec<usize>>)],
    aliases: &mut std::collections::HashMap<String, (String, Vec<String>)>,
    reverse: &std::collections::HashMap<(String, Vec<usize>), Vec<String>>,
) -> CompiledExpr {
    match expr {
        // (get alias :key ...) where alias is a tracked tuple alias
        CompiledExpr::FunctionCall { name, args } if name == "get" && args.len() >= 2 => {
            if let CompiledExpr::Symbol(alias) = &args[0] {
                if let Some((root_param, prefix_path)) = aliases.get(alias) {
                    let mut path = prefix_path.clone();
                    let mut all_kw = true;
                    for arg in &args[1..] {
                        match arg {
                            CompiledExpr::Keyword(k) => path.push(k.clone()),
                            _ => { all_kw = false; break; }
                        }
                    }
                    if all_kw {
                        if let Some(imap) = index_maps.iter().find(|(p, _)| p == root_param).map(|(_, m)| m) {
                            if let Some(indices) = imap.get(&path) {
                                return CompiledExpr::GetTupleElement {
                                    param: root_param.clone(),
                                    indices: indices.clone(),
                                };
                            }
                        }
                    }
                }
            }
            CompiledExpr::FunctionCall {
                name: name.clone(),
                args: args.iter().map(|a| lower_inlined_gets_rec(a, index_maps, aliases, reverse)).collect(),
            }
        }
        CompiledExpr::Let { bindings, body } => {
            let new_bindings: Vec<_> = bindings.iter().map(|(k, v)| {
                let resolved = lower_inlined_gets_rec(v, index_maps, aliases, reverse);
                // Track alias: if the value is GetTupleElement, record what key path it corresponds to
                if let CompiledExpr::GetTupleElement { param, indices } = &resolved {
                    let key = (param.clone(), indices.clone());
                    if let Some(path) = reverse.get(&key) {
                        aliases.insert(k.clone(), (param.clone(), path.clone()));
                    }
                }
                (k.clone(), resolved)
            }).collect();
            CompiledExpr::Let {
                bindings: new_bindings,
                body: Box::new(lower_inlined_gets_rec(body, index_maps, aliases, reverse)),
            }
        }
        CompiledExpr::FunctionCall { name, args } => CompiledExpr::FunctionCall {
            name: name.clone(),
            args: args.iter().map(|a| lower_inlined_gets_rec(a, index_maps, aliases, reverse)).collect(),
        },
        CompiledExpr::Do(exprs) => CompiledExpr::Do(
            exprs.iter().map(|e| lower_inlined_gets_rec(e, index_maps, aliases, reverse)).collect(),
        ),
        CompiledExpr::If { condition, then_branch, else_branch } => CompiledExpr::If {
            condition: Box::new(lower_inlined_gets_rec(condition, index_maps, aliases, reverse)),
            then_branch: Box::new(lower_inlined_gets_rec(then_branch, index_maps, aliases, reverse)),
            else_branch: else_branch.as_ref().map(|e| Box::new(lower_inlined_gets_rec(e, index_maps, aliases, reverse))),
        },
        CompiledExpr::Lambda { params, body } => CompiledExpr::Lambda {
            params: params.clone(),
            body: Box::new(lower_inlined_gets_rec(body, index_maps, aliases, reverse)),
        },
        CompiledExpr::LambdaCall { callee, args } => CompiledExpr::LambdaCall {
            callee: Box::new(lower_inlined_gets_rec(callee, index_maps, aliases, reverse)),
            args: args.iter().map(|a| lower_inlined_gets_rec(a, index_maps, aliases, reverse)).collect(),
        },
        CompiledExpr::Repeat { index_var, count, acc_var, acc_init, body } => CompiledExpr::Repeat {
            index_var: index_var.clone(),
            count: Box::new(lower_inlined_gets_rec(count, index_maps, aliases, reverse)),
            acc_var: acc_var.clone(),
            acc_init: Box::new(lower_inlined_gets_rec(acc_init, index_maps, aliases, reverse)),
            body: Box::new(lower_inlined_gets_rec(body, index_maps, aliases, reverse)),
        },
        CompiledExpr::Vector(elems) => CompiledExpr::Vector(
            elems.iter().map(|e| lower_inlined_gets_rec(e, index_maps, aliases, reverse)).collect(),
        ),
        other => other.clone(),
    }
}

fn resolve_static_constants(
    expr: &CompiledExpr,
    constants: &std::collections::HashMap<(String, Vec<usize>), f64>,
    param_shapes: &std::collections::HashMap<String, Vec<i64>>,
) -> CompiledExpr {
    let mut locals: std::collections::HashMap<String, CompiledExpr> = std::collections::HashMap::new();
    let mut shapes: std::collections::HashMap<String, Vec<i64>> = param_shapes.clone();
    resolve_constants_rec(expr, constants, &mut locals, &mut shapes)
}

fn resolve_constants_rec(
    expr: &CompiledExpr,
    constants: &std::collections::HashMap<(String, Vec<usize>), f64>,
    locals: &mut std::collections::HashMap<String, CompiledExpr>,
    shapes: &mut std::collections::HashMap<String, Vec<i64>>,
) -> CompiledExpr {
    match expr {
        CompiledExpr::GetTupleElement { param, indices } => {
            let key = (param.clone(), indices.clone());
            match constants.get(&key) {
                Some(&val) => f64_to_const(val),
                None => expr.clone(),
            }
        }
        CompiledExpr::Symbol(name) => {
            locals.get(name).cloned().unwrap_or_else(|| expr.clone())
        }
        CompiledExpr::FunctionCall { name, args } if name == "static" && args.len() == 1 => {
            resolve_constants_rec(&args[0], constants, locals, shapes)
        }
        // (shape expr) → Vector([Integer(d0), Integer(d1), ...]) when shape is known
        CompiledExpr::FunctionCall { name, args } if name == "shape" && args.len() == 1 => {
            // Try direct symbol lookup first
            let shape_from_sym = match &args[0] {
                CompiledExpr::Symbol(s) => shapes.get(s.as_str()).cloned(),
                _ => None,
            };
            // Fallback: infer shape from expression (handles inlined code)
            let shape = shape_from_sym.or_else(|| {
                let resolved = resolve_constants_rec(&args[0], constants, locals, shapes);
                try_infer_shape(&resolved, shapes)
            });
            if let Some(sh) = shape {
                return CompiledExpr::Vector(
                    sh.iter().map(|&d| CompiledExpr::Integer(d)).collect(),
                );
            }
            CompiledExpr::FunctionCall {
                name: name.clone(),
                args: args.iter().map(|a| resolve_constants_rec(a, constants, locals, shapes)).collect(),
            }
        }
        // (get Vector idx) → fold with negative index support
        CompiledExpr::FunctionCall { name, args } if name == "get" && args.len() == 2 => {
            let recv = resolve_constants_rec(&args[0], constants, locals, shapes);
            let idx = resolve_constants_rec(&args[1], constants, locals, shapes);
            if let (CompiledExpr::Vector(elems), CompiledExpr::Integer(i)) = (&recv, &idx) {
                let len = elems.len() as i64;
                let norm = if *i < 0 { len + i } else { *i };
                if norm >= 0 && norm < len {
                    return elems[norm as usize].clone();
                }
            }
            CompiledExpr::FunctionCall { name: name.clone(), args: vec![recv, idx] }
        }
        // (first [a b ...]) → a, (last [a b ...]) → last element — fold at compile time
        // Recursively pushes first/last through nested Lets (e.g. inlined functions).
        CompiledExpr::FunctionCall { name, args }
            if (name == "first" || name == "last") && args.len() == 1 =>
        {
            let recv = resolve_constants_rec(&args[0], constants, locals, shapes);
            return push_first_last(name, recv);
        }
        CompiledExpr::FunctionCall { name, args } => {
            let resolved: Vec<_> = args.iter()
                .map(|a| resolve_constants_rec(a, constants, locals, shapes))
                .collect();
            try_fold_arithmetic(name, &resolved)
                .unwrap_or_else(|| CompiledExpr::FunctionCall { name: name.clone(), args: resolved })
        }
        CompiledExpr::Let { bindings, body } => {
            let new_bindings: Vec<_> = bindings.iter().map(|(k, v)| {
                let resolved = resolve_constants_rec(v, constants, locals, shapes);
                match &resolved {
                    CompiledExpr::Integer(_) | CompiledExpr::Float(_) => {
                        locals.insert(k.clone(), resolved.clone());
                    }
                    // Alias: (let [X x]) — propagate shape of x to X
                    CompiledExpr::Symbol(aliased) => {
                        if let Some(sh) = shapes.get(aliased).cloned() {
                            shapes.insert(k.clone(), sh);
                        }
                    }
                    // Known vector shape (e.g. from (shape X)) — track for nested (get (shape ...) -1)
                    CompiledExpr::Vector(elems) if elems.iter().all(|e| matches!(e, CompiledExpr::Integer(_))) => {
                        let sh: Vec<i64> = elems.iter().filter_map(|e| {
                            if let CompiledExpr::Integer(n) = e { Some(*n) } else { None }
                        }).collect();
                        shapes.insert(k.clone(), sh);
                    }
                    _ => {
                        // For FunctionCall/Let RHS (e.g. inlined layer-norm), infer shape transitively
                        if let Some(sh) = try_infer_shape(&resolved, shapes) {
                            shapes.insert(k.clone(), sh);
                        }
                    }
                }
                (k.clone(), resolved)
            }).collect();
            CompiledExpr::Let {
                bindings: new_bindings,
                body: Box::new(resolve_constants_rec(body, constants, locals, shapes)),
            }
        }
        CompiledExpr::Do(exprs) => CompiledExpr::Do(
            exprs.iter().map(|e| resolve_constants_rec(e, constants, locals, shapes)).collect(),
        ),
        CompiledExpr::If { condition, then_branch, else_branch } => CompiledExpr::If {
            condition: Box::new(resolve_constants_rec(condition, constants, locals, shapes)),
            then_branch: Box::new(resolve_constants_rec(then_branch, constants, locals, shapes)),
            else_branch: else_branch.as_ref().map(|e| Box::new(resolve_constants_rec(e, constants, locals, shapes))),
        },
        CompiledExpr::Lambda { params, body } => CompiledExpr::Lambda {
            params: params.clone(),
            body: Box::new(resolve_constants_rec(body, constants, locals, shapes)),
        },
        CompiledExpr::LambdaCall { callee, args } => CompiledExpr::LambdaCall {
            callee: Box::new(resolve_constants_rec(callee, constants, locals, shapes)),
            args: args.iter().map(|a| resolve_constants_rec(a, constants, locals, shapes)).collect(),
        },
        CompiledExpr::Repeat { index_var, count, acc_var, acc_init, body } => CompiledExpr::Repeat {
            index_var: index_var.clone(),
            count: Box::new(resolve_constants_rec(count, constants, locals, shapes)),
            acc_var: acc_var.clone(),
            acc_init: Box::new(resolve_constants_rec(acc_init, constants, locals, shapes)),
            body: Box::new(resolve_constants_rec(body, constants, locals, shapes)),
        },
        CompiledExpr::Vector(elems) => CompiledExpr::Vector(
            elems.iter().map(|e| resolve_constants_rec(e, constants, locals, shapes)).collect(),
        ),
        other => other.clone(),
    }
}

/// Push `first`/`last` through nested Let expressions until a Vector is found.
/// (first (let [binds] (let [binds2] [a b ...]))) → (let [binds] (let [binds2] a))
fn push_first_last(name: &str, expr: CompiledExpr) -> CompiledExpr {
    match expr {
        CompiledExpr::Vector(elems) => {
            if name == "first" {
                elems.into_iter().next()
            } else {
                elems.into_iter().last()
            }
            .unwrap_or_else(|| CompiledExpr::Vector(vec![]))
        }
        CompiledExpr::Let { bindings, body } => CompiledExpr::Let {
            bindings,
            body: Box::new(push_first_last(name, *body)),
        },
        other => CompiledExpr::FunctionCall {
            name: name.to_string(),
            args: vec![other],
        },
    }
}

/// Heuristically infer the output shape of an expression given a known shapes map.
/// Used to propagate shapes through Let-bound intermediate variables after inlining.
fn try_infer_shape(
    expr: &CompiledExpr,
    shapes: &std::collections::HashMap<String, Vec<i64>>,
) -> Option<Vec<i64>> {
    match expr {
        CompiledExpr::Symbol(s) => shapes.get(s).cloned(),
        CompiledExpr::Let { bindings, body } => {
            let mut inner = shapes.clone();
            for (name, rhs) in bindings {
                if let Some(sh) = try_infer_shape(rhs, &inner) {
                    inner.insert(name.clone(), sh);
                }
            }
            try_infer_shape(body, &inner)
        }
        CompiledExpr::FunctionCall { name, args } => match name.as_str() {
            // Element-wise ops: output shape = any known tensor arg's shape
            "+" | "-" | "*" | "/" | "**" | "sqrt" | "exp" | "log" | "abs"
            | "relu" | "gelu" | "tanh" | "sigmoid" | "neg"
            | "maximum" | "minimum" | "clamp"
            | "==" | "!=" | "<" | "<=" | ">" | ">=" | "and" | "or" | "not"
            | "where" => args.iter().find_map(|a| try_infer_shape(a, shapes)),
            // Shape-preserving ops: output has same shape as first tensor arg
            "layer-norm" | "softmax" | "normalize"
                => args.first().and_then(|a| try_infer_shape(a, shapes)),
            "first" => args.first().and_then(|a| try_infer_shape(a, shapes)),
            // reshape: output shape is the second arg (a vector of ints)
            "reshape" if args.len() == 2 => {
                if let CompiledExpr::Vector(elems) = &args[1] {
                    let sh: Option<Vec<i64>> = elems.iter().map(|e| {
                        if let CompiledExpr::Integer(n) = e { Some(*n) } else { None }
                    }).collect();
                    sh
                } else {
                    None
                }
            }
            // swapaxes: permute shape dims
            "swapaxes" if args.len() == 3 => {
                if let (Some(sh), Some(CompiledExpr::Integer(a)), Some(CompiledExpr::Integer(b)))
                    = (try_infer_shape(&args[0], shapes), args.get(1), args.get(2))
                {
                    let ndim = sh.len() as i64;
                    let ai = if *a < 0 { (ndim + a) as usize } else { *a as usize };
                    let bi = if *b < 0 { (ndim + b) as usize } else { *b as usize };
                    if ai < sh.len() && bi < sh.len() {
                        let mut out = sh;
                        out.swap(ai, bi);
                        Some(out)
                    } else {
                        None
                    }
                } else {
                    None
                }
            }
            // matmul (@): [..., M, K] @ [..., K, N] → [..., M, N]
            "@" if args.len() == 2 => {
                let lhs = try_infer_shape(&args[0], shapes)?;
                let rhs = try_infer_shape(&args[1], shapes)?;
                if lhs.len() >= 2 && rhs.len() >= 2 {
                    let mut out = lhs[..lhs.len()-1].to_vec();
                    out.push(*rhs.last().unwrap());
                    Some(out)
                } else {
                    None
                }
            }
            _ => None,
        },
        _ => None,
    }
}

fn f64_to_const(val: f64) -> CompiledExpr {
    if val.fract() == 0.0 && val.abs() < i64::MAX as f64 {
        CompiledExpr::Integer(val as i64)
    } else {
        CompiledExpr::Float(val)
    }
}

fn try_fold_arithmetic(name: &str, args: &[CompiledExpr]) -> Option<CompiledExpr> {
    if args.len() != 2 { return None; }
    let a = extract_numeric(&args[0])?;
    let b = extract_numeric(&args[1])?;
    let result = match name {
        "+" => a + b,
        "-" => a - b,
        "*" => a * b,
        "/" => a / b,
        "//" => (a / b).floor(),
        _ => return None,
    };
    Some(f64_to_const(result))
}

fn extract_numeric(expr: &CompiledExpr) -> Option<f64> {
    match expr {
        CompiledExpr::Integer(n) => Some(*n as f64),
        CompiledExpr::Float(f) => Some(*f),
        _ => None,
    }
}

/// Collect .shf files from a directory, optionally recursively.
fn collect_shf_files(dir: &std::path::Path, recursive: bool) -> Vec<std::path::PathBuf> {
    let mut files = Vec::new();
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("sheaf build: cannot read directory '{}': {}", dir.display(), e);
            return files;
        }
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_file() && path.extension().and_then(|e| e.to_str()) == Some("shf") {
            files.push(path);
        } else if recursive && path.is_dir() {
            files.extend(collect_shf_files(&path, true));
        }
    }
    files.sort();
    files
}

fn find_iree_compile() -> String {
    // 1. Explicit env var
    if let Ok(path) = std::env::var("IREE_COMPILE") {
        return path;
    }
    // 2. Standard SDK install location
    if let Ok(home) = std::env::var("HOME") {
        let candidate = format!("{}/bin/iree-build/tools/iree-compile", home);
        if std::path::Path::new(&candidate).exists() {
            return candidate;
        }
    }
    // 3. PATH
    if let Some(path) = which("iree-compile") {
        return path;
    }
    eprintln!("error: 'sheaf build' requires the Sheaf SDK (iree-compile not found)");
    eprintln!("  Install the SDK or set IREE_COMPILE=/path/to/iree-compile");
    eprintln!("  To emit MLIR only (no SDK): sheaf build FILE -o FILE.mlir -S");
    exit(1);
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

fn run_init_ai() {
    const CONTEXT: &str = include_str!("../../../assets/sheaf-context.md");
    const REFERENCE: &str = include_str!("../../../assets/reference.md");

    let output = std::path::Path::new("sheaf-context.md");
    if output.exists() {
        eprintln!("sheaf-context.md already exists — overwrite? [y/N] ");
        let mut answer = String::new();
        if std::io::stdin().read_line(&mut answer).is_err() || !answer.trim().eq_ignore_ascii_case("y") {
            eprintln!("Aborted.");
            exit(0);
        }
    }

    let combined = format!("{}\n\n---\n\n## REFERENCE\n\n{}", CONTEXT, REFERENCE);
    std::fs::write(output, &combined).unwrap_or_else(|e| {
        eprintln!("sheaf: cannot write '{}': {}", output.display(), e);
        exit(1);
    });

    eprintln!("Wrote sheaf-context.md ({} bytes)", combined.len());
    eprintln!("Add this file to your AI assistant context (e.g. CLAUDE.md, .cursorrules, etc.)");
}

fn resolve_vag_decls(
    compiler: &sheaf_compiler::core::compiler::CompilerContext,
    compiled_exprs: &[sheaf_compiler::core::compiler::CompiledExpr],
    verbosity: u8,
) -> Vec<String> {
    use sheaf_compiler::autodiff::value_and_grad::{GradParam, emit_value_and_grad_func};
    use sheaf_compiler::core::inference::infer_function_signature_with_known;
    use sheaf_compiler::StableHLOType;

    let mut vag_nodes = Vec::new();
    for expr in compiled_exprs {
        collect_vag_nodes(expr, &mut vag_nodes);
    }

    let mut decls = Vec::new();
    for (fn_name, src_fn_name, wrt_params, shape_config) in vag_nodes {
        let func_def = match compiler.registry.get(src_fn_name) {
            Some(fd) => fd,
            None => continue,
        };
        let body_compiled = match &func_def.body_compiled {
            Some(b) => b,
            None => continue,
        };

        let known_types: Vec<(String, StableHLOType)> = shape_config
            .iter()
            .map(|(name, dims)| {
                let ty = if dims.is_empty() {
                    StableHLOType::scalar_f32()
                } else {
                    StableHLOType::f32_tensor(dims.clone())
                };
                (name.clone(), ty)
            })
            .collect();

        let signature = if !known_types.is_empty() {
            match infer_function_signature_with_known(
                compiler,
                &func_def.params,
                body_compiled,
                &known_types,
            ) {
                Ok(sig) => sig,
                Err(e) => {
                    eprintln!(
                        "value-and-grad '{}': signature inference failed: {}",
                        fn_name, e
                    );
                    exit(1);
                }
            }
        } else {
            match &func_def.signature {
                Some(sig) => sig.clone(),
                None => {
                    eprintln!(
                        "value-and-grad '{}': function '{}' has no inferred signature",
                        fn_name, src_fn_name
                    );
                    exit(1);
                }
            }
        };

        let grad_params: Vec<GradParam> = wrt_params
            .iter()
            .map(|wrt_name| {
                let idx = func_def
                    .params
                    .iter()
                    .position(|p| p == wrt_name)
                    .unwrap_or_else(|| {
                        eprintln!(
                            "value-and-grad '{}': '{}' is not a parameter of '{}'",
                            fn_name, wrt_name, src_fn_name
                        );
                        exit(1);
                    });
                GradParam {
                    name: wrt_name.clone(),
                    ty: signature.param_types[idx].clone(),
                }
            })
            .collect();

        if verbosity >= 1 {
            println!("Emitting value-and-grad '{}'...", fn_name);
        }

        let func_decl = emit_value_and_grad_func(
            fn_name,
            &func_def.params,
            &signature.param_types,
            body_compiled,
            &grad_params,
            compiler.registry.clone(),
        )
        .unwrap_or_else(|e| {
            eprintln!("value-and-grad '{}': codegen failed: {}", fn_name, e);
            exit(1);
        });

        decls.push(func_decl);
    }
    decls
}

fn collect_vag_nodes<'a>(
    expr: &'a sheaf_compiler::core::compiler::CompiledExpr,
    out: &mut Vec<(&'a str, &'a str, &'a Vec<String>, &'a Vec<(String, Vec<i64>)>)>,
) {
    use sheaf_compiler::core::compiler::CompiledExpr;
    match expr {
        CompiledExpr::ValueAndGrad {
            fn_name,
            src_fn_name,
            wrt_params,
            shape_config,
        } => {
            out.push((fn_name, src_fn_name, wrt_params, shape_config));
        }
        CompiledExpr::Do(exprs) => {
            for e in exprs {
                collect_vag_nodes(e, out);
            }
        }
        CompiledExpr::Let { bindings, body } => {
            for (_, v) in bindings {
                collect_vag_nodes(v, out);
            }
            collect_vag_nodes(body, out);
        }
        _ => {}
    }
}

/// Scan a lowered body for Let bindings of the form `var = GetTupleElement(param, [i])`.
/// For each such binding, propagate the tuple_key_layout from the dict key name to the
/// variable name. This allows `(get input-layer "W")` to resolve when `input-layer` is
/// a let binding for `(get params "input")`.
fn propagate_let_layouts(
    expr: &sheaf_compiler::core::compiler::CompiledExpr,
    idx_to_key: &std::collections::HashMap<(String, usize), String>,
    layouts: &mut std::collections::HashMap<String, std::collections::BTreeMap<String, usize>>,
) {
    use sheaf_compiler::core::compiler::CompiledExpr;
    match expr {
        CompiledExpr::Let { bindings, body } => {
            for (var_name, value_expr) in bindings {
                if let CompiledExpr::GetTupleElement { param, indices } = value_expr {
                    if indices.len() == 1 {
                        let lookup = (param.clone(), indices[0]);
                        if let Some(key_name) = idx_to_key.get(&lookup) {
                            if let Some(sub_layout) = layouts.get(key_name).cloned() {
                                layouts.insert(var_name.clone(), sub_layout);
                            }
                        }
                    }
                }
                propagate_let_layouts(value_expr, idx_to_key, layouts);
            }
            propagate_let_layouts(body, idx_to_key, layouts);
        }
        CompiledExpr::FunctionCall { args, .. } => {
            for a in args {
                propagate_let_layouts(a, idx_to_key, layouts);
            }
        }
        CompiledExpr::Do(exprs) => {
            for e in exprs {
                propagate_let_layouts(e, idx_to_key, layouts);
            }
        }
        CompiledExpr::If { condition, then_branch, else_branch } => {
            propagate_let_layouts(condition, idx_to_key, layouts);
            propagate_let_layouts(then_branch, idx_to_key, layouts);
            if let Some(e) = else_branch {
                propagate_let_layouts(e, idx_to_key, layouts);
            }
        }
        CompiledExpr::Lambda { body, .. } => {
            propagate_let_layouts(body, idx_to_key, layouts);
        }
        _ => {}
    }
}
