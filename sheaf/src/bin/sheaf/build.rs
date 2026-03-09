// Copyright (c) 2025 Damien Boureille
// Licensed under the MIT License.

//! Build pipeline: compile pure Sheaf functions to StableHLO MLIR and IREE VMFB.

use std::process::exit;

pub(super) fn run_build(args: &[String]) {
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
    -o OUTPUT           Output file (default: module.vmfb)
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

    // Resolve output directory (where module.vmfb goes)
    let output_dir = if input.is_dir() {
        input.canonicalize().unwrap_or_else(|_| input.clone())
    } else {
        input.parent().map(|p| p.to_path_buf()).unwrap_or_else(|| PathBuf::from("."))
            .canonicalize().unwrap_or_else(|_| PathBuf::from("."))
    };

    let output = output.unwrap_or_else(|| {
        if emit_mlir_only {
            output_dir.join("module.mlir")
        } else {
            output_dir.join("module.vmfb")
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

    let extra_decls = super::transforms::resolve_vag_decls(&compiler, &all_compiled_exprs, verbosity);

    let mut all_decls = extra_decls;

    // Shape discovery: interpret source or runner to observe concrete arg shapes
    let (traced_configs, traced_constants) = match trace_with {
        Some(runner_path) => {
            let (configs, constants) = super::trace::trace_with_runner(&runner_path, &compiler, &all_fn_names, verbosity);
            (Some(configs), constants)
        }
        None => {
            // Auto-trace: run source files to discover shapes automatically
            let (configs, constants) = super::trace::auto_trace(&source_files, &compiler, &all_fn_names, verbosity);
            if configs.is_empty() {
                (None, constants)
            } else {
                (Some(configs), constants)
            }
        }
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
                let display = if verbosity >= 2 { &ty_str } else { super::compact_type(&ty_str) };
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
                        let display = if verbosity >= 2 { &ty_str } else { super::compact_type(&ty_str) };
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
        body = super::transforms::lower_inlined_gets(&body, &param_index_maps);

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
        body = super::transforms::resolve_static_constants(&body, fn_consts, &param_shapes);

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
        super::transforms::propagate_let_layouts(&body, &idx_to_key, &mut tuple_key_layouts);

        // Run codegen inside catch_unwind to handle unexpected panics gracefully.
        // Codegen may panic on unsupported type combinations (e.g. shape rank mismatches);
        // catch_unwind lets us skip those functions instead of aborting the entire build.
        let codegen_result = {
            let registry_clone = compiler.registry.clone();
            let params_clone = func_def.params.clone();
            let sig_param_types = sig.param_types.clone();
            let sig_return_type = sig.return_type.clone();
            let body_clone = body.clone();
            let name_clone = name.clone();
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let mut codegen = CodeGenerator::with_function_params(
                    registry_clone,
                    &params_clone,
                    &sig_param_types,
                );
                codegen.set_tuple_key_layouts(tuple_key_layouts);
                codegen.set_idx_to_key(idx_to_key);
                codegen.emit_func_declaration(&name_clone, &body_clone, &sig_param_types, &sig_return_type)
            }))
        };
        match codegen_result {
            Ok(Ok((decl, actual_return_ty))) => {
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
            Ok(Err(e)) => {
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
            Err(_panic) => {
                let src_file = file_functions.iter()
                    .find(|(_, names)| names.contains(name))
                    .map(|(p, _)| p.display().to_string())
                    .unwrap_or_else(|| "?".to_string());
                let reason = if verbosity >= 1 {
                    "codegen: internal error (panic)".to_string()
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
            eprintln!("hint: no function calls were observed during auto-trace.");
            eprintln!("      Ensure source files call the target functions with concrete data,");
            eprintln!("      or use --trace-with <runner.shf> to provide a dedicated runner.");
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

    // VMFB path — requires iree-compile (auto-download if needed)
    let iree_compile = sheaf_compiler::runtime::jit::find_iree_compile()
        .or_else(|| sheaf_compiler::runtime::jit::ensure_toolchain().ok())
        .unwrap_or_else(|| {
            eprintln!("error: iree-compile not found and auto-download failed");
            eprintln!("  Set IREE_COMPILE=/path/to/iree-compile or check network connection");
            eprintln!("  To emit MLIR only (no SDK): sheaf build FILE -o FILE.mlir -S");
            exit(1);
        });

    // Write intermediate MLIR to a temp file
    let mlir_path = output.with_extension("mlir");
    fs::write(&mlir_path, &mlir).unwrap_or_else(|e| {
        eprintln!("error writing '{}': {}", mlir_path.display(), e);
        exit(1);
    });

    if verbosity >= 1 {
        eprintln!("running iree-compile ({})...", iree_compile);
    }

    let mut compile_cmd = Command::new(&iree_compile);
    compile_cmd
        .arg(&mlir_path)
        .arg(format!("--iree-hal-target-backends={}", backend))
        .arg("-o")
        .arg(&output);
    if backend == "llvm-cpu" {
        compile_cmd.arg("--iree-llvmcpu-target-cpu=host");
        compile_cmd.arg("--iree-llvmcpu-enable-ukernels=all");
    }
    let status = compile_cmd.status()
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

/// Write module.json alongside the VMFB with hashes and inferred signatures.
/// Used by `sheaf run` and `(use)` to detect stale artifacts and load signatures.
fn write_manifest(vmfb_path: &std::path::Path, functions: &[ManifestEntry], verbosity: u8) {
    let dir = vmfb_path.parent().unwrap_or(std::path::Path::new("."));
    let manifest_path = dir.join("module.json");
    let vmfb_name = vmfb_path.file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("module.vmfb");

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

