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

mod build;
mod repl;
mod trace;
mod transforms;

use std::process::exit;

fn main() {
    let args: Vec<String> = std::env::args().collect();

    match args.get(1).map(|s| s.as_str()) {
        Some("build") => build::run_build(&args[2..]),
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
    --blame                Profile execution and print timing report
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
    use sheaf_compiler::interpreter::eval::{eval_source_with_blame, eval_source_with_path, eval_source_with_tracing};
    use sheaf_compiler::interpreter::tracer::{CliGuard, LogFormat, TraceLevel, TracerConfig};

    let path = &args[0];

    let mut trace_enabled = false;
    let mut trace_scope: Option<Vec<String>> = None;
    let mut trace_level = TraceLevel::Normal;
    let mut trace_format = LogFormat::Console;
    let mut cli_guards: Vec<CliGuard> = Vec::new();
    let mut blame = false;

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
            "--blame" => {
                blame = true;
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
    let result = if blame {
        let tracer_config = if needs_tracing {
            Some(TracerConfig {
                enabled: trace_enabled,
                scope_filter: trace_scope,
                level: trace_level,
                format: trace_format,
                cli_guards,
            })
        } else {
            None
        };
        eval_source_with_blame(&source, Some(&abs_path), tracer_config)
    } else if needs_tracing {
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
