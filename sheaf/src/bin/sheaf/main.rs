// Copyright (c) 2025 Damien Boureille
// Licensed under the MIT License.

//! Sheaf CLI
//!
//! Usage:
//!   sheaf                               Launch interactive REPL
//!   sheaf file.shf                      Interpret a Sheaf file
//!   sheaf -c '(+ 1 2)'                 Evaluate an expression

mod repl;

use sheaf_compiler::sheaf_msg;
use std::process::exit;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let tail = &args[1..];

    // Handle early-exit flags anywhere in args
    if tail.iter().any(|a| a == "--help" || a == "-h") {
        print_help();
        return;
    }
    if tail.iter().any(|a| a == "--version") {
        println!("Sheaf {}", env!("CARGO_PKG_VERSION"));
        return;
    }

    // Parse global flags and strip them from args
    let mut verbosity: u8 = 0;
    let mut device: Option<String> = None;
    let mut jit_profile = false;
    let mut remaining: Vec<String> = Vec::new();
    let mut i = 0;
    while i < tail.len() {
        match tail[i].as_str() {
            "-v" | "--verbose" => verbosity += 1,
            "-vv" => verbosity = 2,
            "--device" => {
                i += 1;
                match tail.get(i) {
                    Some(d) => device = Some(d.clone()),
                    None => {
                        sheaf_msg!("sheaf: --device requires an argument");
                        exit(1);
                    }
                }
            }
            "--jit-profile" => jit_profile = true,
            other => remaining.push(other.to_string()),
        }
        i += 1;
    }
    sheaf_compiler::core::config::init(verbosity, device, jit_profile);

    // Find the first positional argument (not starting with '-')
    let first_positional = remaining.iter().position(|a| !a.starts_with('-'));

    match first_positional.map(|i| remaining[i].as_str()) {
        Some("init-ai") => run_init_ai(),

        Some("-c") => unreachable!(), // -c starts with '-', won't match

        Some(file) if file.ends_with(".shf") || std::path::Path::new(file).exists() => {
            let pos = first_positional.unwrap();
            let mut reordered = vec![remaining[pos].clone()];
            for (j, a) in remaining.iter().enumerate() {
                if j != pos { reordered.push(a.clone()); }
            }
            run_file(&reordered);
        }

        _ => {
            if let Some(ci) = remaining.iter().position(|a| a == "-c") {
                let mut blame = false;
                let expr = remaining[ci + 1..].iter()
                    .filter(|a| {
                        if *a == "--blame" { blame = true; return false; }
                        !a.starts_with('-') || a.parse::<f64>().is_ok()
                    })
                    .cloned()
                    .collect::<Vec<_>>()
                    .join(" ");
                if expr.is_empty() {
                    sheaf_msg!("sheaf: -c requires an expression");
                    exit(1);
                }
                run_expr(&expr, blame);
            } else if remaining.is_empty() {
                run_repl();
            } else {
                sheaf_msg!("sheaf: unknown command '{}'", remaining[0]);
                eprintln!("Run 'sheaf --help' for usage.");
                exit(1);
            }
        }
    }
}

fn print_help() {
    println!(
        "Sheaf {} - A Functional Language for Differentiable Computation

Usage:
    sheaf                              Launch interactive REPL
    sheaf [OPTIONS] FILE               Run a Sheaf file
    sheaf -c EXPR                      Evaluate an expression
    sheaf init-ai                      Generate context file for AI assistants

Options:
    --trace [FUNCTIONS]    Log calls with tensor shapes and timing
    --trace-level LEVEL    Trace verbosity: fast, normal (default), verbose
    --trace-out FORMAT     Trace output: console (default), json
    --guard SPEC           Runtime guard: [scope:]variable:check (repeatable)
    --blame                Profile execution and print timing report

Advanced options:
    --device DEVICE        Run on specific device: cpu, metal, cuda, vulkan (default: auto)
    -v / -vv               Verbose JIT output (-vv for full MLIR dumps)
    --jit-profile          Show JIT dispatch timing breakdown

Examples:
    sheaf script.shf
    sheaf -c '(+ 1 2)'
    sheaf --blame train.shf
    sheaf --guard no-nan train.shf
    sheaf train.shf --guard loss:range:0:20",
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

fn run_expr(source: &str, blame: bool) {
    use sheaf_compiler::interpreter::eval::{eval_source, eval_source_with_blame};
    let result = if blame {
        eval_source_with_blame(source, None, None)
    } else {
        eval_source(source)
    };
    match result {
        Ok(val) => {
            if !is_silent_result(&val) {
                println!("{}", val);
            }
        }
        Err(e) => {
            sheaf_msg!("{}", sheaf_compiler::core::error_format::format_error(&e));
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
                        sheaf_msg!("sheaf: --trace-level expects fast|normal|verbose");
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
                        sheaf_msg!("sheaf: --trace-out expects console|json");
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
                            sheaf_msg!("sheaf: invalid guard spec '{}': {}", spec, msg);
                            exit(1);
                        }
                    }
                    None => {
                        sheaf_msg!("sheaf: --guard requires a SPEC argument");
                        exit(1);
                    }
                }
            }
            "--blame" => {
                blame = true;
            }
            arg => {
                sheaf_msg!("sheaf: unknown option '{}' for file mode", arg);
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
            sheaf_msg!("sheaf: cannot read '{}': {}", path, e);
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
            sheaf_msg!("{}", sheaf_compiler::core::error_format::format_error(&e));
            exit(1);
        }
    }
}

/// Parse a guard spec: `[scope:]check[:args]`
/// Examples: `no-nan`, `loss:no-nan`, `range:0:10`, `forward:range:-1:1`
fn parse_guard_spec(spec: &str) -> Result<sheaf_compiler::interpreter::tracer::CliGuard, String> {
    use sheaf_compiler::core::expr::GuardCheck;
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

fn run_init_ai() {
    const CONTEXT: &str = include_str!("../../../assets/sheaf-context.md");
    const REFERENCE: &str = include_str!("../../../assets/reference.md");

    let output = std::path::Path::new("sheaf-context.md");
    if output.exists() {
        eprintln!("sheaf-context.md already exists: overwrite? [y/N] ");
        let mut answer = String::new();
        if std::io::stdin().read_line(&mut answer).is_err() || !answer.trim().eq_ignore_ascii_case("y") {
            eprintln!("Aborted.");
            exit(0);
        }
    }

    let combined = format!("{}\n\n---\n\n## REFERENCE\n\n{}", CONTEXT, REFERENCE);
    std::fs::write(output, &combined).unwrap_or_else(|e| {
        sheaf_msg!("sheaf: cannot write '{}': {}", output.display(), e);
        exit(1);
    });

    sheaf_msg!("Wrote sheaf-context.md ({} bytes)", combined.len());
    sheaf_msg!("Add this file to your AI assistant context (e.g. CLAUDE.md, .cursorrules, etc.)");
}
