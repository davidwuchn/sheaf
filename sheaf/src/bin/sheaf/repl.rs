// Copyright (c) 2025 Damien Boureille
// Licensed under the MIT License.

//! Sheaf interactive REPL with autocompletion, multiline, help, and tracing.

use rustyline::completion::Completer;
use rustyline::error::ReadlineError;
use rustyline::highlight::Highlighter;
use rustyline::hint::Hinter;
use rustyline::validate::{ValidationContext, ValidationResult, Validator};
use rustyline::{Editor, Helper};
use sheaf_compiler::forms::special_forms_registry;
use sheaf_compiler::interpreter::eval::Interpreter;
use sheaf_compiler::interpreter::tracer::{Tracer, TraceLevel, LogFormat, TracerConfig};
use sheaf_compiler::interpreter::value::Value;
use std::borrow::Cow;
use std::collections::HashSet;

const REFERENCE: &str = include_str!("../../../assets/reference.md");

const REPL_COMMANDS: &[&str] = &[
    ":help", ":h", ":?",
    ":quit", ":q",
    ":env",
    ":registry", ":reg",
    ":show",
    ":trace",
    ":scope",
    ":blame",
    ":clear",
];

struct SheafHelper {
    names: Vec<String>,
}

impl SheafHelper {
    fn new() -> Self {
        let mut names: Vec<String> = special_forms_registry()
            .keys()
            .map(|k| k.to_string())
            .collect();
        names.sort();
        Self { names }
    }

    fn refresh(&mut self, interp: &Interpreter) {
        let mut set = HashSet::new();
        for name in special_forms_registry().keys() {
            set.insert(name.to_string());
        }
        for name in interp.registry_names() {
            set.insert(name);
        }
        for name in interp.env().all_names() {
            set.insert(name);
        }
        let mut sorted: Vec<String> = set.into_iter().collect();
        sorted.sort();
        self.names = sorted;
    }
}

impl Completer for SheafHelper {
    type Candidate = String;

    fn complete(
        &self,
        line: &str,
        pos: usize,
        _ctx: &rustyline::Context<'_>,
    ) -> rustyline::Result<(usize, Vec<String>)> {
        let delimiters = b" \t\n()[]{}";
        let start = line[..pos]
            .rfind(|c: char| delimiters.contains(&(c as u8)))
            .map(|i| i + 1)
            .unwrap_or(0);
        let word = &line[start..pos];

        let matches = if word.is_empty() {
            self.names.clone()
        } else if word.starts_with(':') {
            REPL_COMMANDS
                .iter()
                .filter(|cmd| cmd.starts_with(word))
                .map(|s| s.to_string())
                .collect()
        } else {
            self.names
                .iter()
                .filter(|name| name.starts_with(word))
                .cloned()
                .collect()
        };

        Ok((start, matches))
    }
}

impl Validator for SheafHelper {
    fn validate(&self, ctx: &mut ValidationContext<'_>) -> rustyline::Result<ValidationResult> {
        let input = ctx.input();
        let trimmed = input.trim();
        if trimmed.is_empty() || trimmed.starts_with(':') {
            return Ok(ValidationResult::Valid(None));
        }
        if is_balanced(trimmed) {
            Ok(ValidationResult::Valid(None))
        } else {
            Ok(ValidationResult::Incomplete)
        }
    }
}

impl Hinter for SheafHelper {
    type Hint = String;
    fn hint(&self, _line: &str, _pos: usize, _ctx: &rustyline::Context<'_>) -> Option<String> {
        None
    }
}

impl Highlighter for SheafHelper {
    fn highlight_prompt<'b, 's: 'b, 'p: 'b>(&'s self, prompt: &'p str, _default: bool) -> Cow<'b, str> {
        Cow::Borrowed(prompt)
    }
}

impl Helper for SheafHelper {}

fn is_balanced(input: &str) -> bool {
    let mut depth = 0i32;
    let mut in_string = false;
    let mut chars = input.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            '"' if !in_string => in_string = true,
            '"' if in_string => in_string = false,
            '\\' if in_string => { chars.next(); }
            ';' if !in_string => {
                // Skip to end of line
                for c2 in chars.by_ref() {
                    if c2 == '\n' { break; }
                }
            }
            '(' | '[' | '{' if !in_string => depth += 1,
            ')' | ']' | '}' if !in_string => depth -= 1,
            _ => {}
        }
    }
    depth <= 0
}

fn is_silent_result(val: &Value) -> bool {
    match val {
        Value::Nil => true,
        Value::List(items) => items.iter().all(|v| matches!(v, Value::Nil)),
        _ => false,
    }
}

pub fn run() {
    println!("Sheaf {} — interactive REPL", env!("CARGO_PKG_VERSION"));
    println!("Type :help or :h for help, :quit or :q to exit.\n");

    let history_file = std::env::var_os("HOME")
        .map(|h| std::path::PathBuf::from(h).join(".sheaf_history"));

    let helper = SheafHelper::new();
    let config = rustyline::Config::builder()
        .completion_type(rustyline::CompletionType::List)
        .build();
    let mut rl = Editor::with_config(config).expect("failed to init line editor");
    rl.set_helper(Some(helper));
    rl.bind_sequence(
        rustyline::KeyEvent(rustyline::KeyCode::Char('d'), rustyline::Modifiers::CTRL),
        rustyline::Cmd::EndOfFile,
    );
    if let Some(ref path) = history_file {
        let _ = rl.load_history(path);
    }

    let mut interp = Interpreter::new();
    // Initial refresh with builtins
    rl.helper_mut().unwrap().refresh(&interp);

    loop {
        match rl.readline("sheaf> ") {
            Ok(line) => {
                let trimmed = line.trim();
                if trimmed.is_empty() {
                    continue;
                }

                rl.add_history_entry(&line).ok();

                if trimmed.starts_with(':') {
                    if handle_command(trimmed, &mut interp) {
                        break;
                    }
                    continue;
                }

                match interp.eval(trimmed) {
                    Ok(val) => {
                        if !is_silent_result(&val) {
                            println!("{}", val);
                        }
                    }
                    Err(e) => eprint!("{}", sheaf_compiler::core::error_format::format_error(&e)),
                }

                rl.helper_mut().unwrap().refresh(&interp);
            }
            Err(ReadlineError::Interrupted) => {
                eprintln!("^C (use :quit to exit)");
                continue;
            }
            Err(ReadlineError::Eof) => {
                println!("\nBye!");
                break;
            }
            Err(e) => {
                eprintln!("sheaf: read error: {}", e);
                break;
            }
        }
    }

    if let Some(ref path) = history_file {
        let _ = rl.save_history(path);
    }
}

/// Handle a REPL command. Returns true if the REPL should exit.
fn handle_command(input: &str, interp: &mut Interpreter) -> bool {
    let parts: Vec<&str> = input.splitn(2, ' ').collect();
    let cmd = parts[0];
    let arg = parts.get(1).map(|s| s.trim()).unwrap_or("");

    match cmd {
        ":quit" | ":q" => return true,

        ":help" | ":h" => {
            if arg.is_empty() {
                print_general_help();
            } else {
                print_symbol_help(arg);
            }
        }

        ":?" => {
            if arg.is_empty() {
                println!("Usage: :? <symbol>");
            } else {
                print_symbol_help(arg);
            }
        }

        ":env" => {
            let reg_names = interp.registry_names();
            let var_names = interp.env().all_names();
            if !reg_names.is_empty() {
                println!("Functions:");
                for name in &reg_names {
                    println!("  {}", name);
                }
            }
            let user_vars: Vec<&String> = var_names
                .iter()
                .filter(|n| {
                    if let Ok(val) = interp.env().get(n) {
                        !matches!(val, Value::BuiltinFn { .. })
                    } else {
                        false
                    }
                })
                .collect();
            if !user_vars.is_empty() {
                println!("Variables:");
                for name in &user_vars {
                    if let Ok(val) = interp.env().get(name) {
                        println!("  {} = {}", name, format_env_value(&val));
                    }
                }
            }
            if reg_names.is_empty() && user_vars.is_empty() {
                println!("(empty environment)");
            }
        }

        ":registry" | ":reg" => {
            let names = interp.registry_names();
            if names.is_empty() {
                println!("(no user-defined functions)");
            } else {
                for name in &names {
                    println!("  {}", name);
                }
            }
        }

        ":show" => {
            if arg.is_empty() {
                println!("Usage: :show <name>");
            } else if let Ok(val) = interp.env().get(arg) {
                println!("{}", val);
            } else if interp.registry_names().contains(&arg.to_string()) {
                println!("<fn:{}>", arg);
            } else {
                eprintln!("Unknown symbol: {}", arg);
            }
        }

        ":trace" => {
            match arg {
                "" => {
                    let status = interp.env().tracer.as_ref()
                        .map(|t| if t.enabled { "on" } else { "off" })
                        .unwrap_or("off");
                    println!("Trace: {}", status);
                }
                "off" => {
                    interp.env_mut().tracer = None;
                    println!("Trace disabled.");
                }
                level_str => {
                    let level = match level_str {
                        "fast" => TraceLevel::Fast,
                        "normal" => TraceLevel::Normal,
                        "verbose" => TraceLevel::Verbose,
                        _ => {
                            eprintln!("Usage: :trace [off|fast|normal|verbose]");
                            return false;
                        }
                    };
                    let config = TracerConfig {
                        enabled: true,
                        scope_filter: None,
                        level,
                        format: LogFormat::Console,
                        cli_guards: Vec::new(),
                    };
                    interp.env_mut().tracer = Some(Tracer::from_config(config));
                    println!("Trace enabled ({}).", level_str);
                }
            }
        }

        ":scope" => {
            match arg {
                "" => {
                    let scope = interp.env().tracer.as_ref()
                        .and_then(|t| t.scope_filter.as_ref())
                        .map(|s| {
                            let mut v: Vec<&String> = s.iter().collect();
                            v.sort();
                            v.iter().map(|s| s.as_str()).collect::<Vec<_>>().join(", ")
                        });
                    match scope {
                        Some(s) => println!("Scope filter: {}", s),
                        None => println!("Scope filter: off (tracing all functions)"),
                    }
                }
                "off" => {
                    if let Some(ref mut tracer) = interp.env_mut().tracer {
                        tracer.scope_filter = None;
                        println!("Scope filter disabled.");
                    } else {
                        eprintln!("Trace is not active. Use :trace to enable first.");
                    }
                }
                names => {
                    if let Some(ref mut tracer) = interp.env_mut().tracer {
                        tracer.scope_filter = Some(
                            names.split(',').map(|s| s.trim().to_string()).collect()
                        );
                        println!("Scope filter: {}", names);
                    } else {
                        eprintln!("Trace is not active. Use :trace to enable first.");
                    }
                }
            }
        }

        ":blame" => {
            match arg {
                "" => {
                    let status = if interp.env().profiler.is_some() { "on" } else { "off" };
                    println!("Blame: {}", status);
                }
                "on" => {
                    interp.env_mut().profiler = Some(
                        sheaf_compiler::interpreter::profiler::Profiler::new()
                    );
                    println!("Profiler enabled.");
                }
                "off" => {
                    if let Some(ref profiler) = interp.env().profiler {
                        profiler.report();
                    }
                    interp.env_mut().profiler = None;
                    println!("Profiler disabled.");
                }
                "report" => {
                    if let Some(ref profiler) = interp.env().profiler {
                        profiler.report();
                    } else {
                        eprintln!("Profiler is not active. Use :blame on to enable first.");
                    }
                }
                _ => {
                    eprintln!("Usage: :blame [on|off|report]");
                }
            }
        }

        ":clear" => {
            print!("\x1b[2J\x1b[H");
        }

        _ => {
            eprintln!("Unknown command: {}", cmd);
            eprintln!("Type :help for a list of commands.");
        }
    }

    false
}

fn format_env_value(val: &Value) -> String {
    match val {
        Value::Function { params, .. } => format!("<fn/{}>", params.len()),
        Value::BuiltinFn { name, .. } => format!("<builtin:{}>", name),
        Value::Tensor { data, .. } => {
            let shape: Vec<usize> = data.shape().to_vec();
            let shape_str = shape.iter().map(|d| d.to_string()).collect::<Vec<_>>().join("x");
            format!("f64[{}]", shape_str)
        }
        Value::Dict(map) => format!("dict({})", map.len()),
        Value::List(items) => format!("list({})", items.len()),
        other => format!("{}", other),
    }
}

fn print_general_help() {
    println!("Sheaf REPL commands:

  :help, :h              Show this help
  :help <name>, :? <name>  Help for a specific function or form
  :env                   List all functions and variables
  :registry, :reg        List user-defined functions
  :show <name>           Show a variable's value
  :trace [off|fast|normal|verbose]  Control execution tracing
  :scope [name|off]      Filter tracing to specific functions
  :blame [on|off|report] Profile execution and print timing report
  :clear                 Clear the screen
  :quit, :q              Exit the REPL

Keyboard:
  Tab                    Autocomplete
  Ctrl-D                 Exit
  Enter on incomplete    Continue on next line");
}

fn print_symbol_help(name: &str) {
    // Search for ### <name> in embedded reference
    let header = format!("### {}", name);
    let start = match REFERENCE.find(&header) {
        Some(pos) => pos,
        None => {
            // Try case-insensitive or partial match
            eprintln!("No help found for '{}'.", name);
            return;
        }
    };

    // Extract until next ### or ## heading
    let content = &REFERENCE[start + header.len()..];
    let end = content.find("\n### ")
        .or_else(|| content.find("\n## "))
        .unwrap_or(content.len());
    let section = content[..end].trim();

    // Parse and display
    println!("\n  {}", name);
    println!("  {}", "-".repeat(name.len()));

    let mut example_count = 0;
    let mut in_code = false;

    for line in section.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("```") {
            if in_code {
                in_code = false;
                println!();
            } else {
                in_code = true;
                example_count += 1;
                if example_count > 3 {
                    continue;
                }
            }
            continue;
        }

        if example_count > 3 {
            continue;
        }

        if in_code {
            println!("    {}", line);
        } else if trimmed.starts_with("**Type:**") {
            println!("  {}", trimmed.replace("**", ""));
        } else if trimmed.starts_with("**Signature:**") {
            println!("  {}", trimmed.replace("**", "").replace('`', ""));
        } else if trimmed == "---" {
            // Skip separators
        } else if !trimmed.is_empty() {
            // Description text — strip markdown
            let clean = trimmed
                .replace("**", "")
                .replace('`', "");
            println!("  {}", clean);
        }
    }
    println!();
}
