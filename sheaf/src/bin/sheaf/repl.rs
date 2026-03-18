// Copyright (c) 2025 Damien Boureille
// Licensed under the MIT License.

//! Sheaf interactive REPL with autocompletion, multiline, help, and tracing.

use rustyline::completion::Completer;
use rustyline::error::ReadlineError;
use rustyline::highlight::Highlighter;
use rustyline::hint::Hinter;
use rustyline::validate::Validator;
use rustyline::{Editor, Helper};
use sheaf_compiler::forms::special_forms_registry;
use sheaf_compiler::interpreter::eval::Interpreter;
use sheaf_compiler::core::color;
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
    ":clear",
];

struct SheafHelper {
    names: Vec<String>,
    term_cols: usize,
    last_was_empty_tab: std::cell::Cell<bool>,
}

impl SheafHelper {
    fn new() -> Self {
        let mut names: Vec<String> = special_forms_registry()
            .keys()
            .map(|k| k.to_string())
            .collect();
        names.sort();
        Self { names, term_cols: 80, last_was_empty_tab: std::cell::Cell::new(false) }
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
            if self.last_was_empty_tab.get() {
                print_columns(&self.names, self.term_cols);
                print!("sheaf> ");
                let _ = std::io::Write::flush(&mut std::io::stdout());
                self.last_was_empty_tab.set(false);
            } else {
                self.last_was_empty_tab.set(true);
            }
            vec![]
        } else if word.starts_with(':') {
            self.last_was_empty_tab.set(false);
            REPL_COMMANDS
                .iter()
                .filter(|cmd| cmd.starts_with(word))
                .map(|s| s.to_string())
                .collect()
        } else {
            self.last_was_empty_tab.set(false);
            self.names
                .iter()
                .filter(|name| name.starts_with(word))
                .cloned()
                .collect()
        };

        Ok((start, matches))
    }
}

impl Validator for SheafHelper {}

impl Hinter for SheafHelper {
    type Hint = String;
    fn hint(&self, _line: &str, _pos: usize, _ctx: &rustyline::Context<'_>) -> Option<String> {
        None
    }
}

impl Highlighter for SheafHelper {
    fn highlight_prompt<'b, 's: 'b, 'p: 'b>(&'s self, prompt: &'p str, default: bool) -> Cow<'b, str> {
        if default {
            Cow::Owned(format!("{}{}{}", color::color(), prompt, color::reset()))
        } else {
            // Continuation prompt for multi-line input
            Cow::Owned(format!("{}     > {}", color::color(), color::reset()))
        }
    }
}

impl Helper for SheafHelper {}

fn print_columns(names: &[String], term_width: usize) {
    if names.is_empty() { return; }
    let col_width = names.iter().map(|n| n.len()).max().unwrap_or(4) + 2;
    let num_cols = (term_width / col_width).max(1);
    let num_rows = (names.len() + num_cols - 1) / num_cols;
    println!();
    for row in 0..num_rows {
        for col in 0..num_cols {
            let idx = col * num_rows + row;
            if idx < names.len() {
                print!("{:<width$}", names[idx], width = col_width);
            }
        }
        println!();
    }
}

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
    println!("Sheaf {} (:h for help)", env!("CARGO_PKG_VERSION"));

    let history_file = std::env::var_os("HOME")
        .map(|h| std::path::PathBuf::from(h).join(".sheaf_history"));

    let helper = SheafHelper::new();
    let config = rustyline::Config::builder()
        .completion_type(rustyline::CompletionType::List)
        .completion_prompt_limit(500)
        .behavior(rustyline::config::Behavior::PreferTerm)
        .build();
    let mut rl = Editor::with_config(config).expect("failed to init line editor");
    let term_cols = rl.dimensions().map(|(c, _)| c).unwrap_or(80);
    let mut helper = helper;
    helper.term_cols = term_cols;
    rl.set_helper(Some(helper));
    rl.bind_sequence(
        rustyline::KeyEvent(rustyline::KeyCode::Char('d'), rustyline::Modifiers::CTRL),
        rustyline::Cmd::EndOfFile,
    );
    if let Some(ref path) = history_file {
        let _ = rl.load_history(path);
    }

    let mut interp = Interpreter::new();

    // Initial refresh with builtins + stdlib (all stdlib loaded by CompilerContext)
    rl.helper_mut().unwrap().refresh(&interp);

    loop {
        match rl.readline("sheaf> ") {
            Ok(line) => {
                let trimmed = line.trim();
                if trimmed.is_empty() {
                    continue;
                }

                if trimmed.starts_with(':') {
                    rl.add_history_entry(&line).ok();
                    if handle_command(trimmed, &mut interp) {
                        break;
                    }
                    continue;
                }

                // Accumulate lines until parentheses are balanced
                let mut input = line.clone();
                while !is_balanced(&input) {
                    match rl.readline("     > ") {
                        Ok(cont) => {
                            input.push('\n');
                            input.push_str(&cont);
                        }
                        Err(ReadlineError::Interrupted) => {
                            eprintln!("^C");
                            input.clear();
                            break;
                        }
                        Err(_) => {
                            input.clear();
                            break;
                        }
                    }
                }

                let trimmed = input.trim();
                if trimmed.is_empty() {
                    continue;
                }

                // Collapse multi-line input to single line for history recall
                let history_entry: String = trimmed.split('\n')
                    .map(|l| l.trim())
                    .collect::<Vec<_>>()
                    .join(" ");
                rl.add_history_entry(&history_entry).ok();

                match interp.eval(trimmed) {
                    Ok(val) => {
                        if !is_silent_result(&val) {
                            if let Some(prefix) = val.repl_type_prefix() {
                                println!("=> {} = {}", prefix, val);
                            } else {
                                println!("=> {}", val);
                            }
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

        ":clear" => {
            print!("\x1b[2J\x1b[H");
            let _ = std::io::Write::flush(&mut std::io::stdout());
        }

        _ => {
            eprintln!("Unknown command: {}", cmd);
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
    println!("Sheaf console usage:
  :help, :h              Show this help
  :help, :h <name>       Help for a specific function or form
  :env                   List all functions and variables
  :registry, :reg        List user-defined functions
  :show <name>           Show a variable's value
  :clear                 Clear screen
  :quit, :q              Exit");
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
            // Description text, strip markdown
            let clean = trimmed
                .replace("**", "")
                .replace('`', "");
            println!("  {}", clean);
        }
    }
    println!();
}
