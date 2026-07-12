// Copyright (c) 2025 Damien Boureille
// Licensed under the MIT License.

//! Pretty-printer for Sheaf source forms.
//!
//! Renders a `SheafValue` AST with line breaks and indentation when it would
//! otherwise exceed the target width. Used by the REPL for `:show <fn>` source
//! display and for readable multi-line history recall.

use sheaf_compiler::SheafValue;

/// Target line width. Forms that fit are kept on a single line.
const MAX_WIDTH: usize = 90;

/// Render a `defn` definition for REPL `:show`.
/// `params` is the flat parameter list (names only); `body` is the AST body.
pub fn format_defn(name: &str, params: &[String], body: &SheafValue) -> String {
    let params_str = params.join(" ");
    let flat = format!("(defn {} [{}] {})", name, params_str, body);
    if flat.len() <= MAX_WIDTH {
        return flat;
    }
    let header = format!("(defn {} [{}]", name, params_str);
    let body_pp = pp(body, 2);
    format!("{}\n  {}", header, body_pp)
}

/// Pretty-print an arbitrary top-level form at column 0.
#[allow(dead_code)]
pub fn pretty_print(sv: &SheafValue) -> String {
    pp(sv, 0)
}

fn pad(n: usize) -> String {
    " ".repeat(n)
}

/// Render a value on a single line (no newlines).
fn flat(sv: &SheafValue) -> String {
    format!("{}", sv)
}

/// Pretty-print `sv`, assuming its first character starts at column `col`.
/// The returned string has no leading indent on the first line; continuation
/// lines carry their own indentation.
fn pp(sv: &SheafValue, col: usize) -> String {
    let f = flat(sv);
    if col + f.len() <= MAX_WIDTH || !is_breakable(sv) {
        return f;
    }
    break_form(sv, col)
}

fn is_breakable(sv: &SheafValue) -> bool {
    match sv {
        SheafValue::List(e, _) | SheafValue::Vector(e, _) => e.len() > 1,
        _ => false,
    }
}

fn break_form(sv: &SheafValue, col: usize) -> String {
    match sv {
        SheafValue::List(elems, _) => {
            if let Some(head) = elems.first().and_then(|h| h.as_symbol()) {
                match head {
                    "let" if elems.len() >= 3 => return break_let(elems, col),
                    "fn" if elems.len() >= 3 => return break_fn(elems, col),
                    "defn" if elems.len() >= 4 => return break_defn(elems, col),
                    "if" if elems.len() >= 3 => return break_if(elems, col),
                    "do" if elems.len() >= 2 => return break_do(elems, col),
                    _ => {}
                }
            }
            break_call(elems, col)
        }
        SheafValue::Vector(elems, _) => break_vector(elems, col),
        _ => flat(sv),
    }
}

/// (let [name val ...] body...)
fn break_let(elems: &[SheafValue], col: usize) -> String {
    let bindings = elems[1].as_vector().unwrap_or(&[]);
    let body_forms = &elems[2..];
    // "(let [" is 6 chars, so the first binding starts at col + 6.
    let bcol = col + 6;
    let body_col = col + 2;

    let mut out = String::from("(let [");
    let mut first = true;
    let mut i = 0;
    while i + 1 < bindings.len() {
        let name = &bindings[i];
        let value = &bindings[i + 1];
        if !first {
            out.push('\n');
            out.push_str(&pad(bcol));
        }
        first = false;
        let name_s = flat(name);
        out.push_str(&name_s);
        out.push(' ');
        let vcol = bcol + name_s.len() + 1;
        out.push_str(&pp(value, vcol));
        i += 2;
    }
    // Stray odd binding (malformed let): emit as-is.
    if i < bindings.len() {
        if !first {
            out.push('\n');
            out.push_str(&pad(bcol));
        }
        out.push_str(&flat(&bindings[i]));
    }
    out.push(']');
    for body in body_forms {
        out.push('\n');
        out.push_str(&pad(body_col));
        out.push_str(&pp(body, body_col));
    }
    out.push(')');
    out
}

/// (defn name [params] body...)
fn break_defn(elems: &[SheafValue], col: usize) -> String {
    let name = flat(&elems[1]);
    let params = flat(&elems[2]);
    let body_forms = &elems[3..];
    let body_col = col + 2;

    let mut out = format!("(defn {} {}", name, params);
    for body in body_forms {
        out.push('\n');
        out.push_str(&pad(body_col));
        out.push_str(&pp(body, body_col));
    }
    out.push(')');
    out
}

/// (fn [params] body...)
fn break_fn(elems: &[SheafValue], col: usize) -> String {
    let params = flat(&elems[1]);
    let body_forms = &elems[2..];
    let mut out = format!("(fn {}", params);

    // Keep a single short body form inline with the params.
    if body_forms.len() == 1 {
        let inline_col = col + out.len() + 1; // after "(fn [params] "
        let body_flat = flat(&body_forms[0]);
        if inline_col + body_flat.len() + 1 <= MAX_WIDTH {
            out.push(' ');
            out.push_str(&body_flat);
            out.push(')');
            return out;
        }
    }

    let body_col = col + 2;
    for body in body_forms {
        out.push('\n');
        out.push_str(&pad(body_col));
        out.push_str(&pp(body, body_col));
    }
    out.push(')');
    out
}

/// (if cond then else?)
fn break_if(elems: &[SheafValue], col: usize) -> String {
    let cond = &elems[1];
    let then_b = &elems[2];
    let else_b = elems.get(3);
    let branch_col = col + 2;

    let mut out = String::from("(if ");
    out.push_str(&pp(cond, col + 4));
    out.push('\n');
    out.push_str(&pad(branch_col));
    out.push_str(&pp(then_b, branch_col));
    if let Some(e) = else_b {
        out.push('\n');
        out.push_str(&pad(branch_col));
        out.push_str(&pp(e, branch_col));
    }
    out.push(')');
    out
}

/// (do form...)
fn break_do(elems: &[SheafValue], col: usize) -> String {
    let forms = &elems[1..];
    let body_col = col + 2;

    let mut out = String::from("(do");
    for form in forms {
        out.push('\n');
        out.push_str(&pad(body_col));
        out.push_str(&pp(form, body_col));
    }
    out.push(')');
    out
}

/// Generic function call: (f arg1 arg2 ...).
/// Short head: align arguments under the first argument.
/// Long head: head alone on the first line, arguments indented 2.
fn break_call(elems: &[SheafValue], col: usize) -> String {
    let head = flat(&elems[0]);
    let args = &elems[1..];
    if args.is_empty() {
        return format!("({})", head);
    }

    let mut out = String::from("(");
    out.push_str(&head);

    // A long head eats horizontal space when aligning under the first arg;
    // fall back to a fixed 2-space indent in that case.
    let long_head = head.len() > 12;
    if long_head {
        let arg_col = col + 2;
        for arg in args {
            out.push('\n');
            out.push_str(&pad(arg_col));
            out.push_str(&pp(arg, arg_col));
        }
    } else {
        let first_arg_col = col + 1 + head.len() + 1;
        for (i, arg) in args.iter().enumerate() {
            if i == 0 {
                out.push(' ');
                out.push_str(&pp(arg, first_arg_col));
            } else {
                out.push('\n');
                out.push_str(&pad(first_arg_col));
                out.push_str(&pp(arg, first_arg_col));
            }
        }
    }
    out.push(')');
    out
}

fn break_vector(elems: &[SheafValue], col: usize) -> String {
    let item_col = col + 1;
    let mut out = String::from("[");
    for (i, e) in elems.iter().enumerate() {
        if i > 0 {
            out.push('\n');
            out.push_str(&pad(item_col));
        }
        out.push_str(&pp(e, item_col));
    }
    out.push(']');
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use sheaf_compiler::core::parse;

    fn first_form(src: &str) -> SheafValue {
        parse(src, "<test>").unwrap().into_iter().next().unwrap()
    }

    #[test]
    fn short_form_stays_inline() {
        let out = pretty_print(&first_form("(* x (sigmoid x))"));
        assert_eq!(out, "(* x (sigmoid x))");
    }

    #[test]
    fn long_call_breaks_aligned() {
        // Build a call well beyond MAX_WIDTH.
        let src = "(foo aaaaaaaaaaaaaaaaa bbbbbbbbbbbbbbbbbb \
                  cccccccccccccccccc dddddddddddddddddd \
                  eeeeeeeeeeeeeeeee)";
        let out = pretty_print(&first_form(src));
        assert!(out.contains('\n'), "expected a line break");
        assert!(out.starts_with("(foo aaaaaaaaaaaaaaaaa"));
        // Every continuation line aligns under the first argument (col 5).
        for line in out.lines().skip(1) {
            assert!(line.starts_with("     "), "misaligned arg: {line:?}");
        }
    }

    #[test]
    fn let_bindings_align_and_body_indented() {
        let body = first_form(
            "(let [new-t (+ t 1)\n\
              new-m (tree-map (fn [md gd] (+ (* b1 md) (* (- 1 b1) gd))) m grads)]\n\
              [new-m new-t])",
        );
        let out = pretty_print(&body);
        assert!(out.starts_with("(let [new-t (+ t 1)"));
        let lines: Vec<&str> = out.lines().collect();
        // Second binding aligned 6 spaces in (after "(let [").
        assert!(lines[1].starts_with("      new-m"), "got: {:?}", lines[1]);
        // Body indented two spaces under the let.
        assert_eq!(lines.last().unwrap(), &"  [new-m new-t])");
    }

    #[test]
    fn format_defn_inline_when_short() {
        let body = first_form("(* x (sigmoid x))");
        let out = format_defn("silu", &["x".to_string()], &body);
        assert_eq!(out, "(defn silu [x] (* x (sigmoid x)))");
    }

    #[test]
    fn format_defn_breaks_body_when_long() {
        let body = first_form(
            "(let [aaa (+ 1 2 3 4 5 6 7 8 9) bbb (* 1 2 3 4 5 6 7 8 9) \
              ccc (- 1 2 3 4 5 6 7 8 9) ddd (/ 1 2 3 4 5 6 7 8 9)] aaa)",
        );
        let out = format_defn("some-very-long-function-name", &["x".to_string()], &body);
        assert!(out.starts_with("(defn some-very-long-function-name [x]\n  "));
    }
}
