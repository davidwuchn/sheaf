// Copyright (c) 2026 Damien Boureille
// Licensed under the MIT License.

//! Macro expansion engine for Sheaf.
//! Handles defmacro registration, quasiquote expansion, and compile-time evaluation.

use std::collections::HashMap;

use crate::core::ast::SheafValue;
use crate::core::expr::FunctionDef;
use crate::core::error::{SheafError, SheafResult, SourceLocation};

/// A user-defined macro registered via (defmacro name [params] body-template).
#[derive(Debug, Clone)]
pub struct MacroDef {
    pub name: String,
    pub positional_params: Vec<String>,
    pub rest_param: Option<String>,
    pub body_template: SheafValue,
}

/// Macro expansion engine.
pub struct MacroEngine {
    pub macros: HashMap<String, MacroDef>,
    expansion_depth: usize,
    max_expansion_depth: usize,
}

/// Internal result from expanding a quasiquote element.
enum ExpandedItem {
    Single(SheafValue),
    Splice(Vec<SheafValue>),
}

type Bindings = HashMap<String, SheafValue>;

impl MacroEngine {
    pub fn new() -> Self {
        Self {
            macros: HashMap::new(),
            expansion_depth: 0,
            max_expansion_depth: 100,
        }
    }

    /// Expand macros in a SheafValue. Returns the (possibly transformed) AST.
    pub fn expand(
        &mut self,
        exp: &SheafValue,
        compiler_env: &HashMap<String, SheafValue>,
        registry: &HashMap<String, FunctionDef>,
    ) -> SheafResult<SheafValue> {
        let elements = match exp {
            SheafValue::List(elems, _) if !elems.is_empty() => elems,
            _ => return Ok(exp.clone()),
        };

        let loc = exp.location().clone();

        // Check if head is a registered macro
        if let Some(op_name) = elements[0].as_symbol() {
            if let Some(macro_def) = self.macros.get(op_name).cloned() {
                if self.expansion_depth >= self.max_expansion_depth {
                    return Err(SheafError::Compile {
                        message: format!(
                            "Macro expansion depth exceeded {} (possible infinite recursion in '{}')",
                            self.max_expansion_depth, op_name
                        ),
                        location: loc,
                    });
                }

                self.expansion_depth += 1;
                let args = &elements[1..];
                let bindings = Self::bind_params(&macro_def, args, &loc)?;
                let expanded =
                    self.expand_template(&macro_def.body_template, &bindings, compiler_env, registry, &loc)?;
                // Recursively expand (the result may contain more macro calls)
                let result = self.expand(&expanded, compiler_env, registry);
                self.expansion_depth -= 1;
                return result;
            }
        }

        // Not a macro call: recursively expand children
        let expanded: SheafResult<Vec<SheafValue>> = elements
            .iter()
            .map(|e| self.expand(e, compiler_env, registry))
            .collect();
        Ok(SheafValue::List(expanded?, loc))
    }

    /// Expand a macro body template with the given bindings.
    fn expand_template(
        &mut self,
        template: &SheafValue,
        bindings: &Bindings,
        compiler_env: &HashMap<String, SheafValue>,
        registry: &HashMap<String, FunctionDef>,
        call_loc: &SourceLocation,
    ) -> SheafResult<SheafValue> {
        match template {
            // Quasiquoted body: the common case for macros
            SheafValue::Quasiquote(inner, _) => {
                self.expand_quasiquote(inner, bindings, 0, compiler_env, registry, call_loc)
            }
            // Non-quasiquoted body (e.g., `(defmacro comment [& body] nil)`)
            SheafValue::Nil(_) => Ok(SheafValue::Nil(call_loc.clone())),
            _ => Ok(substitute_symbols(template, bindings)),
        }
    }

    /// Bind macro arguments to parameter names.
    fn bind_params(
        macro_def: &MacroDef,
        args: &[SheafValue],
        loc: &SourceLocation,
    ) -> SheafResult<Bindings> {
        let mut bindings = Bindings::new();
        let n_positional = macro_def.positional_params.len();

        for (i, param) in macro_def.positional_params.iter().enumerate() {
            if i < args.len() {
                bindings.insert(param.clone(), args[i].clone());
            } else {
                bindings.insert(param.clone(), SheafValue::Nil(loc.clone()));
            }
        }

        if let Some(ref rest_name) = macro_def.rest_param {
            let rest_args: Vec<SheafValue> = if args.len() > n_positional {
                args[n_positional..].to_vec()
            } else {
                vec![]
            };
            bindings.insert(rest_name.clone(), SheafValue::List(rest_args, loc.clone()));
        }

        Ok(bindings)
    }

    /// Expand quasiquote with depth tracking for nested quasiquotes.
    fn expand_quasiquote(
        &mut self,
        template: &SheafValue,
        bindings: &Bindings,
        depth: usize,
        compiler_env: &HashMap<String, SheafValue>,
        registry: &HashMap<String, FunctionDef>,
        loc: &SourceLocation,
    ) -> SheafResult<SheafValue> {
        match template {
            SheafValue::Unquote(inner, uloc) => {
                if depth == 0 {
                    let substituted = substitute_symbols(inner, bindings);
                    eval_at_compile_time(&substituted, bindings, compiler_env, registry, loc)
                } else {
                    let expanded =
                        self.expand_quasiquote(inner, bindings, depth - 1, compiler_env, registry, loc)?;
                    Ok(SheafValue::Unquote(Box::new(expanded), uloc.clone()))
                }
            }

            SheafValue::UnquoteSplicing(inner, uloc) => {
                if depth == 0 {
                    // Should only appear inside a list/vector processing loop,
                    // but handle gracefully
                    let substituted = substitute_symbols(inner, bindings);
                    eval_at_compile_time(&substituted, bindings, compiler_env, registry, loc)
                } else {
                    let expanded =
                        self.expand_quasiquote(inner, bindings, depth - 1, compiler_env, registry, loc)?;
                    Ok(SheafValue::UnquoteSplicing(Box::new(expanded), uloc.clone()))
                }
            }

            SheafValue::Quasiquote(inner, qloc) => {
                let expanded =
                    self.expand_quasiquote(inner, bindings, depth + 1, compiler_env, registry, loc)?;
                Ok(SheafValue::Quasiquote(Box::new(expanded), qloc.clone()))
            }

            SheafValue::List(elements, lloc) => {
                let mut result = Vec::new();
                for elem in elements {
                    match self.expand_quasiquote_element(elem, bindings, depth, compiler_env, registry, loc)? {
                        ExpandedItem::Single(v) => result.push(v),
                        ExpandedItem::Splice(vs) => result.extend(vs),
                    }
                }
                Ok(SheafValue::List(result, lloc.clone()))
            }

            SheafValue::Vector(elements, vloc) => {
                let mut result = Vec::new();
                for elem in elements {
                    match self.expand_quasiquote_element(elem, bindings, depth, compiler_env, registry, loc)? {
                        ExpandedItem::Single(v) => result.push(v),
                        ExpandedItem::Splice(vs) => result.extend(vs),
                    }
                }
                Ok(SheafValue::Vector(result, vloc.clone()))
            }

            SheafValue::Symbol(name, _) => {
                if depth == 0 {
                    if let Some(bound) = bindings.get(name) {
                        return Ok(bound.clone());
                    }
                }
                Ok(template.clone())
            }

            // Literals pass through unchanged
            _ => Ok(template.clone()),
        }
    }

    /// Expand a single element within a list/vector, detecting splice markers.
    fn expand_quasiquote_element(
        &mut self,
        elem: &SheafValue,
        bindings: &Bindings,
        depth: usize,
        compiler_env: &HashMap<String, SheafValue>,
        registry: &HashMap<String, FunctionDef>,
        loc: &SourceLocation,
    ) -> SheafResult<ExpandedItem> {
        if let SheafValue::UnquoteSplicing(inner, _) = elem {
            if depth == 0 {
                let substituted = substitute_symbols(inner, bindings);
                let evaled =
                    eval_at_compile_time(&substituted, bindings, compiler_env, registry, loc)?;
                match evaled {
                    SheafValue::List(items, _) => return Ok(ExpandedItem::Splice(items)),
                    SheafValue::Vector(items, _) => return Ok(ExpandedItem::Splice(items)),
                    SheafValue::Nil(_) => return Ok(ExpandedItem::Splice(vec![])),
                    other => {
                        return Err(SheafError::Compile {
                            message: format!("~@ splice requires a list, got: {}", other),
                            location: loc.clone(),
                        });
                    }
                }
            }
        }

        let expanded = self.expand_quasiquote(elem, bindings, depth, compiler_env, registry, loc)?;
        Ok(ExpandedItem::Single(expanded))
    }
}

/// Replace symbols in an expression with their bindings (no evaluation).
fn substitute_symbols(expr: &SheafValue, bindings: &Bindings) -> SheafValue {
    match expr {
        SheafValue::Symbol(name, _) => {
            if let Some(bound) = bindings.get(name) {
                bound.clone()
            } else {
                expr.clone()
            }
        }
        SheafValue::List(elems, loc) => {
            let substituted: Vec<SheafValue> =
                elems.iter().map(|e| substitute_symbols(e, bindings)).collect();
            SheafValue::List(substituted, loc.clone())
        }
        SheafValue::Vector(elems, loc) => {
            let substituted: Vec<SheafValue> =
                elems.iter().map(|e| substitute_symbols(e, bindings)).collect();
            SheafValue::Vector(substituted, loc.clone())
        }
        _ => expr.clone(),
    }
}

/// Evaluate an expression at macro-expansion time.
/// This is a mini-interpreter that works on SheafValue (AST data).
fn eval_at_compile_time(
    expr: &SheafValue,
    bindings: &Bindings,
    compiler_env: &HashMap<String, SheafValue>,
    registry: &HashMap<String, FunctionDef>,
    loc: &SourceLocation,
) -> SheafResult<SheafValue> {
    match expr {
        // Symbols: look up in bindings, then compiler env
        SheafValue::Symbol(name, _) => {
            if let Some(val) = bindings.get(name) {
                Ok(val.clone())
            } else if let Some(val) = compiler_env.get(name) {
                Ok(val.clone())
            } else {
                // Unknown symbol in compile-time eval: return as-is (it's data)
                Ok(expr.clone())
            }
        }

        // Literals pass through
        SheafValue::Integer(_, _)
        | SheafValue::Float(_, _)
        | SheafValue::String(_, _)
        | SheafValue::Boolean(_, _)
        | SheafValue::Keyword(_, _)
        | SheafValue::Nil(_) => Ok(expr.clone()),

        // Quoted expressions return their content
        SheafValue::Quote(inner, _) => Ok((**inner).clone()),

        // Vectors: eval each element
        SheafValue::Vector(elems, vloc) => {
            let evaled: SheafResult<Vec<SheafValue>> = elems
                .iter()
                .map(|e| eval_at_compile_time(e, bindings, compiler_env, registry, loc))
                .collect();
            Ok(SheafValue::Vector(evaled?, vloc.clone()))
        }

        // Lists: function calls
        SheafValue::List(elems, lloc) => {
            if elems.is_empty() {
                return Ok(expr.clone());
            }

            let func_name = match &elems[0] {
                SheafValue::Symbol(name, _) => name.as_str(),
                _ => {
                    // Not a symbol head: eval each element and return as data
                    let evaled: SheafResult<Vec<SheafValue>> = elems
                        .iter()
                        .map(|e| eval_at_compile_time(e, bindings, compiler_env, registry, loc))
                        .collect();
                    return Ok(SheafValue::List(evaled?, lloc.clone()));
                }
            };

            let args = &elems[1..];

            // Special forms in mini-eval
            match func_name {
                "quote" => {
                    return if args.is_empty() {
                        Ok(SheafValue::Nil(loc.clone()))
                    } else {
                        Ok(args[0].clone())
                    };
                }
                "if" => {
                    if args.len() < 2 {
                        return Err(compile_error("if requires at least 2 arguments", loc));
                    }
                    let cond = eval_at_compile_time(&args[0], bindings, compiler_env, registry, loc)?;
                    if is_truthy(&cond) {
                        return eval_at_compile_time(&args[1], bindings, compiler_env, registry, loc);
                    } else if args.len() > 2 {
                        return eval_at_compile_time(&args[2], bindings, compiler_env, registry, loc);
                    } else {
                        return Ok(SheafValue::Nil(loc.clone()));
                    }
                }
                "let" => {
                    if args.is_empty() {
                        return Ok(SheafValue::Nil(loc.clone()));
                    }
                    let pairs = match &args[0] {
                        SheafValue::Vector(v, _) => v,
                        _ => return Err(compile_error("let requires a vector of bindings", loc)),
                    };
                    let mut local_bindings = bindings.clone();
                    let mut i = 0;
                    while i + 1 < pairs.len() {
                        let name = match &pairs[i] {
                            SheafValue::Symbol(n, _) => n.clone(),
                            _ => return Err(compile_error("let binding name must be a symbol", loc)),
                        };
                        let val = eval_at_compile_time(
                            &pairs[i + 1],
                            &local_bindings,
                            compiler_env,
                            registry,
                            loc,
                        )?;
                        local_bindings.insert(name, val);
                        i += 2;
                    }
                    // Eval body expressions
                    let mut result = SheafValue::Nil(loc.clone());
                    for body_expr in &args[1..] {
                        result = eval_at_compile_time(
                            body_expr,
                            &local_bindings,
                            compiler_env,
                            registry,
                            loc,
                        )?;
                    }
                    return Ok(result);
                }
                _ => {}
            }

            // Eval arguments
            let evaled_args: SheafResult<Vec<SheafValue>> = args
                .iter()
                .map(|a| eval_at_compile_time(a, bindings, compiler_env, registry, loc))
                .collect();
            let evaled_args = evaled_args?;

            // Try user-defined function from registry
            if let Some(func_def) = registry.get(func_name) {
                let mut fn_bindings = Bindings::new();
                for (i, param) in func_def.params.iter().enumerate() {
                    if i < evaled_args.len() {
                        fn_bindings.insert(param.clone(), evaled_args[i].clone());
                    } else {
                        fn_bindings.insert(param.clone(), SheafValue::Nil(loc.clone()));
                    }
                }
                return eval_at_compile_time(&func_def.body, &fn_bindings, compiler_env, registry, loc);
            }

            // Built-in functions
            apply_builtin(func_name, &evaled_args, compiler_env, registry, loc, lloc)
        }

        _ => Ok(expr.clone()),
    }
}

/// Apply a built-in function at compile time (operating on SheafValue data).
fn apply_builtin(
    name: &str,
    args: &[SheafValue],
    compiler_env: &HashMap<String, SheafValue>,
    registry: &HashMap<String, FunctionDef>,
    loc: &SourceLocation,
    list_loc: &SourceLocation,
) -> SheafResult<SheafValue> {
    match name {
        "first" => {
            let items = expect_list_or_vec(args.first(), "first", loc)?;
            Ok(items.first().cloned().unwrap_or_else(|| SheafValue::Nil(loc.clone())))
        }
        "second" => {
            let items = expect_list_or_vec(args.first(), "second", loc)?;
            Ok(items.get(1).cloned().unwrap_or_else(|| SheafValue::Nil(loc.clone())))
        }
        "rest" => {
            let items = expect_list_or_vec(args.first(), "rest", loc)?;
            if items.is_empty() {
                Ok(SheafValue::List(vec![], loc.clone()))
            } else {
                Ok(SheafValue::List(items[1..].to_vec(), loc.clone()))
            }
        }
        "nth" => {
            if args.len() < 2 {
                return Err(compile_error("nth requires 2 arguments", loc));
            }
            let items = expect_list_or_vec(Some(&args[0]), "nth", loc)?;
            let idx = expect_int(&args[1], "nth index", loc)? as usize;
            Ok(items.get(idx).cloned().unwrap_or_else(|| SheafValue::Nil(loc.clone())))
        }
        "count" | "len" => {
            let items = expect_list_or_vec(args.first(), "count", loc)?;
            Ok(SheafValue::Integer(items.len() as i64, loc.clone()))
        }
        "list" => Ok(SheafValue::List(args.to_vec(), loc.clone())),
        "cons" => {
            if args.len() < 2 {
                return Err(compile_error("cons requires 2 arguments", loc));
            }
            let mut items = expect_list_or_vec(Some(&args[1]), "cons", loc)?.to_vec();
            items.insert(0, args[0].clone());
            Ok(SheafValue::List(items, loc.clone()))
        }
        "append" => {
            if args.len() < 2 {
                return Err(compile_error("append requires 2 arguments", loc));
            }
            let mut items = expect_list_or_vec(Some(&args[0]), "append", loc)?.to_vec();
            items.push(args[1].clone());
            Ok(SheafValue::List(items, loc.clone()))
        }
        "concat" => {
            let mut result = Vec::new();
            for arg in args {
                let items = expect_list_or_vec(Some(arg), "concat", loc)?;
                result.extend_from_slice(items);
            }
            Ok(SheafValue::List(result, loc.clone()))
        }
        "map" => {
            if args.len() < 2 {
                return Err(compile_error("map requires 2 arguments (fn, collection)", loc));
            }
            let func = &args[0];
            let items = expect_list_or_vec(Some(&args[1]), "map collection", loc)?;
            let mut result = Vec::new();
            for item in items {
                let call = SheafValue::List(
                    vec![func.clone(), item.clone()],
                    loc.clone(),
                );
                result.push(eval_at_compile_time(
                    &call,
                    &Bindings::new(),
                    compiler_env,
                    registry,
                    loc,
                )?);
            }
            Ok(SheafValue::List(result, loc.clone()))
        }
        ">" | "<" | ">=" | "<=" | "=" | "==" => {
            if args.len() < 2 {
                return Err(compile_error(&format!("{} requires 2 arguments", name), loc));
            }
            let a = to_number(&args[0]);
            let b = to_number(&args[1]);
            let result = match name {
                ">" => a > b,
                "<" => a < b,
                ">=" => a >= b,
                "<=" => a <= b,
                "=" | "==" => (a - b).abs() < 1e-10,
                _ => false,
            };
            Ok(SheafValue::Boolean(result, loc.clone()))
        }
        "+" | "-" | "*" | "/" => {
            if args.len() < 2 {
                return Err(compile_error(&format!("{} requires 2 arguments", name), loc));
            }
            let a = to_number(&args[0]);
            let b = to_number(&args[1]);
            let result = match name {
                "+" => a + b,
                "-" => a - b,
                "*" => a * b,
                "/" => {
                    if b == 0.0 {
                        return Err(compile_error("Division by zero", loc));
                    }
                    a / b
                }
                _ => 0.0,
            };
            // Return as integer if both inputs were integers and result is integral
            if matches!(&args[0], SheafValue::Integer(_, _))
                && matches!(&args[1], SheafValue::Integer(_, _))
                && result.fract() == 0.0
            {
                Ok(SheafValue::Integer(result as i64, loc.clone()))
            } else {
                Ok(SheafValue::Float(result, loc.clone()))
            }
        }
        "not" => {
            let nil = SheafValue::Nil(loc.clone());
            let val = args.first().unwrap_or(&nil);
            Ok(SheafValue::Boolean(!is_truthy(val), loc.clone()))
        }
        "str" => {
            let mut s = String::new();
            for arg in args {
                s.push_str(&sheaf_value_to_string(arg));
            }
            Ok(SheafValue::String(s, loc.clone()))
        }
        _ => {
            // Unknown function: return as data (the list stays as-is)
            let mut elems = vec![SheafValue::Symbol(name.to_string(), loc.clone())];
            elems.extend_from_slice(args);
            Ok(SheafValue::List(elems, list_loc.clone()))
        }
    }
}

/// Check if a SheafValue is truthy (non-nil, non-false).
fn is_truthy(val: &SheafValue) -> bool {
    !matches!(val, SheafValue::Nil(_) | SheafValue::Boolean(false, _))
}

/// Extract number from SheafValue for arithmetic.
fn to_number(val: &SheafValue) -> f64 {
    match val {
        SheafValue::Integer(n, _) => *n as f64,
        SheafValue::Float(f, _) => *f,
        _ => 0.0,
    }
}

/// Extract items from a list or vector.
fn expect_list_or_vec<'a>(
    val: Option<&'a SheafValue>,
    ctx: &str,
    loc: &SourceLocation,
) -> SheafResult<&'a [SheafValue]> {
    match val {
        Some(SheafValue::List(items, _)) => Ok(items),
        Some(SheafValue::Vector(items, _)) => Ok(items),
        Some(other) => Err(compile_error(
            &format!("{}: expected list or vector, got {}", ctx, other),
            loc,
        )),
        None => Err(compile_error(&format!("{}: missing argument", ctx), loc)),
    }
}

/// Extract integer from SheafValue.
fn expect_int(val: &SheafValue, ctx: &str, loc: &SourceLocation) -> SheafResult<i64> {
    match val {
        SheafValue::Integer(n, _) => Ok(*n),
        _ => Err(compile_error(
            &format!("{}: expected integer, got {}", ctx, val),
            loc,
        )),
    }
}

/// Convert a SheafValue to its string representation (for `str` builtin).
fn sheaf_value_to_string(val: &SheafValue) -> String {
    match val {
        SheafValue::String(s, _) => s.clone(),
        SheafValue::Symbol(s, _) => s.clone(),
        SheafValue::Keyword(k, _) => format!(":{}", k),
        SheafValue::Integer(n, _) => n.to_string(),
        SheafValue::Float(f, _) => format!("{}", f),
        SheafValue::Boolean(b, _) => b.to_string(),
        SheafValue::Nil(_) => "nil".to_string(),
        other => format!("{}", other),
    }
}

fn compile_error(msg: &str, loc: &SourceLocation) -> SheafError {
    SheafError::Compile {
        message: msg.to_string(),
        location: loc.clone(),
    }
}
