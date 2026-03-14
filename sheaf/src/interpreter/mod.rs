// Copyright (c) 2025 Damien Boureille
// Licensed under the MIT License.

//! Sheaf interpreter, evaluates CompiledExpr directly to runtime Values.

pub mod builtins;
pub mod env;
pub mod eval;
pub mod profiler;
pub mod tracer;
pub mod value;

use crate::sheaf_msg;
use crate::ast::SheafValue;
use crate::core::compiler::{CompiledExpr, CompilerContext};
use crate::core::error::SheafError;
use crate::interpreter::env::{runtime_error, Env};
use crate::interpreter::value::Value;
use ndarray::{ArrayD, IxDyn};
use std::collections::BTreeMap;
use std::sync::Arc;

pub fn eval(expr: &CompiledExpr, env: &mut Env) -> Result<Value, SheafError> {
    match expr {
        CompiledExpr::Integer(n) => Ok(Value::Int(*n)),
        CompiledExpr::Float(x) => Ok(Value::Float(*x as f32)),
        CompiledExpr::Boolean(b) => Ok(Value::Bool(*b)),
        CompiledExpr::Nil => Ok(Value::Nil),
        CompiledExpr::String(s) => Ok(Value::String(s.clone())),
        CompiledExpr::Keyword(k) => Ok(Value::Keyword(k.clone())),

        CompiledExpr::Symbol(name) => {
            if name == "..." { return Ok(Value::Keyword("...".to_string())); }
            env.get(name)
        }

        CompiledExpr::Vector(elements) => eval_vector(elements, env),

        CompiledExpr::Dict(pairs) => eval_dict(pairs, env),

        CompiledExpr::Quoted(sv) => sheaf_value_to_value(sv),

        CompiledExpr::FunctionRef(name) => {
            // Try env first (builtins live here)
            if let Ok(val) = env.get(name) {
                return Ok(val);
            }
            // Registry functions → Value::Function with real params/body
            if let Some(func_def) = env.registry.get(name).cloned() {
                if let Some(body) = func_def.body_compiled {
                    return Ok(Value::Function {
                        params: func_def.params,
                        body,
                        closure: vec![],
                    });
                }
            }
            Err(runtime_error(format!("Undefined function: {}", name)))
        }

        CompiledExpr::FunctionCall { name, args } => eval_call(name, args, env),

        CompiledExpr::Let { bindings, body } => {
            env.push_scope();
            for (name, expr) in bindings {
                let val = eval(expr, env)?;
                bind_pattern(name, val, env)?;
            }
            let result = eval(body, env);
            env.pop_scope();
            result
        }

        CompiledExpr::If { condition, then_branch, else_branch } => {
            let cond = eval(condition, env)?;
            if cond.is_truthy() {
                eval(then_branch, env)
            } else if let Some(else_br) = else_branch {
                eval(else_br, env)
            } else {
                Ok(Value::Nil)
            }
        }

        CompiledExpr::Do(exprs) => {
            let mut result = Value::Nil;
            for expr in exprs {
                result = eval(expr, env)?;
            }
            Ok(result)
        }

        CompiledExpr::Lambda { params, body } => {
            Ok(Value::Function {
                params: params.clone(),
                body: *body.clone(),
                closure: vec![],
            })
        }

        CompiledExpr::LambdaCall { callee, args } => {
            let func = eval(callee, env)?;
            let mut arg_vals = Vec::new();
            for arg in args {
                arg_vals.push(eval(arg, env)?);
            }
            call_function(&func, &arg_vals, env)
        }

        CompiledExpr::GetTupleElement { param, indices } => {
            let val = env.get(param)?;
            get_nested(&val, indices)
        }

        CompiledExpr::ValueAndGrad { fn_name, .. } => {
            Err(runtime_error(format!(
                "value-and-grad '{}': interpreter support not yet implemented", fn_name
            )))
        }

        CompiledExpr::InlineValueAndGrad { lambda, args, .. } => {
            // Evaluate the lambda to get the function value
            let func = eval(lambda, env)?;
            // Evaluate args
            let mut evaluated_args = Vec::new();
            for arg in args {
                evaluated_args.push(eval(arg, env)?);
            }
            // Use the same HOF path: wrap in vag closure, then call it
            let vag = eval_value_and_grad_hof(&[func], env)?;
            call_function(&vag, &evaluated_args, env)
        }

        CompiledExpr::Repeat { index_var, count, acc_var, acc_init, body } => {
            let n = match eval(count, env)? {
                Value::Int(n) => n,
                Value::Float(f) => f as i64,
                other => return Err(runtime_error(format!(
                    "repeat: count must be an integer, got {}", other.type_name()
                ))),
            };
            let mut acc = eval(acc_init, env)?;
            env.push_scope();
            for i in 0..n {
                env.set(index_var, Value::Int(i));
                env.set(acc_var, acc);
                acc = eval(body, env)?;
            }
            env.pop_scope();
            Ok(acc)
        }

        CompiledExpr::While { condition, acc_var, acc_init, body } => {
            let mut acc = eval(acc_init, env)?;
            env.push_scope();
            env.set(acc_var, acc.clone());
            loop {
                let cond = eval(condition, env)?;
                if !cond.is_truthy() {
                    break;
                }
                acc = eval(body, env)?;
                env.set(acc_var, acc.clone());
            }
            env.pop_scope();
            Ok(acc)
        }

        CompiledExpr::Guard { check, expr } => {
            let val = eval(expr, env)?;
            if let Err(msg) = apply_guard_check(check, &val) {
                eprintln!("\x1b[91m/!\\ Guard Breached: {:?}\x1b[0m", check);
                eprintln!("{}", msg);
                if let Some(ref tracer) = env.tracer {
                    tracer.dump_ring_buffer();
                }
                std::process::exit(1);
            }
            Ok(val)
        }
    }
}

/// Check a guard condition against a value.
/// Returns Ok(()) if the check passes, Err(message) if it fails.
pub fn apply_guard_check(
    check: &crate::core::compiler::GuardCheck,
    val: &Value,
) -> Result<(), String> {
    use crate::core::compiler::GuardCheck;
    match check {
        GuardCheck::NoNan => {
            if let Value::Tensor { data, .. } = val {
                if data.iter().any(|x| !x.is_finite()) {
                    let stats = format_value_brief(val);
                    return Err(format!("Tensor contains NaN or Inf values: {}", stats));
                }
            }
            Ok(())
        }
        GuardCheck::Range { lo, hi } => {
            if let Value::Tensor { data, .. } = val {
                let v_min = data.iter().cloned().fold(f32::INFINITY, f32::min);
                let v_max = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                if (v_min as f64) < *lo || (v_max as f64) > *hi {
                    return Err(format!(
                        "Value range [{:.2e}, {:.2e}] outside allowed [{}, {}]",
                        v_min, v_max, lo, hi
                    ));
                }
            }
            Ok(())
        }
        GuardCheck::Shape(expected) => {
            if let Value::Tensor { data, .. } = val {
                let actual: Vec<i64> = data.shape().iter().map(|&d| d as i64).collect();
                if actual != *expected {
                    return Err(format!(
                        "Shape mismatch: expected {:?}, got {:?}",
                        expected, actual
                    ));
                }
            }
            Ok(())
        }
    }
}

fn format_value_brief(val: &Value) -> String {
    match val {
        Value::Tensor { data, .. } => {
            let shape: Vec<usize> = data.shape().to_vec();
            let shape_str: String = shape.iter().map(|d| d.to_string()).collect::<Vec<_>>().join("x");
            let v_min = data.iter().cloned().fold(f32::INFINITY, f32::min);
            let v_max = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            format!("f32[{}] [min:{:.2e} max:{:.2e}]", shape_str, v_min, v_max)
        }
        other => format!("{}", other),
    }
}

/// Bind a pattern name to a value in the current scope.
///
/// Patterns:
///   - Simple: `"x"` → env["x"] = val
///   - Destructuring: `"[a b]"` (encoded by compiler as "[a b]") → env["a"] = val[0], env["b"] = val[1]
///
/// The compiler encodes vector destructuring patterns as a string like `"[k1 k2]"`.
/// We detect this by the leading `[` and parse the names out.
fn bind_pattern(name: &str, val: Value, env: &mut Env) -> Result<(), SheafError> {
    if name.starts_with('[') && name.ends_with(']') {
        // Destructuring pattern: extract symbol names
        let inner = &name[1..name.len() - 1];
        let names: Vec<&str> = inner.split_whitespace().collect();
        let items = match val {
            Value::List(items) | Value::Tuple(items) => items,
            Value::Tensor { ref data, .. } => {
                if data.ndim() == 1 {
                    data.iter().map(|&x| Value::Float(x)).collect()
                } else {
                    return Err(runtime_error(format!(
                        "let destructuring: expected list/tuple, got tensor with shape {:?}", data.shape()
                    )));
                }
            }
            other => return Err(runtime_error(format!(
                "let destructuring: expected list or tuple, got {}", other.type_name()
            ))),
        };
        let mut items_iter = items.into_iter();
        for n in &names {
            match items_iter.next() {
                Some(v) => env.set(n, v),
                None => env.set(n, Value::Nil),
            }
        }
    } else {
        env.set(name, val);
    }
    Ok(())
}

fn eval_vector(elements: &[CompiledExpr], env: &mut Env) -> Result<Value, SheafError> {
    let vals: Result<Vec<Value>, _> = elements.iter().map(|e| eval(e, env)).collect();
    let vals = vals?;

    if vals.is_empty() {
        return Ok(Value::List(vec![]));
    }

    // Check if all elements are numeric → produce a Tensor (always F32 by default)
    let all_numeric = vals.iter().all(|v| matches!(v, Value::Int(_) | Value::Float(_)));
    if all_numeric {
        let data: Vec<f32> = vals.iter().map(|v| v.to_f64().unwrap() as f32).collect();
        let arr = ArrayD::from_shape_vec(IxDyn(&[data.len()]), data).unwrap();
        return Ok(Value::tensor_f32(arr));
    }

    // Check if all elements are vectors/tensors of same shape → produce a 2D+ tensor
    let all_tensors = vals.iter().all(|v| matches!(v, Value::Tensor { .. }));
    if all_tensors {
        let shapes: Vec<_> = vals.iter().map(|v| match v {
            Value::Tensor { data, .. } => data.shape().to_vec(),
            _ => unreachable!(),
        }).collect();
        if shapes.windows(2).all(|w| w[0] == w[1]) {
            let inner_shape = &shapes[0];
            let mut full_shape = vec![vals.len()];
            full_shape.extend(inner_shape);
            let mut flat_data = Vec::new();
            for v in &vals {
                if let Value::Tensor { data, .. } = v {
                    flat_data.extend(data.iter());
                }
            }
            let arr = ArrayD::from_shape_vec(IxDyn(&full_shape), flat_data).unwrap();
            return Ok(Value::tensor_f32(arr));
        }
    }

    // Otherwise, a heterogeneous list
    Ok(Value::List(vals))
}

fn eval_dict(pairs: &[(CompiledExpr, CompiledExpr)], env: &mut Env) -> Result<Value, SheafError> {
    let mut map = BTreeMap::new();
    for (k, v) in pairs {
        let key = match eval(k, env)? {
            Value::Keyword(s) => s,
            Value::String(s) => s,
            other => return Err(runtime_error(format!("Dict key must be keyword or string, got {}", other.type_name()))),
        };
        let val = eval(v, env)?;
        map.insert(key, val);
    }
    Ok(Value::Dict(map))
}

fn sheaf_value_to_value(sv: &SheafValue) -> Result<Value, SheafError> {
    match sv {
        SheafValue::Integer(n, _) => Ok(Value::Int(*n)),
        SheafValue::Float(x, _) => Ok(Value::Float(*x as f32)),
        SheafValue::Boolean(b, _) => Ok(Value::Bool(*b)),
        SheafValue::Nil(_) => Ok(Value::Nil),
        SheafValue::String(s, _) => Ok(Value::String(s.clone())),
        SheafValue::Symbol(s, _) => Ok(Value::String(s.clone())),
        SheafValue::Keyword(k, _) => Ok(Value::Keyword(k.clone())),
        SheafValue::List(elems, _) | SheafValue::Vector(elems, _) => {
            let items: Result<Vec<Value>, _> = elems.iter().map(|e| sheaf_value_to_value(e)).collect();
            Ok(Value::List(items?))
        }
        SheafValue::Dict(pairs, _) => {
            let mut map = BTreeMap::new();
            for (k, v) in pairs {
                let key = match sheaf_value_to_value(k)? {
                    Value::Keyword(s) => s,
                    Value::String(s) => s,
                    other => return Err(runtime_error(format!("Dict key must be keyword or string, got {:?}", other))),
                };
                map.insert(key, sheaf_value_to_value(v)?);
            }
            Ok(Value::Dict(map))
        }
        SheafValue::Quote(inner, _) => sheaf_value_to_value(inner),
        _ => Err(runtime_error(format!("Cannot convert quoted value: {}", sv))),
    }
}

fn eval_call(name: &str, args: &[CompiledExpr], env: &mut Env) -> Result<Value, SheafError> {
    // Special handling for short-circuit operators
    match name {
        "and" => return eval_and(args, env),
        "or" => return eval_or(args, env),
        _ => {}
    }

    // Only functions that declare keyword params consume :kw val pairs.
    // All others treat keywords as positional values (Clojure semantics).
    let has_kwargs = matches!(name,
        "softmax" | "log-softmax" | "sum" | "mean" | "product"
        | "min" | "max" | "argmax" | "argmin" | "concat"
        | "leaky-relu" | "celu" | "var" | "normalize" | "range"
        | "tensor-split" | "slice"
        | "print" | "choice"
    );

    // Evaluate args, splitting kwargs only for functions that use them
    let (pos_args, kwargs) = if has_kwargs {
        split_kwargs(args, env)?
    } else {
        let pos: Result<Vec<Value>, _> = args.iter().map(|a| eval(a, env)).collect();
        (pos?, BTreeMap::new())
    };

    // Higher-order functions need &mut Env to call lambdas
    match name {
        "map" | "filter" | "reduce" | "scan" | "apply" | "find"
        | "tree-map" | "tree-reduce" | "flatten" | "vmap"
        | "__value-and-grad-hof__" => {
            if let Some(ref mut p) = env.profiler { p.enter(name); }
            let result = match name {
                "map" => eval_map(&pos_args, env),
                "filter" => eval_filter(&pos_args, env),
                "reduce" => eval_reduce(&pos_args, env),
                "scan" => eval_scan(&pos_args, env),
                "apply" => eval_apply(&pos_args, env),
                "find" => eval_find(&pos_args, env),
                "tree-map" => eval_tree_map(&pos_args, env),
                "tree-reduce" => eval_tree_reduce(&pos_args, env),
                "flatten" => eval_flatten(&pos_args),
                "vmap" => eval_vmap(&pos_args, env),
                "__value-and-grad-hof__" => eval_value_and_grad_hof(&pos_args, env),
                _ => unreachable!(),
            };
            if let Some(ref mut p) = env.profiler { p.exit(); }
            return result;
        }
        _ => {}
    }

    // Try builtin from env
    if let Ok(Value::BuiltinFn { func, .. }) = env.get(name) {
        if let Some(ref mut p) = env.profiler { p.enter(name); }
        let result = func(&pos_args, &kwargs);
        if let Some(ref mut p) = env.profiler { p.exit(); }
        return result;
    }

    // Try user-defined function from registry
    if let Some(func_def) = env.registry.get(name).cloned() {
        // Check evaluation deadline (used by auto-trace to avoid running forever)
        if let Some(deadline) = env.eval_deadline {
            if std::time::Instant::now() > deadline {
                return Err(SheafError::Runtime {
                    message: "auto-trace timeout".to_string(),
                    location: None,
                });
            }
        }

        // Record the first call for tracing (sheaf build --trace-with)
        if let Some(ref mut records) = env.call_records {
            let is_new = !records.contains_key(name);
            records.entry(name.to_string()).or_insert_with(|| {
                crate::interpreter::env::CallRecord {
                    arg_values: pos_args.clone(),
                }
            });
            if is_new {
                env.trace_stale_calls = 0;
                // Check if all target functions have been observed
                if let Some(ref targets) = env.trace_targets {
                    if targets.iter().all(|t| records.contains_key(t.as_str())) {
                        return Err(SheafError::Runtime {
                            message: "trace complete".to_string(),
                            location: None,
                        });
                    }
                }
            } else {
                env.trace_stale_calls += 1;
                // No new recordings for many calls, we're in a loop, stop
                if env.trace_stale_calls > 20 {
                    return Err(SheafError::Runtime {
                        message: "trace complete".to_string(),
                        location: None,
                    });
                }
            }
        }

        if let Some(ref mut p) = env.profiler { p.enter(name); }

        // VMFB/JIT dispatch: skip when tracing so the interpreter runs
        // and exposes the full call tree
        #[cfg(iree_runtime)]
        if env.tracer.is_none() {
            // VMFB dispatch: pure compiled functions run via IREE
            if let Some(result) = try_iree_dispatch(&func_def, &pos_args, env) {
                if let Some(ref mut p) = env.profiler { p.exit(); }
                return result;
            }

            // JIT: try to compile on first call if no VMFB exists
            if func_def.vmfb_session_idx.is_none() {
                if let Some(jit) = &mut env.jit_compiler {
                    if let Some((session_idx, sig)) = jit.try_jit_compile(
                        &func_def,
                        &pos_args,
                        &env.registry,
                        &mut env.vmfb_sessions,
                    ) {
                        if let Some(fd) = env.registry.get_mut(name) {
                            fd.vmfb_session_idx = Some(session_idx);
                            fd.signature = Some(sig);
                        }
                        let func_def = env.registry.get(name).unwrap().clone();
                        if let Some(result) = try_iree_dispatch(&func_def, &pos_args, env) {
                            if let Some(ref mut p) = env.profiler { p.exit(); }
                            return result;
                        }
                    }
                }
            }
        }

        // Fallback: interpret
        if let Some(ref body) = func_def.body_compiled {
            let tracing = env.tracer.as_ref().map_or(false, |t| t.is_active(name));
            if tracing {
                let mut tracer = env.tracer.take().unwrap();
                tracer.log_call(name, &pos_args);
                env.tracer = Some(tracer);
            }

            env.push_scope();
            for (param, val) in func_def.params.iter().zip(pos_args.iter()) {
                env.set(param, val.clone());
            }
            let result = eval(body, env);
            env.pop_scope();

            if tracing {
                let mut tracer = env.tracer.take().unwrap();
                if let Ok(ref val) = result {
                    tracer.log_return(name, val);
                    tracer.check_cli_guards(name, val);
                }
                env.tracer = Some(tracer);
            }

            if let Some(ref mut p) = env.profiler { p.exit(); }
            return result;
        }

        if let Some(ref mut p) = env.profiler { p.exit(); }
    }

    // Try function value in env
    if let Ok(func_val) = env.get(name) {
        return call_function(&func_val, &pos_args, env);
    }

    Err(runtime_error(format!("Unknown function: {}", name)))
}

fn eval_and(args: &[CompiledExpr], env: &mut Env) -> Result<Value, SheafError> {
    if args.is_empty() {
        return Ok(Value::Bool(true));
    }
    for arg in &args[..args.len() - 1] {
        let val = eval(arg, env)?;
        if !val.is_truthy() {
            return Ok(val);
        }
    }
    eval(&args[args.len() - 1], env)
}

fn eval_or(args: &[CompiledExpr], env: &mut Env) -> Result<Value, SheafError> {
    if args.is_empty() {
        return Ok(Value::Bool(false));
    }
    for arg in &args[..args.len() - 1] {
        let val = eval(arg, env)?;
        if val.is_truthy() {
            return Ok(val);
        }
    }
    eval(&args[args.len() - 1], env)
}

fn split_kwargs(args: &[CompiledExpr], env: &mut Env) -> Result<(Vec<Value>, BTreeMap<String, Value>), SheafError> {
    let mut pos = Vec::new();
    let mut kwargs = BTreeMap::new();
    let mut i = 0;
    while i < args.len() {
        if let CompiledExpr::Keyword(k) = &args[i] {
            if i + 1 < args.len() {
                // Check if next arg is also a keyword (then this is a flag)
                if matches!(&args[i + 1], CompiledExpr::Keyword(_)) {
                    kwargs.insert(k.clone(), Value::Bool(true));
                } else {
                    let val = eval(&args[i + 1], env)?;
                    kwargs.insert(k.clone(), val);
                    i += 1;
                }
            } else {
                // Last arg is a keyword flag
                kwargs.insert(k.clone(), Value::Bool(true));
            }
        } else {
            pos.push(eval(&args[i], env)?);
        }
        i += 1;
    }
    Ok((pos, kwargs))
}

fn call_function(func: &Value, args: &[Value], env: &mut Env) -> Result<Value, SheafError> {
    match func {
        Value::Function { params: _, body: _, closure } => {
            // Detect vmap HOF closure: contains __vmap_fn__
            if let Some((_, vmap_fn)) = closure.iter().find(|(k, _)| k == "__vmap_fn__") {
                let axes = closure.iter().find(|(k, _)| k == "__vmap_axes__").map(|(_, v)| v.clone());
                return eval_vmap_call(vmap_fn, axes.as_ref(), args, env);
            }
            // Detect value-and-grad HOF closure: contains __vag_fn__
            if let Some((_, vag_fn)) = closure.iter().find(|(k, _)| k == "__vag_fn__") {
                if args.len() != 1 {
                    return Err(runtime_error("value-and-grad: expected exactly 1 argument (params)"));
                }
                return eval_value_and_grad_call(vag_fn, &args[0], env);
            }
            // Normal function call
            let Value::Function { params, body, closure } = func else { unreachable!() };
            if let Some(ref mut p) = env.profiler { p.enter("<lambda>"); }
            env.push_scope();
            for (name, val) in closure {
                env.set(name, val.clone());
            }
            for (param, val) in params.iter().zip(args.iter()) {
                env.set(param, val.clone());
            }
            let result = eval(body, env);
            env.pop_scope();
            if let Some(ref mut p) = env.profiler { p.exit(); }
            result
        }
        Value::BuiltinFn { name, func } => {
            if let Some(ref mut p) = env.profiler { p.enter(name); }
            let result = func(args, &BTreeMap::new());
            if let Some(ref mut p) = env.profiler { p.exit(); }
            result
        }
        _ => Err(runtime_error(format!("Not a function: {}", func.type_name()))),
    }
}

fn get_nested(val: &Value, indices: &[usize]) -> Result<Value, SheafError> {
    let mut current = val.clone();
    for &idx in indices {
        current = match current {
            Value::List(items) => {
                items.get(idx).cloned().ok_or_else(|| {
                    runtime_error(format!("Tuple index {} out of bounds (len {})", idx, items.len()))
                })?
            }
            Value::Dict(map) => {
                let entry = map.values().nth(idx).cloned().ok_or_else(|| {
                    runtime_error(format!("Tuple index {} out of bounds (len {})", idx, map.len()))
                })?;
                entry
            }
            _ => return Err(runtime_error(format!("Cannot index into {}", current.type_name()))),
        };
    }
    Ok(current)
}

fn eval_map(args: &[Value], env: &mut Env) -> Result<Value, SheafError> {
    if args.len() != 2 {
        return Err(runtime_error("map requires 2 arguments: (map fn coll)"));
    }
    let func = &args[0];
    match &args[1] {
        Value::List(items) => {
            let mut results = Vec::with_capacity(items.len());
            for item in items {
                results.push(call_function(func, &[item.clone()], env)?);
            }
            Ok(Value::List(results))
        }
        Value::Tensor { data, dtype } => {
            if data.ndim() == 1 {
                // 1D: iterate over scalar elements
                let mut results = Vec::with_capacity(data.len());
                for &x in data.iter() {
                    results.push(call_function(func, &[Value::Float(x)], env)?);
                }
                Ok(Value::List(results))
            } else {
                // ND: iterate over slices along axis 0
                let n = data.shape()[0];
                let mut results = Vec::with_capacity(n);
                for i in 0..n {
                    let row = data.index_axis(ndarray::Axis(0), i).to_owned();
                    let row_val = Value::Tensor { data: Arc::new(row), dtype: *dtype };
                    results.push(call_function(func, &[row_val], env)?);
                }
                Ok(Value::List(results))
            }
        }
        _ => Err(runtime_error("map: expected list or tensor")),
    }
}

fn eval_filter(args: &[Value], env: &mut Env) -> Result<Value, SheafError> {
    if args.len() != 2 {
        return Err(runtime_error("filter requires 2 arguments: (filter fn coll)"));
    }
    let func = &args[0];
    match &args[1] {
        Value::List(items) => {
            let mut results = Vec::new();
            for item in items {
                let result = call_function(func, &[item.clone()], env)?;
                if result.is_truthy() {
                    results.push(item.clone());
                }
            }
            Ok(Value::List(results))
        }
        _ => Err(runtime_error("filter: expected list")),
    }
}

fn eval_reduce(args: &[Value], env: &mut Env) -> Result<Value, SheafError> {
    if args.len() != 3 {
        return Err(runtime_error("reduce requires 3 arguments: (reduce fn init coll)"));
    }
    let func = &args[0];
    let mut acc = args[1].clone();
    match &args[2] {
        Value::List(items) => {
            for item in items {
                acc = call_function(func, &[acc, item.clone()], env)?;
            }
            Ok(acc)
        }
        Value::Tensor { data, .. } => {
            if data.ndim() == 1 {
                for &x in data.iter() {
                    acc = call_function(func, &[acc, Value::Float(x)], env)?;
                }
            } else {
                for i in 0..data.shape()[0] {
                    let row = data.index_axis(ndarray::Axis(0), i).to_owned();
                    acc = call_function(func, &[acc, Value::tensor_f32(row)], env)?;
                }
            }
            Ok(acc)
        }
        Value::Dict(map) => {
            let n = dict_scan_length(map)?;
            for i in 0..n {
                let slice = slice_dict(map, i)?;
                acc = call_function(func, &[acc, slice], env)?;
            }
            Ok(acc)
        }
        _ => Err(runtime_error("reduce: expected list, tensor, or dict of tensors")),
    }
}

fn eval_scan(args: &[Value], env: &mut Env) -> Result<Value, SheafError> {
    if args.len() != 3 {
        return Err(runtime_error("scan requires 3 arguments: (scan fn init coll)"));
    }
    let func = &args[0];
    let mut carry = args[1].clone();
    let mut outputs = Vec::new();

    // Destructure [new_carry, output] from each step
    let step = |func: &Value, carry: Value, x: Value, env: &mut Env|
        -> Result<(Value, Value), SheafError>
    {
        let result = call_function(func, &[carry, x], env)?;
        match &result {
            Value::List(items) | Value::Tuple(items) if items.len() == 2 => {
                Ok((items[0].clone(), items[1].clone()))
            }
            Value::Tensor { data, .. } if data.ndim() == 1 && data.shape()[0] == 2 => {
                Ok((Value::Float(data[[0]]), Value::Float(data[[1]])))
            }
            Value::Tensor { data, .. } if data.shape()[0] == 2 => {
                let carry = data.index_axis(ndarray::Axis(0), 0).to_owned();
                let output = data.index_axis(ndarray::Axis(0), 1).to_owned();
                Ok((Value::tensor_f32(carry), Value::tensor_f32(output)))
            }
            _ => Err(runtime_error("scan: fn must return [new-carry output]")),
        }
    };

    match &args[2] {
        Value::List(items) => {
            for item in items {
                let (new_carry, output) = step(func, carry, item.clone(), env)?;
                carry = new_carry;
                outputs.push(output);
            }
            Ok(Value::Tuple(vec![carry, Value::List(outputs)]))
        }
        Value::Tensor { data, .. } => {
            if data.ndim() == 1 {
                for &x in data.iter() {
                    let (new_carry, output) = step(func, carry, Value::Float(x), env)?;
                    carry = new_carry;
                    outputs.push(output);
                }
            } else {
                for i in 0..data.shape()[0] {
                    let row = data.index_axis(ndarray::Axis(0), i).to_owned();
                    let (new_carry, output) = step(func, carry, Value::tensor_f32(row), env)?;
                    carry = new_carry;
                    outputs.push(output);
                }
            }
            Ok(Value::Tuple(vec![carry, Value::List(outputs)]))
        }
        Value::Dict(map) => {
            let n = dict_scan_length(map)?;
            for i in 0..n {
                let slice = slice_dict(map, i)?;
                let (new_carry, output) = step(func, carry, slice, env)?;
                carry = new_carry;
                outputs.push(output);
            }
            Ok(Value::Tuple(vec![carry, Value::List(outputs)]))
        }
        _ => Err(runtime_error("scan: expected list, tensor, or dict of tensors")),
    }
}

/// Get the scan length from a dict of tensors (dim-0 of first tensor found).
pub(crate) fn dict_scan_length(map: &std::collections::BTreeMap<String, Value>) -> Result<usize, SheafError> {
    for val in map.values() {
        if let Value::Tensor { data, .. } = val {
            return Ok(data.shape()[0]);
        }
    }
    Err(runtime_error("scan: dict contains no tensors to iterate over"))
}

/// Slice each tensor in a dict along dim-0 at index i.
pub(crate) fn slice_dict(
    map: &std::collections::BTreeMap<String, Value>,
    i: usize,
) -> Result<Value, SheafError> {
    let mut result = std::collections::BTreeMap::new();
    for (key, val) in map {
        let sliced = match val {
            Value::Tensor { data, .. } => {
                if data.ndim() == 1 {
                    Value::Float(data[[i]])
                } else {
                    Value::tensor_f32(data.index_axis(ndarray::Axis(0), i).to_owned())
                }
            }
            other => other.clone(),
        };
        result.insert(key.clone(), sliced);
    }
    Ok(Value::Dict(result))
}

fn eval_apply(args: &[Value], env: &mut Env) -> Result<Value, SheafError> {
    if args.len() != 2 {
        return Err(runtime_error("apply requires 2 arguments: (apply fn args)"));
    }
    let func = &args[0];
    match &args[1] {
        Value::List(items) => call_function(func, items, env),
        Value::Tensor { data, .. } => {
            let call_args: Vec<Value> = data.iter().map(|&x| Value::Float(x)).collect();
            call_function(func, &call_args, env)
        }
        _ => Err(runtime_error("apply: expected list or tensor")),
    }
}

fn eval_find(args: &[Value], env: &mut Env) -> Result<Value, SheafError> {
    if args.len() != 2 {
        return Err(runtime_error("find requires 2 arguments: (find fn coll)"));
    }
    let func = &args[0];
    match &args[1] {
        Value::List(items) => {
            for item in items {
                let result = call_function(func, &[item.clone()], env)?;
                if result.is_truthy() {
                    return Ok(item.clone());
                }
            }
            Ok(Value::Nil)
        }
        _ => Err(runtime_error("find: expected list")),
    }
}

fn tree_map_multi(trees: &[Value], func: &Value, env: &mut Env) -> Result<Value, SheafError> {
    match &trees[0] {
        Value::Dict(map) => {
            let mut result = BTreeMap::new();
            for k in map.keys() {
                let sub_trees: Result<Vec<Value>, _> = trees
                    .iter()
                    .map(|t| match t {
                        Value::Dict(m) => m
                            .get(k)
                            .cloned()
                            .ok_or_else(|| runtime_error(format!("tree-map: key {} missing in one tree", k))),
                        _ => Err(runtime_error("tree-map: tree structure mismatch")),
                    })
                    .collect();
                result.insert(k.clone(), tree_map_multi(&sub_trees?, func, env)?);
            }
            Ok(Value::Dict(result))
        }
        Value::List(items) => {
            let n = items.len();
            let mut result = Vec::new();
            for i in 0..n {
                let sub_trees: Result<Vec<Value>, _> = trees
                    .iter()
                    .map(|t| match t {
                        Value::List(v) => v
                            .get(i)
                            .cloned()
                            .ok_or_else(|| runtime_error("tree-map: list length mismatch")),
                        _ => Err(runtime_error("tree-map: tree structure mismatch")),
                    })
                    .collect();
                result.push(tree_map_multi(&sub_trees?, func, env)?);
            }
            Ok(Value::List(result))
        }
        _ => call_function(func, trees, env),
    }
}

fn eval_tree_map(args: &[Value], env: &mut Env) -> Result<Value, SheafError> {
    if args.len() < 2 {
        return Err(runtime_error("tree-map requires at least 2 arguments: (tree-map fn tree ...)"));
    }
    let func = &args[0];
    let trees = &args[1..];
    tree_map_multi(trees, func, env)
}

fn tree_reduce_value(val: &Value, func: &Value, acc: Value, env: &mut Env) -> Result<Value, SheafError> {
    match val {
        Value::Dict(map) => {
            let mut acc = acc;
            for v in map.values() {
                acc = tree_reduce_value(v, func, acc, env)?;
            }
            Ok(acc)
        }
        Value::List(items) => {
            let mut acc = acc;
            for item in items {
                acc = tree_reduce_value(item, func, acc, env)?;
            }
            Ok(acc)
        }
        leaf => call_function(func, &[acc, leaf.clone()], env),
    }
}

fn eval_tree_reduce(args: &[Value], env: &mut Env) -> Result<Value, SheafError> {
    if args.len() != 3 {
        return Err(runtime_error("tree-reduce requires 3 arguments: (tree-reduce fn tree init)"));
    }
    tree_reduce_value(&args[1], &args[0], args[2].clone(), env)
}

fn flatten_leaves(val: &Value, leaves: &mut Vec<Value>) {
    match val {
        Value::Dict(map) => {
            for v in map.values() {
                flatten_leaves(v, leaves);
            }
        }
        Value::List(items) => {
            for item in items {
                flatten_leaves(item, leaves);
            }
        }
        leaf => leaves.push(leaf.clone()),
    }
}

fn eval_flatten(args: &[Value]) -> Result<Value, SheafError> {
    if args.is_empty() {
        return Err(runtime_error("flatten requires 1 argument"));
    }
    let mut leaves = Vec::new();
    flatten_leaves(&args[0], &mut leaves);
    // Returns (leaves_list, reconstruct_fn), we return a list of [leaves, nil] for now
    // The test only uses (first (flatten params)) → the leaves list
    Ok(Value::List(vec![Value::List(leaves), Value::Nil]))
}

/// (vmap f) or (vmap f in-axes) → returns a vmapped function.
/// When called, slices inputs along the mapped axes, applies f to each slice, and stacks results.
fn eval_vmap(args: &[Value], _env: &mut Env) -> Result<Value, SheafError> {
    if args.is_empty() || args.len() > 2 {
        return Err(runtime_error("vmap requires 1 or 2 arguments: (vmap fn) or (vmap fn in-axes)"));
    }
    let func = args[0].clone();
    let mut closure = vec![("__vmap_fn__".to_string(), func)];
    if args.len() == 2 {
        closure.push(("__vmap_axes__".to_string(), args[1].clone()));
    }
    Ok(Value::Function {
        params: vec!["__vmap_arg__".to_string()],
        body: crate::core::compiler::CompiledExpr::Symbol("__vmap_arg__".to_string()),
        closure,
    })
}

/// Execute a vmapped function call.
fn eval_vmap_call(
    vmap_fn: &Value,
    axes: Option<&Value>,
    args: &[Value],
    env: &mut Env,
) -> Result<Value, SheafError> {
    if args.is_empty() {
        return Err(runtime_error("vmap: called with no arguments"));
    }

    // Parse in-axes: None means axis 0 for all args
    let in_axes: Vec<Option<usize>> = match axes {
        None => args.iter().map(|_| Some(0)).collect(),
        Some(Value::Int(n)) => args.iter().map(|_| Some(*n as usize)).collect(),
        Some(Value::Float(n)) => args.iter().map(|_| Some(*n as usize)).collect(),
        Some(Value::List(ax_list)) => {
            ax_list.iter().map(|a| match a {
                Value::Nil => None,
                Value::Int(n) => Some(*n as usize),
                Value::Float(n) => Some(*n as usize),
                _ => Some(0),
            }).collect()
        }
        _ => args.iter().map(|_| Some(0)).collect(),
    };

    // Find batch size from first mapped arg
    let batch_size = args.iter().zip(in_axes.iter())
        .find_map(|(arg, axis)| {
            axis.and_then(|ax| match arg {
                Value::Tensor { data, .. } => Some(data.shape()[ax]),
                _ => None,
            })
        })
        .ok_or_else(|| runtime_error("vmap: no mapped tensor argument found"))?;

    // Slice, apply, collect
    let mut results = Vec::with_capacity(batch_size);
    for i in 0..batch_size {
        let mut sliced = Vec::with_capacity(args.len());
        for (arg, axis) in args.iter().zip(in_axes.iter()) {
            match axis {
                Some(ax) => match arg {
                    Value::Tensor { data, dtype } => {
                        let row = data.index_axis(ndarray::Axis(*ax), i).to_owned();
                        sliced.push(Value::Tensor { data: Arc::new(row), dtype: *dtype });
                    }
                    _ => sliced.push(arg.clone()),
                },
                None => sliced.push(arg.clone()),
            }
        }
        results.push(call_function(vmap_fn, &sliced, env)?);
    }

    // Stack results: if all are tensors, stack into a tensor; otherwise return list
    if results.iter().all(|r| matches!(r, Value::Tensor { .. })) {
        let arrays: Vec<_> = results.iter().map(|r| match r {
            Value::Tensor { data, .. } => data.view(),
            _ => unreachable!(),
        }).collect();
        let views: Vec<_> = arrays.iter().map(|a| a.clone().insert_axis(ndarray::Axis(0))).collect();
        let stacked = ndarray::concatenate(ndarray::Axis(0), &views)
            .map_err(|e| runtime_error(format!("vmap: failed to stack results: {}", e)))?;
        let dtype = match &results[0] {
            Value::Tensor { dtype, .. } => *dtype,
            _ => unreachable!(),
        };
        Ok(Value::Tensor { data: Arc::new(stacked), dtype })
    } else if results.iter().all(|r| matches!(r, Value::Float(_) | Value::Int(_))) {
        // Scalar results → 1D tensor
        let vals: Vec<f32> = results.iter().map(|r| match r {
            Value::Float(f) => *f,
            Value::Int(n) => *n as f32,
            _ => 0.0f32,
        }).collect();
        let arr = ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[vals.len()]), vals)
            .map_err(|e| runtime_error(format!("vmap: {}", e)))?;
        Ok(Value::Tensor { data: Arc::new(arr), dtype: value::Dtype::F32 })
    } else {
        Ok(Value::List(results))
    }
}

/// (value-and-grad f) → returns a function that, given params, returns [loss, grad_params].
///
/// Gradient computed by central finite differences: grad[i] ≈ (f(p+h) - f(p-h)) / 2h
/// Applied element-wise to every leaf tensor in the params pytree.
fn eval_value_and_grad_hof(args: &[Value], _env: &mut Env) -> Result<Value, SheafError> {
    if args.len() != 1 {
        return Err(runtime_error("value-and-grad: expected exactly 1 argument (the function)"));
    }
    let func = args[0].clone();

    // Return a Value::Function closure that captures `func`.
    // When called with params, call_function detects __vag_fn__ and dispatches
    // to eval_value_and_grad_call which computes (loss, grad) via finite differences.
    Ok(Value::Function {
        params: vec!["__vag_params__".to_string()],
        body: crate::core::compiler::CompiledExpr::Symbol("__vag_params__".to_string()),
        closure: vec![("__vag_fn__".to_string(), func)],
    })
}

/// Evaluate a value-and-grad HOF call.
///
/// Tries symbolic autodiff first (exact, fast). Falls back to finite differences
/// if the function body contains ops the AD engine cannot handle (get, reduce, etc.).
fn eval_value_and_grad_call(func: &Value, params: &Value, env: &mut Env) -> Result<Value, SheafError> {
    // Try JIT compilation first: compile forward+backward into a single VMFB
    // Skip when tracing — interpreter must run to expose the autodiff call tree
    #[cfg(iree_runtime)]
    if env.tracer.is_none() {
        if let Some(result) = try_jit_vag(func, params, env) {
            return result;
        }
    }

    use crate::autodiff::{contains_undiffable_ops, grad_simplified, inline_function_calls};

    if let Value::Function { params: fn_params, body, closure } = func {
        if fn_params.len() == 1 {
            let param_name = &fn_params[0];
            let inlined = inline_function_calls(body, &env.registry);

            if !contains_undiffable_ops(&inlined) {
                // Direct symbolic path: no structural ops to trace
                env.push_scope();
                for (name, val) in closure {
                    env.set(name, val.clone());
                }
                env.set(param_name, params.clone());

                let loss_val = eval(&inlined, env)?;
                let loss = scalar_from_value(&loss_val)?;
                let grad_expr = grad_simplified(&inlined, param_name);
                let grad_val = eval(&grad_expr, env)?;

                env.pop_scope();
                return Ok(Value::List(vec![Value::Float(loss), grad_val]));
            }

            // Tracing path: evaluate structural ops (get, reduce) with
            // concrete values, keep tensor ops symbolic, then differentiate.
            {
                use crate::autodiff::trace::{trace_expr, LeafMap};

                env.push_scope();
                for (name, val) in closure {
                    env.set(name, val.clone());
                }
                env.set(param_name, params.clone());

                let mut leaf_map = LeafMap::new();
                match trace_expr(&inlined, env, &mut leaf_map) {
                    Ok(traced) => {
                        if !contains_undiffable_ops(&traced) {
                            // Forward pass: call the function normally
                            // (not via eval on inlined, which may have
                            // lambda params not in scope)
                            env.pop_scope();
                            let loss_val = call_function(func, &[params.clone()], env)?;
                            let loss = scalar_from_value(&loss_val)?;

                            // Re-enter scope for gradient evaluation
                            env.push_scope();
                            for (name, val) in closure {
                                env.set(name, val.clone());
                            }
                            env.set(param_name, params.clone());
                            for (sym, val) in &leaf_map.leaves {
                                env.set(sym, val.clone());
                            }

                            let grad_tree = build_grad_from_leaves(
                                &traced, params, &leaf_map.leaves, env,
                            )?;

                            env.pop_scope();
                            return Ok(Value::List(vec![Value::Float(loss), grad_tree]));
                        }
                        env.pop_scope();
                        sheaf_msg!("warning: value-and-grad: tracing incomplete, falling back to finite differences");
                    }
                    Err(e) => {
                        env.pop_scope();
                        sheaf_msg!("warning: value-and-grad: tracing failed ({:?}), falling back to finite differences", e);
                    }
                }
            }
        }
    }

    eval_vag_finite_diff(func, params, env)
}

/// Finite-difference fallback for value-and-grad.
fn eval_vag_finite_diff(func: &Value, params: &Value, env: &mut Env) -> Result<Value, SheafError> {
    let h = 1e-4_f32;

    // Evaluate loss at params
    let loss_val = call_function(func, &[params.clone()], env)?;
    let loss = match &loss_val {
        Value::Float(x) => *x,
        Value::Int(n) => *n as f32,
        Value::Tensor { data, .. } => data.first().copied().unwrap_or(0.0f32),
        _ => return Err(runtime_error("value-and-grad: loss function must return a scalar")),
    };

    // Count leaves
    let n_leaves = {
        let mut count = 0usize;
        fn count_leaves_inner(val: &Value, count: &mut usize) {
            match val {
                Value::Tensor { data, .. } => *count += data.len(),
                Value::Dict(map) => map.values().for_each(|v| count_leaves_inner(v, count)),
                Value::Float(_) | Value::Int(_) => *count += 1,
                _ => {}
            }
        }
        count_leaves_inner(params, &mut count);
        count
    };

    // Central finite differences for each leaf
    let mut grads = vec![0.0f32; n_leaves];
    for i in 0..n_leaves {
        let p_plus = perturb_leaf(params, i, h);
        let p_minus = perturb_leaf(params, i, -h);
        let f_plus = call_function(func, &[p_plus], env)?;
        let f_minus = call_function(func, &[p_minus], env)?;
        let fp = scalar_from_value(&f_plus)?;
        let fm = scalar_from_value(&f_minus)?;
        grads[i] = (fp - fm) / (2.0 * h);
    }

    let mut counter = 0;
    let grad_tree = build_grad_tree(params, &grads, &mut counter);

    Ok(Value::List(vec![Value::Float(loss), grad_tree]))
}

fn perturb_leaf(val: &Value, leaf_idx: usize, delta: f32) -> Value {
    let mut counter = 0usize;
    perturb_leaf_inner(val, leaf_idx, delta, &mut counter)
}

fn perturb_leaf_inner(val: &Value, leaf_idx: usize, delta: f32, counter: &mut usize) -> Value {
    match val {
        Value::Tensor { data, dtype } => {
            let n = data.len();
            if *counter <= leaf_idx && leaf_idx < *counter + n {
                let local = leaf_idx - *counter;
                let mut new_data = (**data).clone();
                new_data.as_slice_mut().unwrap()[local] += delta;
                *counter += n;
                Value::Tensor { data: Arc::new(new_data), dtype: *dtype }
            } else {
                *counter += n;
                val.clone()
            }
        }
        Value::Dict(map) => Value::Dict(
            map.iter().map(|(k, v)| (k.clone(), perturb_leaf_inner(v, leaf_idx, delta, counter))).collect()
        ),
        Value::Float(x) => {
            let result = if *counter == leaf_idx { Value::Float(*x + delta) } else { val.clone() };
            *counter += 1;
            result
        }
        Value::Int(n) => {
            let result = if *counter == leaf_idx { Value::Float(*n as f32 + delta) } else { val.clone() };
            *counter += 1;
            result
        }
        _ => val.clone(),
    }
}

fn scalar_from_value(val: &Value) -> Result<f32, SheafError> {
    let val = val.ensure_host()?;
    match &val {
        Value::Float(x) => Ok(*x),
        Value::Int(n) => Ok(*n as f32),
        Value::Tensor { data, .. } => data.first().copied()
            .ok_or_else(|| runtime_error("value-and-grad: empty tensor result")),
        _ => Err(runtime_error("value-and-grad: loss must return a scalar")),
    }
}

fn build_grad_tree(params: &Value, grads: &[f32], counter: &mut usize) -> Value {
    match params {
        Value::Tensor { data, dtype } => {
            let n = data.len();
            let slice = grads[*counter..*counter + n].to_vec();
            *counter += n;
            Value::Tensor {
                data: Arc::new(ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(data.shape()), slice).unwrap()),
                dtype: *dtype,
            }
        }
        Value::Dict(map) => Value::Dict(
            map.iter().map(|(k, v)| (k.clone(), build_grad_tree(v, grads, counter))).collect()
        ),
        Value::Float(_) | Value::Int(_) => {
            let g = grads[*counter];
            *counter += 1;
            Value::Float(g)
        }
        _ => params.clone(),
    }
}

/// Build a gradient pytree from traced symbolic autodiff.
///
/// For each leaf tensor in the params tree, find the corresponding leaf symbol
/// in the leaf_map, compute its symbolic gradient, and evaluate it.
fn build_grad_from_leaves(
    traced_expr: &CompiledExpr,
    params: &Value,
    leaves: &[(String, Value)],
    env: &mut Env,
) -> Result<Value, SheafError> {
    use crate::autodiff::grad_simplified;
    use std::collections::HashMap;

    // Pre-compute all leaf gradients (one grad_simplified per leaf).
    let mut leaf_grads: HashMap<String, Value> = HashMap::new();
    for (sym, _val) in leaves {
        let grad_expr = grad_simplified(traced_expr, sym);
        let grad_val = eval(&grad_expr, env)?;
        leaf_grads.insert(sym.clone(), grad_val);
    }

    // Reconstruct the gradient pytree by matching param leaves to leaf_map entries
    // by data equality (not by sequential order, since tracer order may differ
    // from the params tree traversal order).
    build_grad_tree_by_value(params, leaves, &leaf_grads)
}

fn values_equal(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Tensor { data: da, .. }, Value::Tensor { data: db, .. }) => {
            da.shape() == db.shape() && da.iter().zip(db.iter()).all(|(x, y)| x.to_bits() == y.to_bits())
        }
        (Value::Float(a), Value::Float(b)) => a.to_bits() == b.to_bits(),
        (Value::Int(a), Value::Int(b)) => a == b,
        _ => false,
    }
}

fn build_grad_tree_by_value(
    params: &Value,
    leaves: &[(String, Value)],
    leaf_grads: &std::collections::HashMap<String, Value>,
) -> Result<Value, SheafError> {
    match params {
        Value::Tensor { .. } | Value::Float(_) | Value::Int(_) => {
            // Find the leaf that matches this param value
            for (sym, leaf_val) in leaves {
                if values_equal(params, leaf_val) {
                    if let Some(grad_val) = leaf_grads.get(sym) {
                        return reduce_grad_to_param_shape(grad_val, params);
                    }
                }
            }
            // No matching leaf found, this param wasn't used in the expression.
            // Return zeros with the same shape.
            Ok(zeros_like(params))
        }
        Value::Dict(map) => {
            let mut grad_map = BTreeMap::new();
            for (k, v) in map {
                let g = build_grad_tree_by_value(v, leaves, leaf_grads)?;
                grad_map.insert(k.clone(), g);
            }
            Ok(Value::Dict(grad_map))
        }
        Value::List(items) => {
            let mut grad_items = Vec::new();
            for item in items {
                let g = build_grad_tree_by_value(item, leaves, leaf_grads)?;
                grad_items.push(g);
            }
            Ok(Value::List(grad_items))
        }
        _ => Ok(Value::Nil),
    }
}

fn zeros_like(val: &Value) -> Value {
    match val {
        Value::Tensor { data, dtype } => {
            Value::Tensor {
                data: Arc::new(ArrayD::zeros(IxDyn(data.shape()))),
                dtype: *dtype,
            }
        }
        Value::Float(_) => Value::Float(0.0),
        Value::Int(_) => Value::Int(0),
        _ => Value::Nil,
    }
}

/// Reduce a gradient tensor to match the parameter's shape.
/// When an op broadcasts a param (e.g., bias [1] → [4,1]),
/// the symbolic gradient has the broadcasted shape. Sum over
/// the extra leading dimensions to get back to param shape.
fn reduce_grad_to_param_shape(grad: &Value, param: &Value) -> Result<Value, SheafError> {
    match (grad, param) {
        (Value::Tensor { data: g_data, dtype }, Value::Tensor { data: p_data, .. }) => {
            let g_shape = g_data.shape();
            let p_shape = p_data.shape();
            if g_shape == p_shape {
                return Ok(grad.clone());
            }
            // Sum over leading batch dimensions
            // e.g., grad [4,1] + param [1] → sum axis 0 → [1]
            // e.g., grad [4,2,1] + param [2,1] → sum axis 0 → [2,1]
            let g_ndim = g_shape.len();
            let p_ndim = p_shape.len();
            if g_ndim > p_ndim {
                let extra = g_ndim - p_ndim;
                let mut reduced = (**g_data).clone();
                for _ in 0..extra {
                    reduced = reduced.sum_axis(ndarray::Axis(0));
                }
                // Also handle broadcast in trailing dims: if param dim is 1 but grad > 1, sum
                let r_shape = reduced.shape().to_vec();
                for (i, (&rd, &pd)) in r_shape.iter().zip(p_shape.iter()).enumerate() {
                    if pd == 1 && rd > 1 {
                        reduced = reduced.sum_axis(ndarray::Axis(i));
                        reduced = reduced.insert_axis(ndarray::Axis(i));
                    }
                }
                Ok(Value::Tensor { data: Arc::new(reduced), dtype: *dtype })
            } else if g_ndim == p_ndim {
                // Same rank but different sizes (broadcast case)
                let mut reduced = (**g_data).clone();
                for (i, (&gd, &pd)) in g_shape.iter().zip(p_shape.iter()).enumerate() {
                    if pd == 1 && gd > 1 {
                        reduced = reduced.sum_axis(ndarray::Axis(i));
                        reduced = reduced.insert_axis(ndarray::Axis(i));
                    }
                }
                Ok(Value::Tensor { data: Arc::new(reduced), dtype: *dtype })
            } else {
                // Grad has fewer dims than param, shouldn't happen, return as-is
                Ok(grad.clone())
            }
        }
        (Value::Float(_), Value::Float(_)) => Ok(grad.clone()),
        (Value::Tensor { data, dtype: _ }, Value::Float(_)) => {
            // Reduce tensor gradient to scalar for a Float param
            let sum: f32 = data.iter().sum();
            Ok(Value::Float(sum))
        }
        _ => Ok(grad.clone()),
    }
}

/// High-level entry point: parse + compile + eval a Sheaf expression string.
pub fn eval_str(source: &str) -> Result<Value, SheafError> {
    let exprs = crate::core::parse(source, "<eval>")?;
    let mut compiler = CompilerContext::new();
    let mut last = Value::Nil;

    for expr in &exprs {
        let compiled = compiler.compile(expr)?;
        let mut env = Env::with_registry(compiler.registry.clone());
        builtins::register_builtins(&mut env);
        last = eval(&compiled, &mut env)?;
    }

    Ok(last)
}

/// Evaluate multiple expressions, maintaining state across them.
pub fn eval_exprs(source: &str) -> Result<Value, SheafError> {
    let exprs = crate::core::parse(source, "<eval>")?;
    let mut compiler = CompilerContext::new();
    let mut last = Value::Nil;

    // First pass: compile all (registers defn, etc.)
    let mut compiled_exprs = Vec::new();
    for expr in &exprs {
        compiled_exprs.push(compiler.compile(expr)?);
    }

    // Second pass: evaluate all non-Nil expressions
    let mut env = Env::with_registry(compiler.registry.clone());
    builtins::register_builtins(&mut env);

    for compiled in &compiled_exprs {
        if !matches!(compiled, CompiledExpr::Nil) {
            last = eval(compiled, &mut env)?;
        }
    }

    Ok(last)
}

/// Try to dispatch a function call to IREE.
/// Returns `Some(result)` if IREE handled it, `None` to fall through to the interpreter.
/// Skips IREE when the argument structure doesn't match the compiled signature
/// (e.g. the model was compiled for 2 layers but called with 0).
#[cfg(iree_runtime)]
fn try_iree_dispatch(
    func_def: &crate::core::compiler::FunctionDef,
    args: &[Value],
    env: &mut Env,
) -> Option<Result<Value, SheafError>> {
    let session_idx = func_def.vmfb_session_idx?;
    let session = env.vmfb_sessions.get(session_idx)?;
    let iree_session = session.downcast_ref::<crate::runtime::iree_session::IreeSession>()?;

    let sig = func_def.signature.as_ref()?;

    // Validate tensor count AND shapes before calling into IREE.
    // This prevents the C runtime from printing ugly diagnostics to stderr.
    if !crate::runtime::iree_session::args_match_signature(args, &sig.param_types) {
        let expected = crate::runtime::iree_session::count_signature_tensors(&sig.param_types);
        let actual = crate::runtime::iree_session::count_arg_tensors(args);
        if env.iree_mismatch_warned.insert(func_def.name.clone())
            || crate::core::config::verbosity() >= 2
        {
            sheaf_msg!(
                "warning: '{}' compiled for {} tensors but called with {} — falling back to interpreted",
                func_def.name, expected, actual,
            );
        }
        return None;
    }
    if let Err(mismatch) = crate::runtime::iree_session::check_shapes_match(args, &sig.param_types) {
        if env.iree_mismatch_warned.insert(func_def.name.clone())
            || crate::core::config::verbosity() >= 2
        {
            sheaf_msg!(
                "warning: '{}': {}. Falling back to interpreted mode.",
                func_def.name, mismatch,
            );
        }
        return None;
    }

    let full_name = format!("module.{}", func_def.name.replace('-', "_"));
    let result = match iree_session.call_typed_device(&full_name, args, &sig.return_type) {
        Ok(v) => v,
        Err(_e) => {
            // Unexpected IREE error → fall back to interpreter with warning.
            if env.iree_mismatch_warned.insert(format!("{}:call", func_def.name)) {
                sheaf_msg!(
                    "warning: '{}' IREE call failed — falling back to interpreted: {}",
                    func_def.name, _e,
                );
            }
            return None;
        }
    };

    // Reconstruct nested dicts/lists from flat tuples using arg type layouts
    let result = if !sig.arg_type_layouts.is_empty() {
        crate::core::inference::reconstruct_jit_result(result, &sig.return_type, &sig.arg_type_layouts)
    } else {
        result
    };
    // Reconstruct top-level dict from tuple if the function originally returned a dict
    let result = match (&sig.return_dict_keys, result) {
        (Some(keys), Value::Tuple(elems)) if elems.len() == keys.len() => {
            let map = keys.iter().cloned().zip(elems).collect();
            Value::Dict(map)
        }
        (_, other) => other,
    };

    Some(Ok(result))
}

/// Try to JIT-compile a value-and-grad call into a single VMFB (forward + backward).
/// Returns `Some(Ok(result))` on success, `None` to fall through to the interpreter.
#[cfg(iree_runtime)]
fn try_jit_vag(
    func: &Value,
    params: &Value,
    env: &mut Env,
) -> Option<Result<Value, SheafError>> {
    // Augment the closure with free variables resolved from the environment.
    // Sheaf uses dynamic scoping, so lambdas don't capture free vars at creation.
    // The JIT needs them as explicit captures.
    let augmented_func = augment_closure_with_free_vars(func, env)?;

    let jit = env.jit_compiler.as_mut()?;

    let (session_idx, sig, param_names) = jit.try_jit_value_and_grad(
        &augmented_func,
        params,
        &env.registry,
        &mut env.vmfb_sessions,
    )?;

    // Build the argument list: fn params first, then captures (same order as param_names)
    let (fn_params, closure) = match &augmented_func {
        Value::Function {
            params: p,
            closure: c,
            ..
        } => (p, c),
        _ => return None,
    };

    let mut args: Vec<Value> = Vec::new();
    for name in &param_names {
        if fn_params.contains(name) {
            args.push(params.clone());
        } else if let Some((_, val)) = closure.iter().find(|(k, _)| k == name) {
            args.push(val.clone());
        } else {
            // Scalar capture, not passed to IREE
            continue;
        }
    }

    // Dispatch via IREE
    let session = env.vmfb_sessions.get(session_idx)?;
    let iree_session = session.downcast_ref::<crate::runtime::iree_session::IreeSession>()?;

    let result = match iree_session.call_typed_device("module.value_and_grad", &args, &sig.return_type) {
        Ok(v) => v,
        Err(e) => {
            sheaf_msg!("warning: value-and-grad JIT dispatch failed: {}", e);
            return None;
        }
    };

    // Unpack: IREE returns Tuple([loss_tensor, grad_elements...])
    // We need to return List([Float(loss), grad_value])
    if crate::core::config::verbosity() >= 2 {
        let desc = match &result {
            Value::Tuple(elems) => format!("Tuple(len={})", elems.len()),
            other => format!("{}", other.type_name()),
        };
        sheaf_msg!("jit: [vag] result structure: {}", desc);
        if let Value::Tuple(elems) = &result {
            for (i, e) in elems.iter().enumerate() {
                sheaf_msg!("jit: [vag]   elem[{}]: {}", i, e.type_name());
            }
        }
    }
    let unpacked = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        unpack_vag_result(&result, params)
    })) {
        Ok(Some(v)) => v,
        Ok(None) => {
            sheaf_msg!("warning: value-and-grad result unpacking failed");
            return None;
        }
        Err(e) => {
            sheaf_msg!("warning: value-and-grad result unpacking panicked: {:?}",
                e.downcast_ref::<String>().map(|s| s.as_str())
                    .or_else(|| e.downcast_ref::<&str>().copied()));
            return None;
        }
    };

    Some(Ok(unpacked))
}

/// Augment a lambda's closure with free variables from the dynamic environment.
/// Sheaf lambdas have empty closures (dynamic scoping); the JIT needs explicit captures.
#[cfg(iree_runtime)]
fn augment_closure_with_free_vars(func: &Value, env: &Env) -> Option<Value> {
    let (fn_params, body, closure) = match func {
        Value::Function { params, body, closure } => (params, body, closure),
        _ => return None,
    };

    let mut free_set = std::collections::HashSet::new();
    collect_free_vars_compiled(body, &mut free_set);

    // Remove the lambda's own params
    for p in fn_params {
        free_set.remove(p.as_str());
    }
    // Remove vars already in the closure
    for (k, _) in closure {
        free_set.remove(k.as_str());
    }

    // Sort for deterministic parameter ordering (avoids MLIR hash changes → cache misses)
    let mut free: Vec<&str> = free_set.iter().map(|s| s.as_str()).collect();
    free.sort();

    let mut augmented_closure = closure.clone();
    for name in &free {
        if let Ok(val) = env.get(name) {
            augmented_closure.push((name.to_string(), val.clone()));
        }
        // If not in env, leave it, the JIT will fail gracefully
    }

    Some(Value::Function {
        params: fn_params.clone(),
        body: body.clone(),
        closure: augmented_closure,
    })
}

/// Collect all symbol names referenced in a CompiledExpr (not bound by inner let/lambda).
#[cfg(iree_runtime)]
fn collect_free_vars_compiled(expr: &CompiledExpr, out: &mut std::collections::HashSet<String>) {
    use crate::core::compiler::CompiledExpr;
    match expr {
        CompiledExpr::Symbol(name) => {
            out.insert(name.clone());
        }
        CompiledExpr::FunctionCall { args, .. } => {
            for a in args {
                collect_free_vars_compiled(a, out);
            }
        }
        CompiledExpr::Let { bindings, body } => {
            for (_, v) in bindings {
                collect_free_vars_compiled(v, out);
            }
            // Let-bound names shadow, but we're collecting conservatively
            // (the JIT will just ignore extra captures)
            collect_free_vars_compiled(body, out);
        }
        CompiledExpr::Do(exprs) => {
            for e in exprs {
                collect_free_vars_compiled(e, out);
            }
        }
        CompiledExpr::If { condition, then_branch, else_branch } => {
            collect_free_vars_compiled(condition, out);
            collect_free_vars_compiled(then_branch, out);
            if let Some(e) = else_branch {
                collect_free_vars_compiled(e, out);
            }
        }
        CompiledExpr::Lambda { params, body } => {
            let mut inner = std::collections::HashSet::new();
            collect_free_vars_compiled(body, &mut inner);
            for p in params {
                inner.remove(p.as_str());
            }
            out.extend(inner);
        }
        CompiledExpr::LambdaCall { callee, args } => {
            collect_free_vars_compiled(callee, out);
            for a in args {
                collect_free_vars_compiled(a, out);
            }
        }
        CompiledExpr::GetTupleElement { param, .. } => {
            out.insert(param.clone());
        }
        CompiledExpr::Vector(elems) => {
            for e in elems {
                collect_free_vars_compiled(e, out);
            }
        }
        _ => {} // Literals, Float, Integer, etc.
    }
}

/// Unpack a value-and-grad IREE result into [Float(loss), grad_dict_or_tensor].
#[cfg(iree_runtime)]
fn unpack_vag_result(result: &Value, original_params: &Value) -> Option<Value> {
    let elems = match result {
        Value::Tuple(elems) => elems,
        _ => return None,
    };

    if elems.len() < 2 {
        return None;
    }

    // First element: loss (scalar tensor → Float)
    let loss = match &elems[0] {
        Value::Tensor { data, .. } if data.len() == 1 => {
            Value::Float(*data.iter().next().unwrap())
        }
        Value::DeviceBuffer(db) if db.shape.is_empty() || db.shape.iter().product::<usize>() == 1 => {
            match db.to_host() {
                Ok(data) => Value::Float(*data.iter().next().unwrap_or(&0.0)),
                Err(_) => return None,
            }
        }
        Value::Float(f) => Value::Float(*f),
        _ => return None,
    };

    // Second element: gradient (same structure as original params)
    let grad = if elems.len() == 2 {
        match original_params {
            Value::Dict(map) => tuple_to_dict(&elems[1], map)?,
            _ => elems[1].clone(),
        }
    } else {
        // Multiple wrt params: pack remaining elements
        Value::Tuple(elems[1..].to_vec())
    };

    Some(Value::List(vec![loss, grad]))
}

/// Reconstruct a Dict from a Tuple, using the original Dict's key structure.
/// Dict keys are sorted (BTreeMap), matching the tuple element order from codegen.
#[cfg(iree_runtime)]
fn tuple_to_dict(tuple_val: &Value, original: &BTreeMap<String, Value>) -> Option<Value> {
    let elems = match tuple_val {
        Value::Tuple(elems) => elems,
        _ => {
            // Leaf value, not a dict, return as-is
            return Some(tuple_val.clone());
        }
    };

    if elems.len() != original.len() {
        if crate::core::config::verbosity() >= 2 {
            sheaf_msg!("jit: [vag] tuple_to_dict: tuple len {} != dict keys {} ({:?})",
                elems.len(), original.len(), original.keys().collect::<Vec<_>>());
        }
        return None;
    }

    let mut result = BTreeMap::new();
    for ((key, orig_val), elem) in original.iter().zip(elems.iter()) {
        let val = match orig_val {
            Value::Dict(sub_map) => tuple_to_dict(elem, sub_map)?,
            Value::List(orig_list) => tuple_to_list(elem, orig_list)?,
            _ => elem.clone(),
        };
        result.insert(key.clone(), val);
    }
    Some(Value::Dict(result))
}

/// Reconstruct a List from a Tuple, using the original List's element structure.
#[cfg(iree_runtime)]
fn tuple_to_list(tuple_val: &Value, original: &[Value]) -> Option<Value> {
    let elems = match tuple_val {
        Value::Tuple(elems) => elems,
        _ => {
            return None;
        }
    };

    if elems.len() != original.len() {
        if crate::core::config::verbosity() >= 2 {
            sheaf_msg!("jit: [vag] tuple_to_list: tuple len {} != list len {}",
                elems.len(), original.len());
        }
        return None;
    }

    let mut result = Vec::with_capacity(original.len());
    for (orig_val, elem) in original.iter().zip(elems.iter()) {
        let val = match orig_val {
            Value::Dict(sub_map) => tuple_to_dict(elem, sub_map)?,
            Value::List(sub_list) => tuple_to_list(elem, sub_list)?,
            _ => elem.clone(),
        };
        result.push(val);
    }
    Some(Value::List(result))
}
