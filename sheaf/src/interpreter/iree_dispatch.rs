// Copyright (c) 2025 Damien Boureille
// Licensed under the MIT License.

//! IREE runtime dispatch for JIT-compiled functions and value-and-grad.

#![cfg(iree_runtime)]

use crate::sheaf_msg;
use crate::core::error::SheafError;
use crate::interpreter::env::{runtime_error, Env};
use crate::interpreter::value::Value;
use std::collections::BTreeMap;

/// Try to dispatch a function call to IREE.
/// Returns `Some(result)` if IREE handled it, `None` to fall through to the interpreter.
/// Skips IREE when the argument structure doesn't match the compiled signature
/// (e.g. the model was compiled for 2 layers but called with 0).
pub(super) fn try_iree_dispatch(
    func_def: &crate::core::expr::FunctionDef,
    args: &[Value],
    env: &mut Env,
) -> Option<Result<Value, SheafError>> {
    let session_idx = func_def.vmfb_session_idx?;
    let session = env.vmfb_sessions.get(session_idx)?;
    let iree_session = session.downcast_ref::<crate::runtime::iree_session::IreeSession>()?;

    let sig = func_def.signature.as_ref()?;

    // Validate tensor count AND shapes before calling into IREE.
    // This prevents the C runtime from printing ugly diagnostics to stderr.
    // No warning: the caller handles recompilation and warns only if that fails too.
    if !crate::runtime::iree_session::args_match_signature(args, &sig.param_types) {
        return None;
    }
    if crate::runtime::iree_session::check_shapes_match(args, &sig.param_types).is_err() {
        return None;
    }

    let full_name = format!("module.{}", func_def.name.replace('-', "_").replace('?', "_q").replace('!', "_b"));
    let result = match iree_session.call_typed_device(&full_name, args, &sig.return_type) {
        Ok(v) => v,
        Err(e) => {
            return Some(Err(runtime_error(format!(
                "{}: runtime error: {}", func_def.name, e.short_message()
            ))));
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
pub(super) fn try_jit_vag(
    func: &Value,
    params: &Value,
    env: &mut Env,
) -> Option<Result<Value, SheafError>> {
    // Augment the closure with free variables resolved from the environment.
    // We use dynamic scoping, so lambdas don't capture free vars at creation.
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
            return Some(Err(runtime_error(format!(
                "value-and-grad: runtime error: {}", e.short_message()
            ))));
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
            return Some(Err(runtime_error(
                "value-and-grad: result unpacking failed".to_string()
            )));
        }
        Err(e) => {
            let detail = e.downcast_ref::<String>().map(|s| s.as_str())
                .or_else(|| e.downcast_ref::<&str>().copied())
                .unwrap_or("unknown");
            return Some(Err(runtime_error(format!(
                "value-and-grad: result unpacking panicked: {}", detail
            ))));
        }
    };

    Some(Ok(unpacked))
}

/// Augment a lambda's closure with free variables from the dynamic environment.
/// Sheaf lambdas have empty closures (dynamic scoping) but the JIT needs explicit captures.
fn augment_closure_with_free_vars(func: &Value, env: &Env) -> Option<Value> {
    let (fn_params, body, closure) = match func {
        Value::Function { params, body, closure, .. } => (params, body, closure),
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

    // Sort for deterministic parameter ordering (avoids MLIR hash changes -> cache misses)
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
        name: None,
        params: fn_params.clone(),
        body: body.clone(),
        closure: augmented_closure,
    })
}

/// Collect all symbol names referenced in a CompiledExpr (not bound by inner let/lambda).
fn collect_free_vars_compiled(expr: &crate::core::expr::CompiledExpr, out: &mut std::collections::HashSet<String>) {
    use crate::core::expr::CompiledExpr;
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
fn unpack_vag_result(result: &Value, original_params: &Value) -> Option<Value> {
    let elems = match result {
        Value::Tuple(elems) => elems,
        _ => return None,
    };

    if elems.len() < 2 {
        return None;
    }

    // First element: loss (scalar tensor -> Float)
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
