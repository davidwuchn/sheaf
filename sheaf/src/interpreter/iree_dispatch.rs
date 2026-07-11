// Copyright (c) 2025 Damien Boureille
// Licensed under the MIT License.

//! IREE runtime dispatch for JIT-compiled functions and value-and-grad.

#![cfg(iree_runtime)]

use crate::sheaf_msg;
use crate::runtime::jit::JitVagOutcome;
use crate::core::error::SheafError;
use crate::interpreter::env::{runtime_error, Env};
use crate::interpreter::value::Value;
use std::collections::{BTreeMap, HashSet};
use std::sync::{Mutex, OnceLock};

static STALE_WARNINGS: OnceLock<Mutex<HashSet<(String, String, String)>>> = OnceLock::new();

fn warn_stale_scalars_once(name: &str, diffs: &[ (String, String, f64, f64) ]) {
    let warnings = STALE_WARNINGS.get_or_init(|| Mutex::new(HashSet::new()));
    let mut lock = warnings.lock().unwrap();

    for (param, path, baked, current) in diffs {
        if lock.insert((name.to_string(), param.clone(), path.clone())) {
            sheaf_msg!("WARN {}: scalar {}.{} is baked into the compiled VMFB (value {}) but the current call passed {}. The runtime value is ignored. Delete __sheaf__/ and re-run if you need this to change, or pass this value as a positional argument instead of inside a dict.", 
                name, param, path, baked, current);
        }
    }
}

fn find_scalar_with_path(args: &[Value], param_names: &[String], param: &str, indices: &[usize]) -> Option<(f64, String)> {
    let arg_idx = param_names.iter().position(|p| p == param)?;
    let mut current = &args[arg_idx];
    let mut path_parts = Vec::new();

    for &idx in indices {
        match current {
            Value::Tuple(elems) => {
                if idx < elems.len() {
                    current = &elems[idx];
                } else {
                    return None;
                }
            }
            Value::Dict(map) => {
                let entry = map.iter().nth(idx)?;
                path_parts.push(entry.0.clone());
                current = entry.1;
            }
            _ => return None,
        }
    }

    let val = current.to_f64()?;
    let path_str = if path_parts.is_empty() {
        format!("{:?}", indices)
    } else {
        path_parts.join(".")
    };

    Some((val, path_str))
}

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

    // Check for stale baked scalars in dict arguments
    if !sig.captured_scalars.is_empty() {
        let mut diffs = Vec::new();
        for ((param, indices), baked) in &sig.captured_scalars {
            if let Some((current, path)) = find_scalar_with_path(args, &func_def.params, param, indices) {
                if (current - *baked).abs() > 1e-6 {
                    diffs.push((param.clone(), path, *baked, current));
                }
            }
        }
        if !diffs.is_empty() {
            warn_stale_scalars_once(func_def.name.as_str(), &diffs);
        }
    }

    let full_name = format!("module.{}", func_def.name.replace('-', "_").replace('?', "_q").replace('!', "_b"));
    let result = match iree_session.call_typed_device(&full_name, args, &sig.return_type) {
        Ok(v) => v,
        Err(e) => {
            return Some(Err(e));
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
pub(super) fn try_jit_vag(
    func: &Value,
    params: &Value,
    env: &mut Env,
) -> JitVagOutcome {
    let augmented_func = match augment_closure_with_free_vars(func, env) {
        Some(f) => f,
        None => return JitVagOutcome::Unsupported,
    };

    let jit = match env.jit_compiler.as_mut() {
        Some(j) => j,
        None => return JitVagOutcome::Unsupported,
    };

    let (session_idx, sig, param_names) = match jit.try_jit_value_and_grad(
        &augmented_func,
        params,
        &env.registry,
        &mut env.vmfb_sessions,
    ) {
        Some(x) => x,
        None => return jit.classify_vag_skip(),
    };

    if let Some(ref mut mp) = env.mem_profiler {
        mp.sample("after VAG compile");
    }

    let (fn_params, closure) = match &augmented_func {
        Value::Function { params: p, closure: c, .. } => (p, c),
        _ => return JitVagOutcome::Unsupported,
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

    let session = match env.vmfb_sessions.get(session_idx) {
        Some(s) => s,
        None => return JitVagOutcome::Bug("vmfb session lost".to_string()),
    };
    let iree_session = match session.downcast_ref::<crate::runtime::iree_session::IreeSession>() {
        Some(s) => s,
        None => return JitVagOutcome::Bug("downcast to IreeSession failed".to_string()),
    };

    let result = match iree_session.call_typed_device("module.value_and_grad", &args, &sig.return_type) {
        Ok(v) => v,
        Err(e) => return JitVagOutcome::Success(Err(e)),
    };

    if let Some(ref mut mp) = env.mem_profiler {
        mp.sample("after VAG dispatch");
    }

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
            return JitVagOutcome::Success(Err(runtime_error(
                "value-and-grad: result unpacking failed".to_string(),
            )));
        }
        Err(e) => {
            let detail = e.downcast_ref::<String>().map(|s| s.as_str())
                .or_else(|| e.downcast_ref::<&str>().copied())
                .unwrap_or("unknown");
            return JitVagOutcome::Success(Err(runtime_error(format!(
                "value-and-grad: result unpacking panicked: {}", detail
            ))));
        }
    };

    JitVagOutcome::Success(Ok(unpacked))
}

/// Augment a lambda's closure with free variables from the dynamic environment.
/// Sheaf lambdas have empty closures (dynamic scoping) but the JIT needs explicit captures.
fn augment_closure_with_free_vars(func: &Value, env: &Env) -> Option<Value> {
    let (fn_params, body, closure) = match func {
        Value::Function { params, body, closure, .. } => (params, body, closure),
        _ => return None,
    };

    let mut free_set = std::collections::HashSet::new();
    crate::autodiff::collect_free_vars(body, &mut free_set);

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
            Value::Float(crate::interpreter::builtins::as_scalar(data))
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
