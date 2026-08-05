// Copyright (c) 2025 Damien Boureille
// Licensed under the MIT License.

//! IREE runtime dispatch for JIT-compiled functions and value-and-grad.

#![cfg(iree_runtime)]

use crate::sheaf_msg;
use crate::runtime::jit::JitVagOutcome;
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
    let aot_variant = match func_def.signature.as_ref() {
        Some(signature)
            if crate::runtime::iree_session::args_match_signature(args, &signature.param_types)
                && crate::runtime::iree_session::check_shapes_match(
                    args,
                    &signature.param_types,
                )
                .is_ok() =>
        {
            match crate::runtime::iree_session::initialized_shared_session() {
                Some(session) => {
                    match session.precompiled_module_for(&func_def.name, &func_def.body_hash()) {
                        Ok(module_name) => module_name.map(|name| (name, signature.clone())),
                        Err(e) => return Some(Err(e)),
                    }
                }
                None => None,
            }
        }
        _ => None,
    };
    let (full_name, sig) = match aot_variant {
        Some((module_name, signature)) => (dispatch_name(&module_name, &func_def.name), signature),
        None => {
            let jit = env.jit_compiler.as_mut()?;
            let key = crate::runtime::jit::cache_key_for_function(func_def, args, &env.registry)?;
            let info = jit.module_for_key(&key)?;
            (jit_dispatch_name(&info, &func_def.name), info.sig)
        }
    };

    let iree_session = match crate::runtime::iree_session::shared_session() {
        Ok(session) => session,
        Err(e) => return Some(Err(e)),
    };
    let result = match iree_session.call_typed_device(&full_name, args, &sig.return_type) {
        Ok(v) => v,
        Err(e) => return Some(Err(e)),
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

fn dispatch_name(module_name: &str, function_name: &str) -> String {
    format!(
        "{}.{}",
        module_name,
        function_name
            .replace('-', "_")
            .replace('?', "_q")
            .replace('!', "_b")
    )
}

fn jit_dispatch_name(
    module: &crate::runtime::jit::CompiledModuleInfo,
    function_name: &str,
) -> String {
    dispatch_name(&module.module_name, function_name)
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

    let iree_session = match crate::runtime::iree_session::shared_session() {
        Ok(session) => session,
        Err(e) => return JitVagOutcome::Success(Err(e)),
    };

    let (module_name, sig, param_names) = match jit.try_jit_value_and_grad(
        &augmented_func,
        params,
        &env.registry,
        &iree_session,
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

    let full_name = format!("{}.value_and_grad", module_name);
    let result = match iree_session.call_typed_device(&full_name, &args, &sig.return_type) {
        Ok(v) => v,
        Err(e) => return JitVagOutcome::Success(Err(e)),
    };

    if let Some(ref mut mp) = env.mem_profiler {
        mp.sample("after VAG dispatch");
    }

    if crate::core::config::verbosity() >= 2 {
        let desc = match &result {
            Value::Tuple(elems) => format!("Tuple(len={})", elems.len()),
            other => other.type_name().to_string(),
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

#[cfg(test)]
mod tests {
    use super::{dispatch_name, jit_dispatch_name};
    use crate::core::inference::FunctionSignature;
    use crate::runtime::jit::CompiledModuleInfo;

    fn signature() -> FunctionSignature {
        FunctionSignature {
            param_types: vec![crate::StableHLOType::scalar_f32()],
            return_type: crate::StableHLOType::scalar_f32(),
            return_dict_keys: None,
            arg_type_layouts: Vec::new(),
            captured_scalars: std::collections::HashMap::new(),
        }
    }

    #[test]
    fn dispatch_name_uses_registered_module_name() {
        assert_eq!(
            dispatch_name("aot_model_42", "predict-value?"),
            "aot_model_42.predict_value_q"
        );
    }

    #[test]
    fn jit_dispatch_name_uses_catalogued_variant_namespace() {
        let module = CompiledModuleInfo {
            function_name: "predict-value?".to_string(),
            module_name: "jit_variant_42".to_string(),
            sig: signature(),
        };

        assert_eq!(
            jit_dispatch_name(&module, "predict-value?"),
            "jit_variant_42.predict_value_q"
        );
    }
}
