// Copyright (c) 2025 Damien Boureille
// Licensed under the MIT License.

//! Value-and-grad: automatic differentiation via tracing and finite differences.

use crate::sheaf_msg;
use crate::core::expr::CompiledExpr;
use crate::core::error::SheafError;
use crate::interpreter::env::{runtime_error, Env};
use crate::interpreter::value::Value;
use ndarray::{ArrayD, IxDyn};
use std::collections::BTreeMap;
use std::sync::Arc;

use super::{eval, call_function};

/// (value-and-grad f) -> returns a function that, given params, returns [loss, grad_params].
///
/// Gradient computed by central finite differences: grad[i] ≈ (f(p+h) - f(p-h)) / 2h
/// Applied element-wise to every leaf tensor in the params pytree.
pub(super) fn eval_value_and_grad_hof(args: &[Value], _env: &mut Env) -> Result<Value, SheafError> {
    if args.len() != 1 {
        return Err(runtime_error("value-and-grad: expected exactly 1 argument (the function)"));
    }
    let func = args[0].clone();

    // Return a Value::Function closure that captures `func`.
    // When called with params, call_function detects __vag_fn__ and dispatches
    // to eval_value_and_grad_call which computes (loss, grad) via finite differences.
    Ok(Value::Function {
        name: None,
        params: vec!["__vag_params__".to_string()],
        body: crate::core::expr::CompiledExpr::Symbol("__vag_params__".to_string()),
        closure: vec![("__vag_fn__".to_string(), func)],
    })
}

/// Evaluate a value-and-grad HOF call.
///
/// Tries JIT compilation first, then symbolic autodiff with tracing.
/// Raises a fatal error if differentiation fails.
pub(super) fn eval_value_and_grad_call(func: &Value, params: &Value, env: &mut Env) -> Result<Value, SheafError> {
    // Skip when tracing -- interpreter must run to expose the autodiff call tree
    #[cfg(iree_runtime)]
    if env.tracer.is_none() {
        match super::iree_dispatch::try_jit_vag(func, params, env) {
            Some(result) => return result,
            None => {
                // JIT failed: fall through to symbolic autodiff.
                // This path has incomplete backward rules (softmax, gelu, etc.)
                // and should only be reached for simple expressions.
            }
        }
    }

    use crate::autodiff::{contains_undiffable_ops, grad_simplified, inline_function_calls};

    if let Value::Function { params: fn_params, body, closure, .. } = func {
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

            // Evaluate structural ops (get, reduce) with concrete values,
            // keep tensor ops symbolic, then differentiate.
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
                            env.pop_scope();
                            let loss_val = call_function(func, &[params.clone()], env)?;
                            let loss = scalar_from_value(&loss_val)?;

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
                            ).map_err(|e| {
                                env.pop_scope();
                                runtime_error(format!("value-and-grad -> {}", e.short_message()))
                            })?;
                            env.pop_scope();
                            return Ok(Value::List(vec![Value::Float(loss), grad_tree]));
                        }
                        env.pop_scope();
                        return Err(runtime_error(
                            "value-and-grad: tracing produced undifferentiable ops (this is a bug in the autodiff engine)"
                        ));
                    }
                    Err(e) => {
                        env.pop_scope();
                        return Err(runtime_error(format!(
                            "value-and-grad -> {}", e.short_message()
                        )));
                    }
                }
            }
        }
    }

    Err(runtime_error("value-and-grad: cannot differentiate this function"))
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
    // by data equality. Tracer order may differ from the params tree traversal order.
    build_grad_tree_by_value(params, leaves, &leaf_grads)
}

fn values_equal(a: &Value, b: &Value) -> bool {
    // Materialize DeviceBuffers for comparison
    let a_host = a.ensure_host().ok();
    let b_host = b.ensure_host().ok();
    let a = a_host.as_ref().unwrap_or(a);
    let b = b_host.as_ref().unwrap_or(b);
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
        Value::Tensor { .. } | Value::Float(_) | Value::Int(_) | Value::DeviceBuffer(_) => {
            // Materialize DeviceBuffer for comparison
            let params_host = params.ensure_host().unwrap_or_else(|_| params.clone());
            // Find the leaf that matches this param value
            for (sym, leaf_val) in leaves {
                if values_equal(&params_host, leaf_val) {
                    if let Some(grad_val) = leaf_grads.get(sym) {
                        return reduce_grad_to_param_shape(grad_val, &params_host);
                    }
                }
            }
            // No matching leaf found == this param wasn't used in the expression.
            // Return zeros with the same shape.
            Ok(zeros_like(&params_host))
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
/// When an op broadcasts a param (e.g., bias [1] -> [4,1]),
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
            // e.g., grad [4,1] + param [1] -> sum axis 0 -> [1]
            // e.g., grad [4,2,1] + param [2,1] -> sum axis 0 -> [2,1]
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
