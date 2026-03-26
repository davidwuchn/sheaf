// Copyright (c) 2025 Damien Boureille
// Licensed under the MIT License.

//! Built-in functions for the Sheaf interpreter.

mod arithmetic;
mod activations;
pub(crate) mod pickle;
mod safetensors;
mod comparison;
mod reductions;
mod tensor_ops;
mod tensor_create;
mod collections;
mod io;
mod random;
mod losses;

use crate::interpreter::env::{arity_error, arity_error_with_args, runtime_error, Env};
use crate::interpreter::value::{Dtype, Value};
use ndarray::{ArrayD, Dimension, IxDyn};
use std::collections::BTreeMap;
use std::sync::Arc;

pub(self) type R = Result<Value, crate::core::error::SheafError>;

pub use losses::tree_zeros;

pub fn register_builtins(env: &mut Env) {
    arithmetic::register(env);
    activations::register(env);
    comparison::register(env);
    reductions::register(env);
    tensor_ops::register(env);
    tensor_create::register(env);
    collections::register(env);
    io::register(env);
    random::register(env);
    losses::register(env);
}

/// Resolve a potentially negative index against a dimension length.
/// Negative indices wrap from the end: -1 -> len-1, -2 -> len-2, etc.
pub(self) fn resolve_idx(f: f64, len: usize) -> usize {
    if f < 0.0 {
        (len as i64 + f as i64) as usize
    } else {
        f as usize
    }
}

pub(self) fn to_array(val: &Value) -> Result<(ArrayD<f32>, Dtype), crate::core::error::SheafError> {
    match val {
        Value::Int(n) => Ok((ArrayD::from_elem(IxDyn(&[]), *n as f32), Dtype::I32)),
        Value::Float(f) => Ok((ArrayD::from_elem(IxDyn(&[]), *f), Dtype::F32)),
        Value::Bool(b) => Ok((ArrayD::from_elem(IxDyn(&[]), if *b { 1.0f32 } else { 0.0f32 }), Dtype::I32)),
        Value::Tensor { data, dtype } => Ok(((**data).clone(), *dtype)),
        Value::DeviceBuffer(db) => {
            let data = db.to_host().map_err(|e| runtime_error(format!("materialize: {}", e)))?;
            Ok((data, db.dtype))
        }
        _ => Err(runtime_error(format!("Expected numeric value, got {} ({})", val.type_name(), val))),
    }
}

fn broadcast_shape(a: &[usize], b: &[usize]) -> Option<Vec<usize>> {
    let max_ndim = a.len().max(b.len());
    let mut result = Vec::with_capacity(max_ndim);
    for i in 0..max_ndim {
        let da = if i < a.len() { a[a.len() - 1 - i] } else { 1 };
        let db = if i < b.len() { b[b.len() - 1 - i] } else { 1 };
        if da == db { result.push(da); }
        else if da == 1 { result.push(db); }
        else if db == 1 { result.push(da); }
        else { return None; }
    }
    result.reverse();
    Some(result)
}

pub(self) fn result_dtype(a: Dtype, b: Dtype) -> Dtype {
    if a == b { return a; }
    if a == Dtype::Bool { return b; }
    if b == Dtype::Bool { return a; }
    // Promote: any float wins over I32; F32 wins over BF16/F16
    match (a, b) {
        (Dtype::F32, _) | (_, Dtype::F32) => Dtype::F32,
        (Dtype::BF16, _) | (_, Dtype::BF16) => Dtype::BF16,
        _ => Dtype::I32,
    }
}

pub(self) fn binary_op(args: &[Value], op: fn(f32, f32) -> f32) -> R {
    if args.len() < 2 {
        return Err(runtime_error("Binary operation requires at least 2 arguments"));
    }
    // Fast path for the common 2-arg case: borrow both operands from Arc, zero clones
    if args.len() == 2 {
        return binary_op_two(&args[0], &args[1], op);
    }
    let (mut acc, mut dt) = to_array(&args[0])?;
    for arg in &args[1..] {
        // Borrow RHS from Arc when possible, avoiding clone
        match arg {
            Value::Int(n) => {
                let s = *n as f32;
                dt = result_dtype(dt, Dtype::I32);
                if acc.ndim() == 0 {
                    acc = ArrayD::from_elem(IxDyn(&[]), op(*acc.first().unwrap(), s));
                } else {
                    acc = acc.mapv(|x| op(x, s));
                }
            }
            Value::Float(f) => {
                let s = *f as f32;
                dt = result_dtype(dt, Dtype::F32);
                if acc.ndim() == 0 {
                    acc = ArrayD::from_elem(IxDyn(&[]), op(*acc.first().unwrap(), s));
                } else {
                    acc = acc.mapv(|x| op(x, s));
                }
            }
            Value::Bool(b) => {
                let s = if *b { 1.0 } else { 0.0 };
                dt = result_dtype(dt, Dtype::I32);
                if acc.ndim() == 0 {
                    acc = ArrayD::from_elem(IxDyn(&[]), op(*acc.first().unwrap(), s));
                } else {
                    acc = acc.mapv(|x| op(x, s));
                }
            }
            Value::Tensor { data: b, dtype: bdt } => {
                dt = result_dtype(dt, *bdt);
                if acc.ndim() == 0 && b.ndim() != 0 {
                    let scalar = *acc.first().unwrap();
                    acc = b.mapv(|x| op(scalar, x));
                } else if b.ndim() == 0 && acc.ndim() != 0 {
                    let scalar = *b.first().unwrap();
                    acc = acc.mapv(|x| op(x, scalar));
                } else if acc.shape() == b.shape() {
                    acc = ndarray::Zip::from(&acc).and(b.as_ref()).map_collect(|&a, &b| op(a, b));
                } else {
                    let out_shape = broadcast_shape(acc.shape(), b.shape()).ok_or_else(|| {
                        runtime_error(format!("Cannot broadcast shapes {:?} and {:?}", acc.shape(), b.shape()))
                    })?;
                    let a_bc = acc.broadcast(&out_shape[..]).unwrap();
                    let b_bc = b.broadcast(&out_shape[..]).unwrap();
                    acc = ndarray::Zip::from(&a_bc).and(&b_bc).map_collect(|&a, &b| op(a, b));
                }
            }
            _ => return Err(runtime_error(format!("Expected numeric value, got {}", arg.type_name()))),
        }
    }
    if acc.ndim() == 0 {
        let x = *acc.first().unwrap();
        if dt == Dtype::I32 && x == x.floor() {
            Ok(Value::Int(x as i64))
        } else {
            Ok(Value::Float(x))
        }
    } else {
        Ok(Value::Tensor { data: Arc::new(acc), dtype: dt })
    }
}

/// Zero-clone fast path for binary ops with exactly 2 arguments.
/// Borrows tensor data directly from Arc instead of cloning.
fn binary_op_two(a: &Value, b: &Value, op: fn(f32, f32) -> f32) -> R {
    // Materialize DeviceBuffers to host for arithmetic
    let a_host;
    let b_host;
    let a = if matches!(a, Value::DeviceBuffer(_)) { a_host = a.ensure_host()?; &a_host } else { a };
    let b = if matches!(b, Value::DeviceBuffer(_)) { b_host = b.ensure_host()?; &b_host } else { b };

    let a_scalar = scalar_of(a);
    let b_scalar = scalar_of(b);
    let adt = dtype_of(a);
    let bdt = dtype_of(b);
    let dt = result_dtype(adt, bdt);

    // scalar op scalar
    if let (Some(sa), Some(sb)) = (a_scalar, b_scalar) {
        let x = op(sa, sb);
        return if dt == Dtype::I32 && x == x.floor() {
            Ok(Value::Int(x as i64))
        } else {
            Ok(Value::Float(x))
        };
    }

    // scalar op tensor
    if let (Some(s), Value::Tensor { data, .. }) = (a_scalar, b) {
        let result = if data.ndim() == 0 {
            let x = op(s, *data.first().unwrap());
            return Ok(Value::Float(x));
        } else {
            data.mapv(|x| op(s, x))
        };
        return Ok(Value::Tensor { data: Arc::new(result), dtype: dt });
    }

    // tensor op scalar
    if let (Value::Tensor { data, .. }, Some(s)) = (a, b_scalar) {
        let result = if data.ndim() == 0 {
            let x = op(*data.first().unwrap(), s);
            return Ok(Value::Float(x));
        } else {
            data.mapv(|x| op(x, s))
        };
        return Ok(Value::Tensor { data: Arc::new(result), dtype: dt });
    }

    // tensor op tensor
    if let (Value::Tensor { data: ad, .. }, Value::Tensor { data: bd, .. }) = (a, b) {
        let result = if ad.ndim() == 0 {
            let s = *ad.first().unwrap();
            if bd.ndim() == 0 {
                return Ok(Value::Float(op(s, *bd.first().unwrap())));
            }
            bd.mapv(|x| op(s, x))
        } else if bd.ndim() == 0 {
            let s = *bd.first().unwrap();
            ad.mapv(|x| op(x, s))
        } else if ad.shape() == bd.shape() {
            ndarray::Zip::from(ad.as_ref()).and(bd.as_ref()).map_collect(|&a, &b| op(a, b))
        } else {
            let out_shape = broadcast_shape(ad.shape(), bd.shape()).ok_or_else(|| {
                runtime_error(format!("Cannot broadcast shapes {:?} and {:?}", ad.shape(), bd.shape()))
            })?;
            let a_bc = ad.broadcast(&out_shape[..]).unwrap();
            let b_bc = bd.broadcast(&out_shape[..]).unwrap();
            ndarray::Zip::from(&a_bc).and(&b_bc).map_collect(|&a, &b| op(a, b))
        };
        return Ok(Value::Tensor { data: Arc::new(result), dtype: dt });
    }

    Err(runtime_error(format!("Expected numeric values, got {} and {}", a.type_name(), b.type_name())))
}

fn scalar_of(v: &Value) -> Option<f32> {
    match v {
        Value::Int(n) => Some(*n as f32),
        Value::Float(f) => Some(*f),
        Value::Bool(b) => Some(if *b { 1.0 } else { 0.0 }),
        _ => None,
    }
}

fn dtype_of(v: &Value) -> Dtype {
    match v {
        Value::Tensor { dtype, .. } => *dtype,
        Value::Float(_) => Dtype::F32,
        _ => Dtype::I32,
    }
}

pub(self) fn unary_op(args: &[Value], op: fn(f32) -> f32) -> R {
    if args.is_empty() {
        return Err(runtime_error("Unary operation requires at least 1 argument"));
    }
    let (arr, _dt) = to_array(&args[0])?;
    let result = arr.mapv(op);
    if result.ndim() == 0 {
        Ok(Value::Float(*result.first().unwrap()))
    } else {
        Ok(Value::Tensor { data: Arc::new(result), dtype: Dtype::F32 })
    }
}

pub(self) fn unary_op_f32(args: &[Value], op: fn(f32) -> f32) -> R {
    if args.is_empty() {
        return Err(runtime_error("Unary operation requires at least 1 argument"));
    }
    let (arr, _dt) = to_array(&args[0])?;
    let result = arr.mapv(op);
    if result.ndim() == 0 {
        Ok(Value::Float(*result.first().unwrap()))
    } else {
        Ok(Value::Tensor { data: Arc::new(result), dtype: Dtype::F32 })
    }
}

pub(self) fn get_axis(kw: &BTreeMap<String, Value>) -> Option<i64> {
    kw.get("axis").and_then(|v| match v {
        Value::Int(n) => Some(*n),
        Value::Float(f) => Some(*f as i64),
        _ => None,
    })
}

/// Extract dtype override from kwargs (e.g. `:bf16` flag).
pub(self) fn get_dtype_kwarg(kw: &BTreeMap<String, Value>) -> Option<Dtype> {
    for key in kw.keys() {
        if let Some(dt) = Dtype::from_keyword(key) {
            return Some(dt);
        }
    }
    None
}

/// Apply dtype kwarg override to a result value.
pub(self) fn with_dtype_kwarg(result: R, kw: &BTreeMap<String, Value>) -> R {
    if let Some(dt) = get_dtype_kwarg(kw) {
        match result? {
            Value::Tensor { data, .. } => Ok(Value::Tensor { data, dtype: dt }),
            other => Ok(other),
        }
    } else {
        result
    }
}

pub(self) fn reduce_along_axis(arr: &ArrayD<f32>, axis: usize, op: fn(&[f32]) -> f32) -> ArrayD<f32> {
    let shape = arr.shape();
    if shape.is_empty() || axis >= shape.len() {
        let data: Vec<f32> = arr.iter().copied().collect();
        return ArrayD::from_elem(IxDyn(&[]), op(&data));
    }
    let mut new_shape: Vec<usize> = shape.to_vec();
    new_shape.remove(axis);
    if new_shape.is_empty() {
        let data: Vec<f32> = arr.iter().copied().collect();
        return ArrayD::from_elem(IxDyn(&[]), op(&data));
    }
    let total = new_shape.iter().product::<usize>();
    let mut result_data = Vec::with_capacity(total);
    let n_axis = shape[axis];
    for idx in ndarray::indices(&*new_shape) {
        let mut vals = Vec::with_capacity(n_axis);
        for k in 0..n_axis {
            let mut full_idx: Vec<usize> = idx.as_array_view().to_vec();
            full_idx.insert(axis, k);
            vals.push(arr[IxDyn(&full_idx)]);
        }
        result_data.push(op(&vals));
    }
    ArrayD::from_shape_vec(IxDyn(&new_shape), result_data).unwrap()
}

pub(self) fn argreduce_along_axis(arr: &ArrayD<f32>, axis: usize, cmp: fn(f32, f32) -> bool) -> ArrayD<f32> {
    let shape = arr.shape();
    let mut new_shape: Vec<usize> = shape.to_vec();
    new_shape.remove(axis);
    if new_shape.is_empty() {
        let data: Vec<f32> = arr.iter().copied().collect();
        let idx = data.iter().enumerate().fold(0, |best, (i, &x)| if cmp(x, data[best]) { i } else { best });
        return ArrayD::from_elem(IxDyn(&[]), idx as f32);
    }
    let total = new_shape.iter().product::<usize>();
    let mut result_data = Vec::with_capacity(total);
    let n_axis = shape[axis];
    for idx in ndarray::indices(&*new_shape) {
        let mut best_idx = 0;
        let mut full_idx: Vec<usize> = idx.as_array_view().to_vec();
        full_idx.insert(axis, 0);
        let mut best_val = arr[IxDyn(&full_idx)];
        for k in 1..n_axis {
            full_idx[axis] = k;
            let v = arr[IxDyn(&full_idx)];
            if cmp(v, best_val) {
                best_val = v;
                best_idx = k;
            }
        }
        result_data.push(best_idx as f32);
    }
    ArrayD::from_shape_vec(IxDyn(&new_shape), result_data).unwrap()
}

pub(self) fn shape_from_value(val: &Value) -> Result<Vec<usize>, crate::core::error::SheafError> {
    match val {
        Value::List(items) => {
            items.iter().map(|v| match v {
                Value::Int(n) => Ok(*n as usize),
                Value::Float(f) => Ok(*f as usize),
                _ => Err(runtime_error("shape must contain integers")),
            }).collect()
        }
        Value::Tensor { data, .. } => {
            Ok(data.iter().map(|&x| x as usize).collect())
        }
        _ => Err(runtime_error(format!("Expected shape list, got {}", val.type_name()))),
    }
}

pub(self) fn list_to_tensor(v: &Value) -> Option<(ArrayD<f32>, Dtype)> {
    match v {
        Value::Tensor { data, dtype } => Some(((**data).clone(), *dtype)),
        Value::List(items) => {
            let all_int = items.iter().all(|x| matches!(x, Value::Int(_)));
            let nums: Option<Vec<f32>> = items.iter().map(|x| x.to_f64().map(|v| v as f32)).collect();
            if let Some(data) = nums {
                let dtype = if all_int { Dtype::I32 } else { Dtype::F32 };
                return ArrayD::from_shape_vec(IxDyn(&[data.len()]), data).ok()
                    .map(|a| (a, dtype));
            }
            let rows: Option<Vec<(ArrayD<f32>, Dtype)>> = items.iter().map(list_to_tensor).collect();
            if let Some(rows) = rows {
                let all_i32 = rows.iter().all(|(_, dt)| *dt == Dtype::I32);
                let dtype = if all_i32 { Dtype::I32 } else { Dtype::F32 };
                let stacked: Option<ArrayD<f32>> = ndarray::concatenate(
                    ndarray::Axis(0),
                    &rows.iter().map(|(r, _)| r.view().insert_axis(ndarray::Axis(0))).collect::<Vec<_>>()
                ).ok();
                stacked.map(|a| (a, dtype))
            } else {
                None
            }
        }
        _ => None,
    }
}

pub(self) fn cmp_op(args: &[Value], op: fn(f32, f32) -> f32, _dt: Dtype) -> R {
    if args.len() != 2 { return Err(runtime_error("Comparison requires 2 arguments")); }
    let (a, _) = to_array(&args[0])?;
    let (b, _) = to_array(&args[1])?;
    if a.ndim() == 0 && b.ndim() != 0 {
        let scalar = *a.first().unwrap();
        let result = b.mapv(|x| op(scalar, x));
        return Ok(bool_tensor(result));
    }
    if b.ndim() == 0 && a.ndim() != 0 {
        let scalar = *b.first().unwrap();
        let result = a.mapv(|x| op(x, scalar));
        return Ok(bool_tensor(result));
    }
    if a.ndim() == 0 && b.ndim() == 0 {
        let r = op(*a.first().unwrap(), *b.first().unwrap());
        return Ok(Value::Bool(r != 0.0));
    }
    let result = ndarray::Zip::from(&a).and(&b).map_collect(|&a, &b| op(a, b));
    Ok(bool_tensor(result))
}

pub(self) fn bool_tensor(data: ArrayD<f32>) -> Value {
    Value::Tensor { data: Arc::new(data), dtype: Dtype::Bool }
}
