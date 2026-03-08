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

use crate::interpreter::env::{runtime_error, Env};
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
/// Negative indices wrap from the end: -1 → len-1, -2 → len-2, etc.
pub(self) fn resolve_idx(f: f64, len: usize) -> usize {
    if f < 0.0 {
        (len as f64 + f) as usize
    } else {
        f as usize
    }
}

pub(self) fn to_array(val: &Value) -> Result<(ArrayD<f64>, Dtype), crate::core::error::SheafError> {
    match val {
        Value::Int(n) => Ok((ArrayD::from_elem(IxDyn(&[]), *n as f64), Dtype::I32)),
        Value::Float(f) => Ok((ArrayD::from_elem(IxDyn(&[]), *f), Dtype::F32)),
        Value::Bool(b) => Ok((ArrayD::from_elem(IxDyn(&[]), if *b { 1.0 } else { 0.0 }), Dtype::I32)),
        Value::Tensor { data, dtype } => Ok(((**data).clone(), *dtype)),
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
    if a == Dtype::F32 || b == Dtype::F32 { Dtype::F32 } else if a == Dtype::Bool && b == Dtype::Bool { Dtype::Bool } else { Dtype::I32 }
}

pub(self) fn binary_op(args: &[Value], op: fn(f64, f64) -> f64) -> R {
    if args.len() < 2 {
        return Err(runtime_error("Binary operation requires at least 2 arguments"));
    }
    let (mut acc, mut dt) = to_array(&args[0])?;
    let mut any_tensor = !matches!(&args[0], Value::Int(_) | Value::Float(_) | Value::Bool(_));
    for arg in &args[1..] {
        let (b, bdt) = to_array(arg)?;
        dt = result_dtype(dt, bdt);
        if !matches!(arg, Value::Int(_) | Value::Float(_) | Value::Bool(_)) {
            any_tensor = true;
        }
        if acc.ndim() == 0 && b.ndim() != 0 {
            let scalar = *acc.first().unwrap();
            acc = b.mapv(|x| op(scalar, x));
        } else if b.ndim() == 0 && acc.ndim() != 0 {
            let scalar = *b.first().unwrap();
            acc = acc.mapv(|x| op(x, scalar));
        } else if acc.shape() == b.shape() {
            acc = ndarray::Zip::from(&acc).and(&b).map_collect(|&a, &b| op(a, b));
        } else {
            // Compute broadcast-compatible output shape, then broadcast both operands
            let out_shape = broadcast_shape(acc.shape(), b.shape()).ok_or_else(|| {
                runtime_error(format!("Cannot broadcast shapes {:?} and {:?}", acc.shape(), b.shape()))
            })?;
            let a_bc = acc.broadcast(&out_shape[..]).unwrap().to_owned();
            let b_bc = b.broadcast(&out_shape[..]).unwrap().to_owned();
            acc = ndarray::Zip::from(&a_bc).and(&b_bc).map_collect(|&a, &b| op(a, b));
        }
    }
    if any_tensor {
        dt = Dtype::F32;
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

pub(self) fn unary_op(args: &[Value], op: fn(f64) -> f64) -> R {
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
    let result = arr.mapv(|x| op(x as f32) as f64);
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

pub(self) fn reduce_along_axis(arr: &ArrayD<f64>, axis: usize, op: fn(&[f64]) -> f64) -> ArrayD<f64> {
    let shape = arr.shape();
    if shape.is_empty() || axis >= shape.len() {
        let data: Vec<f64> = arr.iter().copied().collect();
        return ArrayD::from_elem(IxDyn(&[]), op(&data));
    }
    let mut new_shape: Vec<usize> = shape.to_vec();
    new_shape.remove(axis);
    if new_shape.is_empty() {
        let data: Vec<f64> = arr.iter().copied().collect();
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

pub(self) fn argreduce_along_axis(arr: &ArrayD<f64>, axis: usize, cmp: fn(f64, f64) -> bool) -> ArrayD<f64> {
    let shape = arr.shape();
    let mut new_shape: Vec<usize> = shape.to_vec();
    new_shape.remove(axis);
    if new_shape.is_empty() {
        let data: Vec<f64> = arr.iter().copied().collect();
        let idx = data.iter().enumerate().fold(0, |best, (i, &x)| if cmp(x, data[best]) { i } else { best });
        return ArrayD::from_elem(IxDyn(&[]), idx as f64);
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
        result_data.push(best_idx as f64);
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

pub(self) fn list_to_tensor(v: &Value) -> Option<(ArrayD<f64>, Dtype)> {
    match v {
        Value::Tensor { data, dtype } => Some(((**data).clone(), *dtype)),
        Value::List(items) => {
            let all_int = items.iter().all(|x| matches!(x, Value::Int(_)));
            let nums: Option<Vec<f64>> = items.iter().map(|x| x.to_f64()).collect();
            if let Some(data) = nums {
                let dtype = if all_int { Dtype::I32 } else { Dtype::F32 };
                return ArrayD::from_shape_vec(IxDyn(&[data.len()]), data).ok()
                    .map(|a| (a, dtype));
            }
            let rows: Option<Vec<(ArrayD<f64>, Dtype)>> = items.iter().map(list_to_tensor).collect();
            if let Some(rows) = rows {
                let all_i32 = rows.iter().all(|(_, dt)| *dt == Dtype::I32);
                let dtype = if all_i32 { Dtype::I32 } else { Dtype::F32 };
                let stacked: Option<ArrayD<f64>> = ndarray::concatenate(
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

pub(self) fn cmp_op(args: &[Value], op: fn(f64, f64) -> f64, _dt: Dtype) -> R {
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

pub(self) fn bool_tensor(data: ArrayD<f64>) -> Value {
    Value::Tensor { data: Arc::new(data), dtype: Dtype::Bool }
}
