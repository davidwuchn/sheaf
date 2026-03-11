use super::*;
use std::sync::Arc;

pub(super) fn register(env: &mut Env) {
    env.set_builtin("reshape", builtin_reshape);
    env.set_builtin("transpose", builtin_transpose);
    env.set_builtin("concat", builtin_concat);
    env.set_builtin("slice", builtin_slice);
    env.set_builtin("get", builtin_get);
    env.set_builtin("where", builtin_where);
    env.set_builtin("roll", builtin_roll);
    env.set_builtin("index-update", builtin_index_update);
    env.set_builtin("swapaxes", builtin_swapaxes);
    env.set_builtin("dynamic-slice", builtin_dynamic_slice);
    env.set_builtin("tensor-split", builtin_tensor_split);
}

fn builtin_reshape(args: &[Value], _kw: &BTreeMap<String, Value>) -> R {
    let (arr, dt) = to_array(&args[0])?;
    let raw_shape: Vec<i64> = match &args[1] {
        Value::List(items) => items.iter().map(|v| match v {
            Value::Int(n) => *n,
            Value::Float(f) => *f as i64,
            _ => -999,
        }).collect(),
        Value::Tensor { data, .. } => data.iter().map(|&x| x as i64).collect(),
        _ => return Err(runtime_error("reshape: shape must be a list, e.g. (reshape x [2 3]) not (reshape x 2 3)")),
    };
    let total = arr.len() as i64;
    let neg_idx = raw_shape.iter().position(|&x| x < 0);
    let new_shape: Vec<usize> = if let Some(_ni) = neg_idx {
        let known: i64 = raw_shape.iter().filter(|&&x| x > 0).product();
        let inferred = total / known;
        raw_shape.iter().map(|&x| if x < 0 { inferred as usize } else { x as usize }).collect()
    } else {
        raw_shape.iter().map(|&x| x as usize).collect()
    };
    let arr = arr.as_standard_layout().into_owned();
    let result = arr.into_shape_with_order(IxDyn(&new_shape)).map_err(|e| runtime_error(e.to_string()))?;
    Ok(Value::Tensor { data: Arc::new(result), dtype: dt })
}

fn builtin_transpose(args: &[Value], _kw: &BTreeMap<String, Value>) -> R {
    let (arr, _dt) = to_array(&args[0])?;
    if arr.ndim() == 2 {
        Ok(Value::tensor_f32(arr.t().to_owned()))
    } else if arr.ndim() == 1 {
        Ok(Value::tensor_f32(arr))
    } else {
        let mut axes: Vec<usize> = (0..arr.ndim()).rev().collect();
        if args.len() > 1 {
            if let Value::List(items) = &args[1] {
                axes = items.iter().map(|v| v.to_f64().unwrap() as usize).collect();
            }
        }
        Ok(Value::tensor_f32(arr.permuted_axes(IxDyn(&axes))))
    }
}

fn builtin_concat(args: &[Value], kw: &BTreeMap<String, Value>) -> R {
    let axis = get_axis(kw).unwrap_or(0) as usize;
    let has_axis_kw = kw.contains_key("axis");

    let maybe_arrays: Option<Vec<(ArrayD<f32>, Dtype)>> = args.iter().map(|a| list_to_tensor(a)).collect();

    if let Some(arrays) = maybe_arrays {
        if has_axis_kw || args.iter().any(|a| matches!(a, Value::Tensor { .. })) {
            let all_i32 = arrays.iter().all(|(_, dt)| *dt == Dtype::I32);
            let dtype = if all_i32 { Dtype::I32 } else { Dtype::F32 };
            let views: Vec<ndarray::ArrayViewD<f32>> = arrays.iter().map(|(a, _)| a.view()).collect();
            let result = ndarray::concatenate(ndarray::Axis(axis), &views)
                .map_err(|e| runtime_error(e.to_string()))?;
            return Ok(Value::Tensor { data: Arc::new(result), dtype });
        }
    }

    if matches!(&args[0], Value::String(_)) {
        let mut s = String::new();
        for arg in args {
            match arg {
                Value::String(st) => s.push_str(st),
                other => s.push_str(&format!("{}", other)),
            }
        }
        return Ok(Value::String(s));
    }

    if matches!(&args[0], Value::List(_)) {
        let mut all_items = Vec::new();
        for arg in args {
            match arg {
                Value::List(items) => all_items.extend_from_slice(items),
                _ => all_items.push(arg.clone()),
            }
        }
        return Ok(Value::List(all_items));
    }

    let arrays: Vec<ArrayD<f32>> = args.iter().map(|a| {
        to_array(a).map(|(arr, _)| arr)
    }).collect::<Result<Vec<_>, _>>()?;
    let views: Vec<ndarray::ArrayViewD<f32>> = arrays.iter().map(|a| a.view()).collect();
    let result = ndarray::concatenate(ndarray::Axis(axis), &views)
        .map_err(|e| runtime_error(e.to_string()))?;
    Ok(Value::tensor_f32(result))
}

fn builtin_slice(args: &[Value], kw: &BTreeMap<String, Value>) -> R {
    if let Value::String(s) = &args[0] {
        let start = args[1].to_f64().unwrap() as usize;
        let end = if args.len() > 2 { args[2].to_f64().unwrap() as usize } else { s.len() };
        return Ok(Value::String(s[start..end.min(s.len())].to_string()));
    }
    let (arr, _dt) = to_array(&args[0])?;
    let axis_raw = kw.get("axis").and_then(|v| v.to_f64()).map(|v| v as i64).unwrap_or(0);
    let axis = if axis_raw < 0 { (arr.ndim() as i64 + axis_raw) as usize } else { axis_raw as usize };
    let start = args[1].to_f64().unwrap() as usize;
    let end = if args.len() > 2 { args[2].to_f64().unwrap() as usize } else { arr.shape()[axis] };
    let sliced = arr.slice_axis(ndarray::Axis(axis), ndarray::Slice::from(start..end));
    Ok(Value::tensor_f32(sliced.to_owned()))
}

fn builtin_get(args: &[Value], kw: &BTreeMap<String, Value>) -> R {
    match &args[0] {
        Value::Dict(map) => {
            let key = match &args[1] {
                Value::Keyword(k) => k.clone(),
                Value::String(s) => s.clone(),
                _ => return Err(runtime_error("get: key must be keyword or string")),
            };
            match map.get(&key) {
                Some(v) => {
                    // Chained lookup: (get dict :k1 :k2 ...) → nested dict access
                    if args.len() > 2 {
                        let mut cur = v.clone();
                        for extra in &args[2..] {
                            match (&cur, extra) {
                                (Value::Dict(m), Value::Keyword(k)) | (Value::Dict(m), Value::String(k)) => {
                                    cur = m.get(k).cloned().unwrap_or(Value::Nil);
                                }
                                _ => break,
                            }
                        }
                        Ok(cur)
                    } else {
                        Ok(v.clone())
                    }
                }
                None => {
                    if args.len() > 2 { Ok(args[2].clone()) }
                    else if let Some(default) = kw.get("default") { Ok(default.clone()) }
                    else { Ok(Value::Nil) }
                }
            }
        }
        Value::Tensor { data, .. } => {
            if matches!(&args[1], Value::Keyword(k) if k == "...") {
                if args.len() < 3 { return Err(runtime_error("get: ... requires an index argument")); }
                let last_axis = ndarray::Axis(data.ndim() - 1);
                return match &args[2] {
                    v if v.to_f64().is_some() => {
                        let idx = resolve_idx(v.to_f64().unwrap(), data.shape()[data.ndim() - 1]);
                        let sliced = data.index_axis(last_axis, idx).to_owned();
                        Ok(Value::tensor_f32(sliced))
                    }
                    Value::Tensor { data: range_t, .. } if range_t.ndim() == 1 && range_t.len() > 0 => {
                        let start = *range_t.first().unwrap() as usize;
                        let end = *range_t.iter().last().unwrap() as usize + 1;
                        let sliced = data.slice_axis(last_axis, ndarray::Slice::from(start..end));
                        Ok(Value::tensor_f32(sliced.to_owned()))
                    }
                    other => Err(runtime_error(format!("get: ... index must be int or range, got {}", other.type_name()))),
                };
            }
            // Scalar index
            if let Some(f) = args[1].to_f64() {
                let idx = resolve_idx(f, data.shape()[0]);
                let sliced = data.index_axis(ndarray::Axis(0), idx).to_owned();
                return if sliced.shape().is_empty() {
                    Ok(Value::Float(*sliced.first().unwrap()))
                } else {
                    Ok(Value::tensor_f32(sliced))
                };
            }
            // Tensor gather: (get table indices) → gathers rows
            if let Value::Tensor { data: idx_data, .. } = &args[1] {
                let row_shape = &data.shape()[1..];
                let idx_shape = idx_data.shape();
                let mut out_shape = Vec::with_capacity(idx_shape.len() + row_shape.len());
                out_shape.extend_from_slice(idx_shape);
                out_shape.extend_from_slice(row_shape);
                let row_size: usize = row_shape.iter().product::<usize>().max(1);
                let total = idx_data.len() * row_size;
                let mut result = Vec::with_capacity(total);
                for &idx_f in idx_data.iter() {
                    let idx = idx_f as usize;
                    let row = data.index_axis(ndarray::Axis(0), idx);
                    result.extend(row.iter());
                }
                let arr = ArrayD::from_shape_vec(IxDyn(&out_shape), result)
                    .map_err(|e| runtime_error(format!("get: gather reshape: {}", e)))?;
                return Ok(Value::tensor_f32(arr));
            }
            return Err(runtime_error(format!("get: cannot index tensor with {}", args[1])));
        }
        Value::List(items) => {
            let f = args[1].to_f64()
                .ok_or_else(|| runtime_error(format!("get: cannot index list with {}", args[1])))?;
            let idx = resolve_idx(f, items.len());
            items.get(idx).cloned().ok_or_else(|| runtime_error("get: index out of bounds"))
        }
        Value::String(s) => {
            let f = args[1].to_f64()
                .ok_or_else(|| runtime_error(format!("get: cannot index string '{}' with {}", s, args[1])))?;
            let idx = resolve_idx(f, s.chars().count());
            s.chars().nth(idx)
                .map(|c| Value::String(c.to_string()))
                .ok_or_else(|| runtime_error("get: string index out of bounds"))
        }
        Value::DeviceBuffer(_) => {
            // Materialize and retry as host tensor
            let mut host_args = args.to_vec();
            host_args[0] = args[0].ensure_host()?;
            builtin_get(&host_args, kw)
        }
        _ => Err(runtime_error(format!("get: expected dict/tensor/list, got {} (key: {})", args[0].type_name(), args.get(1).map(|v| format!("{}", v)).unwrap_or_default()))),
    }
}

fn broadcast_shape(shapes: &[&[usize]]) -> Result<Vec<usize>, crate::core::error::SheafError> {
    let max_ndim = shapes.iter().map(|s| s.len()).max().unwrap_or(0);
    let mut result = vec![1usize; max_ndim];
    for shape in shapes {
        let offset = max_ndim - shape.len();
        for (i, &dim) in shape.iter().enumerate() {
            let ri = offset + i;
            if result[ri] == 1 {
                result[ri] = dim;
            } else if dim != 1 && dim != result[ri] {
                return Err(runtime_error(format!(
                    "where: cannot broadcast shapes, dimension mismatch {} vs {} at axis {}",
                    result[ri], dim, ri
                )));
            }
        }
    }
    Ok(result)
}

fn builtin_where(args: &[Value], _kw: &BTreeMap<String, Value>) -> R {
    let (cond, _) = to_array(&args[0])?;
    let (on_true, _) = to_array(&args[1])?;
    let (on_false, _) = to_array(&args[2])?;
    let target = broadcast_shape(&[cond.shape(), on_true.shape(), on_false.shape()])?;
    let cond_bc = cond.broadcast(IxDyn(&target)).ok_or_else(|| {
        runtime_error(format!("where: cannot broadcast cond {:?} to {:?}", cond.shape(), target))
    })?.to_owned();
    let true_bc = on_true.broadcast(IxDyn(&target)).ok_or_else(|| {
        runtime_error(format!("where: cannot broadcast on_true {:?} to {:?}", on_true.shape(), target))
    })?.to_owned();
    let false_bc = on_false.broadcast(IxDyn(&target)).ok_or_else(|| {
        runtime_error(format!("where: cannot broadcast on_false {:?} to {:?}", on_false.shape(), target))
    })?.to_owned();
    let result = ndarray::Zip::from(&cond_bc).and(&true_bc).and(&false_bc)
        .map_collect(|&c, &t, &f| if c != 0.0 { t } else { f });
    Ok(Value::tensor_f32(result))
}

fn builtin_roll(args: &[Value], _kw: &BTreeMap<String, Value>) -> R {
    let (arr, _dt) = to_array(&args[0])?;
    let shift = args[1].to_f64().unwrap() as i64;
    let data: Vec<f32> = arr.iter().copied().collect();
    let n = data.len() as i64;
    let shift = ((shift % n) + n) % n;
    let mut result = vec![0.0; data.len()];
    for (i, &v) in data.iter().enumerate() {
        let new_i = ((i as i64 + shift) % n) as usize;
        result[new_i] = v;
    }
    Ok(Value::tensor_f32(ArrayD::from_shape_vec(arr.raw_dim(), result).unwrap()))
}

fn builtin_index_update(args: &[Value], _kw: &BTreeMap<String, Value>) -> R {
    let (mut arr, _dt) = to_array(&args[0])?;
    let idx = args[1].to_f64().unwrap() as usize;
    match &args[2] {
        Value::Tensor { data: new_val, .. } => {
            let mut slice = arr.index_axis_mut(ndarray::Axis(0), idx);
            slice.assign(new_val);
        }
        other => {
            let v = other.to_f64().unwrap() as f32;
            arr[IxDyn(&[idx])] = v;
        }
    }
    Ok(Value::tensor_f32(arr))
}

fn builtin_swapaxes(args: &[Value], _kw: &BTreeMap<String, Value>) -> R {
    let (arr, _dt) = to_array(&args[0])?;
    let ndim = arr.ndim();
    let ax0 = resolve_idx(args[1].to_f64().unwrap(), ndim);
    let ax1 = resolve_idx(args[2].to_f64().unwrap(), ndim);
    let mut axes: Vec<usize> = (0..arr.ndim()).collect();
    axes[ax0] = ax1;
    axes[ax1] = ax0;
    Ok(Value::tensor_f32(arr.permuted_axes(IxDyn(&axes))))
}

fn builtin_tensor_split(args: &[Value], kw: &BTreeMap<String, Value>) -> R {
    let (arr, _dt) = to_array(&args[0])?;
    let num = args[1].to_f64()
        .ok_or_else(|| runtime_error("tensor-split: num-sections must be a number"))? as usize;
    let ax = get_axis(kw).unwrap_or(0);
    let axis = resolve_idx(ax as f64, arr.ndim());
    let dim = arr.shape()[axis];
    if dim % num != 0 {
        return Err(runtime_error(format!(
            "tensor-split: dimension {} ({}) not divisible by {}", axis, dim, num
        )));
    }
    let chunk = dim / num;
    let chunks: Vec<Value> = (0..num).map(|i| {
        let start = i * chunk;
        let end = start + chunk;
        let sliced = arr.slice_axis(ndarray::Axis(axis), ndarray::Slice::from(start..end));
        Value::tensor_f32(sliced.to_owned())
    }).collect();
    Ok(Value::List(chunks))
}

fn builtin_dynamic_slice(args: &[Value], _kw: &BTreeMap<String, Value>) -> R {
    let (arr, _dt) = to_array(&args[0])?;
    let start = args[1].to_f64().unwrap() as usize;
    let end = args[2].to_f64().unwrap() as usize;
    let sliced = arr.slice_axis(ndarray::Axis(0), ndarray::Slice::from(start..=end));
    Ok(Value::tensor_i32(sliced.to_owned()))
}
