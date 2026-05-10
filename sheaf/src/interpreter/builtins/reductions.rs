use super::*;

fn resolve_axis(axis: i64, ndim: usize) -> Result<usize, crate::core::error::SheafError> {
    let ax = if axis < 0 {
        let resolved = ndim as i64 + axis;
        if resolved < 0 {
            return Err(runtime_error(format!("axis {} out of bounds for {}D tensor", axis, ndim)));
        }
        resolved as usize
    } else {
        axis as usize
    };
    if ax >= ndim {
        return Err(runtime_error(format!("axis {} out of bounds for {}D tensor", axis, ndim)));
    }
    Ok(ax)
}

pub(super) fn register(env: &mut Env) {
    env.set_builtin("sum", builtin_sum);
    env.set_builtin("mean", builtin_mean);
    env.set_builtin("product", builtin_product);
    env.set_builtin("min", builtin_min);
    env.set_builtin("max", builtin_max);
    env.set_builtin("minimum", builtin_minimum);
    env.set_builtin("maximum", builtin_maximum);
    env.set_builtin("argmax", builtin_argmax);
    env.set_builtin("argmin", builtin_argmin);
    env.set_builtin("var", builtin_var);
    env.set_builtin("normalize", builtin_normalize);
}

fn keepdims(kw: &BTreeMap<String, Value>) -> bool {
    matches!(kw.get("keepdims"), Some(Value::Bool(true)))
}

fn builtin_sum(args: &[Value], kw: &BTreeMap<String, Value>) -> R {
    let (arr, input_dt) = to_array(&args[0])?;
    let dt = get_dtype_kwarg(kw).unwrap_or(input_dt);
    if let Some(axis) = get_axis(kw) {
        let ax = resolve_axis(axis, arr.ndim())?;
        let result = reduce_along_axis(&arr, ax, |v| v.iter().sum());
        if keepdims(kw) {
            Ok(Value::tensor(result.insert_axis(ndarray::Axis(ax)), dt))
        } else {
            Ok(Value::tensor(result, dt))
        }
    } else {
        Ok(Value::Float(arr.iter().sum::<f32>()))
    }
}

fn builtin_mean(args: &[Value], kw: &BTreeMap<String, Value>) -> R {
    let (arr, input_dt) = to_array(&args[0])?;
    let dt = get_dtype_kwarg(kw).unwrap_or(input_dt);
    if let Some(axis) = get_axis(kw) {
        let ax = resolve_axis(axis, arr.ndim())?;
        let result = reduce_along_axis(&arr, ax, |v| v.iter().sum::<f32>() / v.len() as f32);
        if keepdims(kw) {
            Ok(Value::tensor(result.insert_axis(ndarray::Axis(ax)), dt))
        } else {
            Ok(Value::tensor(result, dt))
        }
    } else {
        let n = arr.len() as f32;
        Ok(Value::Float(arr.iter().sum::<f32>() / n))
    }
}

fn builtin_product(args: &[Value], kw: &BTreeMap<String, Value>) -> R {
    let (arr, input_dt) = to_array(&args[0])?;
    let dt = get_dtype_kwarg(kw).unwrap_or(input_dt);
    if let Some(axis) = get_axis(kw) {
        let ax = resolve_axis(axis, arr.ndim())?;
        let result = reduce_along_axis(&arr, ax, |v| v.iter().product());
        if result.shape().is_empty() {
            Ok(Value::Float(*result.first().unwrap()))
        } else {
            Ok(Value::tensor(result, dt))
        }
    } else {
        Ok(Value::Float(arr.iter().product::<f32>()))
    }
}

fn builtin_min(args: &[Value], kw: &BTreeMap<String, Value>) -> R {
    let (arr, input_dt) = to_array(&args[0])?;
    let dt = get_dtype_kwarg(kw).unwrap_or(input_dt);
    if let Some(axis) = get_axis(kw) {
        let ax = resolve_axis(axis, arr.ndim())?;
        let result = reduce_along_axis(&arr, ax, |v| v.iter().copied().fold(f32::INFINITY, f32::min));
        if keepdims(kw) {
            Ok(Value::tensor(result.insert_axis(ndarray::Axis(ax)), dt))
        } else {
            Ok(Value::tensor(result, dt))
        }
    } else {
        Ok(Value::Float(arr.iter().copied().fold(f32::INFINITY, f32::min)))
    }
}

fn builtin_max(args: &[Value], kw: &BTreeMap<String, Value>) -> R {
    if args.len() > 1 {
        let vals: Result<Vec<f32>, _> = args.iter().map(|a| a.to_f64().map(|v| v as f32).ok_or_else(|| runtime_error("max: expected number"))).collect();
        return Ok(Value::Float(vals?.into_iter().fold(f32::NEG_INFINITY, f32::max)));
    }
    if let Value::List(items) = &args[0] {
        let vals: Result<Vec<f32>, _> = items.iter().map(|a| a.to_f64().map(|v| v as f32).ok_or_else(|| runtime_error("max: list must contain numbers"))).collect();
        return Ok(Value::Float(vals?.into_iter().fold(f32::NEG_INFINITY, f32::max)));
    }
    let (arr, input_dt) = to_array(&args[0])?;
    let dt = get_dtype_kwarg(kw).unwrap_or(input_dt);
    if let Some(axis) = get_axis(kw) {
        let ax = resolve_axis(axis, arr.ndim())?;
        let result = reduce_along_axis(&arr, ax, |v| v.iter().copied().fold(f32::NEG_INFINITY, f32::max));
        if keepdims(kw) {
            Ok(Value::tensor(result.insert_axis(ndarray::Axis(ax)), dt))
        } else {
            Ok(Value::tensor(result, dt))
        }
    } else {
        Ok(Value::Float(arr.iter().copied().fold(f32::NEG_INFINITY, f32::max)))
    }
}

fn builtin_minimum(args: &[Value], _kw: &BTreeMap<String, Value>) -> R {
    binary_op(args, f32::min)
}

fn builtin_maximum(args: &[Value], _kw: &BTreeMap<String, Value>) -> R {
    binary_op(args, f32::max)
}

fn builtin_argmax(args: &[Value], kw: &BTreeMap<String, Value>) -> R {
    let (arr, _dt) = to_array(&args[0])?;
    if let Some(axis) = get_axis(kw) {
        let ax = resolve_axis(axis, arr.ndim())?;
        let result = argreduce_along_axis(&arr, ax, |a, b| a > b);
        Ok(Value::tensor_i32(result))
    } else {
        let data: Vec<f32> = arr.iter().copied().collect();
        let idx = data.iter().enumerate().fold(0, |best, (i, &x)| if x > data[best] { i } else { best });
        Ok(Value::Int(idx as i64))
    }
}

fn builtin_argmin(args: &[Value], kw: &BTreeMap<String, Value>) -> R {
    let (arr, _dt) = to_array(&args[0])?;
    if let Some(axis) = get_axis(kw) {
        let ax = resolve_axis(axis, arr.ndim())?;
        let result = argreduce_along_axis(&arr, ax, |a, b| a < b);
        Ok(Value::tensor_i32(result))
    } else {
        let data: Vec<f32> = arr.iter().copied().collect();
        let idx = data.iter().enumerate().fold(0, |best, (i, &x)| if x < data[best] { i } else { best });
        Ok(Value::Int(idx as i64))
    }
}

fn builtin_var(args: &[Value], kw: &BTreeMap<String, Value>) -> R {
    let (arr, input_dt) = to_array(&args[0])?;
    let dt = get_dtype_kwarg(kw).unwrap_or(input_dt);
    if let Some(axis) = get_axis(kw) {
        let ax = resolve_axis(axis, arr.ndim())?;
        let mean_arr = reduce_along_axis(&arr, ax, |v| v.iter().sum::<f32>() / v.len() as f32);
        let mean_bc = mean_arr.insert_axis(ndarray::Axis(ax));
        let diff = &arr - &mean_bc;
        let sq = &diff * &diff;
        let result = reduce_along_axis(&sq, ax, |v| v.iter().sum::<f32>() / v.len() as f32);
        if keepdims(kw) {
            Ok(Value::tensor(result.insert_axis(ndarray::Axis(ax)), dt))
        } else {
            Ok(Value::tensor(result, dt))
        }
    } else {
        let n = arr.len() as f32;
        let mean = arr.iter().sum::<f32>() / n;
        let var = arr.iter().map(|&x| (x - mean) * (x - mean)).sum::<f32>() / n;
        Ok(Value::Float(var))
    }
}

fn builtin_normalize(args: &[Value], kw: &BTreeMap<String, Value>) -> R {
    let (arr, input_dt) = to_array(&args[0])?;
    let dt = get_dtype_kwarg(kw).unwrap_or(input_dt);
    if let Some(axis) = get_axis(kw) {
        let ax = resolve_axis(axis, arr.ndim())?;
        let sum_arr = reduce_along_axis(&arr, ax, |v| v.iter().sum());
        let sum_bc = sum_arr.insert_axis(ndarray::Axis(ax));
        Ok(Value::tensor(&arr / &sum_bc, dt))
    } else {
        let total: f32 = arr.iter().sum();
        Ok(Value::tensor(arr.mapv(|x| x / total), dt))
    }
}
