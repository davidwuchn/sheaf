use super::*;

pub(super) fn register(env: &mut Env) {
    env.set_builtin("relu", builtin_relu);
    env.set_builtin("leaky-relu", builtin_leaky_relu);
    env.set_builtin("sigmoid", builtin_sigmoid);
    env.set_builtin("tanh", builtin_tanh);
    env.set_builtin("gelu", builtin_gelu);
    env.set_builtin("selu", builtin_selu);
    env.set_builtin("celu", builtin_celu);
    env.set_builtin("silu", builtin_silu);
    env.set_builtin("softmax", builtin_softmax);
    env.set_builtin("log-softmax", builtin_log_softmax);
}

fn builtin_relu(args: &[Value], _kw: &BTreeMap<String, Value>) -> R {
    unary_op(args, |x| if x > 0.0 { x } else { 0.0 })
}

fn builtin_leaky_relu(args: &[Value], kw: &BTreeMap<String, Value>) -> R {
    let slope = kw.get("negative_slope")
        .and_then(|v| v.to_f64())
        .unwrap_or(0.01) as f32;
    let (arr, _dt) = to_array(&args[0])?;
    let result = arr.mapv(|x| {
        let xf = x as f32;
        (if xf > 0.0 { xf } else { slope * xf }) as f64
    });
    if result.ndim() == 0 {
        Ok(Value::Float(*result.first().unwrap()))
    } else {
        Ok(Value::Tensor { data: result, dtype: Dtype::F32 })
    }
}

fn builtin_sigmoid(args: &[Value], _kw: &BTreeMap<String, Value>) -> R {
    unary_op(args, |x| {
        let r = 1.0 / (1.0 + (-x).exp());
        if r < 1e-7 { 0.0 } else if r > 1.0 - 1e-7 { 1.0 } else { r }
    })
}

fn builtin_tanh(args: &[Value], _kw: &BTreeMap<String, Value>) -> R {
    unary_op(args, f64::tanh)
}

fn builtin_gelu(args: &[Value], _kw: &BTreeMap<String, Value>) -> R {
    unary_op(args, |x| {
        0.5 * x * (1.0 + (std::f64::consts::FRAC_2_SQRT_PI * std::f64::consts::FRAC_1_SQRT_2 * (x + 0.044715 * x * x * x)).tanh())
    })
}

fn builtin_selu(args: &[Value], _kw: &BTreeMap<String, Value>) -> R {
    let alpha = 1.6732632423543772_f32;
    let scale = 1.0507009873554805_f32;
    let (arr, _dt) = to_array(&args[0])?;
    let result = arr.mapv(|x| {
        let xf = x as f32;
        (if xf > 0.0 { scale * xf } else { scale * alpha * (xf.exp() - 1.0) }) as f64
    });
    if result.ndim() == 0 {
        Ok(Value::Float(*result.first().unwrap()))
    } else {
        Ok(Value::Tensor { data: result, dtype: Dtype::F32 })
    }
}

fn builtin_celu(args: &[Value], kw: &BTreeMap<String, Value>) -> R {
    let alpha = kw.get("alpha").and_then(|v| v.to_f64()).unwrap_or(1.0) as f32;
    let (arr, _dt) = to_array(&args[0])?;
    let result = arr.mapv(|x| {
        let xf = x as f32;
        (if xf > 0.0 { xf } else { alpha * ((xf / alpha).exp() - 1.0) }) as f64
    });
    if result.ndim() == 0 {
        Ok(Value::Float(*result.first().unwrap()))
    } else {
        Ok(Value::Tensor { data: result, dtype: Dtype::F32 })
    }
}

fn builtin_silu(args: &[Value], _kw: &BTreeMap<String, Value>) -> R {
    unary_op(args, |x| x / (1.0 + (-x).exp()))
}

fn builtin_softmax(args: &[Value], kw: &BTreeMap<String, Value>) -> R {
    let (arr, _dt) = to_array(&args[0])?;
    let axis = get_axis(kw).unwrap_or(-1);
    let ndim = arr.ndim();
    let ax = if axis < 0 { (ndim as i64 + axis) as usize } else { axis as usize };
    let arr_f32 = arr.mapv(|x| x as f32);
    let max_arr = reduce_along_axis(&arr_f32.mapv(|x| x as f64), ax, |v| v.iter().copied().fold(f64::NEG_INFINITY, f64::max));
    let max_bc = max_arr.insert_axis(ndarray::Axis(ax));
    let shifted = (&arr - &max_bc).mapv(|x| (x as f32) as f64);
    let exp_arr = shifted.mapv(|x| (x as f32).exp() as f64);
    let sum_arr = reduce_along_axis(&exp_arr, ax, |v| v.iter().sum::<f64>());
    let sum_bc = sum_arr.insert_axis(ndarray::Axis(ax));
    let result = (&exp_arr / &sum_bc).mapv(|x| (x as f32) as f64);
    Ok(Value::tensor_f32(result))
}

fn builtin_log_softmax(args: &[Value], kw: &BTreeMap<String, Value>) -> R {
    let (arr, _dt) = to_array(&args[0])?;
    let axis = get_axis(kw).unwrap_or(-1);
    let ndim = arr.ndim();
    let ax = if axis < 0 { (ndim as i64 + axis) as usize } else { axis as usize };
    let max_arr = reduce_along_axis(&arr, ax, |v| v.iter().copied().fold(f64::NEG_INFINITY, f64::max));
    let max_bc = max_arr.insert_axis(ndarray::Axis(ax));
    let shifted = (&arr - &max_bc).mapv(|x| (x as f32) as f64);
    let exp_arr = shifted.mapv(|x| (x as f32).exp() as f64);
    let sum_arr = reduce_along_axis(&exp_arr, ax, |v| v.iter().sum::<f64>());
    let log_sum = sum_arr.mapv(|x| (x as f32).ln() as f64);
    let log_sum_bc = log_sum.insert_axis(ndarray::Axis(ax));
    let result = (&shifted - &log_sum_bc).mapv(|x| (x as f32) as f64);
    Ok(Value::tensor_f32(result))
}
