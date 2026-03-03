use super::*;

pub(super) fn register(env: &mut Env) {
    env.set_builtin("zeros", builtin_zeros);
    env.set_builtin("ones", builtin_ones);
    env.set_builtin("arange", builtin_arange);
    env.set_builtin("eye", builtin_eye);
    env.set_builtin("one-hot", builtin_one_hot);
    env.set_builtin("tril", builtin_tril);
    env.set_builtin("tensor", builtin_tensor);
    env.set_builtin("range", builtin_range);
}

fn builtin_zeros(args: &[Value], _kw: &BTreeMap<String, Value>) -> R {
    let shape = shape_from_value(&args[0])?;
    Ok(Value::tensor_f32(ArrayD::zeros(IxDyn(&shape))))
}

fn builtin_ones(args: &[Value], _kw: &BTreeMap<String, Value>) -> R {
    let shape = shape_from_value(&args[0])?;
    Ok(Value::tensor_f32(ArrayD::ones(IxDyn(&shape))))
}

fn builtin_arange(args: &[Value], _kw: &BTreeMap<String, Value>) -> R {
    let (start, stop, step) = match args.len() {
        1 => (0i64, args[0].to_f64().unwrap() as i64, 1i64),
        2 => (args[0].to_f64().unwrap() as i64, args[1].to_f64().unwrap() as i64, 1),
        _ => (args[0].to_f64().unwrap() as i64, args[1].to_f64().unwrap() as i64, args[2].to_f64().unwrap() as i64),
    };
    let mut data = Vec::new();
    let mut i = start;
    while (step > 0 && i < stop) || (step < 0 && i > stop) {
        data.push(i as f64);
        i += step;
    }
    Ok(Value::tensor_i32(ArrayD::from_shape_vec(IxDyn(&[data.len()]), data).unwrap()))
}

fn builtin_eye(args: &[Value], _kw: &BTreeMap<String, Value>) -> R {
    let n = args[0].to_f64().unwrap() as usize;
    let m = if args.len() > 1 { args[1].to_f64().unwrap() as usize } else { n };
    let mut data = vec![0.0; n * m];
    for i in 0..n.min(m) {
        data[i * m + i] = 1.0;
    }
    Ok(Value::tensor_f32(ArrayD::from_shape_vec(IxDyn(&[n, m]), data).unwrap()))
}

fn builtin_one_hot(args: &[Value], _kw: &BTreeMap<String, Value>) -> R {
    let num_classes = args[1].to_f64().unwrap() as usize;
    match &args[0] {
        Value::Int(idx) => {
            let mut data = vec![0.0; num_classes];
            data[*idx as usize] = 1.0;
            Ok(Value::tensor_f32(ArrayD::from_shape_vec(IxDyn(&[num_classes]), data).unwrap()))
        }
        Value::Tensor { data: indices, .. } => {
            let n = indices.len();
            let mut result = vec![0.0; n * num_classes];
            for (i, &idx) in indices.iter().enumerate() {
                result[i * num_classes + idx as usize] = 1.0;
            }
            Ok(Value::tensor_f32(ArrayD::from_shape_vec(IxDyn(&[n, num_classes]), result).unwrap()))
        }
        _ => Err(runtime_error("one-hot: expected int or tensor")),
    }
}

fn builtin_tril(args: &[Value], _kw: &BTreeMap<String, Value>) -> R {
    let (arr, _dt) = to_array(&args[0])?;
    if arr.ndim() != 2 { return Err(runtime_error("tril: expected 2D tensor")); }
    let shape = arr.shape();
    let (n, m) = (shape[0], shape[1]);
    let mut result = arr.clone();
    for i in 0..n {
        for j in (i + 1)..m {
            result[IxDyn(&[i, j])] = 0.0;
        }
    }
    Ok(Value::tensor_f32(result))
}

fn builtin_tensor(args: &[Value], _kw: &BTreeMap<String, Value>) -> R {
    match &args[0] {
        Value::List(items) => {
            let all_numeric = items.iter().all(|v| matches!(v, Value::Int(_) | Value::Float(_)));
            if all_numeric && !items.is_empty() {
                let data: Vec<f64> = items.iter().map(|v| v.to_f64().unwrap()).collect();
                let arr = ArrayD::from_shape_vec(IxDyn(&[data.len()]), data).unwrap();
                Ok(Value::tensor_f32(arr))
            } else {
                Err(runtime_error("tensor: expected list of numbers"))
            }
        }
        Value::Tensor { .. } => Ok(args[0].clone()),
        _ => Err(runtime_error("tensor: expected list or tensor")),
    }
}

fn builtin_range(args: &[Value], kw: &BTreeMap<String, Value>) -> R {
    builtin_arange(args, kw)
}
