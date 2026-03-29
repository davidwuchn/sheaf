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
    env.set_builtin("cast", builtin_cast);
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
        data.push(i as f32);
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
    if args.len() != 2 {
        return Err(runtime_error(format!(
            "one-hot expects 2 arguments (one-hot indices num_classes), got {}",
            args.len()
        )));
    }
    let num_classes = args[1].to_f64().ok_or_else(|| {
        runtime_error(format!(
            "one-hot: num_classes must be an integer, got {}. Example: (one-hot [0 1 2] 10)",
            args[1].type_name()
        ))
    })? as usize;
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
            // Preserve input shape: [...indices_shape, num_classes]
            let mut out_shape: Vec<usize> = indices.shape().to_vec();
            out_shape.push(num_classes);
            Ok(Value::tensor_f32(ArrayD::from_shape_vec(IxDyn(&out_shape), result).unwrap()))
        }
        other => Err(runtime_error(format!("one-hot: expected an integer or a tensor, got {}", other.type_name()))),
    }
}

fn builtin_tril(args: &[Value], _kw: &BTreeMap<String, Value>) -> R {
    let (arr, _dt) = to_array(&args[0])?;
    if arr.ndim() != 2 { return Err(runtime_error(format!("tril: input must be 2D (matrix), got {}D", arr.ndim()))); }
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
                let data: Vec<f32> = items.iter().map(|v| v.to_f64().unwrap() as f32).collect();
                let arr = ArrayD::from_shape_vec(IxDyn(&[data.len()]), data).unwrap();
                Ok(Value::tensor_f32(arr))
            } else {
                Err(runtime_error("tensor: expected a list of numbers, got a list with non-numeric elements"))
            }
        }
        Value::Tensor { .. } => Ok(args[0].clone()),
        other => Err(runtime_error(format!("tensor: expected a list or a tensor, got {}", other.type_name()))),
    }
}

fn builtin_range(args: &[Value], kw: &BTreeMap<String, Value>) -> R {
    builtin_arange(args, kw)
}

/// (cast expr :bf16) or (cast expr :f32) -- convert tensor dtype
fn builtin_cast(args: &[Value], _kw: &BTreeMap<String, Value>) -> R {
    use crate::interpreter::value::Dtype;
    if args.len() != 2 {
        return Err(runtime_error(format!(
            "cast expects 2 arguments (cast tensor :dtype), got {}",
            args.len()
        )));
    }
    let target = match &args[1] {
        Value::Keyword(k) => Dtype::from_keyword(k).ok_or_else(|| {
            runtime_error(format!(
                "cast: unknown dtype :{k}. Valid dtypes: :f32, :f16, :bf16, :i32"
            ))
        })?,
        other => return Err(runtime_error(format!(
            "cast expects a dtype keyword as 2nd argument, got {}. Example: (cast x :f32)",
            other.type_name()
        ))),
    };
    match &args[0] {
        Value::Tensor { data, dtype } => {
            if *dtype == target {
                return Ok(args[0].clone());
            }
            Ok(Value::Tensor { data: data.clone(), dtype: target })
        }
        Value::DeviceBuffer(db) => {
            let host_data = db.to_host()?;
            Ok(Value::Tensor { data: std::sync::Arc::new(host_data), dtype: target })
        }
        Value::Float(f) => {
            Ok(Value::Tensor {
                data: std::sync::Arc::new(ArrayD::from_elem(IxDyn(&[]), *f)),
                dtype: target,
            })
        }
        Value::Int(n) => {
            Ok(Value::Tensor {
                data: std::sync::Arc::new(ArrayD::from_elem(IxDyn(&[]), *n as f32)),
                dtype: target,
            })
        }
        other => Err(runtime_error(format!(
            "cast expects a tensor, float, or int as 1st argument, got {}",
            other.type_name()
        ))),
    }
}
