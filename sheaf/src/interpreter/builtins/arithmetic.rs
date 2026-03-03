use super::*;

pub(super) fn register(env: &mut Env) {
    env.set_builtin("+", builtin_add);
    env.set_builtin("-", builtin_sub);
    env.set_builtin("*", builtin_mul);
    env.set_builtin("/", builtin_div);
    env.set_builtin("//", builtin_floor_div);
    env.set_builtin("mod", builtin_mod);
    env.set_builtin("%", builtin_mod);
    env.set_builtin("**", builtin_pow);
    env.set_builtin("abs", builtin_abs);
    env.set_builtin("ash", builtin_ash);
    env.set_builtin("exp", builtin_exp);
    env.set_builtin("log", builtin_log);
    env.set_builtin("sqrt", builtin_sqrt);
    env.set_builtin("@", builtin_matmul);
    env.set_builtin("einsum", builtin_einsum);
    env.set_builtin("append-and-roll", builtin_append_and_roll);
}

fn builtin_add(args: &[Value], _kw: &BTreeMap<String, Value>) -> R {
    if args.len() == 1 {
        return Ok(args[0].clone());
    }
    binary_op(args, |a, b| a + b)
}

fn builtin_sub(args: &[Value], _kw: &BTreeMap<String, Value>) -> R {
    if args.len() == 1 {
        let (arr, dt) = to_array(&args[0])?;
        let result = arr.mapv(|x| -x);
        if result.ndim() == 0 {
            let x = *result.first().unwrap();
            if dt == Dtype::I32 { return Ok(Value::Int(x as i64)); }
            return Ok(Value::Float(x));
        }
        return Ok(Value::Tensor { data: result, dtype: dt });
    }
    binary_op(args, |a, b| a - b)
}

fn builtin_mul(args: &[Value], _kw: &BTreeMap<String, Value>) -> R {
    binary_op(args, |a, b| a * b)
}

fn builtin_div(args: &[Value], _kw: &BTreeMap<String, Value>) -> R {
    let result = binary_op(args, |a, b| a / b)?;
    match result {
        Value::Int(n) => Ok(Value::Float(n as f64)),
        Value::Tensor { data, .. } => Ok(Value::Tensor { data, dtype: Dtype::F32 }),
        other => Ok(other),
    }
}

fn builtin_floor_div(args: &[Value], _kw: &BTreeMap<String, Value>) -> R {
    binary_op(args, |a, b| (a / b).floor())
}

fn builtin_mod(args: &[Value], _kw: &BTreeMap<String, Value>) -> R {
    binary_op(args, |a, b| ((a % b) + b) % b)
}

fn builtin_pow(args: &[Value], _kw: &BTreeMap<String, Value>) -> R {
    binary_op(args, |a, b| a.powf(b))
}

fn builtin_abs(args: &[Value], _kw: &BTreeMap<String, Value>) -> R {
    unary_op(args, f64::abs)
}

fn builtin_ash(args: &[Value], _kw: &BTreeMap<String, Value>) -> R {
    if args.len() != 2 {
        return Err(runtime_error("ash requires exactly 2 arguments: (ash value shift)"));
    }
    let shift = match &args[1] {
        Value::Int(n) => *n,
        Value::Float(f) => *f as i64,
        _ => return Err(runtime_error(format!("ash: shift amount must be a number, got {}", args[1].type_name()))),
    };
    match &args[0] {
        Value::Int(n) => {
            let result = if shift >= 0 { n << shift } else { n >> (-shift) };
            Ok(Value::Int(result))
        }
        Value::Float(f) => {
            let n = *f as i64;
            let result = if shift >= 0 { n << shift } else { n >> (-shift) };
            Ok(Value::Int(result))
        }
        Value::Tensor { data, .. } => {
            let result = data.mapv(|x| {
                let n = x as i64;
                if shift >= 0 { (n << shift) as f64 } else { (n >> (-shift)) as f64 }
            });
            Ok(Value::Tensor { data: result, dtype: Dtype::I32 })
        }
        _ => Err(runtime_error(format!("ash: expected number or tensor, got {}", args[0].type_name()))),
    }
}

fn builtin_exp(args: &[Value], _kw: &BTreeMap<String, Value>) -> R {
    unary_op(args, f64::exp)
}

fn builtin_log(args: &[Value], _kw: &BTreeMap<String, Value>) -> R {
    unary_op_f32(args, f32::ln)
}

fn builtin_sqrt(args: &[Value], _kw: &BTreeMap<String, Value>) -> R {
    unary_op(args, f64::sqrt)
}

fn builtin_matmul(args: &[Value], _kw: &BTreeMap<String, Value>) -> R {
    if args.len() != 2 {
        return Err(runtime_error("@ requires exactly 2 arguments"));
    }
    let (a, _) = to_array(&args[0])?;
    let (b, _) = to_array(&args[1])?;

    match (a.ndim(), b.ndim()) {
        (1, 1) => {
            let a1 = a.into_dimensionality::<ndarray::Ix1>().map_err(|e| runtime_error(e.to_string()))?;
            let b1 = b.into_dimensionality::<ndarray::Ix1>().map_err(|e| runtime_error(e.to_string()))?;
            Ok(Value::Float(a1.dot(&b1)))
        }
        (2, 2) => {
            let a2 = a.into_dimensionality::<ndarray::Ix2>().map_err(|e| runtime_error(e.to_string()))?;
            let b2 = b.into_dimensionality::<ndarray::Ix2>().map_err(|e| runtime_error(e.to_string()))?;
            Ok(Value::tensor_f32(a2.dot(&b2).into_dyn()))
        }
        (2, 1) => {
            let a2 = a.into_dimensionality::<ndarray::Ix2>().map_err(|e| runtime_error(e.to_string()))?;
            let b1 = b.into_dimensionality::<ndarray::Ix1>().map_err(|e| runtime_error(e.to_string()))?;
            Ok(Value::tensor_f32(a2.dot(&b1).into_dyn()))
        }
        (1, 2) => {
            let a1 = a.into_dimensionality::<ndarray::Ix1>().map_err(|e| runtime_error(e.to_string()))?;
            let b2 = b.into_dimensionality::<ndarray::Ix2>().map_err(|e| runtime_error(e.to_string()))?;
            Ok(Value::tensor_f32(a1.dot(&b2).into_dyn()))
        }
        _ => Err(runtime_error(format!(
            "@ not supported for {}D x {}D", a.ndim(), b.ndim()
        ))),
    }
}

fn expand_einsum_ellipsis(subscript: &str, shape_a: &[usize], shape_b: &[usize]) -> String {
    if !subscript.contains("...") {
        return subscript.to_string();
    }
    let arrow = match subscript.find("->") {
        Some(i) => i,
        None => return subscript.replace("...", ""),
    };
    let lhs = &subscript[..arrow];
    let parts: Vec<&str> = lhs.split(',').collect();
    let explicit_a = parts[0].replace("...", "").len();
    let explicit_b = if parts.len() > 1 { parts[1].replace("...", "").len() } else { 0 };
    let batch_a = shape_a.len().saturating_sub(explicit_a);
    let batch_b = shape_b.len().saturating_sub(explicit_b);
    let n_batch = batch_a.max(batch_b);
    let batch_labels: String = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        .chars()
        .filter(|c| !subscript.contains(*c))
        .take(n_batch)
        .collect();
    subscript.replace("...", &batch_labels)
}

fn builtin_einsum(args: &[Value], _kw: &BTreeMap<String, Value>) -> R {
    if args.len() != 3 {
        return Err(runtime_error("einsum requires exactly 3 arguments: subscript, a, b"));
    }
    let subscript = match &args[0] {
        Value::String(s) => s.as_str(),
        _ => return Err(runtime_error("einsum: first argument must be a subscript string")),
    };
    let (a, _) = to_array(&args[1])?;
    let (b, _) = to_array(&args[2])?;
    let subscript = subscript.replace(' ', "");
    let subscript = expand_einsum_ellipsis(&subscript, a.shape(), b.shape());

    let arrow = subscript.find("->")
        .ok_or_else(|| runtime_error("einsum: subscript must contain '->'"))?;
    let lhs = &subscript[..arrow];
    let rhs = &subscript[arrow + 2..];
    let parts: Vec<&str> = lhs.split(',').collect();
    if parts.len() != 2 {
        return Err(runtime_error("einsum: only two-operand einsum is supported"));
    }
    let idx_a: Vec<char> = parts[0].chars().collect();
    let idx_b: Vec<char> = parts[1].chars().collect();
    let idx_out: Vec<char> = rhs.chars().collect();

    let mut sizes: std::collections::HashMap<char, usize> = std::collections::HashMap::new();
    for (&label, &dim) in idx_a.iter().zip(a.shape().iter()) {
        sizes.insert(label, dim);
    }
    for (&label, &dim) in idx_b.iter().zip(b.shape().iter()) {
        sizes.insert(label, dim);
    }

    let out_shape: Vec<usize> = idx_out.iter()
        .map(|c| *sizes.get(c).unwrap_or(&1))
        .collect();
    let out_len: usize = out_shape.iter().product::<usize>().max(1);
    let mut result = vec![0.0f64; out_len];

    let mut all_labels: Vec<char> = idx_out.clone();
    for &c in idx_a.iter().chain(idx_b.iter()) {
        if !all_labels.contains(&c) {
            all_labels.push(c);
        }
    }

    let label_sizes: Vec<usize> = all_labels.iter()
        .map(|c| *sizes.get(c).unwrap_or(&1))
        .collect();

    let label_pos: std::collections::HashMap<char, usize> = all_labels.iter()
        .enumerate().map(|(i, &c)| (c, i)).collect();

    let out_strides: Vec<usize> = (0..out_shape.len()).map(|i| {
        out_shape[i + 1..].iter().product::<usize>().max(1)
    }).collect();

    let total: usize = label_sizes.iter().product::<usize>().max(1);
    let mut coords = vec![0usize; all_labels.len()];

    for _ in 0..total {
        let a_idx: Vec<usize> = idx_a.iter().map(|c| coords[label_pos[c]]).collect();
        let b_idx: Vec<usize> = idx_b.iter().map(|c| coords[label_pos[c]]).collect();
        let flat_out: usize = idx_out.iter().enumerate()
            .map(|(i, c)| coords[label_pos[c]] * out_strides[i])
            .sum();
        result[flat_out] += a[IxDyn(&a_idx)] * b[IxDyn(&b_idx)];

        for k in (0..coords.len()).rev() {
            coords[k] += 1;
            if coords[k] < label_sizes[k] { break; }
            coords[k] = 0;
        }
    }

    let result_f32: Vec<f64> = result.iter().map(|&x| (x as f32) as f64).collect();

    if out_shape.is_empty() {
        Ok(Value::Float(result_f32[0]))
    } else {
        let arr = ArrayD::from_shape_vec(IxDyn(&out_shape), result_f32)
            .map_err(|e| runtime_error(e.to_string()))?;
        Ok(Value::tensor_f32(arr))
    }
}

fn builtin_append_and_roll(args: &[Value], _kw: &BTreeMap<String, Value>) -> R {
    if args.len() != 2 {
        return Err(runtime_error("append-and-roll requires 2 arguments: tensor, new-element"));
    }
    let (arr, _) = to_array(&args[0])?;
    if arr.ndim() != 1 {
        return Err(runtime_error("append-and-roll: first argument must be a 1D tensor"));
    }
    let new_val = args[1].to_f64()
        .ok_or_else(|| runtime_error("append-and-roll: second argument must be a number"))?;
    let n = arr.shape()[0];
    let mut data: Vec<f64> = arr.iter().skip(1).copied().collect();
    data.push(new_val);
    let result = ArrayD::from_shape_vec(IxDyn(&[n]), data).unwrap();
    Ok(Value::tensor_f32(result))
}
