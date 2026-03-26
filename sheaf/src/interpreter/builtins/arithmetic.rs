use super::*;
use std::sync::Arc;

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
    env.set_builtin("round", builtin_round);
    env.set_builtin("ceil", builtin_ceil);
    env.set_builtin("floor", builtin_floor);
    env.set_builtin("@", builtin_matmul);
    env.set_builtin("@-grad-lhs", builtin_matmul_grad_lhs);
    env.set_builtin("@-grad-rhs", builtin_matmul_grad_rhs);
    env.set_builtin("einsum", builtin_einsum);
    env.set_builtin("append-and-roll", builtin_append_and_roll);
}

fn builtin_add(args: &[Value], kw: &BTreeMap<String, Value>) -> R {
    if args.len() == 1 {
        return Ok(args[0].clone());
    }
    with_dtype_kwarg(binary_op(args, |a, b| a + b), kw)
}

fn builtin_sub(args: &[Value], kw: &BTreeMap<String, Value>) -> R {
    if args.len() == 1 {
        let (arr, dt) = to_array(&args[0])?;
        let result = arr.mapv(|x| -x);
        if result.ndim() == 0 {
            let x = *result.first().unwrap();
            if dt == Dtype::I32 { return Ok(Value::Int(x as i64)); }
            return Ok(Value::Float(x));
        }
        return Ok(Value::Tensor { data: Arc::new(result), dtype: dt });
    }
    with_dtype_kwarg(binary_op(args, |a, b| a - b), kw)
}

fn builtin_mul(args: &[Value], kw: &BTreeMap<String, Value>) -> R {
    with_dtype_kwarg(binary_op(args, |a, b| a * b), kw)
}

fn builtin_div(args: &[Value], kw: &BTreeMap<String, Value>) -> R {
    let result = binary_op(args, |a, b| a / b)?;
    let result = match result {
        Value::Int(n) => Ok(Value::Float(n as f32)),
        Value::Tensor { data, .. } => Ok(Value::Tensor { data, dtype: Dtype::F32 }),
        other => Ok(other),
    };
    with_dtype_kwarg(result, kw)
}

fn builtin_floor_div(args: &[Value], kw: &BTreeMap<String, Value>) -> R {
    with_dtype_kwarg(binary_op(args, |a, b| (a / b).floor()), kw)
}

fn builtin_mod(args: &[Value], kw: &BTreeMap<String, Value>) -> R {
    with_dtype_kwarg(binary_op(args, |a, b| ((a % b) + b) % b), kw)
}

fn builtin_pow(args: &[Value], _kw: &BTreeMap<String, Value>) -> R {
    // Strength reduction: detect small integer exponents and use multiplication
    if args.len() == 2 {
        let exp_int = match &args[1] {
            Value::Int(n) => Some(*n),
            Value::Float(f) if *f == f.floor() && *f >= 0.0 && *f <= 8.0 => Some(*f as i64),
            _ => None,
        };
        if let Some(n) = exp_int {
            match n {
                0 => return unary_op(&args[..1], |_| 1.0),
                1 => return Ok(args[0].clone()),
                2 => return binary_op(&[args[0].clone(), args[0].clone()], |a, b| a * b),
                3 => return unary_op(&args[..1], |a| a * a * a),
                _ => {
                    let exp = n as f32;
                    return binary_op(&[args[0].clone(), Value::Float(exp)], |a, b| a.powf(b));
                }
            }
        }
    }
    binary_op(args, |a, b| a.powf(b))
}

fn builtin_abs(args: &[Value], _kw: &BTreeMap<String, Value>) -> R {
    unary_op(args, f32::abs)
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
                if shift >= 0 { (n << shift) as f32 } else { (n >> (-shift)) as f32 }
            });
            Ok(Value::Tensor { data: Arc::new(result), dtype: Dtype::I32 })
        }
        _ => Err(runtime_error(format!("ash: expected number or tensor, got {}", args[0].type_name()))),
    }
}

fn builtin_exp(args: &[Value], _kw: &BTreeMap<String, Value>) -> R {
    unary_op(args, f32::exp)
}

fn builtin_log(args: &[Value], _kw: &BTreeMap<String, Value>) -> R {
    unary_op_f32(args, f32::ln)
}

fn builtin_sqrt(args: &[Value], _kw: &BTreeMap<String, Value>) -> R {
    unary_op(args, f32::sqrt)
}

fn builtin_round(args: &[Value], _kw: &BTreeMap<String, Value>) -> R {
    unary_op(args, f32::round)
}

fn builtin_ceil(args: &[Value], _kw: &BTreeMap<String, Value>) -> R {
    unary_op(args, f32::ceil)
}

fn builtin_floor(args: &[Value], _kw: &BTreeMap<String, Value>) -> R {
    unary_op(args, f32::floor)
}

fn builtin_matmul(args: &[Value], _kw: &BTreeMap<String, Value>) -> R {
    if args.len() != 2 {
        return Err(arity_error("@", 2, args.len()));
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
        _ if a.ndim() >= 2 && b.ndim() >= 2 => {
            // Batched matmul: [...batch, M, K] @ [...batch, K, N] -> [...batch, M, N]
            // Also handles nD @ 2D and 2D @ nD by broadcasting the 2D operand
            let a_shape = a.shape();
            let b_shape = b.shape();
            let m = a_shape[a.ndim() - 2];
            let k = a_shape[a.ndim() - 1];
            let n = b_shape[b.ndim() - 1];
            let a_batch = &a_shape[..a.ndim() - 2];
            let b_batch = &b_shape[..b.ndim() - 2];
            let batch: Vec<usize> = if a_batch.len() >= b_batch.len() {
                a_batch.to_vec()
            } else {
                b_batch.to_vec()
            };
            let batch_size: usize = batch.iter().product::<usize>().max(1);

            let a_2d = if a.ndim() == 2 {
                Some(a.view().into_dimensionality::<ndarray::Ix2>()
                    .map_err(|e| runtime_error(e.to_string()))?)
            } else { None };
            let b_2d = if b.ndim() == 2 {
                Some(b.view().into_dimensionality::<ndarray::Ix2>()
                    .map_err(|e| runtime_error(e.to_string()))?)
            } else { None };

            // Ensure contiguous layout (swapaxes/permute may produce non-contiguous views)
            let a_c = if a.ndim() > 2 { a.as_standard_layout().into_owned() } else { a.clone() };
            let b_c = if b.ndim() > 2 { b.as_standard_layout().into_owned() } else { b.clone() };
            let a_flat = if a_c.ndim() > 2 {
                Some(a_c.into_shape_with_order((batch_size, m, k))
                    .map_err(|e| runtime_error(format!("@: reshape a: {}", e)))?)
            } else { None };
            let b_flat = if b_c.ndim() > 2 {
                Some(b_c.into_shape_with_order((batch_size, k, n))
                    .map_err(|e| runtime_error(format!("@: reshape b: {}", e)))?)
            } else { None };

            let mut result = Vec::with_capacity(batch_size * m * n);
            for i in 0..batch_size {
                let ai = match (&a_2d, &a_flat) {
                    (Some(a2), _) => a2.view(),
                    (_, Some(af)) => af.index_axis(ndarray::Axis(0), i)
                        .into_dimensionality::<ndarray::Ix2>()
                        .map_err(|e| runtime_error(e.to_string()))?,
                    _ => unreachable!(),
                };
                let bi = match (&b_2d, &b_flat) {
                    (Some(b2), _) => b2.view(),
                    (_, Some(bf)) => bf.index_axis(ndarray::Axis(0), i)
                        .into_dimensionality::<ndarray::Ix2>()
                        .map_err(|e| runtime_error(e.to_string()))?,
                    _ => unreachable!(),
                };
                result.extend(ai.dot(&bi).iter());
            }

            let mut out_shape = batch;
            out_shape.push(m);
            out_shape.push(n);
            let arr = ArrayD::from_shape_vec(IxDyn(&out_shape), result)
                .map_err(|e| runtime_error(format!("@: output reshape: {}", e)))?;
            Ok(Value::tensor_f32(arr))
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

/// Try to decompose an einsum into permute -> reshape 3D -> BLAS matmul -> reshape -> permute.
/// Every 2-operand einsum where each label falls into batch/free_a/free_b/contract
/// can be expressed this way.
fn try_einsum_as_matmul(
    idx_a: &[char], idx_b: &[char], idx_out: &[char],
    a: &ArrayD<f32>, b: &ArrayD<f32>,
    sizes: &std::collections::HashMap<char, usize>,
) -> Option<ArrayD<f32>> {
    use std::collections::HashSet;
    let a_set: HashSet<char> = idx_a.iter().copied().collect();
    let b_set: HashSet<char> = idx_b.iter().copied().collect();
    let out_set: HashSet<char> = idx_out.iter().copied().collect();

    let mut batch = Vec::new();
    let mut free_a = Vec::new();
    let mut free_b = Vec::new();
    let mut contract = Vec::new();

    let mut all_labels = Vec::new();
    for &c in idx_a.iter().chain(idx_b.iter()).chain(idx_out.iter()) {
        if !all_labels.contains(&c) { all_labels.push(c); }
    }
    for &c in &all_labels {
        match (a_set.contains(&c), b_set.contains(&c), out_set.contains(&c)) {
            (true, true, true) => batch.push(c),
            (true, false, true) => free_a.push(c),
            (false, true, true) => free_b.push(c),
            (true, true, false) => contract.push(c),
            _ => return None,
        }
    }

    let batch_dims: Vec<usize> = batch.iter().map(|c| sizes[c]).collect();
    let batch_size: usize = batch_dims.iter().product::<usize>().max(1);
    let free_a_dims: Vec<usize> = free_a.iter().map(|c| sizes[c]).collect();
    let free_a_size: usize = free_a_dims.iter().product::<usize>().max(1);
    let free_b_dims: Vec<usize> = free_b.iter().map(|c| sizes[c]).collect();
    let free_b_size: usize = free_b_dims.iter().product::<usize>().max(1);
    let contract_size: usize = contract.iter().map(|c| sizes[c]).product::<usize>().max(1);

    // Permute A to [batch..., free_a..., contract...]
    let a_order: Vec<char> = batch.iter().chain(free_a.iter()).chain(contract.iter()).copied().collect();
    let a_perm: Vec<usize> = a_order.iter()
        .map(|c| idx_a.iter().position(|x| x == c))
        .collect::<Option<Vec<_>>>()?;
    // Permute B to [batch..., contract..., free_b...]
    let b_order: Vec<char> = batch.iter().chain(contract.iter()).chain(free_b.iter()).copied().collect();
    let b_perm: Vec<usize> = b_order.iter()
        .map(|c| idx_b.iter().position(|x| x == c))
        .collect::<Option<Vec<_>>>()?;

    let a_t = a.view().permuted_axes(IxDyn(&a_perm)).as_standard_layout().into_owned();
    let b_t = b.view().permuted_axes(IxDyn(&b_perm)).as_standard_layout().into_owned();

    let a_3d = a_t.into_shape_with_order((batch_size, free_a_size, contract_size)).ok()?;
    let b_3d = b_t.into_shape_with_order((batch_size, contract_size, free_b_size)).ok()?;

    // Batched matmul via BLAS
    let mut result_data = Vec::with_capacity(batch_size * free_a_size * free_b_size);
    for i in 0..batch_size {
        let ai = a_3d.index_axis(ndarray::Axis(0), i)
            .into_dimensionality::<ndarray::Ix2>().ok()?;
        let bi = b_3d.index_axis(ndarray::Axis(0), i)
            .into_dimensionality::<ndarray::Ix2>().ok()?;
        result_data.extend(ai.dot(&bi).iter());
    }

    // Reshape to [batch..., free_a..., free_b...]
    let mut intermediate_shape = batch_dims;
    intermediate_shape.extend(&free_a_dims);
    intermediate_shape.extend(&free_b_dims);
    let intermediate = ArrayD::from_shape_vec(IxDyn(&intermediate_shape), result_data).ok()?;

    // Permute to match output label order
    let intermediate_labels: Vec<char> = batch.iter()
        .chain(free_a.iter()).chain(free_b.iter()).copied().collect();
    let out_perm: Vec<usize> = idx_out.iter()
        .map(|c| intermediate_labels.iter().position(|x| x == c))
        .collect::<Option<Vec<_>>>()?;

    if out_perm.iter().enumerate().all(|(i, &p)| i == p) {
        Some(intermediate)
    } else {
        Some(intermediate.permuted_axes(IxDyn(&out_perm)).as_standard_layout().into_owned())
    }
}

fn einsum_naive(
    idx_a: &[char], idx_b: &[char], idx_out: &[char],
    a: &ArrayD<f32>, b: &ArrayD<f32>,
    sizes: &std::collections::HashMap<char, usize>,
) -> ArrayD<f32> {
    let out_shape: Vec<usize> = idx_out.iter()
        .map(|c| *sizes.get(c).unwrap_or(&1))
        .collect();
    let out_len: usize = out_shape.iter().product::<usize>().max(1);
    let mut result = vec![0.0f32; out_len];

    let mut all_labels: Vec<char> = idx_out.to_vec();
    for &c in idx_a.iter().chain(idx_b.iter()) {
        if !all_labels.contains(&c) { all_labels.push(c); }
    }
    let label_sizes: Vec<usize> = all_labels.iter()
        .map(|c| *sizes.get(c).unwrap_or(&1)).collect();
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
            .map(|(i, c)| coords[label_pos[c]] * out_strides[i]).sum();
        result[flat_out] += a[IxDyn(&a_idx)] * b[IxDyn(&b_idx)];
        for k in (0..coords.len()).rev() {
            coords[k] += 1;
            if coords[k] < label_sizes[k] { break; }
            coords[k] = 0;
        }
    }
    ArrayD::from_shape_vec(IxDyn(&out_shape), result).unwrap()
}

/// dL/dA for C = A @ B, given (A, B, adj)
/// Handles 1D edge cases: when A is 1D [K], adj is 1D [N], B is [K, N]
/// Standard: adj @ B^T, but when adj is 1D we need reshape.
fn matmul_result_shape(a: &ArrayD<f32>, b: &ArrayD<f32>) -> Vec<usize> {
    match (a.ndim(), b.ndim()) {
        (1, 1) => vec![],
        (1, _) => b.shape()[1..].to_vec(),
        (_, 1) => a.shape()[..a.ndim()-1].to_vec(),
        _ => {
            let mut s = a.shape()[..a.ndim()-1].to_vec();
            s.push(b.shape()[b.ndim()-1]);
            s
        }
    }
}

fn broadcast_adj(adj: ArrayD<f32>, target_shape: &[usize]) -> ArrayD<f32> {
    if adj.shape() == target_shape {
        return adj;
    }
    if adj.ndim() == 0 {
        let scalar = adj.first().copied().unwrap_or(0.0);
        ArrayD::from_elem(IxDyn(target_shape), scalar)
    } else {
        adj
    }
}

fn builtin_matmul_grad_lhs(args: &[Value], _kw: &BTreeMap<String, Value>) -> R {
    if args.len() != 3 { return Err(runtime_error("@-grad-lhs requires 3 arguments: A, B, adj")); }
    let (a, _) = to_array(&args[0])?;
    let (b, _) = to_array(&args[1])?;
    let (adj_raw, _) = to_array(&args[2])?;
    let result_shape = matmul_result_shape(&a, &b);
    let adj = broadcast_adj(adj_raw, &result_shape);
    let a_ndim = a.ndim();
    let b_ndim = b.ndim();

    let result = match (a_ndim, b_ndim) {
        (1, 1) => {
            (&adj * &b).into_dyn()
        }
        (1, 2) => {
            // dA for [K] @ [K, N] -> [N], adj=[N]
            // dA = adj @ B^T: reshape adj to [1, N], matmul with B^T [N, K], squeeze to [K]
            let n = adj.len();
            let k = a.len();
            let adj2d = adj.to_owned().into_shape_with_order((1, n)).unwrap();
            let bt = b.t().to_owned().into_dimensionality::<ndarray::Ix2>()
                .map_err(|e| runtime_error(e.to_string()))?;
            let r = adj2d.dot(&bt);
            r.into_shape_with_order(ndarray::IxDyn(&[k])).unwrap()
        }
        (2, 1) => {
            let m = a.shape()[0];
            let k = b.len();
            let adj_col = adj.to_owned().into_shape_with_order((m, 1)).unwrap();
            let b_row = b.to_owned().into_shape_with_order((1, k)).unwrap();
            adj_col.dot(&b_row).into_dyn()
        }
        _ => {
            let bt = b.t().to_owned().into_dimensionality::<ndarray::Ix2>().map_err(|e| runtime_error(e.to_string()))?;
            let adj2 = adj.into_dimensionality::<ndarray::Ix2>().map_err(|e| runtime_error(e.to_string()))?;
            adj2.dot(&bt).into_dyn()
        }
    };
    Ok(Value::tensor_f32(result))
}

/// dL/dB for C = A @ B, given (A, B, adj)
/// Handles 1D edge cases: when A is 1D [K], need outer product.
fn builtin_matmul_grad_rhs(args: &[Value], _kw: &BTreeMap<String, Value>) -> R {
    if args.len() != 3 { return Err(runtime_error("@-grad-rhs requires 3 arguments: A, B, adj")); }
    let (a, _) = to_array(&args[0])?;
    let (b, _) = to_array(&args[1])?;
    let (adj_raw, _) = to_array(&args[2])?;
    let result_shape = matmul_result_shape(&a, &b);
    let adj = broadcast_adj(adj_raw, &result_shape);
    let a_ndim = a.ndim();
    let b_ndim = b.ndim();

    let result = match (a_ndim, b_ndim) {
        (1, 1) => {
            (&adj * &a).into_dyn()
        }
        (1, 2) => {
            let k = a.len();
            let n = adj.len();
            let a_col = a.to_owned().into_shape_with_order((k, 1)).unwrap();
            let adj_row = adj.to_owned().into_shape_with_order((1, n)).unwrap();
            a_col.dot(&adj_row).into_dyn()
        }
        (2, 1) => {
            let at = a.t().to_owned().into_dimensionality::<ndarray::Ix2>().map_err(|e| runtime_error(e.to_string()))?;
            let adj1 = adj.into_dimensionality::<ndarray::Ix1>().map_err(|e| runtime_error(e.to_string()))?;
            at.dot(&adj1).into_dyn()
        }
        _ => {
            let at = a.t().to_owned().into_dimensionality::<ndarray::Ix2>().map_err(|e| runtime_error(e.to_string()))?;
            let adj2 = adj.into_dimensionality::<ndarray::Ix2>().map_err(|e| runtime_error(e.to_string()))?;
            at.dot(&adj2).into_dyn()
        }
    };
    Ok(Value::tensor_f32(result))
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

    // Try BLAS-accelerated path, fall back to naive loops
    let arr = try_einsum_as_matmul(&idx_a, &idx_b, &idx_out, &a, &b, &sizes)
        .unwrap_or_else(|| einsum_naive(&idx_a, &idx_b, &idx_out, &a, &b, &sizes));

    if arr.ndim() == 0 {
        Ok(Value::Float(*arr.first().unwrap()))
    } else {
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
    let mut data: Vec<f32> = arr.iter().skip(1).copied().collect();
    data.push(new_val as f32);
    let result = ArrayD::from_shape_vec(IxDyn(&[n]), data).unwrap();
    Ok(Value::tensor_f32(result))
}
