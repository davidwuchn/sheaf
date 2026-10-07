use super::*;
use std::sync::Arc;

pub(super) fn register(env: &mut Env) {
    env.set_builtin("random-key", builtin_random_key);
    env.set_builtin("random-split", builtin_random_split);
    env.set_builtin("random-normal", builtin_random_normal);
    env.set_builtin("random-uniform", builtin_random_uniform);
    env.set_builtin("random-randint", builtin_random_randint);
    env.set_builtin("choice", builtin_choice);
    env.set_builtin("top_k", builtin_top_k);
}

fn seed_to_key(seed: u64) -> Value {
    let lo = (seed & 0xFFFFFFFF) as f32;
    let hi = ((seed >> 32) & 0xFFFFFFFF) as f32;
    let arr = ArrayD::from_shape_vec(IxDyn(&[2]), vec![lo, hi]).unwrap();
    Value::tensor_f32(arr)
}

fn builtin_random_key(args: &[Value], _kw: &BTreeMap<String, Value>) -> R {
    let seed = match args.first() {
        Some(Value::Int(n)) => *n as u64,
        Some(Value::Float(f)) => *f as u64,
        _ => return Err(runtime_error("random-key: expected integer seed")),
    };
    Ok(seed_to_key(seed))
}

fn key_to_seed(key: &Value) -> Result<u64, crate::core::error::SheafError> {
    let key = key.ensure_host()?;
    match &key {
        Value::Tensor { data, dtype } if data.shape() == [2] && *dtype != Dtype::Bool
            && data.iter().all(|&word| word.is_finite() && word.fract() == 0.0) => {
            let lo = data[IxDyn(&[0])] as u64;
            let hi = data[IxDyn(&[1])] as u64;
            Ok(lo | (hi << 32))
        }
        Value::List(items) => match items.as_slice() {
            [Value::Int(lo), Value::Int(hi)] => Ok(*lo as u64 | ((*hi as u64) << 32)),
            _ => Err(runtime_error("expected a PRNG key with two integer words")),
        },
        Value::Int(n) => Ok(*n as u64),
        _ => Err(runtime_error(format!("expected a PRNG key, got {}", key.type_name()))),
    }
}

fn builtin_random_split(args: &[Value], _kw: &BTreeMap<String, Value>) -> R {
    if args.is_empty() || args.len() > 2 {
        return Err(runtime_error("random-split: expected (random-split key) or (random-split key n)"));
    }
    let seed = key_to_seed(&args[0])?;
    let n = if args.len() == 2 {
        match &args[1] {
            Value::Int(n) => usize::try_from(*n)
                .map_err(|_| runtime_error("random-split: count must be nonnegative"))?,
            Value::Float(n) => checked_dimension(*n as f64)
                .map_err(|error| runtime_error(format!("random-split: count: {error}")))?,
            _ => return Err(runtime_error("random-split: n must be an integer")),
        }
    } else {
        2
    };
    if n > isize::MAX as usize / std::mem::size_of::<Value>() {
        return Err(runtime_error("random-split: count exceeds addressable memory"));
    }
    let mut keys = Vec::new();
    keys.try_reserve_exact(n)
        .map_err(|_| runtime_error("random-split: cannot allocate keys"))?;
    for i in 0..n {
        let child_seed = seed.wrapping_add(i as u64).wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        keys.push(seed_to_key(child_seed));
    }
    Ok(Value::List(keys))
}

fn splitmix64(state: &mut u64) -> f32 {
    *state = state.wrapping_add(0x9e3779b97f4a7c15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
    z = z ^ (z >> 31);
    (z >> 11) as f32 / (1u64 << 53) as f32
}

fn builtin_random_normal(args: &[Value], _kw: &BTreeMap<String, Value>) -> R {
    if args.len() != 2 {
        return Err(runtime_error("random-normal: expected (random-normal key shape)"));
    }
    let mut state = key_to_seed(&args[0])?;
    let shape = shape_from_value(&args[1])?;
    let n: usize = shape.iter().product();
    let mut data = Vec::with_capacity(n);
    let mut i = 0;
    while i < n {
        let u1 = splitmix64(&mut state).max(1e-10);
        let u2 = splitmix64(&mut state);
        let r = (-2.0f32 * u1.ln()).sqrt();
        let theta = 2.0f32 * std::f32::consts::PI * u2;
        data.push(r * theta.cos());
        if i + 1 < n { data.push(r * theta.sin()); }
        i += 2;
    }
    data.truncate(n);
    let arr = ArrayD::from_shape_vec(IxDyn(&shape), data)
        .map_err(|e| runtime_error(format!("random-normal: shape error: {}", e)))?;
    Ok(Value::tensor_f32(arr))
}

fn builtin_random_uniform(args: &[Value], _kw: &BTreeMap<String, Value>) -> R {
    if args.len() != 2 {
        return Err(runtime_error("random-uniform: expected (random-uniform key shape)"));
    }
    let mut state = key_to_seed(&args[0])?;
    let shape = shape_from_value(&args[1])?;
    let n: usize = shape.iter().product();
    let data: Vec<f32> = (0..n).map(|_| splitmix64(&mut state)).collect();
    let arr = ArrayD::from_shape_vec(IxDyn(&shape), data)
        .map_err(|e| runtime_error(format!("random-uniform: shape error: {}", e)))?;
    Ok(Value::tensor_f32(arr))
}

fn builtin_random_randint(args: &[Value], _kw: &BTreeMap<String, Value>) -> R {
    if args.len() != 4 {
        return Err(runtime_error("random-randint: expected (random-randint key shape low high)"));
    }
    let mut state = key_to_seed(&args[0])?;
    let shape = shape_from_value(&args[1])?;
    let low = match &args[2] {
        Value::Int(n) => *n,
        Value::Float(f) => *f as i64,
        _ => return Err(runtime_error("random-randint: low must be integer")),
    };
    let high = match &args[3] {
        Value::Int(n) => *n,
        Value::Float(f) => *f as i64,
        _ => return Err(runtime_error("random-randint: high must be integer")),
    };
    if high <= low {
        return Err(runtime_error("random-randint: high must be > low"));
    }
    let range = (high - low) as u64;
    let n: usize = shape.iter().product();
    let data: Vec<f32> = (0..n)
        .map(|_| {
            let u = splitmix64(&mut state);
            let idx = (u * range as f32) as i64;
            (low + idx.min(high - low - 1)) as f32
        })
        .collect();
    let arr = ArrayD::from_shape_vec(IxDyn(&shape), data)
        .map_err(|e| runtime_error(format!("random-randint: shape error: {}", e)))?;
    Ok(Value::tensor_i32(arr))
}

fn builtin_choice(args: &[Value], kw: &BTreeMap<String, Value>) -> R {
    if args.len() < 2 {
        return Err(runtime_error("choice: expected (choice key n :p probs)"));
    }
    let seed = key_to_seed(&args[0])?;
    let n = match &args[1] {
        Value::Int(n) => *n as usize,
        Value::Float(f) => *f as usize,
        _ => return Err(runtime_error("choice: n must be integer")),
    };
    let probs = kw.get("p").or_else(|| args.get(2));
    let mut state = seed;
    let u = splitmix64(&mut state);
    match probs {
        Some(Value::Tensor { data, .. }) => {
            if data.is_empty() {
                return Err(runtime_error("choice: empty probability tensor"));
            }
            let mut cumsum = 0.0f32;
            for (i, &p) in data.iter().enumerate() {
                cumsum += p;
                if u < cumsum {
                    return Ok(Value::Int(i as i64));
            }
        }
            Ok(Value::Int((data.len() - 1) as i64))
        }
        Some(Value::DeviceBuffer(db)) => {
            let data = db.to_host().map_err(|e| runtime_error(format!("choice: {}", e)))?;
            if data.is_empty() {
                return Err(runtime_error("choice: empty probability tensor"));
            }
            let mut cumsum = 0.0f32;
            for (i, &p) in data.iter().enumerate() {
                cumsum += p;
                if u < cumsum {
                    return Ok(Value::Int(i as i64));
            }
        }
            Ok(Value::Int((data.len() - 1) as i64))
        }
        Some(Value::List(items)) => {
            let flat: Vec<f32> = items.iter().filter_map(|v| v.to_f64().map(|x| x as f32)).collect();
            if flat.is_empty() {
                return Err(runtime_error("choice: empty probability list"));
            }
            let mut cumsum = 0.0f32;
            for (i, &p) in flat.iter().enumerate() {
                cumsum += p;
                if u < cumsum {
                    return Ok(Value::Int(i as i64));
            }
        }
            Ok(Value::Int((flat.len() - 1) as i64))
        }
        None => {
            Ok(Value::Int((u * n as f32) as i64))
        }
        _ => Err(runtime_error("choice: :p must be a tensor or list of probabilities")),
    }
}

fn builtin_top_k(args: &[Value], _kw: &BTreeMap<String, Value>) -> R {
    if args.len() < 2 {
        return Err(runtime_error("top_k: expected (top_k tensor k)"));
    }
    let (arr, dtype) = to_array(&args[0])?;
    let k = match &args[1] {
        Value::Int(n) => *n as usize,
        Value::Float(f) => *f as usize,
        Value::Tensor { data, .. } if data.ndim() == 0 => as_scalar(data) as usize,
        _ => return Err(runtime_error("top_k: k must be integer")),
    };
    let flat: Vec<f32> = arr.iter().copied().collect();
    let mut indexed: Vec<(usize, f32)> = flat.into_iter().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let k = k.min(indexed.len());
    let top_vals: Vec<f32> = indexed[..k].iter().map(|(_, v)| *v).collect();
    let top_idxs: Vec<f32> = indexed[..k].iter().map(|(i, _)| *i as f32).collect();
    let vals = ArrayD::from_shape_vec(IxDyn(&[k]), top_vals)
        .map_err(|e| runtime_error(format!("top_k: {}", e)))?;
    let idxs = ArrayD::from_shape_vec(IxDyn(&[k]), top_idxs)
        .map_err(|e| runtime_error(format!("top_k: {}", e)))?;
    Ok(Value::Tuple(vec![
        Value::Tensor { data: Arc::new(vals), dtype },
        Value::tensor_i32(idxs),
    ]))
}
