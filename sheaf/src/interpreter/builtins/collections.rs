use super::*;

pub(super) fn register(env: &mut Env) {
    env.set_builtin("first", builtin_first);
    env.set_builtin("second", builtin_second);
    env.set_builtin("last", builtin_last);
    env.set_builtin("rest", builtin_rest);
    env.set_builtin("nth", builtin_nth);
    env.set_builtin("cons", builtin_cons);
    env.set_builtin("append", builtin_append);
    env.set_builtin("empty?", builtin_empty);
    env.set_builtin("get-in", builtin_get_in);
    env.set_builtin("assoc", builtin_assoc);
    env.set_builtin("dissoc", builtin_dissoc);
    env.set_builtin("merge", builtin_merge);
    env.set_builtin("keys", builtin_keys);
    env.set_builtin("vals", builtin_vals);
    env.set_builtin("dict", builtin_dict);
    env.set_builtin("sort", builtin_sort);
    env.set_builtin("chars", builtin_chars);
    env.set_builtin("index-of", builtin_index_of);
}

fn builtin_first(args: &[Value], kw: &BTreeMap<String, Value>) -> R {
    match &args[0] {
        Value::List(items) | Value::Tuple(items) => items.first().cloned().ok_or_else(|| runtime_error("first: empty list")),
        Value::Tensor { data, .. } => {
            if data.shape()[0] == 0 {
                return Err(runtime_error("first: empty tensor"));
        }
            let sliced = data.index_axis(ndarray::Axis(0), 0).to_owned();
            if sliced.shape().is_empty() { Ok(Value::Float(as_scalar(&sliced))) }
            else { Ok(Value::tensor_f32(sliced)) }
        }
        Value::DeviceBuffer(_) => {
            let host = args[0].ensure_host()?;
            builtin_first(std::slice::from_ref(&host), kw)
        }
        other => Err(runtime_error(format!("first: expected a list or a tensor, got {}", other.type_name()))),
    }
}

fn builtin_second(args: &[Value], kw: &BTreeMap<String, Value>) -> R {
    match &args[0] {
        Value::List(items) | Value::Tuple(items) => items.get(1).cloned().ok_or_else(|| runtime_error("second: list too short")),
        Value::Tensor { data, .. } => {
            if data.shape()[0] < 2 {
                return Err(runtime_error("second: tensor too short (need at least 2 elements on axis 0)"));
        }
            let sliced = data.index_axis(ndarray::Axis(0), 1).to_owned();
            if sliced.shape().is_empty() { Ok(Value::Float(as_scalar(&sliced))) }
            else { Ok(Value::tensor_f32(sliced)) }
        }
        Value::DeviceBuffer(_) => {
            let host = args[0].ensure_host()?;
            builtin_second(std::slice::from_ref(&host), kw)
        }
        other => Err(runtime_error(format!("second: expected a list or a tensor, got {}", other.type_name()))),
    }
}

fn builtin_last(args: &[Value], kw: &BTreeMap<String, Value>) -> R {
    match &args[0] {
        Value::List(items) => items.last().cloned().ok_or_else(|| runtime_error("last: empty list")),
        Value::Tensor { data, .. } => {
            let n = data.shape()[0];
            if n == 0 {
                return Err(runtime_error("last: empty tensor"));
        }
            let sliced = data.index_axis(ndarray::Axis(0), n - 1).to_owned();
            if sliced.shape().is_empty() { Ok(Value::Float(as_scalar(&sliced))) }
            else { Ok(Value::tensor_f32(sliced)) }
        }
        Value::DeviceBuffer(_) => {
            let host = args[0].ensure_host()?;
            builtin_last(std::slice::from_ref(&host), kw)
        }
        other => Err(runtime_error(format!("last: expected a list or a tensor, got {}", other.type_name()))),
    }
}

fn builtin_rest(args: &[Value], _kw: &BTreeMap<String, Value>) -> R {
    match &args[0] {
        Value::List(items) => {
            if items.is_empty() { Ok(Value::List(vec![])) }
            else { Ok(Value::List(items[1..].to_vec())) }
        }
        other => Err(runtime_error(format!("rest: expected a list, got {}", other.type_name()))),
    }
}

fn builtin_nth(args: &[Value], kw: &BTreeMap<String, Value>) -> R {
    let f = args[1].to_f64().ok_or_else(|| runtime_error(format!("nth: expected numeric index, got {}", args[1].type_name())))?;
    match &args[0] {
        Value::List(items) => {
            let idx = resolve_idx(f, items.len())?;
            Ok(items[idx].clone())
        }
            Value::Tensor { data, .. } => {
                let dim0 = data.shape()[0];
                let idx = resolve_idx(f, dim0)?;
                let sliced = data.index_axis(ndarray::Axis(0), idx).to_owned();
            if sliced.shape().is_empty() { Ok(Value::Float(as_scalar(&sliced))) }
            else { Ok(Value::tensor_f32(sliced)) }
        }
        Value::DeviceBuffer(_) => {
            let host = args[0].ensure_host()?;
            builtin_nth(&[host, args[1].clone()], kw)
        }
        other => Err(runtime_error(format!("nth: expected a list or a tensor, got {}", other.type_name()))),
    }
}

fn builtin_cons(args: &[Value], _kw: &BTreeMap<String, Value>) -> R {
    match &args[1] {
        Value::List(items) => {
            let mut new = Vec::with_capacity(items.len() + 1);
            new.push(args[0].clone());
            new.extend_from_slice(items);
            Ok(Value::List(new))
        }
        _ => Err(runtime_error("cons: second argument must be a list")),
    }
}

fn builtin_append(args: &[Value], _kw: &BTreeMap<String, Value>) -> R {
    match &args[0] {
        Value::List(items) => {
            let mut new = items.clone();
            new.push(args[1].clone());
            Ok(Value::List(new))
        }
        _ => Err(runtime_error("append: first argument must be a list")),
    }
}

fn builtin_empty(args: &[Value], _kw: &BTreeMap<String, Value>) -> R {
    match &args[0] {
        Value::List(items) => Ok(Value::Bool(items.is_empty())),
        Value::Tensor { data, .. } => Ok(Value::Bool(data.is_empty())),
        Value::Dict(map) => Ok(Value::Bool(map.is_empty())),
        _ => Ok(Value::Bool(false)),
    }
}

fn builtin_get_in(args: &[Value], _kw: &BTreeMap<String, Value>) -> R {
    let owned_path: Vec<Value>;
    let path: &[Value] = match &args[1] {
        Value::List(items) => items,
        Value::Tensor { data, .. } => {
            owned_path = data.iter().map(|&x| Value::Int(x as i64)).collect();
            &owned_path
        }
        _ => return Err(runtime_error("get-in: path must be a list")),
    };
    let default = if args.len() > 2 { Some(args[2].clone()) } else { None };
    let mut current = args[0].clone();
    for key in path {
        current = match (&current, key) {
            (Value::Dict(map), Value::Keyword(k)) | (Value::Dict(map), Value::String(k)) => {
                match map.get(k) {
                    Some(v) => v.clone(),
                    None => return Ok(default.unwrap_or(Value::Nil)),
                }
            }
            (Value::Tensor { data, .. }, Value::Int(idx)) => {
                let dim0 = data.shape()[0];
                let idx_u = if *idx < 0 {
                    let resolved = dim0 as i64 + *idx;
                    if resolved < 0 {
                        return Ok(default.unwrap_or(Value::Nil));
                    }
                    resolved as usize
                } else {
                    *idx as usize
                };
                if idx_u >= dim0 {
                    return Ok(default.unwrap_or(Value::Nil));
            }
                let sliced = data.index_axis(ndarray::Axis(0), idx_u).to_owned();
                if sliced.shape().is_empty() { Value::Float(as_scalar(&sliced)) }
                else { Value::tensor_f32(sliced) }
            }
            (Value::List(items), Value::Int(idx)) => {
                match items.get(*idx as usize) {
                    Some(v) => v.clone(),
                    None => return Ok(default.unwrap_or(Value::Nil)),
                }
            }
            _ => return Ok(default.unwrap_or(Value::Nil)),
        };
    }
    Ok(current)
}

fn builtin_assoc(args: &[Value], _kw: &BTreeMap<String, Value>) -> R {
    if args.len() < 3 || (args.len() - 1) % 2 != 0 {
        return Err(runtime_error("assoc: expected (assoc dict key val ...)"));
    }
    match &args[0] {
        Value::Dict(map) => {
            let mut new = map.clone();
            for pair in args[1..].chunks(2) {
                let key = match &pair[0] {
                    Value::Keyword(k) => k.clone(),
                    Value::String(s) => s.clone(),
                    _ => return Err(runtime_error("assoc: expected key to be a keyword or a string")),
                };
                new.insert(key, pair[1].clone());
            }
            Ok(Value::Dict(new))
        }
        _ => Err(runtime_error("assoc: first argument must be a dict")),
    }
}

fn builtin_dissoc(args: &[Value], _kw: &BTreeMap<String, Value>) -> R {
    match &args[0] {
        Value::Dict(map) => {
            let keys_to_remove: Vec<String> = match &args[1] {
                Value::List(items) => items.iter().filter_map(|v| match v {
                    Value::Keyword(k) | Value::String(k) => Some(k.clone()),
                    _ => None,
                }).collect(),
                _ => return Err(runtime_error("dissoc: keys must be a list")),
            };
            let new: BTreeMap<String, Value> = map.iter()
                .filter(|(k, _)| !keys_to_remove.contains(k))
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect();
            Ok(Value::Dict(new))
        }
        _ => Err(runtime_error("dissoc: first argument must be a dict")),
    }
}

fn builtin_merge(args: &[Value], _kw: &BTreeMap<String, Value>) -> R {
    let mut result = BTreeMap::new();
    for arg in args {
        if let Value::Dict(map) = arg {
            for (k, v) in map {
                result.insert(k.clone(), v.clone());
            }
        } else {
            return Err(runtime_error(format!("merge: expected dicts, got {}", arg.type_name())));
        }
    }
    Ok(Value::Dict(result))
}

fn builtin_keys(args: &[Value], _kw: &BTreeMap<String, Value>) -> R {
    match &args[0] {
        Value::Dict(map) => {
            Ok(Value::List(map.keys().map(|k| Value::String(k.clone())).collect()))
        }
        other => Err(runtime_error(format!("keys: expected a dict, got {}", other.type_name()))),
    }
}

fn builtin_vals(args: &[Value], _kw: &BTreeMap<String, Value>) -> R {
    match &args[0] {
        Value::Dict(map) => {
            Ok(Value::List(map.values().cloned().collect()))
        }
        other => Err(runtime_error(format!("vals: expected a dict, got {}", other.type_name()))),
    }
}

fn builtin_dict(args: &[Value], _kw: &BTreeMap<String, Value>) -> R {
    let mut map = BTreeMap::new();
    let mut i = 0;
    while i + 1 < args.len() {
        let key = match &args[i] {
            Value::Keyword(k) => k.clone(),
            Value::String(s) => s.clone(),
            _ => return Err(runtime_error("dict: expected key to be a keyword or a string")),
        };
        map.insert(key, args[i + 1].clone());
        i += 2;
    }
    Ok(Value::Dict(map))
}

fn builtin_sort(args: &[Value], kw: &BTreeMap<String, Value>) -> R {
    let reverse = matches!(kw.get("reverse"), Some(Value::Bool(true)));

    match args.first() {
        Some(Value::List(items)) => {
            let mut sorted = items.clone();
            sorted.sort_by(|a, b| {
                let ka = sort_key(a);
                let kb = sort_key(b);
                let ord = ka.partial_cmp(&kb).unwrap_or(std::cmp::Ordering::Equal);
                if reverse { ord.reverse() } else { ord }
            });
            Ok(Value::List(sorted))
        }
        Some(Value::Tensor { data, dtype }) => {
            let axis = super::get_axis(kw).unwrap_or(-1);
            let ax = if axis < 0 { (data.ndim() as i64 + axis) as usize } else { axis as usize };
            if ax >= data.ndim() {
                return Err(runtime_error(format!("sort: axis {} out of bounds for {}D tensor", axis, data.ndim())));
            }

            let mut result = (**data).clone();
            for mut lane in result.lanes_mut(ndarray::Axis(ax)) {
                let mut v: Vec<f32> = lane.iter().copied().collect();
                if reverse {
                    v.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
                } else {
                    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                }
                for (dst, src) in lane.iter_mut().zip(v.iter()) {
                    *dst = *src;
                }
            }
            Ok(Value::Tensor { data: std::sync::Arc::new(result), dtype: *dtype })
        }
        other => Err(runtime_error(format!("sort: expected a list or tensor, got {}", other.map(|v| v.type_name()).unwrap_or("nothing")))),
    }
}

fn sort_key(val: &Value) -> String {
    match val {
        Value::String(s) => s.clone(),
        Value::Int(n) => format!("{:020}", n),
        Value::Float(f) => format!("{:020.10}", f),
        Value::Keyword(k) => k.clone(),
        _ => format!("{}", val),
    }
}

fn builtin_chars(args: &[Value], _kw: &BTreeMap<String, Value>) -> R {
    match args.first() {
        Some(Value::String(s)) => {
            let chars: Vec<Value> = s.chars().map(|c| Value::String(c.to_string())).collect();
            Ok(Value::List(chars))
        }
        _ => Err(runtime_error("chars: expected a string")),
    }
}

fn builtin_index_of(args: &[Value], _kw: &BTreeMap<String, Value>) -> R {
    match &args[0] {
        Value::List(items) => {
            let target = &args[1];
            for (i, item) in items.iter().enumerate() {
                let eq = match (item, target) {
                    (Value::Int(a), Value::Int(b)) => a == b,
                    (Value::Float(a), Value::Float(b)) => (a - b).abs() < 1e-6,
                    (Value::Int(a), Value::Float(b)) | (Value::Float(b), Value::Int(a)) => (*a as f32 - b).abs() < 1e-6,
                    (Value::String(a), Value::String(b)) => a == b,
                    (Value::Keyword(a), Value::Keyword(b)) => a == b,
                    (Value::Bool(a), Value::Bool(b)) => a == b,
                    _ => false,
                };
                if eq { return Ok(Value::Int(i as i64)); }
            }
            Ok(Value::Int(-1))
        }
        other => Err(runtime_error(format!("index-of: expected a list, got {}", other.type_name()))),
    }
}
