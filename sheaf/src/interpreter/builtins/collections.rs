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

fn builtin_first(args: &[Value], _kw: &BTreeMap<String, Value>) -> R {
    match &args[0] {
        Value::List(items) | Value::Tuple(items) => items.first().cloned().ok_or_else(|| runtime_error("first: empty list")),
        Value::Tensor { data, .. } => {
            let sliced = data.index_axis(ndarray::Axis(0), 0).to_owned();
            if sliced.shape().is_empty() { Ok(Value::Float(*sliced.first().unwrap())) }
            else { Ok(Value::tensor_f32(sliced)) }
        }
        _ => Err(runtime_error("first: expected list or tensor")),
    }
}

fn builtin_second(args: &[Value], _kw: &BTreeMap<String, Value>) -> R {
    match &args[0] {
        Value::List(items) | Value::Tuple(items) => items.get(1).cloned().ok_or_else(|| runtime_error("second: list too short")),
        Value::Tensor { data, .. } => {
            let sliced = data.index_axis(ndarray::Axis(0), 1).to_owned();
            if sliced.shape().is_empty() { Ok(Value::Float(*sliced.first().unwrap())) }
            else { Ok(Value::tensor_f32(sliced)) }
        }
        _ => Err(runtime_error("second: expected list or tensor")),
    }
}

fn builtin_last(args: &[Value], _kw: &BTreeMap<String, Value>) -> R {
    match &args[0] {
        Value::List(items) => items.last().cloned().ok_or_else(|| runtime_error("last: empty list")),
        Value::Tensor { data, .. } => {
            let n = data.shape()[0];
            let sliced = data.index_axis(ndarray::Axis(0), n - 1).to_owned();
            if sliced.shape().is_empty() { Ok(Value::Float(*sliced.first().unwrap())) }
            else { Ok(Value::tensor_f32(sliced)) }
        }
        _ => Err(runtime_error("last: expected list or tensor")),
    }
}

fn builtin_rest(args: &[Value], _kw: &BTreeMap<String, Value>) -> R {
    match &args[0] {
        Value::List(items) => {
            if items.is_empty() { Ok(Value::List(vec![])) }
            else { Ok(Value::List(items[1..].to_vec())) }
        }
        _ => Err(runtime_error("rest: expected list")),
    }
}

fn builtin_nth(args: &[Value], _kw: &BTreeMap<String, Value>) -> R {
    let f = args[1].to_f64().unwrap();
    match &args[0] {
        Value::List(items) => {
            let idx = resolve_idx(f, items.len());
            items.get(idx).cloned().ok_or_else(|| runtime_error("nth: index out of bounds"))
        }
        Value::Tensor { data, .. } => {
            let idx = resolve_idx(f, data.shape()[0]);
            let sliced = data.index_axis(ndarray::Axis(0), idx).to_owned();
            if sliced.shape().is_empty() { Ok(Value::Float(*sliced.first().unwrap())) }
            else { Ok(Value::tensor_f32(sliced)) }
        }
        _ => Err(runtime_error("nth: expected list or tensor")),
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
        _ => Err(runtime_error("cons: second arg must be list")),
    }
}

fn builtin_append(args: &[Value], _kw: &BTreeMap<String, Value>) -> R {
    match &args[0] {
        Value::List(items) => {
            let mut new = items.clone();
            new.push(args[1].clone());
            Ok(Value::List(new))
        }
        _ => Err(runtime_error("append: first arg must be list")),
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
                let sliced = data.index_axis(ndarray::Axis(0), *idx as usize).to_owned();
                if sliced.shape().is_empty() { Value::Float(*sliced.first().unwrap()) }
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
                    _ => return Err(runtime_error("assoc: key must be keyword or string")),
                };
                new.insert(key, pair[1].clone());
            }
            Ok(Value::Dict(new))
        }
        _ => Err(runtime_error("assoc: expected dict")),
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
        _ => Err(runtime_error("dissoc: expected dict")),
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
            return Err(runtime_error("merge: expected dicts"));
        }
    }
    Ok(Value::Dict(result))
}

fn builtin_keys(args: &[Value], _kw: &BTreeMap<String, Value>) -> R {
    match &args[0] {
        Value::Dict(map) => {
            Ok(Value::List(map.keys().map(|k| Value::String(k.clone())).collect()))
        }
        _ => Err(runtime_error("keys: expected dict")),
    }
}

fn builtin_vals(args: &[Value], _kw: &BTreeMap<String, Value>) -> R {
    match &args[0] {
        Value::Dict(map) => {
            Ok(Value::List(map.values().cloned().collect()))
        }
        _ => Err(runtime_error("vals: expected dict")),
    }
}

fn builtin_dict(args: &[Value], _kw: &BTreeMap<String, Value>) -> R {
    let mut map = BTreeMap::new();
    let mut i = 0;
    while i + 1 < args.len() {
        let key = match &args[i] {
            Value::Keyword(k) => k.clone(),
            Value::String(s) => s.clone(),
            _ => return Err(runtime_error("dict: key must be keyword or string")),
        };
        map.insert(key, args[i + 1].clone());
        i += 2;
    }
    Ok(Value::Dict(map))
}

fn builtin_sort(args: &[Value], _kw: &BTreeMap<String, Value>) -> R {
    match args.first() {
        Some(Value::List(items)) => {
            let mut sorted = items.clone();
            sorted.sort_by(|a, b| {
                let ka = sort_key(a);
                let kb = sort_key(b);
                ka.partial_cmp(&kb).unwrap_or(std::cmp::Ordering::Equal)
            });
            Ok(Value::List(sorted))
        }
        _ => Err(runtime_error("sort: expected a list")),
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
        _ => Err(runtime_error("index-of: expected list")),
    }
}
