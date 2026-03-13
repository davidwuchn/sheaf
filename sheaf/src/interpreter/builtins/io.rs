use super::*;
use std::sync::Arc;

pub(super) fn register(env: &mut Env) {
    env.set_builtin("print", builtin_print);
    env.set_builtin("str", builtin_str);
    env.set_builtin("str-call", builtin_str_call);
    env.set_builtin("io", builtin_io);
    env.set_builtin("gensym", builtin_gensym);
    env.set_builtin("symbol?", builtin_symbol_q);
}

fn builtin_print(args: &[Value], kw: &BTreeMap<String, Value>) -> R {
    let end = match kw.get("end") {
        Some(Value::String(s)) => s.as_str(),
        _ => "\n",
    };
    if args.is_empty() {
        print!("{}", end);
        flush_stdout();
        return Ok(Value::Nil);
    }
    let output = if args.len() == 1 {
        format!("{}", args[0])
    } else if let Value::String(s) = &args[0] {
        if s.contains("{}") || s.contains("{:") {
            format_string(s, &args[1..])
        } else {
            args.iter().map(|a| format!("{}", a)).collect::<Vec<_>>().join(" ")
        }
    } else {
        args.iter().map(|a| format!("{}", a)).collect::<Vec<_>>().join(" ")
    };
    print!("{}{}", output, end);
    if !end.ends_with('\n') {
        flush_stdout();
    }
    Ok(Value::Nil)
}

fn flush_stdout() {
    use std::io::Write;
    let _ = std::io::stdout().flush();
}

fn format_string(fmt: &str, vals: &[Value]) -> String {
    let mut result = String::new();
    let mut val_idx = 0;
    let mut chars = fmt.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '{' {
            if chars.peek() == Some(&'{') {
                chars.next();
                result.push('{');
                continue;
            }
            let mut spec = String::new();
            let mut closed = false;
            for ch in chars.by_ref() {
                if ch == '}' { closed = true; break; }
                spec.push(ch);
            }
            if !closed {
                result.push('{');
                result.push_str(&spec);
                continue;
            }
            if let Some(val) = vals.get(val_idx) {
                val_idx += 1;
                result.push_str(&format_value_with_spec(val, &spec));
            } else {
                result.push_str("{}");
            }
        } else if c == '}' && chars.peek() == Some(&'}') {
            chars.next();
            result.push('}');
        } else {
            result.push(c);
        }
    }
    result
}

fn format_value_with_spec(val: &Value, spec: &str) -> String {
    if spec.is_empty() {
        return format!("{}", val);
    }
    let spec = spec.strip_prefix(':').unwrap_or(spec);
    if let Some(rest) = spec.strip_prefix('.') {
        if let Some(prec_str) = rest.strip_suffix('f') {
            if let Ok(prec) = prec_str.parse::<usize>() {
                let f = match val {
                    Value::Float(x) => *x as f64,
                    Value::Int(n) => *n as f64,
                    Value::Tensor { data, .. } => data.first().copied().unwrap_or(0.0) as f64,
                    _ => return format!("{}", val),
                };
                return format!("{:.prec$}", f, prec = prec);
            }
        }
    }
    format!("{}", val)
}

fn builtin_str(args: &[Value], _kw: &BTreeMap<String, Value>) -> R {
    if args.is_empty() { return Ok(Value::String(String::new())); }
    let mut s = String::new();
    for a in args {
        use std::fmt::Write;
        write!(s, "{}", a).unwrap();
    }
    Ok(Value::String(s))
}

fn builtin_str_call(args: &[Value], _kw: &BTreeMap<String, Value>) -> R {
    if args.len() < 2 {
        return Err(runtime_error("str-call: expected (str-call method string ...)"));
    }
    let method = match &args[0] {
        Value::String(s) => s.as_str(),
        _ => return Err(runtime_error("str-call: first arg must be a string method name")),
    };
    let s = match &args[1] {
        Value::String(s) => s.clone(),
        other => format!("{}", other),
    };
    let extra = &args[2..];
    match method {
        "format" => {
            let result = format_string(&s, extra);
            Ok(Value::String(result))
        }
        "upper" => Ok(Value::String(s.to_uppercase())),
        "lower" => Ok(Value::String(s.to_lowercase())),
        "trim" => Ok(Value::String(s.trim().to_string())),
        "replace" => {
            if extra.len() != 2 {
                return Err(runtime_error("str-call replace: expected (str-call \"replace\" s from to)"));
            }
            let from = format!("{}", extra[0]);
            let to = format!("{}", extra[1]);
            Ok(Value::String(s.replace(&from, &to)))
        }
        "startswith" => {
            let prefix = extra.first().map(|v| format!("{}", v)).unwrap_or_default();
            Ok(Value::Bool(s.starts_with(&prefix)))
        }
        "endswith" => {
            let suffix = extra.first().map(|v| format!("{}", v)).unwrap_or_default();
            Ok(Value::Bool(s.ends_with(&suffix)))
        }
        "contains" => {
            let sub = extra.first().map(|v| format!("{}", v)).unwrap_or_default();
            Ok(Value::Bool(s.contains(&sub)))
        }
        "split" => {
            let sep = extra.first().map(|v| format!("{}", v)).unwrap_or_else(|| " ".to_string());
            let parts: Vec<Value> = s.split(&sep).map(|p| Value::String(p.to_string())).collect();
            Ok(Value::List(parts))
        }
        _ => Err(runtime_error(format!("str-call: unknown method '{}'", method))),
    }
}

fn builtin_io(args: &[Value], _kw: &BTreeMap<String, Value>) -> R {
    let verb = match args.first() {
        Some(Value::String(s)) => s.as_str(),
        _ => return Err(runtime_error("io: first argument must be a string verb")),
    };
    match verb {
        "entropy" => {
            let mut bytes = [0u8; 8];
            getrandom::getrandom(&mut bytes).map_err(|e| {
                runtime_error(format!("io: entropy: {}", e))
            })?;
            let seed = u64::from_le_bytes(bytes) as i64;
            Ok(Value::Int(seed))
        }
        "read" => {
            let path = match args.get(1) {
                Some(Value::String(s)) => s,
                _ => return Err(runtime_error("io read: expected path string")),
            };
            let contents = std::fs::read_to_string(path)
                .map_err(|e| runtime_error(format!("io read '{}': {}", path, e)))?;
            Ok(Value::String(contents))
        }
        "exists" => {
            let path = match args.get(1) {
                Some(Value::String(s)) => s,
                _ => return Err(runtime_error("io exists: expected path string")),
            };
            Ok(Value::Bool(std::path::Path::new(path).exists()))
        }
        "save" => {
            let path = match args.get(1) {
                Some(Value::String(s)) => s,
                _ => return Err(runtime_error("io save: expected path string")),
            };
            let value = match args.get(2) {
                Some(v) => v,
                None => return Err(runtime_error("io save: expected value to save")),
            };
            let json = value_to_json(value)?;
            if let Some(parent) = std::path::Path::new(path).parent() {
                if !parent.exists() {
                    std::fs::create_dir_all(parent)
                        .map_err(|e| runtime_error(format!("io save: mkdir '{}': {}", parent.display(), e)))?;
                }
            }
            let data = serde_json::to_vec(&json)
                .map_err(|e| runtime_error(format!("io save: serialize: {}", e)))?;
            std::fs::write(path, data)
                .map_err(|e| runtime_error(format!("io save '{}': {}", path, e)))?;
            Ok(Value::Nil)
        }
        "load" => {
            let path = match args.get(1) {
                Some(Value::String(s)) => s,
                _ => return Err(runtime_error("io load: expected path string")),
            };
            let data = std::fs::read(path)
                .map_err(|e| runtime_error(format!("io load '{}': {}", path, e)))?;
            // Detect format by extension first, then by magic bytes
            if path.ends_with(".safetensors") {
                return super::safetensors::load_safetensors(&data);
            }
            if data.first() == Some(&0x80) {
                return super::pickle::load_pickle_bytes(&data);
            }
            let json: serde_json::Value = serde_json::from_slice(&data)
                .map_err(|e| runtime_error(format!("io load: parse '{}': {}", path, e)))?;
            json_to_value(&json)
        }
        _ => Err(runtime_error(format!("io: unknown verb '{}'", verb))),
    }
}

fn value_to_json(val: &Value) -> Result<serde_json::Value, crate::core::error::SheafError> {
    match val {
        Value::Int(n) => Ok(serde_json::Value::Number((*n).into())),
        Value::Float(f) => Ok(serde_json::json!(*f)),
        Value::Bool(b) => Ok(serde_json::Value::Bool(*b)),
        Value::Nil => Ok(serde_json::Value::Null),
        Value::String(s) => Ok(serde_json::Value::String(s.clone())),
        Value::Keyword(k) => Ok(serde_json::json!({"__keyword": k})),
        Value::List(items) => {
            let arr: Result<Vec<_>, _> = items.iter().map(value_to_json).collect();
            Ok(serde_json::Value::Array(arr?))
        }
        Value::Tuple(items) => {
            let arr: Result<Vec<_>, _> = items.iter().map(value_to_json).collect();
            Ok(serde_json::json!({"__tuple": arr?}))
        }
        Value::Dict(map) => {
            let mut obj = serde_json::Map::new();
            for (k, v) in map {
                obj.insert(k.clone(), value_to_json(v)?);
            }
            Ok(serde_json::Value::Object(obj))
        }
        Value::Tensor { data, dtype } => {
            let shape: Vec<usize> = data.shape().to_vec();
            let flat: Vec<f32> = data.iter().copied().collect();
            let dtype_str = match dtype {
                Dtype::F32 => "f32",
                Dtype::I32 => "i32",
                Dtype::Bool => "bool",
            };
            Ok(serde_json::json!({
                "__tensor": true,
                "shape": shape,
                "dtype": dtype_str,
                "data": flat,
            }))
        }
        Value::DeviceBuffer(db) => {
            let data = db.to_host().map_err(|e| runtime_error(format!("io save: {}", e)))?;
            value_to_json(&Value::Tensor { data: std::sync::Arc::new(data), dtype: *&db.dtype })
        }
        Value::Function { .. } | Value::BuiltinFn { .. } => {
            Err(runtime_error("io save: cannot serialize functions"))
        }
    }
}

fn json_to_value(json: &serde_json::Value) -> Result<Value, crate::core::error::SheafError> {
    match json {
        serde_json::Value::Null => Ok(Value::Nil),
        serde_json::Value::Bool(b) => Ok(Value::Bool(*b)),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Ok(Value::Int(i))
            } else if let Some(f) = n.as_f64() {
                Ok(Value::Float(f as f32))
            } else {
                Err(runtime_error("io load: invalid number"))
            }
        }
        serde_json::Value::String(s) => Ok(Value::String(s.clone())),
        serde_json::Value::Array(arr) => {
            let items: Result<Vec<_>, _> = arr.iter().map(json_to_value).collect();
            Ok(Value::List(items?))
        }
        serde_json::Value::Object(obj) => {
            if obj.contains_key("__tensor") {
                let shape: Vec<usize> = obj.get("shape")
                    .and_then(|v| v.as_array())
                    .map(|a| a.iter().filter_map(|x| x.as_u64().map(|n| n as usize)).collect())
                    .unwrap_or_default();
                let flat: Vec<f32> = obj.get("data")
                    .and_then(|v| v.as_array())
                    .map(|a| a.iter().filter_map(|x| x.as_f64().map(|v| v as f32)).collect())
                    .unwrap_or_default();
                let dtype = match obj.get("dtype").and_then(|v| v.as_str()) {
                    Some("i32") => Dtype::I32,
                    Some("bool") => Dtype::Bool,
                    _ => Dtype::F32,
                };
                let arr = ArrayD::from_shape_vec(IxDyn(&shape), flat)
                    .map_err(|e| runtime_error(format!("io load: tensor reshape: {}", e)))?;
                Ok(Value::Tensor { data: Arc::new(arr), dtype })
            } else if let Some(items) = obj.get("__tuple") {
                let arr = items.as_array()
                    .ok_or_else(|| runtime_error("io load: __tuple must be array"))?;
                let vals: Result<Vec<_>, _> = arr.iter().map(json_to_value).collect();
                Ok(Value::Tuple(vals?))
            } else if let Some(kw) = obj.get("__keyword") {
                let s = kw.as_str()
                    .ok_or_else(|| runtime_error("io load: __keyword must be string"))?;
                Ok(Value::Keyword(s.to_string()))
            } else {
                let mut map = BTreeMap::new();
                for (k, v) in obj {
                    map.insert(k.clone(), json_to_value(v)?);
                }
                Ok(Value::Dict(map))
            }
        }
    }
}

fn builtin_gensym(args: &[Value], _kw: &BTreeMap<String, Value>) -> R {
    let prefix = if args.is_empty() {
        "g".to_string()
    } else {
        match &args[0] {
            Value::String(s) => s.clone(),
            _ => format!("{}", args[0]),
        }
    };
    use std::time::{SystemTime, UNIX_EPOCH};
    let t = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default();
    let hash = format!("{:08x}", (t.as_nanos() & 0xFFFFFFFF) as u32);
    Ok(Value::String(format!("{}{}", prefix, hash)))
}

fn builtin_symbol_q(args: &[Value], _kw: &BTreeMap<String, Value>) -> R {
    if args.is_empty() { return Err(runtime_error("symbol? requires 1 argument")); }
    match &args[0] {
        Value::String(_) => Ok(Value::Bool(true)),
        _ => Ok(Value::Bool(false)),
    }
}

