use super::*;
use std::sync::Arc;

pub(super) fn register(env: &mut Env) {
    env.set_builtin("tree-map-zeros", builtin_tree_map_zeros);
}

pub fn tree_zeros(val: &Value) -> Value {
    match val {
        Value::Dict(map) => {
            Value::Dict(map.iter().map(|(k, v)| (k.clone(), tree_zeros(v))).collect())
        }
        Value::Tensor { data, dtype } => {
            Value::Tensor { data: Arc::new(ArrayD::zeros(data.raw_dim())), dtype: *dtype }
        }
        Value::Float(_) => Value::Float(0.0),
        Value::Int(_) => Value::Int(0),
        Value::List(items) => Value::List(items.iter().map(tree_zeros).collect()),
        other => other.clone(),
    }
}

fn builtin_tree_map_zeros(args: &[Value], _kw: &BTreeMap<String, Value>) -> R {
    if args.is_empty() { return Err(runtime_error(format!("tree-map-zeros: expected 1 argument, got {}", args.len()))); }
    Ok(tree_zeros(&args[0]))
}
