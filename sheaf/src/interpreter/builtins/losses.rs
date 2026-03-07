use super::*;
use std::sync::Arc;

pub(super) fn register(env: &mut Env) {
    env.set_builtin("mse-loss", builtin_mse_loss);
    env.set_builtin("mae-loss", builtin_mae_loss);
    env.set_builtin("sparse-cross-entropy", builtin_sparse_cross_entropy);
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
    if args.is_empty() { return Err(runtime_error("tree-map-zeros requires 1 argument")); }
    Ok(tree_zeros(&args[0]))
}

fn builtin_mse_loss(args: &[Value], _kw: &BTreeMap<String, Value>) -> R {
    let (pred, _) = to_array(&args[0])?;
    let (target, _) = to_array(&args[1])?;
    let diff = &pred - &target;
    let mse_f64 = diff.iter().map(|&x| x * x).sum::<f64>() / pred.len() as f64;
    Ok(Value::Float((mse_f64 as f32) as f64))
}

fn builtin_mae_loss(args: &[Value], _kw: &BTreeMap<String, Value>) -> R {
    let (pred, _) = to_array(&args[0])?;
    let (target, _) = to_array(&args[1])?;
    let diff = &pred - &target;
    let mae_f64 = diff.iter().map(|&x| x.abs()).sum::<f64>() / pred.len() as f64;
    Ok(Value::Float((mae_f64 as f32) as f64))
}

fn builtin_sparse_cross_entropy(args: &[Value], _kw: &BTreeMap<String, Value>) -> R {
    let (logits, _) = to_array(&args[0])?;
    let (labels, _) = to_array(&args[1])?;
    if logits.ndim() != 2 {
        return Err(runtime_error("sparse-cross-entropy: logits must be 2D [batch, classes]"));
    }
    let batch = logits.shape()[0];
    let num_classes = logits.shape()[1];
    let mut total_loss = 0.0f64;
    for i in 0..batch {
        let class_idx = labels[IxDyn(&[i])] as usize;
        if class_idx >= num_classes {
            return Err(runtime_error(format!(
                "sparse-cross-entropy: label {} out of range [0, {})", class_idx, num_classes
            )));
        }
        let row: Vec<f64> = (0..num_classes).map(|j| logits[IxDyn(&[i, j])]).collect();
        let max_val = row.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let shifted: Vec<f64> = row.iter().map(|&x| x - max_val).collect();
        let log_sum = shifted.iter().map(|&x| x.exp()).sum::<f64>().ln();
        let log_prob = shifted[class_idx] - log_sum;
        total_loss += -log_prob;
    }
    let mean_loss = total_loss / batch as f64;
    Ok(Value::Float((mean_loss as f32) as f64))
}
