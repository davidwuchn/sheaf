#![allow(dead_code)]

use crate::interpreter::value::Value;

/// Count the flat tensor leaves expected by a compiled signature.
pub fn count_signature_tensors(types: &[crate::lowering::stablehlo::StableHLOType]) -> usize {
    types.iter().map(count_type_leaves).sum()
}

fn count_type_leaves(ty: &crate::lowering::stablehlo::StableHLOType) -> usize {
    match ty {
        crate::lowering::stablehlo::StableHLOType::Tuple(elems, _) => {
            elems.iter().map(count_type_leaves).sum()
        }
        _ => 1,
    }
}

/// Count the flat tensor leaves in runtime values.
pub fn count_arg_tensors(values: &[Value]) -> usize {
    values.iter().map(count_one_value).sum()
}

fn count_one_value(val: &Value) -> usize {
    match val {
        Value::Dict(map) => map.values().map(count_one_value).sum(),
        Value::Tuple(elems) => elems.iter().map(count_one_value).sum(),
        // List of all scalars -> single tensor f32[N] (matches value_to_stablehlo_type)
        // Empty list -> 0 leaves (consistent with flatten_value and trace.rs)
        Value::List(elems)
            if !elems.is_empty() && elems
                .iter()
                .all(|v| matches!(v, Value::Float(_) | Value::Int(_))) =>
        {
            1
        }
        Value::List(elems) => elems.iter().map(count_one_value).sum(),
        Value::Tensor { .. } | Value::Float(_) | Value::Int(_) | Value::Bool(_)
        | Value::DeviceBuffer(_) => 1,
        _ => 0,
    }
}

/// Check if runtime args are structurally compatible with a compiled signature.
pub fn args_match_signature(
    args: &[Value],
    param_types: &[crate::lowering::stablehlo::StableHLOType],
) -> bool {
    count_arg_tensors(args) == count_signature_tensors(param_types)
}

/// Validate that runtime arg shapes match the compiled signature shapes.
/// Returns `Ok(())` on match, or `Err(description)` with a human-readable
/// mismatch message suitable for display.
pub fn check_shapes_match(
    args: &[Value],
    param_types: &[crate::lowering::stablehlo::StableHLOType],
) -> Result<(), String> {
    let mut expected_shapes: Vec<Vec<i64>> = Vec::new();
    collect_leaf_shapes(param_types, &mut expected_shapes);

    let mut actual_shapes: Vec<Vec<i64>> = Vec::new();
    for val in args {
        collect_value_shapes(val, &mut actual_shapes);
    }

    if expected_shapes.len() != actual_shapes.len() {
        return Err(format!(
            "tensor count mismatch: expected {} but have {}",
            expected_shapes.len(),
            actual_shapes.len(),
        ));
    }

    for (i, (exp, act)) in expected_shapes.iter().zip(actual_shapes.iter()).enumerate() {
        if exp != act {
            let fmt = |s: &[i64]| -> String {
                if s.is_empty() { "scalar".to_string() }
                else { s.iter().map(|d| d.to_string()).collect::<Vec<_>>().join("x") }
            };
            return Err(format!(
                "input{} shape mismatch: expected {} but have {}",
                i, fmt(exp), fmt(act),
            ));
        }
    }

    Ok(())
}

fn collect_leaf_shapes(types: &[crate::lowering::stablehlo::StableHLOType], out: &mut Vec<Vec<i64>>) {
    use crate::lowering::stablehlo::StableHLOType;
    for ty in types {
        match ty {
            StableHLOType::Tuple(elems, _) => collect_leaf_shapes(elems, out),
            _ => out.push(ty.shape().to_vec()),
        }
    }
}

fn collect_value_shapes(val: &Value, out: &mut Vec<Vec<i64>>) {
    match val {
        Value::Dict(map) => {
            for v in map.values() {
                collect_value_shapes(v, out);
            }
        }
        Value::Tuple(elems) | Value::List(elems) => {
            for v in elems {
                collect_value_shapes(v, out);
            }
        }
        Value::Tensor { data, .. } => {
            out.push(data.shape().iter().map(|&d| d as i64).collect());
        }
        Value::DeviceBuffer(db) => {
            out.push(db.shape.iter().map(|&d| d as i64).collect());
        }
        Value::Float(_) | Value::Int(_) | Value::Bool(_) => {
            out.push(vec![]);
        }
        _ => {}
    }
}
