// Copyright (c) 2025 Damien Boureille
// Licensed under the MIT License.

//! Dictionary layout lowering.

use std::collections::BTreeMap;

use serde_json::Value as JsonValue;

use crate::lowering::stablehlo::StableHLOType;
use crate::core::expr::CompiledExpr;
use crate::core::error::{SheafError, SheafResult, SourceLocation};

pub type DictIndexMap = BTreeMap<Vec<String>, Vec<usize>>;
pub type ParamIndexMap = (String, DictIndexMap);
pub type ParamIndexMaps = Vec<ParamIndexMap>;

pub fn json_to_stablehlo_type(val: &JsonValue) -> SheafResult<StableHLOType> {
    match val {
        JsonValue::Object(map) => {
            let sorted: BTreeMap<&str, &JsonValue> =
                map.iter().map(|(k, v)| (k.as_str(), v)).collect();
            let elems: SheafResult<Vec<StableHLOType>> =
                sorted.values().map(|v| json_to_stablehlo_type(v)).collect();
            Ok(StableHLOType::Tuple(elems?, None))
        }
        JsonValue::Array(dims) => {
            if dims.is_empty() {
                Ok(StableHLOType::ScalarF32)
            } else {
                let shape: SheafResult<Vec<i64>> = dims
                    .iter()
                    .map(|d| {
                        d.as_i64().ok_or_else(|| SheafError::Compile {
                            message: format!("config: shape dimension must be integer, got {}", d),
                            location: SourceLocation::unknown(),
                        })
                    })
                    .collect();
                Ok(StableHLOType::f32_tensor(shape?))
            }
        }
        JsonValue::Number(_) => Ok(StableHLOType::ScalarF32),
        JsonValue::Bool(_) => Ok(StableHLOType::scalar_f32()),
        other => Err(SheafError::Compile {
            message: format!("config: unsupported JSON value {}", other),
            location: SourceLocation::unknown(),
        }),
    }
}

pub fn build_index_map(val: &JsonValue) -> DictIndexMap {
    let mut map = BTreeMap::new();
    build_index_map_rec(val, &[], &[], &mut map);
    map
}

fn build_index_map_rec(
    val: &JsonValue,
    path: &[String],
    indices: &[usize],
    map: &mut DictIndexMap,
) {
    if !path.is_empty() {
        map.insert(path.to_vec(), indices.to_vec());
    }
    if let JsonValue::Object(obj) = val {
        let sorted: BTreeMap<&str, &JsonValue> = obj.iter().map(|(k, v)| (k.as_str(), v)).collect();
        for (i, (key, child)) in sorted.iter().enumerate() {
            let mut child_path = path.to_vec();
            child_path.push(key.to_string());
            let mut child_indices = indices.to_vec();
            child_indices.push(i);
            build_index_map_rec(child, &child_path, &child_indices, map);
        }
    }
}

pub fn layout_to_index_map(layout: &crate::core::expr::ParamLayout) -> DictIndexMap {
    let mut map = BTreeMap::new();
    for field in &layout.fields {
        map.insert(field.path.clone(), field.tuple_index.clone());
        for depth in 1..field.path.len() {
            let prefix: Vec<String> = field.path[..depth].to_vec();
            let idx_prefix: Vec<usize> = field.tuple_index[..depth].to_vec();
            map.entry(prefix).or_insert(idx_prefix);
        }
    }
    map
}

pub fn lower_get_calls(
    expr: &CompiledExpr,
    param_name: &str,
    index_map: &DictIndexMap,
) -> CompiledExpr {
    if let Some(indices) = try_extract_get_chain(expr, param_name, index_map) {
        return CompiledExpr::GetTupleElement {
            param: param_name.to_string(),
            indices,
        };
    }
    expr.map_children(|e| lower_get_calls(e, param_name, index_map))
}

fn try_extract_get_chain(
    expr: &CompiledExpr,
    param_name: &str,
    index_map: &DictIndexMap,
) -> Option<Vec<usize>> {
    let path = extract_key_path(expr, param_name)?;
    index_map.get(&path).cloned()
}

fn extract_key_path(expr: &CompiledExpr, param_name: &str) -> Option<Vec<String>> {
    match expr {
        CompiledExpr::Symbol(name) if name == param_name => Some(vec![]),
        CompiledExpr::FunctionCall { name, args, .. } if name == "get" && args.len() >= 2 => {
            let mut path = extract_key_path(&args[0], param_name)?;
            for arg in &args[1..] {
                match arg {
                    CompiledExpr::Keyword(k) | CompiledExpr::String(k) => path.push(k.clone()),
                    _ => return None,
                }
            }
            Some(path)
        }
        CompiledExpr::FunctionCall { name, args, .. } if name == "get-in" && args.len() >= 2 => {
            let base_path = extract_key_path(&args[0], param_name)?;
            let keys = match &args[1] {
                CompiledExpr::Vector(elems) => {
                    let mut ks = Vec::new();
                    for elem in elems {
                        match elem {
                            CompiledExpr::Keyword(k) => ks.push(k.clone()),
                            _ => return None,
                        }
                    }
                    ks
                }
                _ => return None,
            };
            let mut path = base_path;
            path.extend(keys);
            Some(path)
        }
        _ => None,
    }
}
