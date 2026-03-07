//! SafeTensors format loader.
//! Format: 8-byte LE header size + JSON header + raw tensor data.
//! Flat dot-separated keys are auto-nested into hierarchical dicts;
//! contiguous numeric siblings (0, 1, 2, ...) become Value::List.

use crate::core::error::SheafError;
use crate::interpreter::env::runtime_error;
use crate::interpreter::value::{Dtype, Value};
use ndarray::{ArrayD, IxDyn};
use std::collections::BTreeMap;
use std::sync::Arc;

pub fn load_safetensors(data: &[u8]) -> Result<Value, SheafError> {
    if data.len() < 8 {
        return Err(runtime_error("safetensors: file too small"));
    }

    let header_size = u64::from_le_bytes([
        data[0], data[1], data[2], data[3],
        data[4], data[5], data[6], data[7],
    ]) as usize;

    let header_end = 8 + header_size;
    if header_end > data.len() {
        return Err(runtime_error(format!(
            "safetensors: header size {} exceeds file length {}", header_size, data.len()
        )));
    }

    let header: serde_json::Map<String, serde_json::Value> =
        serde_json::from_slice(&data[8..header_end])
            .map_err(|e| runtime_error(format!("safetensors: invalid header: {}", e)))?;

    let tensor_data = &data[header_end..];
    let mut flat: BTreeMap<String, Value> = BTreeMap::new();

    for (key, info) in &header {
        if key == "__metadata__" {
            continue;
        }

        let obj = info.as_object()
            .ok_or_else(|| runtime_error(format!("safetensors: '{}' must be object", key)))?;

        let dtype_str = obj.get("dtype").and_then(|v| v.as_str())
            .ok_or_else(|| runtime_error(format!("safetensors: '{}' missing dtype", key)))?;

        let shape: Vec<usize> = obj.get("shape").and_then(|v| v.as_array())
            .ok_or_else(|| runtime_error(format!("safetensors: '{}' missing shape", key)))?
            .iter().filter_map(|x| x.as_u64().map(|n| n as usize)).collect();

        let offsets = obj.get("data_offsets").and_then(|v| v.as_array())
            .ok_or_else(|| runtime_error(format!("safetensors: '{}' missing data_offsets", key)))?;
        let start = offsets.first().and_then(|v| v.as_u64()).unwrap_or(0) as usize;
        let end = offsets.get(1).and_then(|v| v.as_u64()).unwrap_or(0) as usize;

        if end > tensor_data.len() {
            return Err(runtime_error(format!(
                "safetensors: '{}' offsets [{}, {}] exceed data length {}",
                key, start, end, tensor_data.len()
            )));
        }

        let (values, sheaf_dtype) = decode_raw(&tensor_data[start..end], dtype_str)
            .map_err(|msg| runtime_error(format!("safetensors: '{}': {}", key, msg)))?;

        let expected: usize = if shape.is_empty() { 1 } else { shape.iter().product() };
        if values.len() != expected {
            return Err(runtime_error(format!(
                "safetensors: '{}' shape {:?} expects {} elements, got {}",
                key, shape, expected, values.len()
            )));
        }

        let arr = ArrayD::from_shape_vec(IxDyn(&shape), values)
            .map_err(|e| runtime_error(format!("safetensors: '{}' reshape: {}", key, e)))?;

        flat.insert(key.clone(), Value::Tensor { data: Arc::new(arr), dtype: sheaf_dtype });
    }

    Ok(nest_flat_keys(flat))
}

fn decode_raw(raw: &[u8], dtype: &str) -> Result<(Vec<f64>, Dtype), String> {
    match dtype {
        "F32" => Ok((
            raw.chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f64)
                .collect(),
            Dtype::F32,
        )),
        "F64" => Ok((
            raw.chunks_exact(8)
                .map(|c| f64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
                .collect(),
            Dtype::F32,
        )),
        "F16" => Ok((
            raw.chunks_exact(2)
                .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])) as f64)
                .collect(),
            Dtype::F32,
        )),
        "BF16" => Ok((
            raw.chunks_exact(2)
                .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16) as f64)
                .collect(),
            Dtype::F32,
        )),
        "I32" => Ok((
            raw.chunks_exact(4)
                .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f64)
                .collect(),
            Dtype::I32,
        )),
        "I64" => Ok((
            raw.chunks_exact(8)
                .map(|c| i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]) as f64)
                .collect(),
            Dtype::I32,
        )),
        "BOOL" => Ok((
            raw.iter().map(|&b| if b != 0 { 1.0 } else { 0.0 }).collect(),
            Dtype::Bool,
        )),
        "U8" => Ok((
            raw.iter().map(|&b| b as f64).collect(),
            Dtype::I32,
        )),
        other => Err(format!("unsupported dtype '{}'", other)),
    }
}

fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) & 1) as u32;
    let exp = ((h >> 10) & 0x1F) as u32;
    let frac = (h & 0x3FF) as u32;
    if exp == 0 {
        if frac == 0 {
            f32::from_bits(sign << 31)
        } else {
            let mut e = 0u32;
            let mut f = frac;
            while f & 0x400 == 0 { f <<= 1; e += 1; }
            f &= 0x3FF;
            let e = 127 - 15 + 1 - e;
            f32::from_bits((sign << 31) | (e << 23) | (f << 13))
        }
    } else if exp == 31 {
        if frac == 0 { f32::from_bits((sign << 31) | (0xFF << 23)) } else { f32::NAN }
    } else {
        f32::from_bits((sign << 31) | ((exp + 127 - 15) << 23) | (frac << 13))
    }
}

enum TreeNode {
    Leaf(Value),
    Branch(BTreeMap<String, TreeNode>),
}

fn nest_flat_keys(flat: BTreeMap<String, Value>) -> Value {
    let mut root = TreeNode::Branch(BTreeMap::new());
    for (key, value) in flat {
        let segments: Vec<&str> = key.split('.').collect();
        insert_tree(&mut root, &segments, value);
    }
    tree_to_value(root)
}

fn insert_tree(node: &mut TreeNode, segs: &[&str], value: Value) {
    if segs.is_empty() {
        *node = TreeNode::Leaf(value);
        return;
    }
    if let TreeNode::Branch(children) = node {
        let child = children.entry(segs[0].to_string())
            .or_insert_with(|| TreeNode::Branch(BTreeMap::new()));
        insert_tree(child, &segs[1..], value);
    }
}

fn tree_to_value(node: TreeNode) -> Value {
    match node {
        TreeNode::Leaf(v) => v,
        TreeNode::Branch(children) => {
            let n = children.len();
            if n == 0 {
                return Value::Dict(BTreeMap::new());
            }
            // Contiguous numeric keys 0..n → Value::List
            let all_numeric = children.keys().all(|k| k.parse::<usize>().is_ok());
            if all_numeric {
                let max_idx = children.keys()
                    .filter_map(|k| k.parse::<usize>().ok())
                    .max()
                    .unwrap_or(0);
                if max_idx + 1 == n {
                    let mut items: Vec<(usize, TreeNode)> = children.into_iter()
                        .map(|(k, v)| (k.parse::<usize>().unwrap(), v))
                        .collect();
                    items.sort_by_key(|(idx, _)| *idx);
                    return Value::List(
                        items.into_iter().map(|(_, v)| tree_to_value(v)).collect()
                    );
                }
            }
            Value::Dict(
                children.into_iter()
                    .map(|(k, v)| (k, tree_to_value(v)))
                    .collect()
            )
        }
    }
}
