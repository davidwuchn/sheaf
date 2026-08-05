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

fn decode_raw(raw: &[u8], dtype: &str) -> Result<(Vec<f32>, Dtype), String> {
    match dtype {
        "F32" => Ok((
            raw.chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect(),
            Dtype::F32,
        )),
        "F64" => Ok((
            raw.chunks_exact(8)
                .map(|c| f64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]) as f32)
                .collect(),
            Dtype::F32,
        )),
        "F16" => Ok((
            raw.chunks_exact(2)
                .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
                .collect(),
            Dtype::F32,
        )),
        "BF16" => Ok((
            raw.chunks_exact(2)
                .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
                .collect(),
            Dtype::F32,
        )),
        "I32" => Ok((
            raw.chunks_exact(4)
                .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f32)
                .collect(),
            Dtype::I32,
        )),
        "I64" => Ok((
            raw.chunks_exact(8)
                .map(|c| i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]) as f32)
                .collect(),
            Dtype::I32,
        )),
        "BOOL" => Ok((
            raw.iter().map(|&b| if b != 0 { 1.0 } else { 0.0 }).collect(),
            Dtype::Bool,
        )),
        "U8" => Ok((
            raw.iter().map(|&b| b as f32).collect(),
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
            // Contiguous numeric keys 0..n -> Value::List
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

/// Serialize `Value` to the safetensors format.
///
/// Nested values are flattened into dot-separated keys, while tensor leaves
/// store the raw tensor data.
///
/// This is the inverse of `load_safetensors` for tensor-containing values.
pub fn save_safetensors(value: &Value) -> Result<Vec<u8>, SheafError> {
    let mut entries: Vec<(String, TensorEntry)> = Vec::new();
    flatten_into(value, "", &mut entries)?;
    // Deterministic, sorted-by-key output (matches typical HF checkpoints).
    entries.sort_by(|a, b| a.0.cmp(&b.0));

    let mut header = serde_json::Map::new();
    let mut data_buf: Vec<u8> = Vec::new();
    for (key, entry) in &entries {
        let start = data_buf.len();
        data_buf.extend_from_slice(&entry.bytes);
        let end = data_buf.len();
        header.insert(
            key.clone(),
            serde_json::json!({
                "dtype": entry.dtype_str,
                "shape": entry.shape,
                "data_offsets": [start, end],
            }),
        );
    }

    let mut header_bytes = serde_json::to_vec(&serde_json::Value::Object(header))
        .map_err(|e| runtime_error(format!("safetensors: serialize header: {}", e)))?;

    // Pad header with spaces so the data section is 8-byte aligned (safetensors
    // convention; trailing whitespace is ignored by JSON parsers).
    while (8 + header_bytes.len()) % 8 != 0 {
        header_bytes.push(b' ');
    }

    let mut out = Vec::with_capacity(8 + header_bytes.len() + data_buf.len());
    out.extend_from_slice(&(header_bytes.len() as u64).to_le_bytes());
    out.extend_from_slice(&header_bytes);
    out.extend_from_slice(&data_buf);
    Ok(out)
}

struct TensorEntry {
    dtype_str: &'static str,
    shape: Vec<usize>,
    bytes: Vec<u8>,
}

fn prefixed(prefix: &str, seg: &str) -> String {
    if prefix.is_empty() {
        seg.to_string()
    } else {
        format!("{}.{}", prefix, seg)
    }
}

fn leaf_key(prefix: &str, fallback: &str) -> String {
    if prefix.is_empty() { fallback.to_string() } else { prefix.to_string() }
}

fn flatten_into(
    value: &Value,
    prefix: &str,
    out: &mut Vec<(String, TensorEntry)>,
) -> Result<(), SheafError> {
    match value {
        Value::Dict(map) => {
            for (k, v) in map {
                flatten_into(v, &prefixed(prefix, k), out)?;
            }
            Ok(())
        }
        Value::List(items) | Value::Tuple(items) => {
            for (i, v) in items.iter().enumerate() {
                flatten_into(v, &prefixed(prefix, &i.to_string()), out)?;
            }
            Ok(())
        }
        Value::Tensor { data, dtype } => {
            out.push((leaf_key(prefix, "tensor"), tensor_to_entry(data, *dtype)));
            Ok(())
        }
        Value::DeviceBuffer(db) => {
            let host = db
                .to_host()
                .map_err(|e| runtime_error(format!("safetensors: D2H: {}", e)))?;
            out.push((leaf_key(prefix, "tensor"), tensor_to_entry(&host, db.dtype)));
            Ok(())
        }
        // Scalars are stored as 0-d tensors so they survive a round trip.
        Value::Int(n) => {
            out.push((
                leaf_key(prefix, "value"),
                TensorEntry { dtype_str: "I32", shape: vec![], bytes: (*n as i32).to_le_bytes().to_vec() },
            ));
            Ok(())
        }
        Value::Float(f) => {
            out.push((
                leaf_key(prefix, "value"),
                TensorEntry { dtype_str: "F32", shape: vec![], bytes: f.to_le_bytes().to_vec() },
            ));
            Ok(())
        }
        Value::Bool(b) => {
            out.push((
                leaf_key(prefix, "value"),
                TensorEntry { dtype_str: "BOOL", shape: vec![], bytes: vec![if *b { 1 } else { 0 }] },
            ));
            Ok(())
        }
        Value::String(_) | Value::Keyword(_) | Value::Nil | Value::Function { .. } | Value::BuiltinFn { .. } => {
            Err(runtime_error(format!(
                "safetensors: cannot serialize {} (only tensors and nested dicts/lists)",
                value.type_name()
            )))
        }
    }
}

fn tensor_to_entry(data: &ArrayD<f32>, dtype: Dtype) -> TensorEntry {
    let shape: Vec<usize> = data.shape().to_vec();
    let flat: Vec<f32> = data.iter().copied().collect();
    let (dtype_str, bytes) = match dtype {
        Dtype::F32 => {
            let mut b = Vec::with_capacity(flat.len() * 4);
            for f in &flat {
                b.extend_from_slice(&f.to_le_bytes());
            }
            ("F32", b)
        }
        Dtype::BF16 => {
            let mut b = Vec::with_capacity(flat.len() * 2);
            for f in &flat {
                b.extend_from_slice(&f32_to_bf16_bits(*f).to_le_bytes());
            }
            ("BF16", b)
        }
        Dtype::I32 => {
            let mut b = Vec::with_capacity(flat.len() * 4);
            for f in &flat {
                b.extend_from_slice(&(*f as i32).to_le_bytes());
            }
            ("I32", b)
        }
        Dtype::Bool => (
            "BOOL",
            flat.iter().map(|f| if *f != 0.0 { 1u8 } else { 0u8 }).collect(),
        ),
    };
    TensorEntry { dtype_str, shape, bytes }
}

/// Round-to-nearest-even conversion f32 -> bf16 bit pattern.
fn f32_to_bf16_bits(f: f32) -> u16 {
    let bits = f.to_bits();
    let lsb = (bits >> 16) & 1;
    let rounding_bias = 0x7FFF + lsb;
    (bits.wrapping_add(rounding_bias) >> 16) as u16
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tensor(vals: &[f32], shape: &[usize]) -> Value {
        Value::Tensor {
            data: Arc::new(ArrayD::from_shape_vec(IxDyn(shape), vals.to_vec()).unwrap()),
            dtype: Dtype::F32,
        }
    }

    fn get<'a>(v: &'a Value, key: &str) -> &'a Value {
        match v {
            Value::Dict(m) => m.get(key).expect("missing key"),
            _ => panic!("expected dict"),
        }
    }

    fn at<'a>(v: &'a Value, idx: usize) -> &'a Value {
        match v {
            Value::List(items) => &items[idx],
            _ => panic!("expected list"),
        }
    }

    fn tdata(v: &Value) -> &ArrayD<f32> {
        match v {
            Value::Tensor { data, .. } => data,
            _ => panic!("expected tensor"),
        }
    }

    /// Checks that a nested pytree can be saved and loaded back correctly.
    /// Uses a GPT-like pytree {:head {:w [2 2]} :blocks [{:w} {:w}]}.
    #[test]
    fn save_load_roundtrip_nested() {
        let head = Value::Dict(
            [("w".to_string(), tensor(&[1.0, 2.0, 3.0, 4.0], &[2, 2]))]
                .into_iter()
                .collect(),
        );
        let block = |a: f32| -> Value {
            Value::Dict(
                [("w".to_string(), tensor(&[a, a + 1.0, a + 2.0, a + 3.0], &[2, 2]))]
                    .into_iter()
                    .collect(),
            )
        };
        let blocks = Value::List(vec![block(10.0), block(20.0)]);
        let root = Value::Dict(
            [("head".to_string(), head), ("blocks".to_string(), blocks)]
                .into_iter()
                .collect(),
        );

        let bytes = save_safetensors(&root).unwrap();
        assert!(bytes.len() > 8);
        let loaded = load_safetensors(&bytes).unwrap();

        assert_eq!(tdata(get(get(&loaded, "head"), "w")), tdata(get(get(&root, "head"), "w")));
        assert_eq!(
            tdata(get(at(get(&loaded, "blocks"), 0), "w")),
            tdata(get(at(get(&root, "blocks"), 0), "w"))
        );
        assert_eq!(
            tdata(get(at(get(&loaded, "blocks"), 1), "w")),
            tdata(get(at(get(&root, "blocks"), 1), "w"))
        );
    }

    /// Checks the output is well-formed: 8-byte LE header length, valid JSON, and the
    /// data section begins on an 8-byte boundary.
    #[test]
    fn save_emits_valid_layout() {
        let bytes = save_safetensors(&tensor(&[1.5, -2.25, 0.0, 7.0], &[4])).unwrap();
        let n = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
        assert!(8 + n <= bytes.len());
        let _: serde_json::Value = serde_json::from_slice(&bytes[8..8 + n]).unwrap();
        assert_eq!((8 + n) % 8, 0, "data section should be 8-byte aligned");
    }
}
