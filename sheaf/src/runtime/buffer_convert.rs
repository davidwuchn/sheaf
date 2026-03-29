#![allow(dead_code)]

use std::collections::BTreeMap;
use std::ffi::c_void;
use std::sync::Arc;

use crate::core::error::SheafError;
use crate::interpreter::value::{Dtype, Value};
use crate::runtime::iree_ffi::*;
use ndarray::ArrayD;

use super::device_buffer::libc_stderr;

pub(super) unsafe fn value_to_buffer_view(
    device: *mut iree_hal_device_t,
    allocator: *mut iree_hal_allocator_t,
    val: &Value,
) -> Result<*mut iree_hal_buffer_view_t, SheafError> {
    unsafe {
        match val {
            Value::Tensor { data, dtype } => {
                let shape: Vec<iree_hal_dim_t> =
                    data.shape().iter().map(|&d| d as iree_hal_dim_t).collect();

                let f32_slice = data.as_slice().unwrap();
                let (byte_data, element_type) = match dtype {
                    Dtype::BF16 => {
                        // f32 -> bf16: truncate to top 16 bits
                        let bytes: Vec<u8> = f32_slice
                            .iter()
                            .flat_map(|f| {
                                let bits = f.to_bits();
                                let bf16_bits = (bits >> 16) as u16;
                                bf16_bits.to_ne_bytes()
                            })
                            .collect();
                        (bytes, IREE_HAL_ELEMENT_TYPE_BFLOAT_16)
                    }
                    _ => {
                        let bytes: Vec<u8> = f32_slice
                            .iter()
                            .flat_map(|f| f.to_ne_bytes())
                            .collect();
                        (bytes, IREE_HAL_ELEMENT_TYPE_FLOAT_32)
                    }
                };

                let params = iree_hal_buffer_params_t {
                    usage: 3 | 3072,   // TRANSFER | DISPATCH_STORAGE
                    access: 7,         // ALL (read|write|discard)
                    type_: 50,         // DEVICE_LOCAL | HOST_VISIBLE
                    queue_affinity: IREE_HAL_QUEUE_AFFINITY_ANY,
                    min_alignment: 0,
                };

                let span = iree_const_byte_span_t {
                    data: byte_data.as_ptr(),
                    data_length: byte_data.len(),
                };

                let mut bv: *mut iree_hal_buffer_view_t = std::ptr::null_mut();
                let status = iree_hal_buffer_view_allocate_buffer_copy(
                    device,
                    allocator,
                    shape.len(),
                    shape.as_ptr(),
                    element_type,
                    IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR,
                    params,
                    span,
                    &mut bv,
                );
                if !iree_status_is_ok(status) {
                    return Err(iree_err("failed to allocate IREE buffer view"));
                }
                Ok(bv)
            }
            Value::Float(f) => {
                let tensor = Value::Tensor {
                    data: Arc::new(ArrayD::from_elem(vec![], *f)),
                    dtype: Dtype::F32,
                };
                value_to_buffer_view(device, allocator, &tensor)
            }
            Value::Int(n) => {
                let tensor = Value::Tensor {
                    data: Arc::new(ArrayD::from_elem(vec![], *n as f32)),
                    dtype: Dtype::F32,
                };
                value_to_buffer_view(device, allocator, &tensor)
            }
            Value::Bool(b) => {
                let tensor = Value::Tensor {
                    data: Arc::new(ArrayD::from_elem(vec![], if *b { 1.0f32 } else { 0.0f32 })),
                    dtype: Dtype::F32,
                };
                value_to_buffer_view(device, allocator, &tensor)
            }
            Value::DeviceBuffer(db) => {
                // Already an IREE buffer view, pass through directly.
                // Caller must not cache/release this pointer (not owned by cache).
                Ok(db.buffer_view())
            }
            _ => Err(iree_err(&format!(
                "cannot convert {} to tensor buffer",
                val.type_name()
            ))),
        }
    }
}

pub(super) unsafe fn buffer_view_to_value(
    device: *mut iree_hal_device_t,
    bv: *mut iree_hal_buffer_view_t,
) -> Result<Value, SheafError> {
    unsafe {
        let rank = iree_hal_buffer_view_shape_rank(bv);
        let shape: Vec<usize> = (0..rank)
            .map(|i| iree_hal_buffer_view_shape_dim(bv, i) as usize)
            .collect();
        let elem_type = iree_hal_buffer_view_element_type(bv);
        let n_elems: usize = shape.iter().product::<usize>().max(1);

        let buf = iree_hal_buffer_view_buffer(bv);

        let (f32_data, dtype) = if elem_type == IREE_HAL_ELEMENT_TYPE_FLOAT_32 {
            let byte_len = n_elems * 4;
            let mut f32_buf: Vec<f32> = vec![0.0; n_elems];
            let status = iree_hal_device_transfer_d2h(
                device, buf, 0,
                f32_buf.as_mut_ptr() as *mut c_void,
                byte_len as u64, 0, iree_timeout_t::infinite(),
            );
            if !iree_status_is_ok(status) {
                iree_status_fprint(libc_stderr(), status);
                return Err(iree_err("failed to read IREE buffer data"));
            }
            (f32_buf, Dtype::F32)
        } else if elem_type == IREE_HAL_ELEMENT_TYPE_BFLOAT_16 {
            let byte_len = n_elems * 2;
            let mut raw: Vec<u16> = vec![0; n_elems];
            let status = iree_hal_device_transfer_d2h(
                device, buf, 0,
                raw.as_mut_ptr() as *mut c_void,
                byte_len as u64, 0, iree_timeout_t::infinite(),
            );
            if !iree_status_is_ok(status) {
                iree_status_fprint(libc_stderr(), status);
                return Err(iree_err("failed to read IREE buffer data (bf16)"));
            }
            let f32_buf: Vec<f32> = raw.iter()
                .map(|&bits| f32::from_bits((bits as u32) << 16))
                .collect();
            (f32_buf, Dtype::BF16)
        } else if elem_type == IREE_HAL_ELEMENT_TYPE_INT_32 {
            let byte_len = n_elems * 4;
            let mut i32_buf: Vec<i32> = vec![0; n_elems];
            let status = iree_hal_device_transfer_d2h(
                device, buf, 0,
                i32_buf.as_mut_ptr() as *mut c_void,
                byte_len as u64, 0, iree_timeout_t::infinite(),
            );
            if !iree_status_is_ok(status) {
                iree_status_fprint(libc_stderr(), status);
                return Err(iree_err("failed to read IREE buffer data"));
            }
            (i32_buf.iter().map(|&x| x as f32).collect::<Vec<f32>>(), Dtype::I32)
        } else {
            return Err(iree_err(&format!(
                "unsupported tensor element type: 0x{:08x}",
                elem_type
            )));
        };

        let data = ArrayD::from_shape_vec(shape, f32_data)
            .map_err(|e| iree_err(&format!("shape mismatch: {}", e)))?;

        Ok(Value::Tensor {
            data: Arc::new(data),
            dtype,
        })
    }
}

/// Flatten a list of values into individual tensor leaf references.
/// Dicts are sorted by key (matching codegen convention), then recursed.
/// Tuples are recursed. Scalars/tensors pass through.
/// Returns references to avoid cloning tensor data.
pub(super) fn flatten_values<'a>(inputs: &'a [Value]) -> Result<Vec<&'a Value>, SheafError> {
    let mut flat = Vec::new();
    for val in inputs {
        flatten_value(val, &mut flat)?;
    }
    Ok(flat)
}

fn flatten_value<'a>(val: &'a Value, out: &mut Vec<&'a Value>) -> Result<(), SheafError> {
    match val {
        Value::Dict(map) => {
            // Keys are already sorted (BTreeMap)
            for v in map.values() {
                flatten_value(v, out)?;
            }
            Ok(())
        }
        Value::Tuple(elems) | Value::List(elems) => {
            for v in elems {
                flatten_value(v, out)?;
            }
            Ok(())
        }
        Value::Tensor { .. } | Value::Float(_) | Value::Int(_) | Value::Bool(_)
        | Value::DeviceBuffer(_) => {
            out.push(val);
            Ok(())
        }
        _ => Err(iree_err(&format!(
            "cannot flatten {} for function call",
            val.type_name()
        ))),
    }
}

/// Reconstruct a nested Value from a flat list of tensor Values,
/// guided by a StableHLOType structure.
pub(super) fn unflatten_value(
    ty: &crate::lowering::stablehlo::StableHLOType,
    flat: &[Value],
    cursor: &mut usize,
) -> Result<Value, SheafError> {
    use crate::lowering::stablehlo::StableHLOType;
    match ty {
        StableHLOType::Tuple(elem_tys, keys) => {
            let mut elems = Vec::new();
            for elem_ty in elem_tys {
                elems.push(unflatten_value(elem_ty, flat, cursor)?);
            }
            if let Some(key_names) = keys {
                let map: BTreeMap<String, Value> = key_names
                    .iter()
                    .zip(elems)
                    .map(|(k, v)| (k.clone(), v))
                    .collect();
                Ok(Value::Dict(map))
            } else {
                Ok(Value::Tuple(elems))
            }
        }
        _ => {
            if *cursor < flat.len() {
                let val = flat[*cursor].clone();
                *cursor += 1;
                Ok(val)
            } else {
                Err(iree_err("not enough IREE outputs to reconstruct tuple structure"))
            }
        }
    }
}

pub(crate) fn iree_err(msg: &str) -> SheafError {
    SheafError::Runtime {
        message: msg.to_string(),
        location: None,
    }
}
