#![allow(dead_code)]

use std::sync::Arc;

use crate::interpreter::value::Value;
use crate::runtime::iree_ffi::*;
use super::device_buffer::DeviceBufferInner;

/// Identity-based fingerprint for buffer view caching.
/// Uses Arc identity for tensors (O(1), no false positives).
/// Same Arc = same data = cache hit. New Arc = miss (correct for computed values).
/// Stores a clone of the Arc to keep it alive and prevent address reuse (ABA problem).
#[derive(Clone)]
pub(super) enum TensorFingerprint {
    Tensor(Arc<ndarray::ArrayD<f32>>),
    DeviceBuffer(Arc<DeviceBufferInner>),
    Scalar(u64),
}

impl TensorFingerprint {
    pub(super) fn from_value(val: &Value) -> Option<Self> {
        match val {
            Value::Tensor { data, .. } => Some(Self::Tensor(Arc::clone(data))),
            Value::DeviceBuffer(db) => Some(Self::DeviceBuffer(Arc::clone(db))),
            Value::Float(f) => Some(Self::Scalar(f.to_bits() as u64)),
            Value::Int(n) => Some(Self::Scalar(*n as u64)),
            Value::Bool(b) => Some(Self::Scalar(*b as u64)),
            _ => None,
        }
    }

    pub(super) fn matches(&self, val: &Value) -> bool {
        match (self, val) {
            (Self::Tensor(cached), Value::Tensor { data, .. }) => Arc::ptr_eq(cached, data),
            (Self::DeviceBuffer(cached), Value::DeviceBuffer(db)) => Arc::ptr_eq(cached, db),
            (Self::Scalar(s), Value::Float(f)) => *s == (f.to_bits() as u64),
            (Self::Scalar(s), Value::Int(n)) => *s == (*n as u64),
            (Self::Scalar(s), Value::Bool(b)) => *s == (*b as u64),
            _ => false,
        }
    }
}

pub(super) struct CachedBufferView {
    pub(super) fingerprint: TensorFingerprint,
    pub(super) bv: *mut iree_hal_buffer_view_t,
}
