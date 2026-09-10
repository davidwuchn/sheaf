#![allow(dead_code)]

use std::sync::Arc;

use crate::interpreter::value::{Dtype, Value};
use crate::runtime::iree_ffi::*;
use super::device_buffer::DeviceBufferInner;

/// Identity-based fingerprint for buffer view caching.
/// Uses Arc identity for tensors (O(1), no false positives).
/// Same Arc = same data = cache hit. New Arc = miss (correct for computed values).
/// Stores a clone of the Arc to keep it alive and prevent address reuse (ABA problem).
#[derive(Clone)]
pub(super) enum TensorFingerprint {
    Tensor {
        data: Arc<ndarray::ArrayD<f32>>,
        dtype: Dtype,
    },
    DeviceBuffer(Arc<DeviceBufferInner>),
    Float(u32),
    Int(i64),
    Bool(bool),
}

impl TensorFingerprint {
    pub(super) fn from_value(val: &Value) -> Option<Self> {
        match val {
            Value::Tensor { data, dtype } => Some(Self::Tensor {
                data: Arc::clone(data),
                dtype: *dtype,
            }),
            Value::DeviceBuffer(db) => Some(Self::DeviceBuffer(Arc::clone(db))),
            Value::Float(f) => Some(Self::Float(f.to_bits())),
            Value::Int(n) => Some(Self::Int(*n)),
            Value::Bool(b) => Some(Self::Bool(*b)),
            _ => None,
        }
    }

    pub(super) fn matches(&self, val: &Value) -> bool {
        match (self, val) {
            (
                Self::Tensor {
                    data: cached,
                    dtype: cached_dtype,
                },
                Value::Tensor { data, dtype },
            ) => Arc::ptr_eq(cached, data) && cached_dtype == dtype,
            (Self::DeviceBuffer(cached), Value::DeviceBuffer(db)) => Arc::ptr_eq(cached, db),
            (Self::Float(cached), Value::Float(value)) => *cached == value.to_bits(),
            (Self::Int(cached), Value::Int(value)) => cached == value,
            (Self::Bool(cached), Value::Bool(value)) => cached == value,
            _ => false,
        }
    }
}

pub(super) struct CachedBufferView {
    pub(super) fingerprint: TensorFingerprint,
    pub(super) bv: *mut iree_hal_buffer_view_t,
}

pub(super) struct BufferViewCache {
    entries: Vec<CachedBufferView>,
    capacity: usize,
}

impl BufferViewCache {
    pub(super) fn new(capacity: usize) -> Self {
        assert!(capacity > 0, "buffer cache capacity must be positive");
        Self {
            entries: Vec::new(),
            capacity,
        }
    }

    pub(super) fn get(&mut self, value: &Value) -> Option<*mut iree_hal_buffer_view_t> {
        let index = self
            .entries
            .iter()
            .position(|entry| entry.fingerprint.matches(value))?;
        let entry = self.entries.remove(index);
        let buffer_view = entry.bv;
        self.entries.insert(0, entry);
        Some(buffer_view)
    }

    pub(super) fn insert(
        &mut self,
        fingerprint: TensorFingerprint,
        buffer_view: *mut iree_hal_buffer_view_t,
    ) -> Option<*mut iree_hal_buffer_view_t> {
        let evicted = if self.entries.len() >= self.capacity {
            self.entries.pop().map(|entry| entry.bv)
        } else {
            None
        };
        self.entries.insert(0, CachedBufferView {
            fingerprint,
            bv: buffer_view,
        });
        evicted
    }

    pub(super) fn buffer_views(&self) -> impl Iterator<Item = *mut iree_hal_buffer_view_t> + '_ {
        self.entries.iter().map(|entry| entry.bv)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{ArrayD, IxDyn};

    fn tensor(data: &Arc<ArrayD<f32>>, dtype: Dtype) -> Value {
        Value::Tensor {
            data: Arc::clone(data),
            dtype,
        }
    }

    fn buffer_view(address: usize) -> *mut iree_hal_buffer_view_t {
        address as *mut iree_hal_buffer_view_t
    }

    #[test]
    fn tensor_fingerprint_requires_identity_and_dtype() {
        let data = Arc::new(ArrayD::zeros(IxDyn(&[2])));
        let same = tensor(&data, Dtype::F32);
        let other = Value::Tensor {
            data: Arc::new(ArrayD::zeros(IxDyn(&[2]))),
            dtype: Dtype::F32,
        };
        let different_dtype = tensor(&data, Dtype::F16);
        let fingerprint = TensorFingerprint::from_value(&same).unwrap();

        assert!(fingerprint.matches(&same));
        assert!(!fingerprint.matches(&other));
        assert!(!fingerprint.matches(&different_dtype));
    }

    #[test]
    fn scalar_fingerprints_preserve_value_kinds() {
        let float = TensorFingerprint::from_value(&Value::Float(1.0)).unwrap();
        let int = TensorFingerprint::from_value(&Value::Int(1)).unwrap();
        let boolean = TensorFingerprint::from_value(&Value::Bool(true)).unwrap();

        assert!(float.matches(&Value::Float(1.0)));
        assert!(!float.matches(&Value::Int(1)));
        assert!(int.matches(&Value::Int(1)));
        assert!(!int.matches(&Value::Bool(true)));
        assert!(boolean.matches(&Value::Bool(true)));
    }

    #[test]
    fn cache_evicts_the_least_recently_used_entry() {
        let first_data = Arc::new(ArrayD::zeros(IxDyn(&[1])));
        let second_data = Arc::new(ArrayD::zeros(IxDyn(&[2])));
        let third_data = Arc::new(ArrayD::zeros(IxDyn(&[3])));
        let first = tensor(&first_data, Dtype::F32);
        let second = tensor(&second_data, Dtype::F32);
        let third = tensor(&third_data, Dtype::F32);
        let mut cache = BufferViewCache::new(2);

        assert_eq!(
            cache.insert(
                TensorFingerprint::from_value(&first).unwrap(),
                buffer_view(1),
            ),
            None
        );
        assert_eq!(
            cache.insert(
                TensorFingerprint::from_value(&second).unwrap(),
                buffer_view(2),
            ),
            None
        );
        assert_eq!(cache.get(&first), Some(buffer_view(1)));
        assert_eq!(
            cache.insert(
                TensorFingerprint::from_value(&third).unwrap(),
                buffer_view(3),
            ),
            Some(buffer_view(2))
        );
        assert_eq!(cache.get(&second), None);
        assert_eq!(cache.get(&first), Some(buffer_view(1)));
        assert_eq!(cache.get(&third), Some(buffer_view(3)));
    }

    #[test]
    #[should_panic(expected = "buffer cache capacity must be positive")]
    fn cache_rejects_zero_capacity() {
        let _ = BufferViewCache::new(0);
    }
}
