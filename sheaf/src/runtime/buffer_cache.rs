#![allow(dead_code)]

use std::collections::HashMap;
use std::sync::Arc;

use crate::interpreter::value::{Dtype, Value};
use crate::runtime::iree_ffi::*;
use super::device_buffer::DeviceBufferInner;

/// Identity-based fingerprint for buffer view caching.
/// Uses Arc identity for tensors (O(1), no false positives).
/// Same Arc = same data = cache hit. New Arc = miss (correct for computed values).
/// Stores a clone of the Arc to keep it alive and prevent address reuse (ABA problem).
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
enum BufferCacheKey {
    Tensor { address: usize, dtype: Dtype },
    DeviceBuffer(usize),
    Float(u32),
    Int(i64),
    Bool(bool),
}

impl BufferCacheKey {
    fn from_value(value: &Value) -> Option<Self> {
        match value {
            Value::Tensor { data, dtype } => Some(Self::Tensor {
                address: Arc::as_ptr(data) as usize,
                dtype: *dtype,
            }),
            Value::DeviceBuffer(buffer) => {
                Some(Self::DeviceBuffer(Arc::as_ptr(buffer) as usize))
            }
            Value::Float(value) => Some(Self::Float(value.to_bits())),
            Value::Int(value) => Some(Self::Int(*value)),
            Value::Bool(value) => Some(Self::Bool(*value)),
            _ => None,
        }
    }
}

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

    fn key(&self) -> BufferCacheKey {
        match self {
            Self::Tensor { data, dtype } => BufferCacheKey::Tensor {
                address: Arc::as_ptr(data) as usize,
                dtype: *dtype,
            },
            Self::DeviceBuffer(buffer) => {
                BufferCacheKey::DeviceBuffer(Arc::as_ptr(buffer) as usize)
            }
            Self::Float(value) => BufferCacheKey::Float(*value),
            Self::Int(value) => BufferCacheKey::Int(*value),
            Self::Bool(value) => BufferCacheKey::Bool(*value),
        }
    }
}

pub(super) struct CachedBufferView {
    pub(super) fingerprint: TensorFingerprint,
    pub(super) bv: *mut iree_hal_buffer_view_t,
    last_used: u64,
}

pub(super) struct BufferViewCache {
    entries: HashMap<BufferCacheKey, CachedBufferView>,
    capacity: usize,
    next_use: u64,
    evictions: u64,
}

impl BufferViewCache {
    pub(super) fn new(capacity: usize) -> Self {
        assert!(capacity > 0, "buffer cache capacity must be positive");
        Self {
            entries: HashMap::with_capacity(capacity),
            capacity,
            next_use: 0,
            evictions: 0,
        }
    }

    pub(super) fn ensure_capacity(&mut self, required: usize) {
        if required <= self.capacity {
            return;
        }
        let new_capacity = required.checked_next_power_of_two().unwrap_or(required);
        self.entries
            .reserve(new_capacity.saturating_sub(self.entries.len()));
        self.capacity = new_capacity;
    }

    fn take_use(&mut self) -> u64 {
        let current = self.next_use;
        self.next_use = self
            .next_use
            .checked_add(1)
            .expect("buffer cache use counter overflowed");
        current
    }

    pub(super) fn get(&mut self, value: &Value) -> Option<*mut iree_hal_buffer_view_t> {
        let key = BufferCacheKey::from_value(value)?;
        let last_used = self.take_use();
        let entry = self.entries.get_mut(&key)?;
        entry.last_used = last_used;
        Some(entry.bv)
    }

    pub(super) fn insert(
        &mut self,
        fingerprint: TensorFingerprint,
        buffer_view: *mut iree_hal_buffer_view_t,
    ) -> Option<*mut iree_hal_buffer_view_t> {
        let key = fingerprint.key();
        let last_used = self.take_use();

        if let Some(entry) = self.entries.get_mut(&key) {
            entry.fingerprint = fingerprint;
            entry.last_used = last_used;
            return Some(std::mem::replace(&mut entry.bv, buffer_view));
        }

        let evicted = if self.entries.len() >= self.capacity {
            let lru_key = self
                .entries
                .iter()
                .min_by_key(|(_, entry)| entry.last_used)
                .map(|(key, _)| *key)
                .expect("a full buffer cache must contain an entry");
            self.evictions += 1;
            self.entries.remove(&lru_key).map(|entry| entry.bv)
        } else {
            None
        };

        self.entries.insert(
            key,
            CachedBufferView {
                fingerprint,
                bv: buffer_view,
                last_used,
            },
        );
        evicted
    }

    pub(super) fn len(&self) -> usize {
        self.entries.len()
    }

    pub(super) fn capacity(&self) -> usize {
        self.capacity
    }

    pub(super) fn evictions(&self) -> u64 {
        self.evictions
    }

    pub(super) fn buffer_views(&self) -> impl Iterator<Item = *mut iree_hal_buffer_view_t> + '_ {
        self.entries.values().map(|entry| entry.bv)
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
    fn tensor_cache_keys_require_identity_and_dtype() {
        let data = Arc::new(ArrayD::zeros(IxDyn(&[2])));
        let same = tensor(&data, Dtype::F32);
        let other = Value::Tensor {
            data: Arc::new(ArrayD::zeros(IxDyn(&[2]))),
            dtype: Dtype::F32,
        };
        let different_dtype = tensor(&data, Dtype::F16);

        assert_eq!(
            BufferCacheKey::from_value(&same),
            BufferCacheKey::from_value(&same)
        );
        assert_ne!(
            BufferCacheKey::from_value(&same),
            BufferCacheKey::from_value(&other)
        );
        assert_ne!(
            BufferCacheKey::from_value(&same),
            BufferCacheKey::from_value(&different_dtype)
        );
    }

    #[test]
    fn scalar_cache_keys_preserve_value_kinds() {
        assert_eq!(
            BufferCacheKey::from_value(&Value::Float(1.0)),
            BufferCacheKey::from_value(&Value::Float(1.0))
        );
        assert_ne!(
            BufferCacheKey::from_value(&Value::Float(1.0)),
            BufferCacheKey::from_value(&Value::Int(1))
        );
        assert_ne!(
            BufferCacheKey::from_value(&Value::Int(1)),
            BufferCacheKey::from_value(&Value::Bool(true))
        );
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
        assert_eq!(cache.evictions(), 1);
        assert_eq!(cache.get(&second), None);
        assert_eq!(cache.get(&first), Some(buffer_view(1)));
        assert_eq!(cache.get(&third), Some(buffer_view(3)));
    }

    #[test]
    fn cache_grows_to_fit_a_complete_working_set() {
        let mut cache = BufferViewCache::new(2);

        cache.ensure_capacity(3);

        assert_eq!(cache.capacity, 4);
    }

    #[test]
    fn cache_keeps_a_working_set_larger_than_its_initial_capacity() {
        let values = (0..600)
            .map(|_| Value::Tensor {
                data: Arc::new(ArrayD::zeros(IxDyn(&[1]))),
                dtype: Dtype::F32,
            })
            .collect::<Vec<_>>();
        let mut cache = BufferViewCache::new(512);
        cache.ensure_capacity(values.len());

        for (index, value) in values.iter().enumerate() {
            assert_eq!(
                cache.insert(
                    TensorFingerprint::from_value(value).unwrap(),
                    buffer_view(index + 1),
                ),
                None
            );
        }

        assert_eq!(cache.capacity, 1024);
        assert_eq!(cache.entries.len(), values.len());
        for (index, value) in values.iter().enumerate() {
            assert_eq!(cache.get(value), Some(buffer_view(index + 1)));
        }
    }

    #[test]
    fn cache_replaces_an_existing_identity_without_growing() {
        let data = Arc::new(ArrayD::zeros(IxDyn(&[1])));
        let value = tensor(&data, Dtype::F32);
        let mut cache = BufferViewCache::new(2);

        assert_eq!(
            cache.insert(
                TensorFingerprint::from_value(&value).unwrap(),
                buffer_view(1),
            ),
            None
        );
        assert_eq!(
            cache.insert(
                TensorFingerprint::from_value(&value).unwrap(),
                buffer_view(2),
            ),
            Some(buffer_view(1))
        );
        assert_eq!(cache.entries.len(), 1);
        assert_eq!(cache.get(&value), Some(buffer_view(2)));
    }

    #[test]
    #[should_panic(expected = "buffer cache capacity must be positive")]
    fn cache_rejects_zero_capacity() {
        let _ = BufferViewCache::new(0);
    }
}
