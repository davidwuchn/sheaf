#![allow(dead_code)]

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

/// Cached target backend string, set once on first IreeSession creation.
static CACHED_BACKEND: OnceLock<String> = OnceLock::new();

/// Cumulative number of attempts to construct an IREE session.
///
/// Incremented once for every call to `IreeSession::new()`, including calls
/// that fail. It never decreases.
static SESSION_CREATION_ATTEMPTS: AtomicUsize = AtomicUsize::new(0);

/// Number of successfully constructed IREE sessions that have not been
/// dropped. This never underflows.
static LIVE_SESSION_COUNT: AtomicUsize = AtomicUsize::new(0);

/// Returns the cumulative number of `IreeSession::new()` attempts.
pub fn session_creation_attempt_count() -> usize {
    SESSION_CREATION_ATTEMPTS.load(Ordering::Relaxed)
}

/// Returns the number of successfully constructed sessions that are live.
pub fn live_session_count() -> usize {
    LIVE_SESSION_COUNT.load(Ordering::Relaxed)
}

fn record_session_creation_attempt() {
    SESSION_CREATION_ATTEMPTS.fetch_add(1, Ordering::Relaxed);
}

fn record_live_session() {
    LIVE_SESSION_COUNT.fetch_add(1, Ordering::Relaxed);
}

fn record_session_drop() {
    let _ = LIVE_SESSION_COUNT.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |count| {
        count.checked_sub(1)
    });
}

static SHARED_SESSION: OnceLock<Arc<IreeSession>> = OnceLock::new();
static SHARED_SESSION_INIT: Mutex<()> = Mutex::new(());

/// Returns the lazily initialized process-wide IREE session.
pub fn shared_session() -> Result<Arc<IreeSession>, SheafError> {
    if let Some(session) = SHARED_SESSION.get() {
        return Ok(Arc::clone(session));
    }
    let _init = SHARED_SESSION_INIT.lock().map_err(|_| SheafError::Runtime {
        message: "IREE shared session initialization lock is poisoned".to_string(),
        location: None,
    })?;
    if let Some(session) = SHARED_SESSION.get() {
        return Ok(Arc::clone(session));
    }
    let session = Arc::new(IreeSession::new()?);
    SHARED_SESSION.set(Arc::clone(&session)).map_err(|_| SheafError::Runtime {
        message: "failed to publish IREE shared session".to_string(),
        location: None,
    })?;
    Ok(session)
}

/// Returns the shared session without initializing it.
pub fn initialized_shared_session() -> Option<Arc<IreeSession>> {
    SHARED_SESSION.get().map(Arc::clone)
}

use crate::core::error::SheafError;
use crate::sheaf_msg;
use crate::interpreter::value::{Dtype, Value};
use crate::runtime::iree_ffi::*;

use super::buffer_cache::{CachedBufferView, TensorFingerprint};
use super::buffer_convert::{
    buffer_view_to_value, flatten_values, iree_err, unflatten_value, value_to_buffer_view,
};
use super::device_buffer::{libc_stderr, suppress_stderr, restore_stderr};

// Re-export public types so external callers can still use iree_session::*
pub use super::device_buffer::{DeviceBufferInner, IreeDeviceHandle};
pub use super::signature::{
    args_match_signature, check_shapes_match, count_arg_tensors, count_signature_tensors,
};

type PrecompiledFunctionKey = (String, String);

#[derive(Default)]
struct PrecompiledModuleRegistry {
    modules: HashMap<PrecompiledFunctionKey, String>,
    module_names: HashSet<String>,
    reservations: HashMap<PrecompiledFunctionKey, String>,
}

impl PrecompiledModuleRegistry {
    fn reserve(&mut self, module_name: &str, keys: &[PrecompiledFunctionKey]) -> bool {
        if self.module_names.contains(module_name)
            || keys
                .iter()
                .any(|key| self.modules.contains_key(key) || self.reservations.contains_key(key))
        {
            return false;
        }

        self.module_names.insert(module_name.to_string());
        for key in keys {
            self.reservations
                .insert(key.clone(), module_name.to_string());
        }
        true
    }

    fn release(&mut self, module_name: &str) {
        self.module_names.remove(module_name);
        self.reservations
            .retain(|_, reserved_name| reserved_name != module_name);
    }

    fn publish(&mut self, module_name: &str, keys: &[PrecompiledFunctionKey]) -> bool {
        if keys
            .iter()
            .any(|key| self.reservations.get(key).map(String::as_str) != Some(module_name))
        {
            return false;
        }

        for key in keys {
            self.modules.insert(key.clone(), module_name.to_string());
        }
        self.reservations
            .retain(|_, reserved_name| reserved_name != module_name);
        true
    }

    fn module_for(&self, function_name: &str, body_hash: &str) -> Option<String> {
        self.modules
            .get(&(function_name.to_string(), body_hash.to_string()))
            .cloned()
    }
}

pub struct IreeSession {
    instance: *mut iree_runtime_instance_t,
    device_handle: Arc<IreeDeviceHandle>,
    session: *mut iree_runtime_session_t,
    /// Source buffers remain live because IREE may retain the input span after
    /// appending a bytecode module.
    _vmfb_data: Mutex<Vec<Vec<u8>>>,
    /// HAL driver name: "metal", "local-task", etc.
    driver_name: String,
    /// Per-function buffer view cache: fn_name -> per-position cached buffer views.
    /// Each position holds up to MAX_CACHE_ENTRIES entries to avoid thrashing when
    /// the same function is called with different weight sets (e.g. transformer layers).
    buffer_cache: Mutex<HashMap<String, Vec<Vec<CachedBufferView>>>>,
    precompiled_modules: Mutex<PrecompiledModuleRegistry>,
    /// Dispatch timing (nanoseconds, accumulated). Enabled by --jit-profile.
    profile: bool,
    t_flatten_ns: AtomicU64,
    t_buffers_ns: AtomicU64,
    t_call_ns: AtomicU64,
    t_output_ns: AtomicU64,
    n_calls: AtomicU64,
    n_cache_hits: AtomicU64,
    n_cache_misses: AtomicU64,
}

unsafe impl Send for IreeSession {}
unsafe impl Sync for IreeSession {}

impl IreeSession {
    pub fn new() -> Result<Self, SheafError> {
        record_session_creation_attempt();
        unsafe {
            let alloc = system_allocator();

            let mut opts: iree_runtime_instance_options_t = std::mem::zeroed();
            iree_runtime_instance_options_initialize(&mut opts);
            iree_runtime_instance_options_use_all_available_drivers(&mut opts);

            let mut instance: *mut iree_runtime_instance_t = std::ptr::null_mut();
            let status = iree_runtime_instance_create(&opts, alloc, &mut instance);
            if !iree_status_is_ok(status) {
                return Err(iree_err("failed to create IREE instance"));
            }

            // Try drivers in preference order: CUDA > Metal > Vulkan > CPU
            let device_override = crate::core::config::device_override();
            let driver_names: Vec<&str> = match device_override {
                Some("cpu") => vec!["local-task"],
                Some(d) => vec![d, "local-task"],
                None => vec!["cuda", "metal", "vulkan", "local-task"],
            };
            let mut device: *mut iree_hal_device_t = std::ptr::null_mut();
            let mut chosen_driver = "";
            for name in &driver_names {
                let driver = iree_string_view_t::from_utf8(name);
                let status =
                    iree_runtime_instance_try_create_default_device(instance, driver, &mut device);
                if iree_status_is_ok(status) {
                    chosen_driver = name;
                    break;
                } else if crate::core::config::verbosity() >= 2 {
                    sheaf_msg!("jit: driver '{}' not available", name);
                }
            }
            if device.is_null() {
                iree_runtime_instance_release(instance);
                let tried = driver_names.join(", ");
                return Err(iree_err(&format!(
                    "failed to create IREE device (tried: {})", tried
                )));
            }
            // Cache the target backend for JIT (avoids re-probing drivers)
            let backend = match chosen_driver {
                "metal" => "metal-spirv",
                "vulkan" => "vulkan-spirv",
                "cuda" => "cuda",
                _ => "llvm-cpu",
            };
            let _ = CACHED_BACKEND.set(backend.to_string());

            {
                use std::sync::atomic::AtomicBool;
                static PRINTED: AtomicBool = AtomicBool::new(false);
                if !PRINTED.swap(true, Ordering::Relaxed) {
                    let display = match chosen_driver {
                        "local-task" => "CPU".to_string(),
                        "metal" => "Metal GPU".to_string(),
                        "cuda" => match super::jit::detect_cuda_target() {
                            Some(target) => format!("CUDA GPU ({})", target),
                            None => "CUDA GPU".to_string(),
                        },
                        "vulkan" => "Vulkan GPU".to_string(),
                        other => other.to_string(),
                    };
                    sheaf_msg!("sheaf: running on {}", display);
                }
            }

            // Retain the device for our Arc handle (session also holds its own ref)
            iree_hal_device_retain(device);
            let device_handle = Arc::new(IreeDeviceHandle { device });

            let mut session_opts: iree_runtime_session_options_t = std::mem::zeroed();
            iree_runtime_session_options_initialize(&mut session_opts);

            let mut session: *mut iree_runtime_session_t = std::ptr::null_mut();
            let status = iree_runtime_session_create_with_device(
                instance,
                &session_opts,
                device,
                alloc,
                &mut session,
            );
            if !iree_status_is_ok(status) {
                drop(device_handle); // releases our retained ref
                iree_hal_device_release(device); // release the original ref
                iree_runtime_instance_release(instance);
                return Err(iree_err("failed to create IREE session"));
            }

            let result = IreeSession {
                instance,
                device_handle,
                session,
                _vmfb_data: Mutex::new(Vec::new()),
                driver_name: chosen_driver.to_string(),
                buffer_cache: Mutex::new(HashMap::new()),
                precompiled_modules: Mutex::new(PrecompiledModuleRegistry::default()),
                profile: crate::core::config::jit_profile(),
                t_flatten_ns: AtomicU64::new(0),
                t_buffers_ns: AtomicU64::new(0),
                t_call_ns: AtomicU64::new(0),
                t_output_ns: AtomicU64::new(0),
                n_calls: AtomicU64::new(0),
                n_cache_hits: AtomicU64::new(0),
                n_cache_misses: AtomicU64::new(0),
            };
            record_live_session();
            Ok(result)
        }
    }

    /// Returns the iree-compile target backend for the active HAL driver.
    pub fn target_backend(&self) -> &str {
        match self.driver_name.as_str() {
            "metal" => "metal-spirv",
            "vulkan" => "vulkan-spirv",
            "cuda" => "cuda",
            _ => "llvm-cpu",
        }
    }

    /// Returns the cached target backend without creating a new session.
    /// Available after the first IreeSession::new() call.
    pub fn cached_target_backend() -> Option<&'static str> {
        CACHED_BACKEND.get().map(|s| s.as_str())
    }

    /// Returns the HAL driver name ("metal", "local-task", etc.)
    pub fn driver_name(&self) -> &str {
        &self.driver_name
    }

    /// Returns a clone of the device handle for DeviceBuffer lifetime management.
    pub fn device_handle(&self) -> &Arc<IreeDeviceHandle> {
        &self.device_handle
    }

    /// Returns the raw IREE device allocator pointer for memory statistics queries.
    pub fn device_allocator_ptr(&self) -> *mut crate::runtime::iree_ffi::iree_hal_allocator_t {
        unsafe { iree_runtime_session_device_allocator(self.session) }
    }

    pub fn reserve_precompiled_module(
        &self,
        module_name: &str,
        keys: &[PrecompiledFunctionKey],
    ) -> Result<bool, SheafError> {
        let mut registry = self
            .precompiled_modules
            .lock()
            .map_err(|_| precompiled_registry_error())?;
        Ok(registry.reserve(module_name, keys))
    }

    pub fn release_precompiled_module(&self, module_name: &str) -> Result<(), SheafError> {
        let mut registry = self
            .precompiled_modules
            .lock()
            .map_err(|_| precompiled_registry_error())?;
        registry.release(module_name);
        Ok(())
    }

    pub fn register_precompiled_functions(
        &self,
        module_name: &str,
        keys: &[PrecompiledFunctionKey],
    ) -> Result<(), SheafError> {
        let mut registry = self
            .precompiled_modules
            .lock()
            .map_err(|_| precompiled_registry_error())?;
        if registry.publish(module_name, keys) {
            Ok(())
        } else {
            Err(SheafError::Runtime {
                message: format!(
                    "IREE precompiled module '{}' has no matching reservation",
                    module_name
                ),
                location: None,
            })
        }
    }

    pub fn precompiled_module_for(
        &self,
        function_name: &str,
        body_hash: &str,
    ) -> Result<Option<String>, SheafError> {
        let registry = self
            .precompiled_modules
            .lock()
            .map_err(|_| precompiled_registry_error())?;
        Ok(registry.module_for(function_name, body_hash))
    }

    pub fn load_vmfb(&self, data: Vec<u8>) -> Result<(), SheafError> {
        let mut vmfb_data = self._vmfb_data.lock().map_err(|_| SheafError::Runtime {
            message: "IREE VMFB retention lock is poisoned".to_string(),
            location: None,
        })?;
        vmfb_data.push(data);
        unsafe {
            let bytes = vmfb_data
                .last()
                .ok_or_else(|| iree_err("failed to retain VMFB source data"))?;
            let span = iree_const_byte_span_t::from_slice(bytes);
            let null_alloc = iree_allocator_t {
                self_: std::ptr::null_mut(),
                ctl: None,
            };
            // Suppress IREE C runtime diagnostics on stderr in non-verbose mode
            let suppress = crate::core::config::verbosity() < 2;
            let saved_stderr = if suppress { suppress_stderr() } else { None };
            let status = iree_runtime_session_append_bytecode_module_from_memory(
                self.session,
                span,
                null_alloc,
            );
            if let Some(fd) = saved_stderr { restore_stderr(fd); }
            if !iree_status_is_ok(status) {
                if crate::core::config::verbosity() >= 2 {
                    iree_status_fprint(libc_stderr(), status);
                }
                vmfb_data.pop();
                return Err(iree_err("failed to load compiled module"));
            }

            Ok(())
        }
    }

    pub fn call(&self, fn_name: &str, inputs: &[Value]) -> Result<Value, SheafError> {
        unsafe {
            let alloc = system_allocator();
            let device = iree_runtime_session_device(self.session);
            let device_alloc = iree_runtime_session_device_allocator(self.session);

            let t0 = if self.profile { Some(std::time::Instant::now()) } else { None };

            // Flatten tuples/dicts into individual tensor leaves for IREE
            let flat_inputs = flatten_values(inputs)?;

            let t1 = t0.map(|_| std::time::Instant::now());

            let mut input_list: *mut iree_vm_list_t = std::ptr::null_mut();
            let variant_type = iree_vm_type_def_t { value: 0 };
            let status =
                iree_vm_list_create(variant_type, flat_inputs.len(), alloc, &mut input_list);
            if !iree_status_is_ok(status) {
                return Err(iree_err("failed to create input list"));
            }

            // Build input buffer views with caching.
            // Each position holds up to 8 entries to avoid thrashing when the same
            // function is called with different weight sets (e.g. 6 transformer layers).
            const MAX_CACHE_ENTRIES: usize = 8;
            let mut cache = self.buffer_cache.lock().unwrap();
            let cached_fn = match cache.get_mut(fn_name) {
                Some(c) => c,
                None => cache.entry(fn_name.to_string()).or_default(),
            };
            if cached_fn.len() < flat_inputs.len() {
                cached_fn.resize_with(flat_inputs.len(), Vec::new);
            }

            for (i, val) in flat_inputs.iter().enumerate() {
                let hit_idx = cached_fn[i].iter().position(|entry| entry.fingerprint.matches(val));

                let bv = if let Some(idx) = hit_idx {
                    if self.profile { self.n_cache_hits.fetch_add(1, Ordering::Relaxed); }
                    if idx > 0 { cached_fn[i].swap(0, idx); }
                    cached_fn[i][0].bv
                } else {
                    if self.profile { self.n_cache_misses.fetch_add(1, Ordering::Relaxed); }
                    let new_bv = match value_to_buffer_view(device, device_alloc, val) {
                        Ok(view) => view,
                        Err(error) => {
                            // The VM list owns retained references for earlier arguments.
                            iree_vm_list_release(input_list);
                            return Err(error);
                        }
                    };
                    if let Some(fp) = TensorFingerprint::from_value(val) {
                        if cached_fn[i].len() >= MAX_CACHE_ENTRIES {
                            let evicted = cached_fn[i].pop().unwrap();
                            iree_hal_buffer_view_release(evicted.bv);
                        }
                        cached_fn[i].insert(0, CachedBufferView { fingerprint: fp, bv: new_bv });
                    }
                    new_bv
                };

                // Create a VM ref for the input list (retains the buffer view)
                let mut ref_ = iree_hal_buffer_view_retain_ref(bv);
                let status = iree_vm_list_push_ref_retain(input_list, &ref_);
                iree_vm_ref_release(&mut ref_);
                if !iree_status_is_ok(status) {
                    iree_vm_list_release(input_list);
                    return Err(iree_err("failed to push input to list"));
                }
            }

            // Drop cache lock before the IREE call
            drop(cache);

            let t2 = t0.map(|_| std::time::Instant::now());

            let mut output_list: *mut iree_vm_list_t = std::ptr::null_mut();
            let status =
                iree_vm_list_create(variant_type, 16, alloc, &mut output_list);
            if !iree_status_is_ok(status) {
                iree_vm_list_release(input_list);
                return Err(iree_err("failed to create output list"));
            }

            let name = iree_string_view_t::from_utf8(fn_name);
            let status = iree_runtime_session_call_by_name(
                self.session,
                name,
                input_list,
                output_list,
            );
            iree_vm_list_release(input_list);
            if !iree_status_is_ok(status) {
                iree_vm_list_release(output_list);
                let clean_name = fn_name.strip_prefix("module.").unwrap_or(fn_name).replace('_', "-");
                let got = inputs.iter().map(|v| v.short_desc()).collect::<Vec<_>>().join(", ");
                return Err(iree_err(&format!(
                    "{}: shape mismatch (compiled signature does not match arguments).\n  Called with: ({})",
                    clean_name, got
                )));
            }

            let t3 = t0.map(|_| std::time::Instant::now());

            let n_outputs = iree_vm_list_size(output_list);
            let mut results = Vec::with_capacity(n_outputs);
            for i in 0..n_outputs {
                let mut ref_: iree_vm_ref_t = std::mem::zeroed();
                let status = iree_vm_list_get_ref_retain(output_list, i, &mut ref_);
                if !iree_status_is_ok(status) {
                    iree_vm_list_release(output_list);
                    return Err(iree_err("failed to get output from list"));
                }
                let bv = ref_.ptr as *mut iree_hal_buffer_view_t;
                let val = match buffer_view_to_value(device, bv) {
                    Ok(value) => value,
                    Err(error) => {
                        iree_vm_ref_release(&mut ref_);
                        iree_vm_list_release(output_list);
                        return Err(error);
                    }
                };
                iree_vm_ref_release(&mut ref_);
                results.push(val);
            }
            iree_vm_list_release(output_list);

            if let (Some(t0), Some(t1), Some(t2), Some(t3)) = (t0, t1, t2, t3) {
                let t4 = std::time::Instant::now();
                self.t_flatten_ns.fetch_add((t1 - t0).as_nanos() as u64, Ordering::Relaxed);
                self.t_buffers_ns.fetch_add((t2 - t1).as_nanos() as u64, Ordering::Relaxed);
                self.t_call_ns.fetch_add((t3 - t2).as_nanos() as u64, Ordering::Relaxed);
                self.t_output_ns.fetch_add((t4 - t3).as_nanos() as u64, Ordering::Relaxed);
                self.n_calls.fetch_add(1, Ordering::Relaxed);
            }

            match results.len() {
                0 => Ok(Value::Nil),
                1 => Ok(results.into_iter().next().unwrap()),
                _ => Ok(Value::Tuple(results)),
            }
        }
    }

    /// Call with a known return type to reconstruct nested tuple/dict structure
    /// from IREE's flattened output buffers.
    pub fn call_typed(
        &self,
        fn_name: &str,
        inputs: &[Value],
        return_type: &crate::lowering::stablehlo::StableHLOType,
    ) -> Result<Value, SheafError> {
        let flat_result = self.call(fn_name, inputs)?;
        // Unpack the flat result into the expected structure
        let flat_values = match flat_result {
            Value::Tuple(vals) => vals,
            other => vec![other],
        };
        let mut cursor = 0;
        let structured = unflatten_value(return_type, &flat_values, &mut cursor)?;
        Ok(structured)
    }

    /// Like call(), but returns DeviceBuffer values instead of host tensors.
    /// DeviceBuffer inputs are passed through without h2d copy.
    pub fn call_device(&self, fn_name: &str, inputs: &[Value]) -> Result<Value, SheafError> {
        unsafe {
            let alloc = system_allocator();
            let device = iree_runtime_session_device(self.session);
            let device_alloc = iree_runtime_session_device_allocator(self.session);

            let t0 = if self.profile { Some(std::time::Instant::now()) } else { None };

            let flat_inputs = flatten_values(inputs)?;

            let t1 = t0.map(|_| std::time::Instant::now());

            let mut input_list: *mut iree_vm_list_t = std::ptr::null_mut();
            let variant_type = iree_vm_type_def_t { value: 0 };
            let status =
                iree_vm_list_create(variant_type, flat_inputs.len(), alloc, &mut input_list);
            if !iree_status_is_ok(status) {
                return Err(iree_err("failed to create input list"));
            }

            // Build input buffer views: DeviceBuffers pass through, others use cache.
            let all_device = flat_inputs.iter().all(|v| matches!(v, Value::DeviceBuffer(_)));

            if all_device {
                // Fast path: all inputs already on device, skip cache entirely
                for val in flat_inputs.iter() {
                    if let Value::DeviceBuffer(db) = val {
                        if self.profile { self.n_cache_hits.fetch_add(1, Ordering::Relaxed); }
                        let mut ref_ = iree_hal_buffer_view_retain_ref(db.buffer_view());
                        let status = iree_vm_list_push_ref_retain(input_list, &ref_);
                        iree_vm_ref_release(&mut ref_);
                        if !iree_status_is_ok(status) {
                            iree_vm_list_release(input_list);
                            return Err(iree_err("failed to push input to list"));
                        }
                    }
                }
            } else {
                const MAX_CACHE_ENTRIES: usize = 8;
                let mut cache = self.buffer_cache.lock().unwrap();
                let cached_fn = cache.entry(fn_name.to_string()).or_default();
                if cached_fn.len() < flat_inputs.len() {
                    cached_fn.resize_with(flat_inputs.len(), Vec::new);
                }

                for (i, val) in flat_inputs.iter().enumerate() {
                    let bv = match val {
                        Value::DeviceBuffer(db) => {
                            if self.profile { self.n_cache_hits.fetch_add(1, Ordering::Relaxed); }
                            db.buffer_view()
                        }
                        _ => {
                            let hit_idx = cached_fn[i].iter().position(|entry| entry.fingerprint.matches(val));
                            if let Some(idx) = hit_idx {
                                if self.profile { self.n_cache_hits.fetch_add(1, Ordering::Relaxed); }
                                if idx > 0 { cached_fn[i].swap(0, idx); }
                                cached_fn[i][0].bv
                            } else {
                                if self.profile { self.n_cache_misses.fetch_add(1, Ordering::Relaxed); }
                                let new_bv = match value_to_buffer_view(device, device_alloc, val) {
                                    Ok(view) => view,
                                    Err(error) => {
                                        // The VM list owns retained references for earlier arguments.
                                        iree_vm_list_release(input_list);
                                        return Err(error);
                                    }
                                };
                                if let Some(fp) = TensorFingerprint::from_value(val) {
                                    if cached_fn[i].len() >= MAX_CACHE_ENTRIES {
                                        let evicted = cached_fn[i].pop().unwrap();
                                        iree_hal_buffer_view_release(evicted.bv);
                                    }
                                    cached_fn[i].insert(0, CachedBufferView { fingerprint: fp, bv: new_bv });
                                }
                                new_bv
                            }
                        }
                    };

                    let mut ref_ = iree_hal_buffer_view_retain_ref(bv);
                    let status = iree_vm_list_push_ref_retain(input_list, &ref_);
                    iree_vm_ref_release(&mut ref_);
                    if !iree_status_is_ok(status) {
                        iree_vm_list_release(input_list);
                        return Err(iree_err("failed to push input to list"));
                    }
                }

                drop(cache);
            }

            let t2 = t0.map(|_| std::time::Instant::now());

            let mut output_list: *mut iree_vm_list_t = std::ptr::null_mut();
            let status =
                iree_vm_list_create(variant_type, 16, alloc, &mut output_list);
            if !iree_status_is_ok(status) {
                iree_vm_list_release(input_list);
                return Err(iree_err("failed to create output list"));
            }

            let name = iree_string_view_t::from_utf8(fn_name);
            let status = iree_runtime_session_call_by_name(
                self.session,
                name,
                input_list,
                output_list,
            );
            iree_vm_list_release(input_list);
            if !iree_status_is_ok(status) {
                iree_vm_list_release(output_list);
                let clean_name = fn_name.strip_prefix("module.").unwrap_or(fn_name).replace('_', "-");
                let got = inputs.iter().map(|v| v.short_desc()).collect::<Vec<_>>().join(", ");
                return Err(iree_err(&format!(
                    "{}: shape mismatch (compiled signature does not match arguments).\n  Called with: ({})",
                    clean_name, got
                )));
            }

            let t3 = t0.map(|_| std::time::Instant::now());

            // Wrap outputs as DeviceBuffers instead of d2h transfer
            let n_outputs = iree_vm_list_size(output_list);
            let mut results = Vec::with_capacity(n_outputs);
            for i in 0..n_outputs {
                let mut ref_: iree_vm_ref_t = std::mem::zeroed();
                let status = iree_vm_list_get_ref_retain(output_list, i, &mut ref_);
                if !iree_status_is_ok(status) {
                    iree_vm_list_release(output_list);
                    return Err(iree_err("failed to get output from list"));
                }
                let bv = ref_.ptr as *mut iree_hal_buffer_view_t;

                // Read shape and dtype metadata (no data transfer)
                let rank = iree_hal_buffer_view_shape_rank(bv);
                let shape: Vec<usize> = (0..rank)
                    .map(|j| iree_hal_buffer_view_shape_dim(bv, j) as usize)
                    .collect();
                let elem_type = iree_hal_buffer_view_element_type(bv);
                let dtype = if elem_type == IREE_HAL_ELEMENT_TYPE_BFLOAT_16 {
                    Dtype::BF16
                } else if elem_type == IREE_HAL_ELEMENT_TYPE_INT_32 {
                    Dtype::I32
                } else {
                    Dtype::F32
                };

                // Eagerly transfer scalars to host. Avoids GPU sync in
                // interpreter hot loops (e.g. nth on generate-token output).
                if shape.is_empty() {
                    let buf = iree_hal_buffer_view_buffer(bv);
                    let byte_len = if dtype == Dtype::BF16 { 2u64 } else { 4u64 };
                    let mut raw = [0u8; 4];
                    let status = iree_hal_device_transfer_d2h(
                        self.device_handle.device,
                        buf,
                        0,
                        raw.as_mut_ptr() as *mut std::ffi::c_void,
                        byte_len,
                        0,
                        iree_timeout_t::infinite(),
                    );
                    iree_hal_buffer_view_release(bv);
                    ref_.ptr = std::ptr::null_mut();
                    iree_vm_ref_release(&mut ref_);
                    if iree_status_is_ok(status) {
                        let val = if dtype == Dtype::BF16 {
                            let bits = u16::from_le_bytes([raw[0], raw[1]]);
                            f32::from_bits((bits as u32) << 16)
                        } else {
                            f32::from_le_bytes(raw)
                        };
                        if dtype == Dtype::I32 {
                            results.push(Value::Int(val as i64));
                        } else {
                            results.push(Value::Float(val));
                        }
                    } else {
                        results.push(Value::Float(0.0));
                    }
                } else {
                    let db = Arc::new(DeviceBufferInner::new(
                        bv,
                        Arc::clone(&self.device_handle),
                        shape,
                        dtype,
                    ));
                    // Transfer ownership: nullify ptr to prevent double-free on ref release
                    ref_.ptr = std::ptr::null_mut();
                    iree_vm_ref_release(&mut ref_);

                    results.push(Value::DeviceBuffer(db));
                }
            }
            iree_vm_list_release(output_list);

            if let (Some(t0), Some(t1), Some(t2), Some(t3)) = (t0, t1, t2, t3) {
                let t4 = std::time::Instant::now();
                self.t_flatten_ns.fetch_add((t1 - t0).as_nanos() as u64, Ordering::Relaxed);
                self.t_buffers_ns.fetch_add((t2 - t1).as_nanos() as u64, Ordering::Relaxed);
                self.t_call_ns.fetch_add((t3 - t2).as_nanos() as u64, Ordering::Relaxed);
                self.t_output_ns.fetch_add((t4 - t3).as_nanos() as u64, Ordering::Relaxed);
                self.n_calls.fetch_add(1, Ordering::Relaxed);
            }

            match results.len() {
                0 => Ok(Value::Nil),
                1 => Ok(results.into_iter().next().unwrap()),
                _ => Ok(Value::Tuple(results)),
            }
        }
    }

    /// Like call_typed(), but returns DeviceBuffers instead of host tensors.
    pub fn call_typed_device(
        &self,
        fn_name: &str,
        inputs: &[Value],
        return_type: &crate::lowering::stablehlo::StableHLOType,
    ) -> Result<Value, SheafError> {
        let flat_result = self.call_device(fn_name, inputs)?;
        let flat_values = match flat_result {
            Value::Tuple(vals) => vals,
            other => vec![other],
        };
        let mut cursor = 0;
        let structured = unflatten_value(return_type, &flat_values, &mut cursor)?;
        Ok(structured)
    }
}


fn precompiled_registry_error() -> SheafError {
    SheafError::Runtime {
        message: "IREE precompiled module registry lock is poisoned".to_string(),
        location: None,
    }
}

impl Drop for IreeSession {
    fn drop(&mut self) {
        record_session_drop();
        if self.profile {
            let n = self.n_calls.load(Ordering::Relaxed);
            if n > 0 {
                let flatten = self.t_flatten_ns.load(Ordering::Relaxed) as f64 / 1e6;
                let buffers = self.t_buffers_ns.load(Ordering::Relaxed) as f64 / 1e6;
                let call = self.t_call_ns.load(Ordering::Relaxed) as f64 / 1e6;
                let output = self.t_output_ns.load(Ordering::Relaxed) as f64 / 1e6;
                let total = flatten + buffers + call + output;
                let hits = self.n_cache_hits.load(Ordering::Relaxed);
                let misses = self.n_cache_misses.load(Ordering::Relaxed);
                sheaf_msg!(
                    "\njit: dispatch profile ({} calls, {:.1}ms total):",
                    n,
                    total
                );
                sheaf_msg!(
                    "  flatten:  {:7.1}ms ({:4.1}%)",
                    flatten,
                    flatten / total * 100.0
                );
                sheaf_msg!(
                    "  buffers:  {:7.1}ms ({:4.1}%)  [hits: {}, misses: {}]",
                    buffers,
                    buffers / total * 100.0,
                    hits,
                    misses
                );
                sheaf_msg!("  call:     {:7.1}ms ({:4.1}%)", call, call / total * 100.0);
                sheaf_msg!(
                    "  output:   {:7.1}ms ({:4.1}%)",
                    output,
                    output / total * 100.0
                );
            }
        }
        unsafe {
            // Release all cached buffer views before tearing down the session
            if let Ok(cache) = self.buffer_cache.lock() {
                for positions in cache.values() {
                    for slot in positions {
                        for entry in slot {
                            iree_hal_buffer_view_release(entry.bv);
                        }
                    }
                }
            }
            if !self.session.is_null() {
                iree_runtime_session_release(self.session);
            }
            // device_handle (Arc<IreeDeviceHandle>) releases the device when
            // the last DeviceBuffer is dropped. No explicit release here.
            if !self.instance.is_null() {
                iree_runtime_instance_release(self.instance);
            }
        }
    }
}

#[cfg(test)]
mod lifetime_counter_tests {
    use super::{
        LIVE_SESSION_COUNT, SESSION_CREATION_ATTEMPTS, live_session_count, record_live_session,
        record_session_creation_attempt, record_session_drop, session_creation_attempt_count,
    };
    use std::sync::atomic::Ordering;
    use std::sync::{Mutex, OnceLock};

    fn counter_test_lock() -> &'static Mutex<()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
    }

    struct CounterRestore {
        attempts: usize,
        live: usize,
    }

    impl Drop for CounterRestore {
        fn drop(&mut self) {
            SESSION_CREATION_ATTEMPTS.store(self.attempts, Ordering::Relaxed);
            LIVE_SESSION_COUNT.store(self.live, Ordering::Relaxed);
        }
    }

    #[test]
    fn precompiled_registry_distinguishes_bodies_with_the_same_name() {
        let mut registry = super::PrecompiledModuleRegistry::default();
        let first = vec![("predict".to_string(), "first".to_string())];
        let second = vec![("predict".to_string(), "second".to_string())];

        assert!(registry.reserve("aot_first", &first));
        assert!(registry.publish("aot_first", &first));
        assert!(registry.reserve("aot_second", &second));
        assert!(registry.publish("aot_second", &second));
        assert_eq!(
            registry.module_for("predict", "first"),
            Some("aot_first".to_string())
        );
        assert_eq!(
            registry.module_for("predict", "second"),
            Some("aot_second".to_string())
        );
    }

    #[test]
    fn precompiled_registry_rejects_name_and_function_collisions() {
        let mut registry = super::PrecompiledModuleRegistry::default();
        let first = vec![("predict".to_string(), "first".to_string())];
        let second = vec![("predict".to_string(), "second".to_string())];

        assert!(registry.reserve("aot_first", &first));
        assert!(!registry.reserve("aot_first", &second));
        assert!(!registry.reserve("aot_second", &first));
        assert!(registry.publish("aot_first", &first));
        assert!(!registry.reserve("aot_third", &first));
    }

    #[test]
    fn precompiled_registry_release_restores_reservations() {
        let mut registry = super::PrecompiledModuleRegistry::default();
        let keys = vec![("predict".to_string(), "first".to_string())];

        assert!(registry.reserve("aot_first", &keys));
        registry.release("aot_first");
        assert!(registry.reserve("aot_first", &keys));
    }

    #[test]
    fn lifetime_counters_use_deltas_and_live_count_does_not_underflow() {
        let _lock = counter_test_lock().lock().unwrap();
        let restore = CounterRestore {
            attempts: session_creation_attempt_count(),
            live: live_session_count(),
        };
        record_session_creation_attempt();
        assert_eq!(session_creation_attempt_count(), restore.attempts + 1);
        assert_eq!(live_session_count(), restore.live);
        record_live_session();
        assert_eq!(live_session_count(), restore.live + 1);
        record_session_drop();
        assert_eq!(live_session_count(), restore.live);
        LIVE_SESSION_COUNT.store(0, Ordering::Relaxed);
        record_session_drop();
        assert_eq!(live_session_count(), 0);
    }
}
