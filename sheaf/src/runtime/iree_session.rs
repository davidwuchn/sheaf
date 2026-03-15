#![allow(dead_code)]

use std::collections::{BTreeMap, HashMap};
use std::ffi::c_void;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

/// Cached target backend string, set once on first IreeSession creation.
static CACHED_BACKEND: OnceLock<String> = OnceLock::new();

use crate::core::error::SheafError;
use crate::sheaf_msg;
use crate::interpreter::value::{Dtype, Value};
use crate::runtime::iree_ffi::*;
use ndarray::ArrayD;

/// Shared handle to the IREE device, allowing DeviceBuffers to outlive
/// the IreeSession safely. Released when the last reference drops.
pub struct IreeDeviceHandle {
    pub(crate) device: *mut iree_hal_device_t,
}

unsafe impl Send for IreeDeviceHandle {}
unsafe impl Sync for IreeDeviceHandle {}

impl Drop for IreeDeviceHandle {
    fn drop(&mut self) {
        unsafe { iree_hal_device_release(self.device); }
    }
}

/// An IREE buffer view living on device. Wraps raw FFI pointers behind Arc
/// to make it Clone + Send + Sync, matching Value's requirements.
pub struct DeviceBufferInner {
    bv: *mut iree_hal_buffer_view_t,
    device: Arc<IreeDeviceHandle>,
    pub shape: Vec<usize>,
    pub dtype: Dtype,
}

unsafe impl Send for DeviceBufferInner {}
unsafe impl Sync for DeviceBufferInner {}

impl Drop for DeviceBufferInner {
    fn drop(&mut self) {
        unsafe { iree_hal_buffer_view_release(self.bv); }
    }
}

impl DeviceBufferInner {
    /// Transfer data to host, returning an ndarray.
    pub fn to_host(&self) -> Result<ArrayD<f32>, SheafError> {
        unsafe {
            let n_elems: usize = self.shape.iter().product::<usize>().max(1);
            let byte_len = n_elems * 4; // f32 = 4 bytes
            let mut f32_buf: Vec<f32> = vec![0.0; n_elems];

            let buf = iree_hal_buffer_view_buffer(self.bv);
            let status = iree_hal_device_transfer_d2h(
                self.device.device,
                buf,
                0,
                f32_buf.as_mut_ptr() as *mut c_void,
                byte_len as u64,
                0,
                iree_timeout_t::infinite(),
            );
            if !iree_status_is_ok(status) {
                iree_status_fprint(libc_stderr(), status);
                return Err(iree_err("d2h transfer failed"));
            }

            ArrayD::from_shape_vec(self.shape.clone(), f32_buf)
                .map_err(|e| iree_err(&format!("shape mismatch: {}", e)))
        }
    }

    /// Get the raw buffer view pointer (for passing back to IREE).
    pub fn buffer_view(&self) -> *mut iree_hal_buffer_view_t {
        self.bv
    }
}

unsafe fn libc_stderr() -> *mut c_void {
    unsafe {
        #[cfg(target_os = "macos")]
        {
            unsafe extern "C" {
                static __stderrp: *mut c_void;
            }
            __stderrp
        }
        #[cfg(not(target_os = "macos"))]
        {
            unsafe extern "C" {
                static stderr: *mut c_void;
            }
            stderr
        }
    }
}

/// Temporarily redirect fd 2 (stderr) to /dev/null, returning the saved fd.
/// Used to suppress IREE C runtime diagnostics in non-verbose mode.
fn suppress_stderr() -> Option<i32> {
    unsafe {
        unsafe extern "C" {
            fn open(path: *const u8, oflag: i32) -> i32;
            fn close(fd: i32) -> i32;
            fn dup(fd: i32) -> i32;
            fn dup2(fd: i32, fd2: i32) -> i32;
        }
        const O_WRONLY: i32 = 1;
        let devnull = open(b"/dev/null\0".as_ptr(), O_WRONLY);
        if devnull < 0 { return None; }
        let saved = dup(2);
        if saved < 0 { close(devnull); return None; }
        dup2(devnull, 2);
        close(devnull);
        Some(saved)
    }
}

/// Restore stderr from a saved fd (from suppress_stderr).
fn restore_stderr(saved_fd: i32) {
    unsafe {
        unsafe extern "C" {
            fn close(fd: i32) -> i32;
            fn dup2(fd: i32, fd2: i32) -> i32;
        }
        dup2(saved_fd, 2);
        close(saved_fd);
    }
}

/// Identity-based fingerprint for buffer view caching.
/// Uses Arc identity for tensors (O(1), no false positives).
/// Same Arc = same data = cache hit. New Arc = miss (correct for computed values).
/// Stores a clone of the Arc to keep it alive and prevent address reuse (ABA problem).
#[derive(Clone)]
enum TensorFingerprint {
    Tensor(Arc<ndarray::ArrayD<f32>>),
    DeviceBuffer(Arc<DeviceBufferInner>),
    Scalar(u64),
}

impl TensorFingerprint {
    fn from_value(val: &Value) -> Option<Self> {
        match val {
            Value::Tensor { data, .. } => Some(Self::Tensor(Arc::clone(data))),
            Value::DeviceBuffer(db) => Some(Self::DeviceBuffer(Arc::clone(db))),
            Value::Float(f) => Some(Self::Scalar(f.to_bits() as u64)),
            Value::Int(n) => Some(Self::Scalar(*n as u64)),
            Value::Bool(b) => Some(Self::Scalar(*b as u64)),
            _ => None,
        }
    }

    fn matches(&self, val: &Value) -> bool {
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

struct CachedBufferView {
    fingerprint: TensorFingerprint,
    bv: *mut iree_hal_buffer_view_t,
}

pub struct IreeSession {
    instance: *mut iree_runtime_instance_t,
    device_handle: Arc<IreeDeviceHandle>,
    session: *mut iree_runtime_session_t,
    _vmfb_data: Option<Vec<u8>>,
    /// HAL driver name: "metal", "local-task", etc.
    driver_name: String,
    /// Per-function buffer view cache: fn_name -> per-position cached buffer views.
    /// Each position holds up to MAX_CACHE_ENTRIES entries to avoid thrashing when
    /// the same function is called with different weight sets (e.g. transformer layers).
    buffer_cache: Mutex<HashMap<String, Vec<Vec<CachedBufferView>>>>,
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

            // Try drivers in preference order: GPU > CPU
            let device_override = crate::core::config::device_override();
            let driver_names: Vec<&str> = match device_override {
                Some("cpu") => vec!["local-task"],
                Some(d) => vec![d, "local-task"],
                None => vec!["cuda", "metal", "vulkan", "local-task"],
            };
            let mut device: *mut iree_hal_device_t = std::ptr::null_mut();
            let mut chosen_driver = "";
            for name in &driver_names {
                let driver = iree_string_view_t::from_str(name);
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

            Ok(IreeSession {
                instance,
                device_handle,
                session,
                _vmfb_data: None,
                driver_name: chosen_driver.to_string(),
                buffer_cache: Mutex::new(HashMap::new()),
                profile: crate::core::config::jit_profile(),
                t_flatten_ns: AtomicU64::new(0),
                t_buffers_ns: AtomicU64::new(0),
                t_call_ns: AtomicU64::new(0),
                t_output_ns: AtomicU64::new(0),
                n_calls: AtomicU64::new(0),
                n_cache_hits: AtomicU64::new(0),
                n_cache_misses: AtomicU64::new(0),
            })
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

    pub fn load_vmfb(&mut self, data: Vec<u8>) -> Result<(), SheafError> {
        unsafe {
            self._vmfb_data = Some(data);
            let bytes = self._vmfb_data.as_ref().unwrap();
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
                return Err(iree_err("failed to load compiled module"));
            }

            // Print backend once, after first successful VMFB load
            {
                use std::sync::atomic::AtomicBool;
                static PRINTED: AtomicBool = AtomicBool::new(false);
                if !PRINTED.swap(true, Ordering::Relaxed) {
                    let display = match self.driver_name.as_str() {
                        "local-task" => "CPU",
                        "metal" => "Metal GPU",
                        "cuda" => "CUDA GPU",
                        "vulkan" => "Vulkan GPU",
                        other => other,
                    };
                    sheaf_msg!("sheaf: running on {}", display);
                }
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
                    let new_bv = value_to_buffer_view(device, device_alloc, val)?;
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

            let name = iree_string_view_t::from_str(fn_name);
            let status = iree_runtime_session_call_by_name(
                self.session,
                name,
                input_list,
                output_list,
            );
            iree_vm_list_release(input_list);
            if !iree_status_is_ok(status) {
                iree_status_fprint(libc_stderr(), status);
                iree_vm_list_release(output_list);
                return Err(iree_err(&format!("IREE call '{}' failed", fn_name)));
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
                let val = buffer_view_to_value(device, bv)?;
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
        return_type: &crate::compiler::stablehlo::StableHLOType,
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
                                let new_bv = value_to_buffer_view(device, device_alloc, val)?;
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

            let name = iree_string_view_t::from_str(fn_name);
            let status = iree_runtime_session_call_by_name(
                self.session,
                name,
                input_list,
                output_list,
            );
            iree_vm_list_release(input_list);
            if !iree_status_is_ok(status) {
                iree_status_fprint(libc_stderr(), status);
                iree_vm_list_release(output_list);
                return Err(iree_err(&format!("IREE call '{}' failed", fn_name)));
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

                // Read shape metadata (no data transfer)
                let rank = iree_hal_buffer_view_shape_rank(bv);
                let shape: Vec<usize> = (0..rank)
                    .map(|j| iree_hal_buffer_view_shape_dim(bv, j) as usize)
                    .collect();

                let db = Arc::new(DeviceBufferInner {
                    bv,
                    device: Arc::clone(&self.device_handle),
                    shape,
                    dtype: Dtype::F32,
                });
                // Transfer ownership: nullify ptr to prevent double-free on ref release
                ref_.ptr = std::ptr::null_mut();
                iree_vm_ref_release(&mut ref_);

                results.push(Value::DeviceBuffer(db));
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
        return_type: &crate::compiler::stablehlo::StableHLOType,
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

impl Drop for IreeSession {
    fn drop(&mut self) {
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
                sheaf_msg!("\njit: dispatch profile ({} calls, {:.1}ms total):", n, total);
                sheaf_msg!("  flatten:  {:7.1}ms ({:4.1}%)", flatten, flatten / total * 100.0);
                sheaf_msg!("  buffers:  {:7.1}ms ({:4.1}%)  [hits: {}, misses: {}]",
                    buffers, buffers / total * 100.0, hits, misses);
                sheaf_msg!("  call:     {:7.1}ms ({:4.1}%)", call, call / total * 100.0);
                sheaf_msg!("  output:   {:7.1}ms ({:4.1}%)", output, output / total * 100.0);
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

unsafe fn value_to_buffer_view(
    device: *mut iree_hal_device_t,
    allocator: *mut iree_hal_allocator_t,
    val: &Value,
) -> Result<*mut iree_hal_buffer_view_t, SheafError> {
    unsafe {
        match val {
            Value::Tensor { data, dtype: _ } => {
                let shape: Vec<iree_hal_dim_t> =
                    data.shape().iter().map(|&d| d as iree_hal_dim_t).collect();

                let f32_slice = data.as_slice().unwrap();
                let byte_data: Vec<u8> = f32_slice
                    .iter()
                    .flat_map(|f| f.to_ne_bytes())
                    .collect();
                let element_type = IREE_HAL_ELEMENT_TYPE_FLOAT_32;

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
                "cannot convert {} to IREE buffer",
                val.type_name()
            ))),
        }
    }
}

unsafe fn buffer_view_to_value(
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
        let byte_len = n_elems * 4; // both f32 and i32 are 4 bytes

        let buf = iree_hal_buffer_view_buffer(bv);

        let (f32_data, dtype) = if elem_type == IREE_HAL_ELEMENT_TYPE_FLOAT_32 {
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
        } else if elem_type == IREE_HAL_ELEMENT_TYPE_INT_32 {
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
                "unsupported IREE element type: 0x{:08x}",
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

/// Count the flat tensor leaves expected by a compiled signature.
pub fn count_signature_tensors(types: &[crate::compiler::stablehlo::StableHLOType]) -> usize {
    types.iter().map(count_type_leaves).sum()
}

fn count_type_leaves(ty: &crate::compiler::stablehlo::StableHLOType) -> usize {
    match ty {
        crate::compiler::stablehlo::StableHLOType::Tuple(elems, _) => {
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
    param_types: &[crate::compiler::stablehlo::StableHLOType],
) -> bool {
    count_arg_tensors(args) == count_signature_tensors(param_types)
}

/// Validate that runtime arg shapes match the compiled signature shapes.
/// Returns `Ok(())` on match, or `Err(description)` with a human-readable
/// mismatch message suitable for display.
pub fn check_shapes_match(
    args: &[Value],
    param_types: &[crate::compiler::stablehlo::StableHLOType],
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

fn collect_leaf_shapes(types: &[crate::compiler::stablehlo::StableHLOType], out: &mut Vec<Vec<i64>>) {
    use crate::compiler::stablehlo::StableHLOType;
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

/// Flatten a list of values into individual tensor leaf references.
/// Dicts are sorted by key (matching codegen convention), then recursed.
/// Tuples are recursed. Scalars/tensors pass through.
/// Returns references to avoid cloning tensor data.
fn flatten_values<'a>(inputs: &'a [Value]) -> Result<Vec<&'a Value>, SheafError> {
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
            "cannot flatten {} for IREE call",
            val.type_name()
        ))),
    }
}

/// Reconstruct a nested Value from a flat list of tensor Values,
/// guided by a StableHLOType structure.
fn unflatten_value(
    ty: &crate::compiler::stablehlo::StableHLOType,
    flat: &[Value],
    cursor: &mut usize,
) -> Result<Value, SheafError> {
    use crate::compiler::stablehlo::StableHLOType;
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

fn iree_err(msg: &str) -> SheafError {
    SheafError::Runtime {
        message: msg.to_string(),
        location: None,
    }
}
