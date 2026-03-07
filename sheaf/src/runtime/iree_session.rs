#![allow(dead_code)]

use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use crate::core::error::SheafError;
use crate::interpreter::value::{Dtype, Value};
use crate::runtime::iree_ffi::*;
use ndarray::ArrayD;

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

/// Lightweight fingerprint for detecting tensor changes between calls.
/// Stores shape + 4 spread-sampled elements (as exact f64 bits).
struct TensorFingerprint {
    shape: Vec<usize>,
    num_elems: usize,
    sample: [u64; 4],
}

impl TensorFingerprint {
    /// Build a fingerprint from a value (allocates shape Vec). Use only on cache miss.
    fn from_value(val: &Value) -> Option<Self> {
        match val {
            Value::Tensor { data, .. } => {
                let shape = data.shape().to_vec();
                let num_elems = data.len();
                let sample = Self::sample_tensor(data);
                Some(Self { shape, num_elems, sample })
            }
            Value::Float(f) => Some(Self {
                shape: vec![],
                num_elems: 1,
                sample: [f.to_bits(), 0, 0, 0],
            }),
            Value::Int(n) => Some(Self {
                shape: vec![],
                num_elems: 1,
                sample: [(*n as f64).to_bits(), 0, 0, 0],
            }),
            Value::Bool(b) => Some(Self {
                shape: vec![],
                num_elems: 1,
                sample: [(*b as u64), 0, 0, 0],
            }),
            _ => None,
        }
    }

    /// Check if a value matches this fingerprint without allocating.
    fn matches(&self, val: &Value) -> bool {
        match val {
            Value::Tensor { data, .. } => {
                let n = data.len();
                n == self.num_elems
                    && data.shape() == self.shape.as_slice()
                    && Self::sample_tensor(data) == self.sample
            }
            Value::Float(f) => self.num_elems == 1 && self.sample[0] == f.to_bits(),
            Value::Int(n) => self.num_elems == 1 && self.sample[0] == (*n as f64).to_bits(),
            Value::Bool(b) => self.num_elems == 1 && self.sample[0] == (*b as u64),
            _ => false,
        }
    }

    fn sample_tensor(data: &ArrayD<f64>) -> [u64; 4] {
        let n = data.len();
        let mut sample = [0u64; 4];
        if n == 0 { return sample; }
        if n <= 4 {
            for (i, &x) in data.iter().take(4).enumerate() {
                sample[i] = x.to_bits();
            }
        } else if let Some(slice) = data.as_slice() {
            let indices = [0, n / 4, 3 * n / 4, n - 1];
            for (i, &idx) in indices.iter().enumerate() {
                sample[i] = slice[idx].to_bits();
            }
        } else {
            let flat: Vec<f64> = data.iter().copied().collect();
            let indices = [0, n / 4, 3 * n / 4, n - 1];
            for (i, &idx) in indices.iter().enumerate() {
                sample[i] = flat[idx].to_bits();
            }
        }
        sample
    }
}

struct CachedBufferView {
    fingerprint: TensorFingerprint,
    bv: *mut iree_hal_buffer_view_t,
}

pub struct IreeSession {
    instance: *mut iree_runtime_instance_t,
    device: *mut iree_hal_device_t,
    session: *mut iree_runtime_session_t,
    _vmfb_data: Option<Vec<u8>>,
    /// HAL driver name: "metal", "local-task", etc.
    driver_name: String,
    /// Per-function buffer view cache: fn_name -> per-position cached buffer views.
    /// Static arguments (e.g. model weights) are allocated once and reused across calls.
    buffer_cache: Mutex<HashMap<String, Vec<Option<CachedBufferView>>>>,
    /// Dispatch timing (nanoseconds, accumulated). Enabled by SHEAF_PROFILE_IREE=1.
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
            // SHEAF_DEVICE overrides: "cpu", "metal", "cuda", "vulkan", etc.
            let device_override = std::env::var("SHEAF_DEVICE").ok();
            let driver_names: Vec<&str> = match device_override.as_deref() {
                Some("cpu") => vec!["local-task"],
                Some(d) => vec![d, "local-task"],
                None => vec!["cuda", "metal", "vulkan", "local-task"],
            };
            let mut device: *mut iree_hal_device_t = std::ptr::null_mut();
            let mut chosen_driver = "";
            let verbose = std::env::var("SHEAF_JIT_VERBOSE").is_ok();
            for name in &driver_names {
                let driver = iree_string_view_t::from_str(name);
                let status =
                    iree_runtime_instance_try_create_default_device(instance, driver, &mut device);
                if iree_status_is_ok(status) {
                    chosen_driver = name;
                    break;
                } else if verbose {
                    eprintln!("iree: driver '{}' not available", name);
                }
            }
            if device.is_null() {
                iree_runtime_instance_release(instance);
                let tried = driver_names.join(", ");
                return Err(iree_err(&format!(
                    "failed to create IREE device (tried: {})", tried
                )));
            }
            if chosen_driver != "local-task" {
                eprintln!("iree: using {} device", chosen_driver);
            }

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
                iree_hal_device_release(device);
                iree_runtime_instance_release(instance);
                return Err(iree_err("failed to create IREE session"));
            }

            Ok(IreeSession {
                instance,
                device,
                session,
                _vmfb_data: None,
                driver_name: chosen_driver.to_string(),
                buffer_cache: Mutex::new(HashMap::new()),
                profile: std::env::var("SHEAF_PROFILE_IREE").is_ok(),
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

    /// Returns the HAL driver name ("metal", "local-task", etc.)
    pub fn driver_name(&self) -> &str {
        &self.driver_name
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
            let status = iree_runtime_session_append_bytecode_module_from_memory(
                self.session,
                span,
                null_alloc,
            );
            if !iree_status_is_ok(status) {
                unsafe extern "C" { static __stderrp: *mut std::ffi::c_void; }
                iree_status_fprint(__stderrp, status);
                return Err(iree_err("failed to load VMFB module"));
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
            // Static arguments (model weights) are fingerprinted and reused across calls.
            let mut cache = self.buffer_cache.lock().unwrap();
            // Avoid String allocation on cache hit: try get_mut first
            let cached_fn = match cache.get_mut(fn_name) {
                Some(c) => c,
                None => cache.entry(fn_name.to_string()).or_default(),
            };
            // Ensure cache vec is large enough
            cached_fn.resize_with(cached_fn.len().max(flat_inputs.len()), || None);

            for (i, val) in flat_inputs.iter().enumerate() {
                // Check cache without allocating a fingerprint
                let is_cached = cached_fn[i]
                    .as_ref()
                    .map_or(false, |entry| entry.fingerprint.matches(val));

                let bv = if is_cached {
                    if self.profile { self.n_cache_hits.fetch_add(1, Ordering::Relaxed); }
                    cached_fn[i].as_ref().unwrap().bv
                } else {
                    if self.profile { self.n_cache_misses.fetch_add(1, Ordering::Relaxed); }
                    // Cache miss: allocate new buffer view + fingerprint
                    let new_bv = value_to_buffer_view(device, device_alloc, val)?;
                    if let Some(old) = cached_fn[i].take() {
                        iree_hal_buffer_view_release(old.bv);
                    }
                    if let Some(fp) = TensorFingerprint::from_value(val) {
                        cached_fn[i] = Some(CachedBufferView { fingerprint: fp, bv: new_bv });
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
                eprintln!("\nIREE dispatch profile ({} calls, {:.1}ms total):", n, total);
                eprintln!("  flatten:  {:7.1}ms ({:4.1}%)", flatten, flatten / total * 100.0);
                eprintln!("  buffers:  {:7.1}ms ({:4.1}%)  [hits: {}, misses: {}]",
                    buffers, buffers / total * 100.0, hits, misses);
                eprintln!("  call:     {:7.1}ms ({:4.1}%)", call, call / total * 100.0);
                eprintln!("  output:   {:7.1}ms ({:4.1}%)", output, output / total * 100.0);
            }
        }
        unsafe {
            // Release all cached buffer views before tearing down the session
            if let Ok(cache) = self.buffer_cache.lock() {
                for entries in cache.values() {
                    for entry in entries.iter().flatten() {
                        iree_hal_buffer_view_release(entry.bv);
                    }
                }
            }
            if !self.session.is_null() {
                iree_runtime_session_release(self.session);
            }
            if !self.device.is_null() {
                iree_hal_device_release(self.device);
            }
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
            Value::Tensor { data, dtype } => {
                let shape: Vec<iree_hal_dim_t> =
                    data.shape().iter().map(|&d| d as iree_hal_dim_t).collect();

                let (element_type, byte_data) = match dtype {
                    Dtype::F32 => {
                        let f32_data: Vec<f32> = data.iter().map(|&x| x as f32).collect();
                        let bytes: Vec<u8> = f32_data
                            .iter()
                            .flat_map(|f| f.to_ne_bytes())
                            .collect();
                        (IREE_HAL_ELEMENT_TYPE_FLOAT_32, bytes)
                    }
                    _ => {
                        return Err(iree_err(&format!(
                            "unsupported dtype {:?} for IREE buffer",
                            dtype
                        )));
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
                    data: Arc::new(ArrayD::from_elem(vec![], *n as f64)),
                    dtype: Dtype::F32,
                };
                value_to_buffer_view(device, allocator, &tensor)
            }
            Value::Bool(b) => {
                let tensor = Value::Tensor {
                    data: Arc::new(ArrayD::from_elem(vec![], if *b { 1.0 } else { 0.0 })),
                    dtype: Dtype::F32,
                };
                value_to_buffer_view(device, allocator, &tensor)
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

        if elem_type != IREE_HAL_ELEMENT_TYPE_FLOAT_32 {
            return Err(iree_err(&format!(
                "unsupported IREE element type: 0x{:08x}",
                elem_type
            )));
        }

        let n_elems: usize = shape.iter().product::<usize>().max(1);
        let byte_len = n_elems * 4;
        let mut f32_buf: Vec<f32> = vec![0.0; n_elems];

        let buf = iree_hal_buffer_view_buffer(bv);
        // Use device_transfer_d2h which works for both CPU and GPU buffers.
        // iree_hal_buffer_map_read only works for HOST_VISIBLE buffers.
        let status = iree_hal_device_transfer_d2h(
            device,
            buf,
            0,
            f32_buf.as_mut_ptr() as *mut c_void,
            byte_len as u64,
            0, // flags
            iree_timeout_t::infinite(),
        );
        if !iree_status_is_ok(status) {
            unsafe extern "C" { static __stderrp: *mut std::ffi::c_void; }
            unsafe { iree_status_fprint(__stderrp, status); }
            return Err(iree_err("failed to read IREE buffer data"));
        }

        let f64_data: Vec<f64> = f32_buf.iter().map(|&x| x as f64).collect();
        let data = ArrayD::from_shape_vec(shape, f64_data)
            .map_err(|e| iree_err(&format!("shape mismatch: {}", e)))?;

        Ok(Value::Tensor {
            data: Arc::new(data),
            dtype: Dtype::F32,
        })
    }
}

/// Count the flat tensor leaves expected by a compiled signature.
pub fn count_signature_tensors(types: &[crate::compiler::stablehlo::StableHLOType]) -> usize {
    types.iter().map(count_type_leaves).sum()
}

fn count_type_leaves(ty: &crate::compiler::stablehlo::StableHLOType) -> usize {
    match ty {
        crate::compiler::stablehlo::StableHLOType::Tuple(elems) => {
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
        Value::Tuple(elems) | Value::List(elems) => elems.iter().map(count_one_value).sum(),
        Value::Tensor { .. } | Value::Float(_) | Value::Int(_) | Value::Bool(_) => 1,
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
            StableHLOType::Tuple(elems) => collect_leaf_shapes(elems, out),
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
        Value::Tensor { .. } | Value::Float(_) | Value::Int(_) | Value::Bool(_) => {
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
        StableHLOType::Tuple(elem_tys) => {
            let mut elems = Vec::new();
            for elem_ty in elem_tys {
                elems.push(unflatten_value(elem_ty, flat, cursor)?);
            }
            Ok(Value::Tuple(elems))
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
