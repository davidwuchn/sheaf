// Copyright (c) 2026 Damien Boureille
// Licensed under the MIT License.

//! Process and IREE allocator memory profiling for `--mem-profile`.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

struct Snapshot {
    label: String,
    rss_bytes: usize,
    delta_bytes: i64,
}

/// ABI mirror of `iree_hal_allocator_statistics_t` when statistics are enabled.
#[cfg(iree_runtime)]
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
struct IreeAllocStats {
    host_bytes_peak: usize,
    host_bytes_allocated: usize,
    host_bytes_freed: usize,
    device_bytes_peak: usize,
    device_bytes_allocated: usize,
    device_bytes_freed: usize,
}

#[cfg(iree_runtime)]
unsafe extern "C" {
    fn iree_hal_allocator_query_statistics(
        allocator: *mut crate::runtime::iree_ffi::iree_hal_allocator_t,
        out_stats: *mut IreeAllocStats,
    );
}

/// Tracks process RSS in a background thread and records named checkpoints.
pub struct MemProfiler {
    snapshots: Vec<Snapshot>,
    start_rss: usize,
    peak_sampled: Arc<AtomicUsize>,
    stop_flag: Arc<AtomicBool>,
    _thread_handle: Option<thread::JoinHandle<()>>,
    #[cfg(iree_runtime)]
    iree_device_peak: AtomicUsize,
    #[cfg(iree_runtime)]
    iree_host_peak: AtomicUsize,
    #[cfg(iree_runtime)]
    // Device peak, device live, host peak, host live.
    iree_last: Mutex<(usize, usize, usize, usize)>,
}

impl MemProfiler {
    /// Starts RSS sampling with the current usage as the baseline.
    pub fn new() -> Self {
        let start_rss = current_rss();

        let peak_sampled = Arc::new(AtomicUsize::new(start_rss));
        let stop_flag = Arc::new(AtomicBool::new(false));

        let peak_clone = peak_sampled.clone();
        let stop_clone = stop_flag.clone();
        let handle = thread::spawn(move || {
            while !stop_clone.load(Ordering::Relaxed) {
                let rss = current_rss();
                let mut prev = peak_clone.load(Ordering::Relaxed);
                while rss > prev {
                    match peak_clone.compare_exchange_weak(
                        prev, rss, Ordering::SeqCst, Ordering::Relaxed
                    ) {
                        Ok(_) => break,
                        Err(cur) => prev = cur,
                    }
                }
                thread::sleep(Duration::from_millis(100));
            }
        });

        Self {
            snapshots: Vec::new(),
            start_rss,
            peak_sampled,
            stop_flag,
            _thread_handle: Some(handle),
            #[cfg(iree_runtime)]
            iree_device_peak: AtomicUsize::new(0),
            #[cfg(iree_runtime)]
            iree_host_peak: AtomicUsize::new(0),
            #[cfg(iree_runtime)]
            iree_last: Mutex::new((0, 0, 0, 0)),
        }
    }

    /// Records RSS and its change from the previous checkpoint.
    pub fn sample(&mut self, label: &str) {
        let rss = current_rss();
        let prev = self
            .snapshots
            .last()
            .map(|s| s.rss_bytes)
            .unwrap_or(self.start_rss);
        let delta = rss as i64 - prev as i64;
        self.snapshots.push(Snapshot {
            label: label.to_string(),
            rss_bytes: rss,
            delta_bytes: delta,
        });
    }

    /// Samples allocator totals from the session's IREE device.
    #[cfg(iree_runtime)]
    pub fn sample_iree(&self, session: &crate::runtime::iree_session::IreeSession) {
        let allocator = session.device_allocator_ptr();
        if allocator.is_null() { return; }
        let mut stats = IreeAllocStats::default();
        unsafe { iree_hal_allocator_query_statistics(allocator, &mut stats) };
        let dev_peak = stats.device_bytes_peak;
        let host_peak = stats.host_bytes_peak;
        let dev_live = stats.device_bytes_allocated.saturating_sub(stats.device_bytes_freed);
        let host_live = stats.host_bytes_allocated.saturating_sub(stats.host_bytes_freed);
        if dev_peak > self.iree_device_peak.load(Ordering::SeqCst) {
            self.iree_device_peak.store(dev_peak, Ordering::SeqCst);
        }
        if host_peak > self.iree_host_peak.load(Ordering::SeqCst) {
            self.iree_host_peak.store(host_peak, Ordering::SeqCst);
        }
        if let Ok(mut g) = self.iree_last.lock() {
            *g = (dev_peak, dev_live, host_peak, host_live);
        }
    }

    /// Formats the process and allocator measurements.
    pub fn report(&self) -> String {
        let start_human = format_bytes(self.start_rss);

        let peak = std::cmp::max(
            self.peak_sampled.load(Ordering::SeqCst),
            std::cmp::max(
                self.snapshots.iter().map(|s| s.rss_bytes).max().unwrap_or(0),
                peak_rss_from_getrusage(),
            ),
        );

        let mut lines: Vec<String> = Vec::new();
        lines.push("\nMemory profile:".to_string());
        lines.push(format!(
            "  {:<28} {:>12} {:>14}",
            "Checkpoint", "RSS", "Delta"
        ));
        lines.push(format!("  {}", "-".repeat(56)));
        lines.push(format!("  {:<28} {:>12}", "start", start_human));

        for snap in &self.snapshots {
            let rss_h = format_bytes(snap.rss_bytes);
            let delta_h = format_delta(snap.delta_bytes);
            lines.push(format!(
                "  {:<28} {:>12} {:>14}",
                snap.label, rss_h, delta_h
            ));
        }

        lines.push(format!(
            "  {:<28} {:>12}",
            "peak",
            format_bytes(peak)
        ));
        lines.push("".to_string());

        #[cfg(target_os = "macos")]
        {
            let footprint = current_phys_footprint();
            if footprint > 0 {
                lines.push(format!("  {:<28} {:>12}", "phys footprint (dirty)", format_bytes(footprint)));
            }
        }

        #[cfg(iree_runtime)]
        {
            let dev_peak = self.iree_device_peak.load(Ordering::SeqCst);
            if dev_peak > 0 {
                lines.push("  IREE device allocator:".to_string());
                lines.push(format!("  {:<28} {:>12}", "device memory peak", format_bytes(dev_peak)));
                if let Ok(g) = self.iree_last.lock() {
                    lines.push(format!("  {:<28} {:>12}", "device memory live", format_bytes(g.1)));
                    let hpeak = self.iree_host_peak.load(Ordering::SeqCst);
                    if hpeak > 0 {
                        lines.push(format!("  {:<28} {:>12}", "IREE host peak", format_bytes(hpeak)));
                        lines.push(format!("  {:<28} {:>12}", "IREE host live", format_bytes(g.3)));
                    }
                }
            }
        }

        lines.join("\n")
    }

}

impl Drop for MemProfiler {
    fn drop(&mut self) {
        self.stop_flag.store(true, Ordering::SeqCst);
        if let Some(h) = self._thread_handle.take() {
            let _ = h.join();
        }
    }
}

/// Reads current resident memory rather than the monotonic `ru_maxrss` value.
fn current_rss() -> usize {
    #[cfg(target_os = "macos")]
    {
        use std::mem;

        #[allow(non_camel_case_types)]
        type mach_port_t = u32;
        #[allow(non_camel_case_types)]
        type kern_return_t = i32;
        #[allow(non_camel_case_types)]
        type natural_t = u32;

        const MACH_TASK_BASIC_INFO: u32 = 20;

        #[repr(C)]
        struct MachTaskBasicInfo {
            virtual_size:       u64,
            resident_size:      u64,
            resident_size_max:  u64,
            user_time:          [u32; 2],
            system_time:        [u32; 2],
            policy:             i32,
            suspend_count:      i32,
        }

        unsafe extern "C" {
            fn task_info(
                target_task: mach_port_t,
                flavor: u32,
                task_info_out: *mut MachTaskBasicInfo,
                task_info_outCnt: *mut natural_t,
            ) -> kern_return_t;
            fn mach_task_self() -> mach_port_t;
        }

        let mut info: MachTaskBasicInfo = unsafe { mem::zeroed() };
        let mut count = (mem::size_of::<MachTaskBasicInfo>() / 4) as natural_t;
        let ret = unsafe {
            task_info(mach_task_self(), MACH_TASK_BASIC_INFO, &mut info, &mut count)
        };

        if ret == 0 { info.resident_size as usize } else { 0 }
    }
    #[cfg(target_os = "linux")]
    {
        std::fs::read_to_string("/proc/self/status")
            .ok()
            .and_then(|s| {
                s.lines()
                    .find(|l| l.starts_with("VmRSS:"))
                    .and_then(|l| {
                        l.split_whitespace()
                            .nth(1)
                            .and_then(|v| v.parse::<usize>().ok())
                    })
            })
            .map(|kb| kb * 1024)
            .unwrap_or(0)
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    {
        0
    }
}

#[cfg(target_os = "macos")]
fn current_phys_footprint() -> usize {
    use std::mem;

    #[allow(non_camel_case_types)]
    type mach_port_t = u32;
    #[allow(non_camel_case_types)]
    type kern_return_t = i32;
    #[allow(non_camel_case_types)]
    type task_flavor_t = u32;
    #[allow(non_camel_case_types)]
    type natural_t = u32;

    const TASK_VM_INFO_PURGEABLE: task_flavor_t = 23;

    #[repr(C)]
    struct TaskVmInfo {
        virtual_size:                u64,
        region_count:                i32,
        page_size:                   i32,
        resident_size:               u64,
        resident_size_peak:          u64,
        device:                      u64,
        device_peak:                 u64,
        internal:                    u64,
        internal_peak:               u64,
        external:                    u64,
        external_peak:               u64,
        reusable:                    u64,
        reusable_peak:               u64,
        purgeable_volatile_pmap:     u64,
        purgeable_volatile_resident: u64,
        purgeable_volatile_virtual:  u64,
        compressed:                  u64,
        compressed_peak:             u64,
        compressed_lifetime:         u64,
        phys_footprint:              u64,
    }

    #[allow(clashing_extern_declarations)]
    unsafe extern "C" {
        fn task_info(
            target_task: mach_port_t,
            flavor: task_flavor_t,
            task_info_out: *mut TaskVmInfo,
            task_info_outCnt: *mut natural_t,
        ) -> kern_return_t;
        fn mach_task_self() -> mach_port_t;
    }

    let mut info: TaskVmInfo = unsafe { mem::zeroed() };
    let mut count = (mem::size_of::<TaskVmInfo>() / 4) as natural_t;
    let ret = unsafe {
        task_info(mach_task_self(), TASK_VM_INFO_PURGEABLE, &mut info, &mut count)
    };

    if ret == 0 { info.phys_footprint as usize } else { 0 }
}

/// Reads the process RSS high-water mark.
fn peak_rss_from_getrusage() -> usize {
    let mut usage = std::mem::MaybeUninit::<libc::rusage>::uninit();
    let ret = unsafe { libc::getrusage(libc::RUSAGE_SELF, usage.as_mut_ptr()) };
    if ret == 0 {
        #[cfg(target_os = "macos")]
        { unsafe { usage.assume_init().ru_maxrss as usize } }
        #[cfg(target_os = "linux")]
        { unsafe { (usage.assume_init().ru_maxrss as usize) * 1024 } }
    } else {
        0
    }
}

fn format_bytes(bytes: usize) -> String {
    if bytes == 0 {
        return "?".to_string();
    }
    let mb = bytes as f64 / (1024.0 * 1024.0);
    if mb < 1.0 {
        format!("{:.0} KB", bytes as f64 / 1024.0)
    } else if mb < 1024.0 {
        format!("{:.1} MB", mb)
    } else {
        format!("{:.2} GB", mb / 1024.0)
    }
}

fn format_delta(delta: i64) -> String {
    if delta == 0 {
        "+0".to_string()
    } else {
        let sign = if delta > 0 { "+" } else { "" };
        format!(
            "{}{}",
            sign,
            format_bytes(delta.unsigned_abs() as usize)
        )
    }
}
