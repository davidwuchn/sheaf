#[cfg(iree_runtime)]
mod process_memory {
    #[cfg(target_os = "macos")]
    use std::ffi::c_void;
    #[cfg(target_os = "macos")]
    use std::mem;

    #[cfg(target_os = "macos")]
    type MachPortT = u32;
    #[cfg(target_os = "macos")]
    type KernReturnT = i32;
    #[cfg(target_os = "macos")]
    type NaturalT = u32;

    #[cfg(target_os = "macos")]
    unsafe extern "C" {
        fn mach_task_self() -> MachPortT;
        fn task_info(
            target_task: MachPortT,
            flavor: u32,
            task_info_out: *mut c_void,
            task_info_out_count: *mut NaturalT,
        ) -> KernReturnT;
    }

    #[cfg(target_os = "macos")]
    #[repr(C)]
    struct MachTaskBasicInfo {
        virtual_size: u64,
        resident_size: u64,
        resident_size_max: u64,
        user_time: [u32; 2],
        system_time: [u32; 2],
        policy: i32,
        suspend_count: i32,
    }

    #[cfg(target_os = "macos")]
    pub fn rss_bytes() -> Option<usize> {
        const MACH_TASK_BASIC_INFO: u32 = 20;
        unsafe {
            let mut info: MachTaskBasicInfo = mem::zeroed();
            let mut count =
                (mem::size_of::<MachTaskBasicInfo>() / mem::size_of::<NaturalT>()) as NaturalT;
            let result = task_info(
                mach_task_self(),
                MACH_TASK_BASIC_INFO,
                (&mut info as *mut MachTaskBasicInfo).cast(),
                &mut count,
            );
            (result == 0).then_some(info.resident_size as usize)
        }
    }

    #[cfg(target_os = "linux")]
    pub fn rss_bytes() -> Option<usize> {
        let pages = std::fs::read_to_string("/proc/self/statm")
            .ok()?
            .split_whitespace()
            .nth(1)?
            .parse::<usize>()
            .ok()?;
        let page_size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
        (page_size > 0).then_some(pages * page_size as usize)
    }

    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    pub fn rss_bytes() -> Option<usize> {
        None
    }
}

#[cfg(iree_runtime)]
fn rss_mb() -> Option<f64> {
    process_memory::rss_bytes().map(|bytes| bytes as f64 / (1024.0 * 1024.0))
}

#[test]
#[ignore]
#[cfg(iree_runtime)]
fn iree_session_creation_memory_attribution() {
    use sheaf_compiler::runtime::iree_session::IreeSession;

    let Some(mut previous_rss) = rss_mb() else {
        eprintln!("skipping: RSS measurement is unavailable on this platform");
        return;
    };
    let n = std::env::var("SHEAF_MEM_ATTRIB_N")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(8);
    let vmfb = std::env::var("SHEAF_MEM_ATTRIB_VMFB")
        .ok()
        .and_then(|path| std::fs::read(path).ok());
    let mut sessions = Vec::with_capacity(n);

    eprintln!("IREE session creation memory attribution");
    for i in 1..=n {
        let mut session = IreeSession::new().expect("IreeSession::new()");
        if let Some(bytes) = &vmfb {
            session.load_vmfb(bytes.clone()).expect("load_vmfb()");
        }
        sessions.push(session);
        std::thread::sleep(std::time::Duration::from_millis(20));

        let rss = rss_mb().expect("RSS measurement became unavailable");
        eprintln!("  session {i}: rss={rss:.1} MB ({:+.1})", rss - previous_rss);
        previous_rss = rss;
    }

    drop(sessions);
    let rss = rss_mb().expect("RSS measurement became unavailable");
    eprintln!("  after drop: rss={rss:.1} MB");
}

#[test]
#[ignore]
#[cfg(iree_runtime)]
fn iree_session_drop_memory_attribution() {
    use ndarray::{ArrayD, IxDyn};
    use sheaf_compiler::core::config;
    use sheaf_compiler::interpreter::value::Value;
    use sheaf_compiler::runtime::iree_session::IreeSession;

    let Some(baseline_rss) = rss_mb() else {
        eprintln!("skipping: RSS measurement is unavailable on this platform");
        return;
    };
    let vmfb = match std::env::var("SHEAF_MEM_ATTRIB_VMFB") {
        Ok(path) => std::fs::read(path).expect("read SHEAF_MEM_ATTRIB_VMFB"),
        Err(_) => {
            eprintln!("skipping: SHEAF_MEM_ATTRIB_VMFB=<path to .vmfb> is required");
            return;
        }
    };
    config::init(0, Some("cpu".to_string()), false);

    let n = std::env::var("SHEAF_MEM_DROP_N")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(50);
    let shape = std::env::var("SHEAF_MEM_ATTRIB_SHAPE")
        .ok()
        .map(|value| {
            value
                .split(',')
                .filter_map(|dimension| dimension.parse().ok())
                .collect()
        })
        .unwrap_or_else(|| vec![1, 5, 1]);
    let elements = shape.iter().product::<usize>().max(1);
    let input = Value::tensor_f32(
        ArrayD::from_shape_vec(IxDyn(&shape), vec![0.5; elements]).expect("input shape"),
    );
    let mut max_rss = baseline_rss;

    eprintln!("IREE session drop memory attribution ({n} iterations)");
    for i in 1..=n {
        let mut session = IreeSession::new().expect("IreeSession::new()");
        session.load_vmfb(vmfb.clone()).expect("load_vmfb()");
        let _ = session.call("module.sigmoid", &[input.clone()]);
        drop(session);

        let rss = rss_mb().expect("RSS measurement became unavailable");
        max_rss = max_rss.max(rss);
        if i <= 20 || i % 50 == 0 || i == n {
            eprintln!("  {i}: rss={rss:.1} MB ({:+.1})", rss - baseline_rss);
        }
    }

    eprintln!("  peak delta: rss={:+.1} MB", max_rss - baseline_rss);
}
