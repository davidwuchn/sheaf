// Copyright (c) 2026 Damien Boureille
// Licensed under the MIT License.

//! Sheaf terminal output: violet system messages on stderr.

use std::sync::OnceLock;

const COLOR: &str = "\x1b[35m";
const RESET: &str = "\x1b[0m";

static USE_COLOR: OnceLock<bool> = OnceLock::new();

pub fn use_color() -> bool {
    *USE_COLOR.get_or_init(|| {
        #[cfg(unix)]
        { unsafe extern "C" { fn isatty(fd: i32) -> i32; } unsafe { isatty(2) != 0 } }
        #[cfg(not(unix))]
        { true }
    })
}

pub fn color() -> &'static str { if use_color() { COLOR } else { "" } }
pub fn reset() -> &'static str { if use_color() { RESET } else { "" } }

#[macro_export]
macro_rules! sheaf_msg {
    ($($arg:tt)*) => {
        eprintln!("{}{}{}", $crate::core::color::color(), format_args!($($arg)*), $crate::core::color::reset())
    };
}
