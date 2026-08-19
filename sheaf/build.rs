use std::env;
use std::path::{Path, PathBuf};

fn main() {
    // Search order for IREE runtime libraries:
    // 1. IREE_BUILD_DIR env var (CMake build tree)
    // 2. IREE_DIST_DIR env var (nightly dist flat layout)
    // 3. IREE_RUNTIME_LIB_DIR env var (explicit lib directory)
    // 4. iree-runtime/ directory next to Cargo.toml (convention)

    // Expose [package.metadata] iree-version as compile-time env var
    let cargo_toml = std::fs::read_to_string("Cargo.toml").expect("cannot read Cargo.toml");
    let iree_version = cargo_toml.lines()
        .find(|l| l.starts_with("iree-version"))
        .and_then(|l| l.split('"').nth(1))
        .expect("missing iree-version in [package.metadata]");
    println!("cargo:rustc-env=IREE_VERSION={}", iree_version);
    println!("cargo:rerun-if-changed=Cargo.toml");

    let found = try_cmake_layout()
        || try_dist_layout()
        || try_explicit_lib_dir()
        || try_conventional_dir();

    if !found {
        panic!(
            "\n\nIREE runtime not found. Place libraries in sheaf/iree-runtime/ or set IREE_RUNTIME_LIB_DIR.\n\
             Required files: libiree_runtime_unified.a, libflatcc_parsing.a\n"
        );
    }

    // macOS system frameworks
    if cfg!(target_os = "macos") {
        println!("cargo:rustc-link-lib=framework=Accelerate");
        println!("cargo:rustc-link-lib=framework=CoreFoundation");
        println!("cargo:rustc-link-lib=framework=Foundation");
        println!("cargo:rustc-link-lib=framework=Metal");
        println!("cargo:rustc-link-lib=framework=IOKit");
    }

    // System libs
    if cfg!(target_os = "macos") {
        println!("cargo:rustc-link-lib=dylib=c++");
    } else {
        println!("cargo:rustc-link-lib=dylib=stdc++");
    }

    // Tell Rust code that IREE is available
    println!("cargo:rustc-check-cfg=cfg(iree_runtime)");
    println!("cargo:rustc-check-cfg=cfg(sheaf_frontend)");
    println!("cargo:rustc-cfg=iree_runtime");

    println!("cargo:rerun-if-env-changed=IREE_BUILD_DIR");
    println!("cargo:rerun-if-env-changed=IREE_DIST_DIR");
    println!("cargo:rerun-if-env-changed=IREE_RUNTIME_LIB_DIR");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=iree-runtime");

    stamp_version();
}

/// Stamp the binary with a version string.
///
/// By default this is the Cargo package version. Nightly builds can override
/// it with `SHEAF_BUILD_VERSION`.
///
/// Generates `OUT_DIR/generated_version.rs` with a `SHEAF_VERSION` constant.
fn stamp_version() {
    let out_dir = env::var("OUT_DIR").expect("OUT_DIR not set");
    let dest = Path::new(&out_dir).join("generated_version.rs");

    let version = match env::var("SHEAF_BUILD_VERSION") {
        Ok(v) if !v.trim().is_empty() => v,
        // Fall back to the Cargo package version
        _ => env!("CARGO_PKG_VERSION").to_string(),
    };

    let src = format!("pub const SHEAF_VERSION: &str = {version:?};\n");

    println!("cargo:rerun-if-env-changed=SHEAF_BUILD_VERSION");

    // Only rewrite if the content changed, so we don't needlessly invalidate
    // downstream caches.
    if std::fs::read_to_string(&dest).unwrap_or_default() != src {
        std::fs::write(&dest, src)
            .unwrap_or_else(|e| panic!("cannot write {dest:?}: {e}"));
    }
}

/// CMake build tree layout: IREE_BUILD_DIR/runtime/src/iree/runtime/libiree_runtime_unified.a
fn try_cmake_layout() -> bool {
    let iree_build_dir = match env::var("IREE_BUILD_DIR") {
        Ok(d) => PathBuf::from(d),
        Err(_) => return false,
    };

    let runtime_dir = iree_build_dir.join("runtime/src/iree/runtime");
    let unified_lib = runtime_dir.join("libiree_runtime_unified.a");

    if !unified_lib.exists() {
        return false;
    }

    println!("cargo:rustc-link-search=native={}", runtime_dir.display());
    println!("cargo:rustc-link-lib=static:+whole-archive=iree_runtime_unified");

    let flatcc_dir = iree_build_dir.join("build_tools/third_party/flatcc");
    if flatcc_dir.join("libflatcc_parsing.a").exists() {
        println!("cargo:rustc-link-search=native={}", flatcc_dir.display());
        println!("cargo:rustc-link-lib=static=flatcc_parsing");
    }

    true
}

/// Nightly dist flat layout: iree-dist-*/lib/libiree_runtime_unified.a
fn try_dist_layout() -> bool {
    let dist_dir = match env::var("IREE_DIST_DIR") {
        Ok(d) => PathBuf::from(d),
        Err(_) => return false,
    };

    let lib_dir = dist_dir.join("lib");
    let unified_lib = lib_dir.join("libiree_runtime_unified.a");

    if !unified_lib.exists() {
        return false;
    }

    link_from_dir(&lib_dir)
}

/// Explicit lib directory: all .a files in one place
fn try_explicit_lib_dir() -> bool {
    let lib_dir = match env::var("IREE_RUNTIME_LIB_DIR") {
        Ok(d) => PathBuf::from(d),
        Err(_) => return false,
    };

    let unified_lib = lib_dir.join("libiree_runtime_unified.a");
    if !unified_lib.exists() {
        return false;
    }

    link_from_dir(&lib_dir)
}

/// Convention: iree-runtime/ next to Cargo.toml
fn try_conventional_dir() -> bool {
    let lib_dir = PathBuf::from("iree-runtime");
    if lib_dir.join("libiree_runtime_unified.a").exists() {
        link_from_dir(&lib_dir)
    } else {
        false
    }
}

/// Link `iree_runtime_unified` and `flatcc_parsing` from a single directory.
///
/// Use `+whole-archive` so driver modules (Metal, CUDA, etc.) aren't dropped
/// by the linker. IREE discovers them through its driver registry at runtime.
fn link_from_dir(lib_dir: &Path) -> bool {
    println!("cargo:rustc-link-search=native={}", lib_dir.display());
    println!("cargo:rustc-link-lib=static:+whole-archive=iree_runtime_unified");

    let flatcc_parsing = lib_dir.join("libflatcc_parsing.a");
    let flatcc_runtime = lib_dir.join("libflatcc_runtime.a");
    if flatcc_parsing.exists() {
        println!("cargo:rustc-link-lib=static=flatcc_parsing");
    }
    if flatcc_runtime.exists() {
        println!("cargo:rustc-link-lib=static=flatcc_runtime");
    }
    // Force flatcc onto linker command line for GNU ld cross-compilation
    if flatcc_parsing.exists() {
        println!("cargo:rustc-link-arg={}", flatcc_parsing.display());
    }
    if flatcc_runtime.exists() {
        println!("cargo:rustc-link-arg={}", flatcc_runtime.display());
    }

    true
}
