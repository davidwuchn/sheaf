use std::env;
use std::path::PathBuf;

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
    println!("cargo:rustc-cfg=iree_runtime");

    println!("cargo:rerun-if-env-changed=IREE_BUILD_DIR");
    println!("cargo:rerun-if-env-changed=IREE_DIST_DIR");
    println!("cargo:rerun-if-env-changed=IREE_RUNTIME_LIB_DIR");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=iree-runtime");
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

/// Link iree_runtime_unified + flatcc_parsing from a single directory.
/// We use +whole-archive to ensure all driver modules (Metal, CUDA, etc.)
/// are included even when not directly referenced — IREE discovers them
/// at runtime via its driver registry.
fn link_from_dir(lib_dir: &PathBuf) -> bool {
    println!("cargo:rustc-link-search=native={}", lib_dir.display());
    println!("cargo:rustc-link-lib=static:+whole-archive=iree_runtime_unified");

    if lib_dir.join("libflatcc_parsing.a").exists() {
        println!("cargo:rustc-link-lib=static=flatcc_parsing");
    }
    if lib_dir.join("libflatcc_runtime.a").exists() {
        println!("cargo:rustc-link-lib=static=flatcc_runtime");
    }

    true
}
