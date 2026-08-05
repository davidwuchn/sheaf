// Copyright (c) 2025 Damien Boureille
// Licensed under the MIT License.

//! IREE compiler toolchain download, version management, and path resolution.

use std::path::PathBuf;

use crate::sheaf_msg;

/// IREE compiler version from Cargo.toml [package.metadata]
pub(crate) const IREE_COMPILER_VERSION: &str = env!("IREE_VERSION");

/// Locate `iree-compile`. Returns None if not found.
pub fn find_iree_compile() -> Option<String> {
    // Explicit env var
    if let Ok(path) = std::env::var("IREE_COMPILE")
        && std::path::Path::new(&path).exists() {
            return Some(path);
        }
    // Auto-downloaded toolchain cache
    if let Some(path) = find_cached_toolchain() {
        return Some(path);
    }
    // PATH lookup
    which("iree-compile")
}

fn toolchain_dir() -> Option<PathBuf> {
    std::env::var("HOME").ok().map(|h| PathBuf::from(h).join(".sheaf/toolchain"))
}

fn find_cached_toolchain() -> Option<String> {
    let dir = toolchain_dir()?;
    let binary = dir.join("iree-compile");
    if !binary.exists() {
        return None;
    }
    // Check version matches
    let version_file = dir.join("version");
    if let Ok(cached_version) = std::fs::read_to_string(&version_file) {
        if cached_version.trim() != IREE_COMPILER_VERSION {
            return None; // stale version, will trigger re-download
        }
    } else {
        return None;
    }
    Some(binary.to_string_lossy().to_string())
}

fn platform_wheel_tag() -> Option<&'static str> {
    match (std::env::consts::OS, std::env::consts::ARCH) {
        ("macos", _) => Some("macosx_13_0_universal2"),
        ("linux", "x86_64") => Some("manylinux_2_28_x86_64"),
        ("linux", "aarch64") => Some("manylinux_2_28_aarch64"),
        _ => None,
    }
}

fn compiler_lib_name() -> &'static str {
    if cfg!(target_os = "macos") { "libIREECompiler.dylib" }
    else { "libIREECompiler.so" }
}

/// Download and install the IREE compiler toolchain from PyPI.
pub fn ensure_toolchain() -> Result<String, Box<dyn std::error::Error>> {
    let platform_tag = platform_wheel_tag()
        .ok_or("unsupported platform for auto-download")?;
    let dir = toolchain_dir()
        .ok_or("cannot determine home directory")?;
    std::fs::create_dir_all(&dir)?;

    for tool in &["curl", "unzip"] {
        if which(tool).is_none() {
            return Err(format!("'{}' is required to download the compiler toolchain", tool).into());
        }
    }

    sheaf_msg!("sheaf: downloading compiler toolchain...");

    // Fetch PyPI JSON metadata to find the wheel URL
    let pypi_url = format!(
        "https://pypi.org/pypi/iree-base-compiler/{}/json",
        IREE_COMPILER_VERSION
    );
    let json_path = std::env::temp_dir().join("sheaf-pypi-metadata.json");
    let curl_status = std::process::Command::new("curl")
        .args(["-sSf", "-o"])
        .arg(&json_path)
        .arg(&pypi_url)
        .status()?;
    if !curl_status.success() {
        return Err("failed to fetch PyPI metadata (check network connection)".into());
    }

    let json_str = std::fs::read_to_string(&json_path)?;
    let _ = std::fs::remove_file(&json_path);

    // Parse JSON to find matching wheel URL
    let json: serde_json::Value = serde_json::from_str(&json_str)?;
    let urls = json["urls"].as_array()
        .ok_or("unexpected PyPI JSON format")?;

    let wheel_url = urls.iter()
        .filter_map(|entry| {
            let filename = entry["filename"].as_str()?;
            if filename.ends_with(".whl") && filename.contains(platform_tag) {
                entry["url"].as_str().map(|s| s.to_string())
            } else {
                None
            }
        })
        .next()
        .ok_or_else(|| format!(
            "no wheel found for platform '{}' at version {}",
            platform_tag, IREE_COMPILER_VERSION
        ))?;

    // Download the wheel
    let wheel_path = std::env::temp_dir().join("sheaf-iree-compiler.whl");
    let curl_status = std::process::Command::new("curl")
        .args(["-sSfL", "-o"])
        .arg(&wheel_path)
        .arg(&wheel_url)
        .status()?;
    if !curl_status.success() {
        return Err("failed to download IREE compiler wheel".into());
    }

    // Extract iree-compile and libIREECompiler from the wheel (ZIP file)
    let lib_name = compiler_lib_name();
    let unzip_status = std::process::Command::new("unzip")
        .args(["-j", "-o"])
        .arg(&wheel_path)
        .arg("iree/compiler/_mlir_libs/iree-compile")
        .arg("iree/compiler/_mlir_libs/iree-lld")
        .arg(format!("iree/compiler/_mlir_libs/{}", lib_name))
        .arg("-d")
        .arg(&dir)
        .stdout(std::process::Stdio::null())
        .status()?;
    let _ = std::fs::remove_file(&wheel_path);
    if !unzip_status.success() {
        return Err("failed to extract iree-compile from wheel".into());
    }

    // Ensure binaries are executable
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        for bin in &["iree-compile", "iree-lld"] {
            let path = dir.join(bin);
            if let Ok(meta) = std::fs::metadata(&path) {
                let mut perms = meta.permissions();
                perms.set_mode(0o755);
                let _ = std::fs::set_permissions(&path, perms);
            }
        }
    }

    // Write version file
    std::fs::write(dir.join("version"), IREE_COMPILER_VERSION)?;

    let binary = dir.join("iree-compile");
    sheaf_msg!("sheaf: compiler successfully installed in {}", dir.display());
    Ok(binary.to_string_lossy().to_string())
}

fn which(name: &str) -> Option<String> {
    std::env::var("PATH").ok().and_then(|path_var| {
        path_var.split(':').find_map(|dir| {
            let candidate = format!("{}/{}", dir, name);
            if std::path::Path::new(&candidate).exists() {
                Some(candidate)
            } else {
                None
            }
        })
    })
}
