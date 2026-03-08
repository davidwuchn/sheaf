// Copyright (c) 2026 Damien Boureille
// Licensed under the MIT License.

//! Shared VMFB loading logic: manifest validation, IREE session creation,
//! function tagging, and signature loading.
//! Used by both `eval_source_with_path` (sheaf run) and the `(use)` form.

use std::path::Path;
use std::sync::Arc;

use crate::compiler::effects::has_side_effects;
use crate::core::compiler::CompilerContext;
use crate::core::inference::FunctionSignature;
use crate::runtime::iree_session::IreeSession;

/// Try to load a module.vmfb from the directory of a Sheaf source file.
///
/// Looks for `module.json` in the same directory as `shf_path` to determine
/// which VMFB to load and validates freshness via content hashes.
/// Falls back to `module.vmfb` with timestamp check if no manifest.
///
/// `candidate_fns`: function names to consider for IREE dispatch.
/// Only pure (side-effect-free) functions among these will be tagged.
///
/// Returns `true` if functions were successfully tagged for IREE dispatch.
pub fn try_load_vmfb(
    compiler: &mut CompilerContext,
    shf_path: &Path,
    candidate_fns: &[String],
) -> bool {
    if compiler.disable_vmfb {
        return false;
    }
    let dir = shf_path.parent().unwrap_or_else(|| Path::new("."));

    // Skip if we already checked this directory (prevents duplicate warnings)
    let canon_dir = dir.canonicalize().unwrap_or_else(|_| dir.to_path_buf());
    if !compiler.checked_vmfb_dirs.insert(canon_dir) {
        return false;
    }

    // Filter to pure functions only
    let pure_fns: Vec<String> = candidate_fns
        .iter()
        .filter(|name| {
            compiler
                .registry
                .get(*name)
                .and_then(|fd| fd.body_compiled.as_ref())
                .map(|body| !has_side_effects(body))
                .unwrap_or(false)
        })
        .cloned()
        .collect();

    if pure_fns.is_empty() {
        return false;
    }

    // Try module.json first, then fallback to bare VMFB with timestamp check
    let manifest_path = dir.join("module.json");
    let (vmfb_path, valid_fns, signatures) = match std::fs::read_to_string(&manifest_path) {
        Ok(manifest_str) => {
            match validate_manifest(&manifest_str, dir, compiler, &pure_fns) {
                Some((vmfb, fns, sigs)) => (vmfb, fns, sigs),
                None => return false,
            }
        }
        Err(_) => {
            // No manifest — try module.vmfb with timestamp fallback
            let vmfb_path = dir.join("module.vmfb");
            if !vmfb_path.exists() {
                return false;
            }
            let is_fresh = match (
                std::fs::metadata(shf_path).and_then(|m| m.modified()),
                std::fs::metadata(&vmfb_path).and_then(|m| m.modified()),
            ) {
                (Ok(shf_time), Ok(vmfb_time)) => vmfb_time >= shf_time,
                _ => false,
            };
            if !is_fresh {
                return false;
            }
            (vmfb_path, pure_fns, std::collections::HashMap::new())
        }
    };

    if valid_fns.is_empty() {
        return false;
    }

    // Load the VMFB into an IREE session
    let vmfb_data = match std::fs::read(&vmfb_path) {
        Ok(data) => data,
        Err(e) => {
            eprintln!("warning: cannot read '{}': {}", vmfb_path.display(), e);
            return false;
        }
    };
    let mut session = match IreeSession::new() {
        Ok(s) => s,
        Err(e) => {
            eprintln!("warning: IREE init failed: {}", e);
            return false;
        }
    };
    if let Err(e) = session.load_vmfb(vmfb_data) {
        eprintln!("warning: failed to load '{}': {}", vmfb_path.display(), e);
        return false;
    }

    let session_idx = compiler.vmfb_sessions.len();
    compiler.vmfb_sessions.push(Arc::new(session));

    // Tag functions for IREE dispatch and load signatures from manifest
    for fn_name in &valid_fns {
        if let Some(fd) = compiler.registry.get_mut(fn_name) {
            fd.vmfb_session_idx = Some(session_idx);
            if let Some(sig) = signatures.get(fn_name) {
                fd.signature = Some(sig.clone());
            }
        }
    }

    // For functions without manifest signatures, trace at runtime
    for fn_name in &valid_fns {
        let needs_trace = compiler.registry.get(fn_name)
            .map(|fd| fd.signature.is_none())
            .unwrap_or(false);
        if needs_trace {
            if let Some(fd) = compiler.registry.get(fn_name).cloned() {
                if let Ok(traced_sig) = crate::core::trace::trace_function_signature(compiler, &fd) {
                    if let Some(fd_mut) = compiler.registry.get_mut(fn_name) {
                        fd_mut.signature = Some(traced_sig);
                    }
                }
            }
        }
    }

    eprintln!(
        "using module.vmfb: {}",
        valid_fns.join(", ")
    );
    true
}

/// Validate a module.json and return (vmfb_path, valid_fns, signatures).
/// Returns None if stale or invalid.
fn validate_manifest(
    manifest_str: &str,
    dir: &Path,
    compiler: &CompilerContext,
    candidate_fns: &[String],
) -> Option<(std::path::PathBuf, Vec<String>, std::collections::HashMap<String, FunctionSignature>)> {
    let manifest: serde_json::Value = match serde_json::from_str(manifest_str) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("warning: invalid module.json: {}", e);
            return None;
        }
    };

    // Resolve VMFB path from manifest
    let vmfb_name = manifest.get("vmfb")
        .and_then(|v| v.as_str())
        .unwrap_or("module.vmfb");
    let vmfb_path = dir.join(vmfb_name);
    if !vmfb_path.exists() {
        eprintln!("warning: '{}' referenced by module.json not found", vmfb_path.display());
        return None;
    }

    let functions = manifest.get("functions").and_then(|f| f.as_object())?;

    // Validate hashes for every function in the manifest
    for (name, entry) in functions {
        let expected_hash = match entry.get("hash").and_then(|h| h.as_str()) {
            Some(h) => h,
            None => continue,
        };
        match compiler.registry.get(name) {
            Some(fd) => {
                if fd.body_hash() != expected_hash {
                    eprintln!(
                        "warning: module.json is stale ('{}' changed), run `sheaf build`",
                        name
                    );
                    return None;
                }
            }
            None => {
                eprintln!(
                    "warning: module.json is stale ('{}' not found), run `sheaf build`",
                    name
                );
                return None;
            }
        }
    }

    // Extract signatures from manifest
    let mut signatures = std::collections::HashMap::new();
    for (name, entry) in functions {
        if let Some(sig) = parse_manifest_signature(entry, compiler, name) {
            signatures.insert(name.clone(), sig);
        }
    }

    // Return only candidate functions that are both pure AND in the manifest
    let manifest_names: std::collections::HashSet<&str> =
        functions.keys().map(|s| s.as_str()).collect();
    let valid: Vec<String> = candidate_fns
        .iter()
        .filter(|name| manifest_names.contains(name.as_str()))
        .cloned()
        .collect();

    Some((vmfb_path, valid, signatures))
}

/// Parse a FunctionSignature from a manifest entry's params/returns fields.
fn parse_manifest_signature(
    entry: &serde_json::Value,
    compiler: &CompilerContext,
    fn_name: &str,
) -> Option<FunctionSignature> {
    let params_obj = entry.get("params")?.as_object()?;
    let return_str = entry.get("returns")?.as_str()?;

    // Get param order from the function definition in the registry
    let fd = compiler.registry.get(fn_name)?;

    let param_types: Vec<crate::compiler::stablehlo::StableHLOType> = fd.params.iter()
        .map(|p| {
            params_obj.get(p)
                .and_then(|v| v.as_str())
                .and_then(|s| crate::compiler::stablehlo::StableHLOType::parse(s))
                .unwrap_or_else(crate::compiler::stablehlo::StableHLOType::scalar_f32)
        })
        .collect();

    let return_type = crate::compiler::stablehlo::StableHLOType::parse(return_str)
        .unwrap_or_else(crate::compiler::stablehlo::StableHLOType::scalar_f32);

    Some(FunctionSignature {
        param_types,
        return_type,
        return_dict_keys: None,
    })
}
