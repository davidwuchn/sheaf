// Copyright (c) 2026 Damien Boureille
// Licensed under the MIT License.

//! Shared VMFB loading logic: manifest validation, IREE session creation,
//! function tagging, and signature loading.
//! Used by both `eval_source_with_path` (sheaf run) and the `(use)` form.

use std::path::Path;
use crate::lowering::effects::has_side_effects;
use crate::core::expr::CompilerContext;
use crate::core::inference::FunctionSignature;

/// Try to load a module.vmfb from the directory of a Sheaf source file.
///
/// Looks for `module.json` in the same directory as `shf_path` to determine
/// which VMFB to load and validates freshness via content hashes.
/// Requires a manifest with the module name exported by the VMFB.
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

    let manifest_path = dir.join("module.json");
    let (vmfb_path, module_name, valid_fns, signatures) =
        match std::fs::read_to_string(&manifest_path) {
            Ok(manifest_str) => match validate_manifest(&manifest_str, dir, compiler, &pure_fns) {
                Some(validated) => validated,
                None => return false,
            },
            Err(_) => {
                if dir.join("module.vmfb").exists() {
                    crate::sheaf_msg!(
                        "warning: refusing module.vmfb without module.json module_name"
                    );
                }
                return false;
            }
        };

    if valid_fns.is_empty() {
        return false;
    }

    // Load the VMFB into the process-wide IREE session.
    let vmfb_data = match std::fs::read(&vmfb_path) {
        Ok(data) => data,
        Err(e) => {
            crate::sheaf_msg!("warning: cannot read '{}': {}", vmfb_path.display(), e);
            return false;
        }
    };
    let session = match crate::runtime::iree_session::shared_session() {
        Ok(s) => s,
        Err(e) => {
            crate::sheaf_msg!("warning: JIT engine init failed: {}", e);
            return false;
        }
    };
    let function_keys: Vec<(String, String)> = valid_fns
        .iter()
        .filter_map(|name| {
            compiler
                .registry
                .get(name)
                .map(|function| (name.clone(), function.body_hash()))
        })
        .collect();
    if function_keys.len() != valid_fns.len() {
        return false;
    }

    match session.reserve_precompiled_module(&module_name, &function_keys) {
        Ok(true) => {}
        Ok(false) => {
            crate::sheaf_msg!(
                "warning: precompiled module '{}' conflicts with a loaded or pending AOT module",
                module_name
            );
            return false;
        }
        Err(e) => {
            crate::sheaf_msg!(
                "warning: failed to reserve precompiled module '{}': {}",
                module_name,
                e
            );
            return false;
        }
    }

    if let Err(e) = session.load_vmfb(vmfb_data) {
        if let Err(release_error) = session.release_precompiled_module(&module_name) {
            crate::sheaf_msg!(
                "warning: failed to release precompiled module '{}': {}",
                module_name,
                release_error
            );
        }
        crate::sheaf_msg!("warning: failed to load '{}': {}", vmfb_path.display(), e);
        return false;
    }

    // Tag functions for IREE dispatch and load signatures from manifest
    for fn_name in &valid_fns {
        if let Some(fd) = compiler.registry.get_mut(fn_name)
            && let Some(sig) = signatures.get(fn_name) {
                fd.signature = Some(sig.clone());
            }
    }

    // For functions without manifest signatures, trace at runtime
    for fn_name in &valid_fns {
        let needs_trace = compiler.registry.get(fn_name)
            .map(|fd| fd.signature.is_none())
            .unwrap_or(false);
        if needs_trace
            && let Some(fd) = compiler.registry.get(fn_name).cloned()
            && let Ok(traced_sig) = crate::core::trace::trace_function_signature(compiler, &fd)
            && let Some(fd_mut) = compiler.registry.get_mut(fn_name) {
                        fd_mut.signature = Some(traced_sig);
                    }
    }

    if let Err(e) = session.register_precompiled_functions(&module_name, &function_keys) {
        // IREE cannot unload an appended module, so this module remains unused.
        if let Err(release_error) = session.release_precompiled_module(&module_name) {
            crate::sheaf_msg!(
                "warning: failed to release unusable precompiled module '{}': {}",
                module_name,
                release_error
            );
        }
        crate::sheaf_msg!(
            "warning: loaded but could not register precompiled module '{}': {}",
            module_name,
            e
        );
        return false;
    }

    crate::sheaf_msg!("using module.vmfb: {}", valid_fns.join(", "));
    true
}

/// Validate a module.json and return its path, module name, functions, and signatures.
/// Returns None if stale or invalid.
fn validate_manifest(
    manifest_str: &str,
    dir: &Path,
    compiler: &CompilerContext,
    candidate_fns: &[String],
) -> Option<(
    std::path::PathBuf,
    String,
    Vec<String>,
    std::collections::HashMap<String, FunctionSignature>,
)> {
    let manifest: serde_json::Value = match serde_json::from_str(manifest_str) {
        Ok(v) => v,
        Err(e) => {
            crate::sheaf_msg!("warning: invalid module.json: {}", e);
            return None;
        }
    };

    let module_name = match manifest.get("module_name").and_then(|value| value.as_str()) {
        Some(name) if is_safe_module_name(name) => name.to_string(),
        Some(name) => {
            crate::sheaf_msg!("warning: invalid module.json module_name '{}'", name);
            return None;
        }
        None => {
            crate::sheaf_msg!("warning: module.json is missing module_name");
            return None;
        }
    };

    // Resolve VMFB path from manifest
    let vmfb_name = manifest.get("vmfb")
        .and_then(|v| v.as_str())
        .unwrap_or("module.vmfb");
    let vmfb_path = dir.join(vmfb_name);
    if !vmfb_path.exists() {
        crate::sheaf_msg!(
            "warning: '{}' referenced by module.json not found",
            vmfb_path.display()
        );
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
                    crate::sheaf_msg!(
                        "warning: module.json is stale ('{}' changed), run `sheaf build`",
                        name
                    );
                    return None;
                }
            }
            None => {
                crate::sheaf_msg!(
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

    Some((vmfb_path, module_name, valid, signatures))
}

fn is_safe_module_name(name: &str) -> bool {
    let mut chars = name.chars();
    matches!(chars.next(), Some(c) if c.is_ascii_alphabetic() || c == '_')
        && chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
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

    let param_types: Vec<crate::lowering::stablehlo::StableHLOType> = fd.params.iter()
        .map(|p| {
            params_obj.get(p)
                .and_then(|v| v.as_str())
                .and_then(crate::lowering::stablehlo::StableHLOType::parse)
                .unwrap_or_else(crate::lowering::stablehlo::StableHLOType::scalar_f32)
        })
        .collect();

    let return_type = crate::lowering::stablehlo::StableHLOType::parse(return_str)
        .unwrap_or_else(crate::lowering::stablehlo::StableHLOType::scalar_f32);

    Some(FunctionSignature {
        param_types,
        return_type,
        return_dict_keys: None,
        arg_type_layouts: vec![],
        captured_scalars: std::collections::HashMap::new(),
    })
}

#[cfg(test)]
mod tests {
    use super::{is_safe_module_name, validate_manifest};
    use crate::core::expr::CompilerContext;
    use std::path::Path;

    #[test]
    fn validates_module_names() {
        for name in ["module", "aot_model_42", "_private"] {
            assert!(is_safe_module_name(name));
        }
        for name in ["", "42model", "module.name", "module-name", "module name"] {
            assert!(!is_safe_module_name(name));
        }
    }

    #[test]
    fn rejects_manifest_without_module_name_before_loading() {
        let manifest = r#"{"vmfb":"missing.vmfb","functions":{}}"#;
        assert!(
            validate_manifest(manifest, Path::new("."), &CompilerContext::new(), &[]).is_none()
        );
    }
}
