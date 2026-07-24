// Copyright (c) 2026 Damien Boureille
// Licensed under the MIT License.

//! JIT auto-compilation: transparently compile pure functions on first call.
//!
//! When the interpreter calls a function that has no pre-compiled VMFB,
//! the JIT attempts to compile it on the fly via the following pipeline:
//! type inference -> dict lowering -> inlining -> codegen -> MLIR
//! -> iree-compile -> load VMFB. On success, subsequent calls dispatch via IREE.
//! On failure the result is remembered so the same artifact is not retried:
//! structural rejections by compiled definition identity, per-variant
//! failures by cache key, and value-and-grad failures by their own key.
//! The interpreter then handles the call normally.

use crate::SheafError;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::PathBuf;
use std::sync::{Condvar, Mutex, MutexGuard, OnceLock};

/// Serializes cache publication and module loading for the shared IREE session.
static JIT_COMPILATION_LOCK: Mutex<()> = Mutex::new(());

/// Metadata for a module loaded in the process-wide IREE session.
#[derive(Clone, Debug, PartialEq)]
pub struct CompiledModuleInfo {
    pub module_name: String,
    pub signature: FunctionSignature,
}

/// Maps JIT variant identities to metadata for modules loaded in the shared session.
static JIT_MODULE_CATALOG: OnceLock<Mutex<HashMap<JitCacheKey, CompiledModuleInfo>>> = OnceLock::new();

/// Cached CUDA target architecture (e.g. "sm_75"), detected once via NVML.
static CUDA_TARGET: OnceLock<Option<String>> = OnceLock::new();

pub fn detect_cuda_target() -> Option<String> {
    CUDA_TARGET.get_or_init(|| {
        let nvml = nvml_wrapper::Nvml::init()
            .or_else(|_| nvml_wrapper::Nvml::builder()
                .lib_path(std::ffi::OsStr::new("libnvidia-ml.so.1"))
                .init());
        let nvml = match nvml {
            Ok(n) => n,
            Err(e) => { eprintln!("nvml: init failed: {}", e); return None; }
        };
        let device = match nvml.device_by_index(0) {
            Ok(d) => d,
            Err(e) => { eprintln!("nvml: device_by_index failed: {}", e); return None; }
        };
        let cc = match device.cuda_compute_capability() {
            Ok(c) => c,
            Err(e) => { eprintln!("nvml: compute_capability failed: {}", e); return None; }
        };
        let target = format!("sm_{}{}", cc.major, cc.minor);
        Some(target)
    }).clone()
}

use super::toolchain::{ensure_toolchain, find_iree_compile, IREE_COMPILER_VERSION};

use crate::autodiff::{reverse::{reverse_grad, to_anf}, simplify, transforms::cse};
use crate::core::expr::{CompiledExpr, FunctionDef};
use crate::lowering::codegen::{collect_tuple_leaves, expand_tuple_to_symbols, CodeGenerator};
use crate::lowering::config::{layout_to_index_map, lower_get_calls};
use crate::lowering::effects::{collect_effects, collect_hof_calls};
use crate::lowering::jit_eligibility;
use crate::lowering::stablehlo::{Register, StableHLOEmitter};
use crate::lowering::transforms::{
    collect_shape_gtes, extract_scalar_constants, filter_constants_for_shape_positions,
    lower_inlined_gets, propagate_let_layouts, resolve_static_constants, substitute_scalar_param,
    unroll_reduces,
};
use crate::sheaf_msg;

/// Identity of one JIT-compiled runtime variant.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct JitCacheKey {
    definition_identity: String,
    function_name: String,
    argument_types: Vec<String>,
    argument_layouts: Vec<Vec<(Vec<String>, Vec<i64>, Vec<usize>)>>,
    shape_scalars: Vec<(String, Vec<usize>, u64)>,
}

impl JitCacheKey {
    fn suffix(&self) -> String {
        use std::hash::{Hash, Hasher};

        let mut first = StableHasher(0xcbf29ce484222325);
        let mut second = StableHasher(0x84222325cbf29ce4);
        self.hash(&mut first);
        self.hash(&mut second);
        format!("{:016x}{:016x}", first.finish(), second.finish())
    }
}

struct StableHasher(u64);

impl std::hash::Hasher for StableHasher {
    fn finish(&self) -> u64 {
        self.0
    }

    fn write(&mut self, bytes: &[u8]) {
        for byte in bytes {
            self.0 ^= u64::from(*byte);
            self.0 = self.0.wrapping_mul(0x100000001b3);
        }
    }
}

fn function_definition_identity(func_def: &FunctionDef, registry: &HashMap<String, FunctionDef>) -> String {
    let mut pending = vec![func_def.name.clone()];
    let mut seen = HashSet::new();
    let mut identities = Vec::new();

    while let Some(name) = pending.pop() {
        if !seen.insert(name.clone()) {
            continue;
        }
        let Some(definition) = registry.get(&name) else {
            continue;
        };
        identities.push((name, definition.body_hash()));
        if let Some(body) = &definition.body_compiled {
            let mut callees: Vec<_> = crate::lowering::call_graph::direct_callees(body)
                .into_iter()
                .filter(|callee| registry.contains_key(callee))
                .collect();
            callees.sort();
            pending.extend(callees);
        }
    }
    identities.sort();
    format!("{:?}", identities)
}

type ScalarPosition = (String, Vec<usize>);
type RuntimeLayout = Vec<(Vec<String>, Vec<i64>, Vec<usize>)>;

#[derive(Clone, Eq, Hash, PartialEq)]
struct ShapeAnalysisCacheKey {
    definition_identity: String,
    argument_types: Vec<String>,
    argument_layouts: Vec<RuntimeLayout>,
}

enum ShapeAnalysisState {
    Computing,
    Ready(Vec<ScalarPosition>),
}

struct ShapeAnalysisCache {
    entries: Mutex<HashMap<ShapeAnalysisCacheKey, ShapeAnalysisState>>,
    ready: Condvar,
}

struct ShapeAnalysisReservation {
    key: ShapeAnalysisCacheKey,
    active: bool,
}

impl ShapeAnalysisReservation {
    fn finish(&mut self) {
        self.active = false;
    }
}

impl Drop for ShapeAnalysisReservation {
    fn drop(&mut self) {
        if !self.active {
            return;
        }
        let cache = shape_analysis_cache();
        if let Ok(mut entries) = cache.entries.lock() {
            entries.remove(&self.key);
            cache.ready.notify_all();
        }
    }
}

/// Process-local cache of sorted scalar positions consumed by shape expressions.
static SHAPE_ANALYSIS_CACHE: OnceLock<ShapeAnalysisCache> = OnceLock::new();

fn shape_analysis_cache() -> &'static ShapeAnalysisCache {
    SHAPE_ANALYSIS_CACHE.get_or_init(|| ShapeAnalysisCache {
        entries: Mutex::new(HashMap::new()),
        ready: Condvar::new(),
    })
}

fn shape_analysis_lock(
    cache: &ShapeAnalysisCache,
) -> Result<MutexGuard<'_, HashMap<ShapeAnalysisCacheKey, ShapeAnalysisState>>, SheafError> {
    cache.entries.lock().map_err(|_| SheafError::Compile {
        message: "JIT cache key construction failed: shape analysis cache lock is poisoned".to_string(),
        location: crate::core::error::SourceLocation::unknown(),
    })
}

fn runtime_layouts(args: &[Value], params: &[String]) -> Vec<RuntimeLayout> {
    args.iter()
        .enumerate()
        .map(|(index, arg)| {
            let param = params.get(index).map(String::as_str).unwrap_or("");
            value_to_param_layout(param, arg)
                .map(|layout| {
                    layout.fields.into_iter().map(|field| {
                        (field.path, field.shape, field.tuple_index)
                    }).collect()
                })
                .unwrap_or_default()
        })
        .collect()
}

/// Builds the identity used for both JIT lookup and compilation.
pub fn cache_key_for_function(
    func_def: &FunctionDef,
    args: &[Value],
    registry: &HashMap<String, FunctionDef>,
) -> Option<JitCacheKey> {
    let body = func_def.body_compiled.clone()?;
    let definition_identity = function_definition_identity(func_def, registry);
    let argument_types = args.iter().map(|arg| {
        value_to_stablehlo_type(arg).map(|ty| ty.to_mlir()).unwrap_or_else(|_| "<invalid>".to_string())
    }).collect::<Vec<_>>();
    let argument_layouts = runtime_layouts(args, &func_def.params);
    let analysis_key = ShapeAnalysisCacheKey {
        definition_identity: definition_identity.clone(),
        argument_types: argument_types.clone(),
        argument_layouts: argument_layouts.clone(),
    };

    let mut constants = HashMap::new();
    let mut index_maps = Vec::new();
    for (index, arg) in args.iter().enumerate() {
        let param = func_def.params.get(index)?.clone();
        let index_map = value_to_param_layout(&param, arg)
            .map(|layout| layout_to_index_map(&layout))
            .unwrap_or_else(|| std::iter::once((Vec::new(), Vec::new())).collect());
        extract_scalar_constants(arg, &param, &index_map, &mut constants);
        index_maps.push((param, index_map));
    }

    let shape_positions = match cached_shape_positions(&analysis_key, body, registry, &index_maps) {
        Ok(positions) => positions,
        Err(error) => {
            sheaf_msg!("jit: cache key construction failed: {}", error);
            return None;
        }
    };
    let shape_scalars = constants.into_iter()
        .filter(|(position, _)| shape_positions.binary_search(position).is_ok())
        .collect::<BTreeMap<_, _>>()
        .into_iter()
        .map(|((param, indices), value)| (param, indices, value.to_bits()))
        .collect();

    Some(JitCacheKey {
        definition_identity,
        function_name: func_def.name.clone(),
        argument_types,
        argument_layouts,
        shape_scalars,
    })
}

fn cached_shape_positions(
    key: &ShapeAnalysisCacheKey,
    mut body: CompiledExpr,
    registry: &HashMap<String, FunctionDef>,
    index_maps: &[(String, BTreeMap<Vec<String>, Vec<usize>>)],
) -> Result<Vec<ScalarPosition>, SheafError> {
    let cache = shape_analysis_cache();
    let mut reservation = loop {
        let entries = shape_analysis_lock(cache)?;
        match entries.get(key) {
            Some(ShapeAnalysisState::Ready(positions)) => return Ok(positions.clone()),
            Some(ShapeAnalysisState::Computing) => {
                let entries = cache.ready.wait(entries).map_err(|_| SheafError::Compile {
                    message: "JIT cache key construction failed: shape analysis cache lock is poisoned".to_string(),
                    location: crate::core::error::SourceLocation::unknown(),
                })?;
                drop(entries);
            }
            None => {
                drop(entries);
                let mut entries = shape_analysis_lock(cache)?;
                if entries.contains_key(key) {
                    continue;
                }
                entries.insert(key.clone(), ShapeAnalysisState::Computing);
                break ShapeAnalysisReservation { key: key.clone(), active: true };
            }
        }
    };

    for (param, index_map) in index_maps {
        body = lower_get_calls(&body, param, index_map);
    }
    body = crate::autodiff::inline_function_calls(&body, registry);
    for (param, index_map) in index_maps {
        body = lower_get_calls(&body, param, index_map);
    }
    body = lower_inlined_gets(&body, index_maps);
    let mut positions = collect_shape_gtes(&body).into_iter().collect::<Vec<_>>();
    positions.sort();

    let mut entries = shape_analysis_lock(cache)?;
    entries.insert(key.clone(), ShapeAnalysisState::Ready(positions.clone()));
    cache.ready.notify_all();
    reservation.finish();
    Ok(positions)
}

/// Returns an MLIR-safe module name for a JIT variant.
pub fn module_name_for(key: &JitCacheKey) -> String {
    let safe_name = key.function_name.replace('-', "_").replace('?', "_q").replace('!', "_b");
    format!("jit_{}__{}", safe_name, key.suffix())
}

/// Count AST nodes in an expression tree. Used to bail out of JIT
/// when inlining or unrolling produces a graph that is too large.
fn expr_node_count(expr: &CompiledExpr) -> usize {
    match expr {
        CompiledExpr::FunctionCall { args, .. } => {
            1 + args.iter().map(expr_node_count).sum::<usize>()
        }
        CompiledExpr::Let { bindings, body } => {
            1 + bindings
                .iter()
                .map(|(_, v)| expr_node_count(v))
                .sum::<usize>()
                + expr_node_count(body)
        }
        CompiledExpr::Do(exprs) => 1 + exprs.iter().map(expr_node_count).sum::<usize>(),
        CompiledExpr::If {
            condition,
            then_branch,
            else_branch,
        } => {
            1 + expr_node_count(condition)
                + expr_node_count(then_branch)
                + else_branch.as_ref().map_or(0, |e| expr_node_count(e))
        }
        CompiledExpr::Lambda { body, .. } => 1 + expr_node_count(body),
        CompiledExpr::LambdaCall { callee, args } => {
            1 + expr_node_count(callee) + args.iter().map(expr_node_count).sum::<usize>()
        }
        CompiledExpr::Vector(elems) => 1 + elems.iter().map(expr_node_count).sum::<usize>(),
        _ => 1,
    }
}

const MAX_VAG_GRAPH_NODES: usize = 10_000;
use crate::core::inference::{infer_function_signature_with_known, FunctionSignature};
use crate::core::trace::{value_to_param_layout, value_to_stablehlo_type};
use crate::interpreter::value::Value;
use crate::StableHLOType;

fn vag_module_name(key: &str) -> String {
    use std::hash::{Hash, Hasher};

    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    key.hash(&mut hasher);
    format!("vag_{:016x}", hasher.finish())
}

pub struct JitCompiler {
    iree_compile_path: Option<String>,
    target_backend: String,
    /// Structural rejections independent of runtime arguments, keyed by
    /// compiled definition identity.
    failed_definitions: HashSet<String>,
    /// Per-variant compilation failures, keyed by the exact cache key.
    failed_keys: HashSet<JitCacheKey>,
    /// Value-and-grad failures, keyed by their `__vag_` identity.
    failed_vag: HashSet<String>,
    variants: HashMap<JitCacheKey, (String, FunctionSignature)>,
    last_vag_fail_reason: Option<String>,
    /// Cache compiled VAG metadata: vag_key -> (module_name, signature, param_names)
    vag_cache: HashMap<String, (String, FunctionSignature, Vec<String>)>,
}

impl JitCompiler {
    pub fn new() -> Self {
        let target_backend = Self::detect_target_backend();
        let iree_compile_path = find_iree_compile().or_else(|| {
            match ensure_toolchain() {
                Ok(path) => Some(path),
                Err(e) => {
                    // Check if cached VMFBs exist, if so, we can still run
                    let has_cache = std::path::Path::new("__sheaf__/manifest.json").exists();
                    if has_cache {
                        sheaf_msg!("sheaf: compiler toolchain not available, using cached compilations");
                    } else {
                        eprintln!("error: Could not download the compiler toolchain and no cached compilation found.");
                        eprintln!("       Reason: {}", e);
                        eprintln!("       Install curl and unzip, or set IREE_COMPILE to a local iree-compile binary.");
                        std::process::exit(1);
                    }
                    None
                }
            }
        });
        Self {
            iree_compile_path,
            target_backend,
            failed_definitions: HashSet::new(),
            failed_keys: HashSet::new(),
            failed_vag: HashSet::new(),
            variants: HashMap::new(),
            last_vag_fail_reason: None,
            vag_cache: HashMap::new(),
        }
    }

    fn detect_target_backend() -> String {
        if let Some(d) = crate::core::config::device_override() {
            return match d {
                "cpu" => "llvm-cpu",
                "metal" => "metal-spirv",
                "cuda" => "cuda",
                "vulkan" => "vulkan-spirv",
                _ => "llvm-cpu",
            }.to_string();
        }
        // Use the backend selected by the shared session when available
        if let Some(backend) = crate::runtime::iree_session::IreeSession::cached_target_backend() {
            return backend.to_string();
        }
        // Fallback: create a session to probe available drivers
        if let Ok(session) = crate::runtime::iree_session::shared_session() {
            return session.target_backend().to_string();
        }
        "llvm-cpu".to_string()
    }


    pub(crate) fn variant_for(&self, key: &JitCacheKey) -> Option<&(String, FunctionSignature)> {
        self.variants.get(key)
    }

    /// Returns cloned metadata for a module loaded by any JIT compiler in this process.
    pub fn module_for_key(
        &self,
        key: &JitCacheKey,
    ) -> Result<Option<CompiledModuleInfo>, SheafError> {
        let catalog = JIT_MODULE_CATALOG.get_or_init(|| Mutex::new(HashMap::new()));
        let catalog = catalog.lock().map_err(|_| SheafError::Runtime {
            message: "JIT module catalog lock is poisoned".to_string(),
            location: None,
        })?;
        Ok(catalog.get(key).cloned())
    }

    fn publish_module(
        &self,
        key: JitCacheKey,
        module: CompiledModuleInfo,
    ) -> Result<(), SheafError> {
        let catalog = JIT_MODULE_CATALOG.get_or_init(|| Mutex::new(HashMap::new()));
        let mut catalog = catalog.lock().map_err(|_| SheafError::Runtime {
            message: "JIT module catalog lock is poisoned".to_string(),
            location: None,
        })?;
        catalog.insert(key, module);
        Ok(())
    }

    fn register_module(
        &mut self,
        key: JitCacheKey,
        module: CompiledModuleInfo,
    ) -> Result<(), SheafError> {
        self.publish_module(key.clone(), module.clone())?;
        self.variants.insert(key, (module.module_name, module.signature));
        Ok(())
    }

    pub(crate) fn preflight_jit_eligibility(
        &mut self,
        func_def: &FunctionDef,
        registry: &HashMap<String, FunctionDef>,
    ) -> bool {
        let name = &func_def.name;
        if func_def.body_compiled.is_none() {
            return false;
        }
        let definition_identity = function_definition_identity(func_def, registry);
        if self.failed_definitions.contains(&definition_identity) {
            return false;
        }
        if let Err(reason) = jit_eligibility(name, registry) {
            self.failed_definitions.insert(definition_identity);
            self.jit_fail(name, &reason);
            return false;
        }
        true
    }

    pub fn try_jit_compile(
        &mut self,
        func_def: &FunctionDef,
        args: &[Value],
        registry: &HashMap<String, FunctionDef>,
    ) -> Option<FunctionSignature> {
        if !self.preflight_jit_eligibility(func_def, registry) {
            return None;
        }

        let name = &func_def.name;
        let iree_compile = self.iree_compile_path.clone()?;
        let cache_key = cache_key_for_function(func_def, args, registry)?;
        if let Some((_, signature)) = self.variants.get(&cache_key) {
            return Some(signature.clone());
        }
        let module_name = module_name_for(&cache_key);

        // Structural rejections are remembered by definition identity so one
        // bad call graph does not poison a distinct specialization; a failed
        // variant is remembered by its exact cache key instead.
        if self.failed_definitions.contains(&cache_key.definition_identity) {
            return None;
        }
        if self.failed_keys.contains(&cache_key) {
            return None;
        }

        // Skip scalar-only functions (no benefit from IREE)
        let has_tensor = args.iter().any(|a| {
            matches!(a, Value::Tensor { .. } | Value::DeviceBuffer(_) | Value::Dict(_) | Value::Tuple(_))
        });
        if !has_tensor {
            self.failed_definitions
                .insert(cache_key.definition_identity.clone());
            self.jit_fail(name, "scalar-only args");
            return None;
        }

        let transaction = match JIT_COMPILATION_LOCK.lock() {
            Ok(guard) => guard,
            Err(_) => {
                self.jit_fail(name, "JIT compilation lock is poisoned");
                return None;
            }
        };

        let backend = self.target_backend.clone();
        let result = self.compile_function(
            iree_compile.clone(),
            func_def,
            args,
            registry,
            &backend,
            &module_name,
        );
        drop(transaction);
        if let Some(signature) = result {
            let module = CompiledModuleInfo {
                module_name: module_name.clone(),
                signature: signature.clone(),
            };
            if let Err(error) = self.register_module(cache_key, module) {
                self.jit_fail(name, error.short_message());
                return None;
            }
            Some(signature)
        } else {
            // A failed compile_function attempt is specific to this runtime
            // variant: record it by cache key so other shapes stay eligible.
            self.failed_keys.insert(cache_key);
            None
        }
    }

    fn compile_function(
        &mut self,
        iree_compile: String,
        func_def: &FunctionDef,
        args: &[Value],
        registry: &HashMap<String, FunctionDef>,
        target_backend: &str,
        module_name: &str,
    ) -> Option<FunctionSignature> {
        let name = &func_def.name;
        let mut body = func_def.body_compiled.clone()?;

        // Type inference from runtime args
        let mut known_types: Vec<(String, StableHLOType)> = Vec::new();
        let mut param_index_maps: Vec<(String, BTreeMap<Vec<String>, Vec<usize>>)> = Vec::new();
        let mut constants: HashMap<(String, Vec<usize>), f64> = HashMap::new();

        for (param_name, arg_val) in func_def.params.iter().zip(args) {
            let ty = match value_to_stablehlo_type(arg_val) {
                Ok(ty) => ty,
                Err(e) => {
                    self.jit_fail(
                        name,
                        &format!("arg '{}' has {}", param_name, e.short_message()),
                    );
                    return None;
                }
            };
            let imap = value_to_param_layout(param_name, arg_val)
                .map(|layout| layout_to_index_map(&layout))
                .unwrap_or_else(|| std::iter::once((Vec::new(), Vec::new())).collect());
            extract_scalar_constants(arg_val, param_name, &imap, &mut constants);
            body = lower_get_calls(&body, param_name, &imap);
            param_index_maps.push((param_name.clone(), imap));

            known_types.push((param_name.clone(), ty));
        }

        // Signature inference
        let dummy_compiler = crate::core::expr::CompilerContext::new();
        let mut sig = match infer_function_signature_with_known(
            &dummy_compiler,
            &func_def.params,
            &body,
            &known_types,
        ) {
            Ok(s) => s,
            Err(e) => {
                self.jit_fail(name, &format!("signature inference: {}", e));
                return None;
            }
        };

        // Override param types for dict/tuple params
        for (param_name, ty) in &known_types {
            if let Some(idx) = func_def.params.iter().position(|p| p == param_name) {
                sig.param_types[idx] = ty.clone();
            }
        }

        // -vv: log signature and lowered params
        if crate::core::config::verbosity() >= 2 {
            for (pname, pty) in func_def.params.iter().zip(sig.param_types.iter()) {
                if let Some((_, imap)) = param_index_maps.iter().find(|(n, _)| n == pname) {
                    // Collect top-level fields with their types from the tuple
                    let mut top_fields: BTreeMap<usize, (String, String)> = BTreeMap::new();
                    for (path, indices) in imap.iter() {
                        if path.len() == 1 {
                            // Leaf field: resolve type from tuple
                            let ty_str = if let StableHLOType::Tuple(elems, _) = pty {
                                if let Some(t) = elems.get(indices[0]) {
                                    t.to_mlir()
                                } else {
                                    "?".to_string()
                                }
                            } else {
                                "?".to_string()
                            };
                            top_fields
                                .entry(indices[0])
                                .or_insert((path[0].clone(), ty_str));
                        } else if !top_fields.contains_key(&indices[0]) {
                            // Nested field: show as tuple<...>
                            let ty_str = if let StableHLOType::Tuple(elems, _) = pty {
                                if let Some(StableHLOType::Tuple(sub, _)) = elems.get(indices[0]) {
                                    format!("tuple<...> ({} fields)", sub.len())
                                } else {
                                    "tuple<...>".to_string()
                                }
                            } else {
                                "tuple<...>".to_string()
                            };
                            top_fields.insert(indices[0], (path[0].clone(), ty_str));
                        }
                    }
                    for (_, (key, ty_str)) in &top_fields {
                        sheaf_msg!("jit: {} | {}.{}: {}", name, pname, key, ty_str);
                    }
                } else {
                    sheaf_msg!("jit: {} | {}: {}", name, pname, pty.to_mlir());
                }
            }
            sheaf_msg!("jit: {} | return: {}", name, sig.return_type.to_mlir());
        }

        // Capture value layouts for dict/tuple args (for return value reconstruction)
        {
            use crate::core::inference::ValueLayout;
            let mut seen_types = std::collections::HashSet::new();
            for (arg_val, ty) in args.iter().zip(sig.param_types.iter()) {
                let layout = ValueLayout::from_value(arg_val);
                if !matches!(layout, ValueLayout::Leaf) {
                    let type_key = format!("{:?}", ty);
                    if seen_types.insert(type_key) {
                        sig.arg_type_layouts.push((ty.clone(), layout));
                    }
                }
            }
        }

        // Inline user-defined function calls
        body = crate::autodiff::inline_function_calls(&body, registry);

        // Post-inline: re-lower dict access from inlined bodies
        for (param_name, index_map) in &param_index_maps {
            body = lower_get_calls(&body, param_name, index_map);
        }
        body = lower_inlined_gets(&body, &param_index_maps);

        // Compute param_shapes (needed by both preprocess_vag_lambda and resolve_static_constants)
        let param_shapes: HashMap<String, Vec<i64>> = func_def
            .params
            .iter()
            .zip(sig.param_types.iter())
            .filter_map(|(p, ty)| {
                let shape = ty.shape();
                if shape.is_empty() {
                    None
                } else {
                    Some((p.clone(), shape.to_vec()))
                }
            })
            .collect();

        // Filter constants: only bake scalars used in shape-bearing positions
        // (reshape dims, slice bounds, repeat count, ...), resolving Let-bound
        // symbol aliases. Scalars that only feed arithmetic stay live runtime
        // arguments. If a shape-bearing scalar eludes the classifier it will
        // surface as a codegen error (not a silent freeze).
        let filtered_constants =
            filter_constants_for_shape_positions(&constants, &body);

        // Preprocess VAG lambda bodies BEFORE resolve_static_constants so that
        // VAG lambdas get their own resolve_static_constants with correct (unshadowed)
        // param_shapes. Then the outer resolve_static_constants can safely skip
        // Lambda bodies since they're already resolved.
        let arity_err: std::cell::Cell<Option<SheafError>> = std::cell::Cell::new(None);
        body = self.preprocess_vag_lambda(
            &body, registry, &filtered_constants, &param_shapes, &param_index_maps, &known_types,
            &arity_err,
        );
        if let Some(err) = arity_err.into_inner() {
            self.jit_fail(name, &err.short_message());
            return None;
        }

        body = resolve_static_constants(&body, &filtered_constants, &param_shapes, false);

        body = match crate::lowering::transforms::lower_tuples_and_destructuring(
            body, &param_shapes,
        ) {
            Ok(b) => b,
            Err(e) => {
                self.jit_fail(name, &e.short_message());
                return None;
            }
        };

        // Build key layouts for codegen
        let mut tuple_key_layouts: HashMap<String, BTreeMap<String, usize>> = HashMap::new();
        let mut idx_to_key: HashMap<(String, usize), String> = HashMap::new();
        for (param_name, index_map) in &param_index_maps {
            for (key_path, indices) in index_map {
                for depth in 0..key_path.len() {
                    let parent = if depth == 0 {
                        param_name.clone()
                    } else {
                        key_path[depth - 1].clone()
                    };
                    let child = &key_path[depth];
                    let idx = indices[depth];
                    tuple_key_layouts
                        .entry(parent.clone())
                        .or_default()
                        .entry(child.clone())
                        .or_insert(idx);
                    idx_to_key
                        .entry((parent, idx))
                        .or_insert_with(|| child.clone());
                }
            }
        }
        propagate_let_layouts(&body, &idx_to_key, &mut tuple_key_layouts);

        // Collect scalar param values for constant propagation in codegen.
        // Scalar f32 params (e.g. top-k=40, temp=0.8) get their values recorded
        // so shape-critical ops (top_k slice sizes) can resolve K at compile time.
        let scalar_param_values: Vec<(String, f64)> = func_def
            .params
            .iter()
            .zip(args.iter())
            .filter_map(|(name, val)| match val {
                Value::Float(f) => Some((name.clone(), *f as f64)),
                Value::Int(n) => Some((name.clone(), *n as f64)),
                _ => None,
            })
            .collect();

        // Codegen (catch panics gracefully)
        let codegen_result = {
            let registry_clone = registry.clone();
            let params_clone = func_def.params.clone();
            let param_types = sig.param_types.clone();
            let return_type = sig.return_type.clone();
            let body_clone = body.clone();
            let name_clone = name.clone();
            let constants_clone = constants.clone();
            let param_index_maps_clone = param_index_maps.clone();
            let scalar_values_clone = scalar_param_values.clone();
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
                let mut codegen = CodeGenerator::with_function_params(
                    registry_clone,
                    &params_clone,
                    &param_types,
                );
                codegen.set_tuple_key_layouts(tuple_key_layouts);
                codegen.set_idx_to_key(idx_to_key);
                codegen.set_scalar_constants(constants_clone);
                codegen.set_param_index_maps(param_index_maps_clone);
                codegen.set_scalar_param_values(&scalar_values_clone);
                codegen.emit_func_declaration(&name_clone, &body_clone, &param_types, &return_type)
            }))
        };

        let (mlir_decl, actual_return_ty) = match codegen_result {
            Ok(Ok(result)) => result,
            Ok(Err(e)) => {
                self.jit_fail(name, &format!("codegen: {}", e));
                return None;
            }
            Err(_) => {
                self.jit_fail(name, "codegen: internal panic");
                return None;
            }
        };

        sig.return_type = actual_return_ty;

        if crate::core::config::verbosity() >= 2 {
            sheaf_msg!("jit: {} | codegen return: {}", name, sig.return_type.to_mlir());
            sheaf_msg!("jit: {} | return_dict_keys: {:?}", name, sig.return_dict_keys);
        }

        // Register layout for return type (may differ from param type due to
        // scalar promotion, e.g. ScalarI64->scalar_f32 after adam-step)
        if let StableHLOType::Tuple(ret_elems, ret_keys) = &sig.return_type {
            // Only match return layout to a param layout if the return type already
            // has dict keys AND those keys match a param layout's keys.
            // A plain tuple (ret_keys == None) or mismatched keys must NOT inherit
            // dict structure from an unrelated param.
            if ret_keys.is_some()
                && !sig
                    .arg_type_layouts
                    .iter()
                    .any(|(t, _)| t == &sig.return_type)
            {
                for (t, layout) in sig.arg_type_layouts.clone() {
                    if let StableHLOType::Tuple(param_elems, param_keys) = &t {
                        if param_elems.len() == ret_elems.len() && param_keys == ret_keys {
                            sig.arg_type_layouts.push((sig.return_type.clone(), layout));
                            break;
                        }
                    }
                }
            }
        }

        let mlir = StableHLOEmitter::emit_module_named(Some(module_name), &[mlir_decl]);

        if crate::core::config::verbosity() >= 2 {
            sheaf_msg!("jit: {} | MLIR {} lines", name, mlir.lines().count());
        }

        // Content hash for staleness check
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        mlir.hash(&mut hasher);
        target_backend.hash(&mut hasher);
        IREE_COMPILER_VERSION.hash(&mut hasher);
        let content_hash = format!("{:016x}", hasher.finish());

        let cache_dir = PathBuf::from("__sheaf__");
        let backend_suffix = target_backend.replace('-', "_");
        let artifact_identity = format!("{}.{}", module_name, backend_suffix);
        let cached_vmfb = cache_dir.join(format!("{artifact_identity}.vmfb"));

        // Check manifest for staleness (-vv forces recompile for full debug output)
        let force_recompile = crate::core::config::verbosity() >= 2;
        let vmfb_data = if !force_recompile && cached_vmfb.exists() && manifest_hash_matches(&cache_dir, &artifact_identity, &content_hash) {
            match std::fs::read(&cached_vmfb) {
                Ok(d) => {
                    if crate::core::config::verbosity() >= 2 {
                        sheaf_msg!("jit: {} (cached, {}KB, {})", name, d.len() / 1024, target_backend);
                    } else if crate::core::config::verbosity() >= 1 {
                        sheaf_msg!("jit: {} (cached)", name);
                    }
                    d
                }
                Err(_) => {
                    self.jit_fail(name, "failed to read cached compilation");
                    return None;
                }
            }
        } else {
            let data = match self.run_iree_compile(&iree_compile, name, &mlir, target_backend) {
                Some(d) => d,
                None => {
                    self.jit_fail(name, "iree-compile failed on all backends");
                    return None;
                }
            };

            // Cache: named VMFB + manifest entry
            let _ = std::fs::create_dir_all(&cache_dir);
            let _ = std::fs::write(&cached_vmfb, &data);
            update_manifest(&cache_dir, &artifact_identity, &content_hash);

            data
        };

        // Load into the process-wide IREE session.
        let session = match crate::runtime::iree_session::shared_session() {
            Ok(s) => s,
            Err(e) => {
                self.jit_fail(name, &format!("JIT engine init: {}", e));
                return None;
            }
        };
        if let Err(e) = session.load_vmfb(vmfb_data) {
            let _ = std::fs::remove_file(&cached_vmfb);
            self.jit_fail(name, &format!("JIT load: {}", e));
            return None;
        }

        // Populate captured_scalars so the dispatcher knows which scalars were baked.
        sig.captured_scalars = filtered_constants.clone();

        Some(sig)
    }

    /// Preprocess VAG lambda bodies found in a function body.
    ///
    /// When the body contains `((value-and-grad loss-fn) params)`, the Lambda
    /// inside `__value-and-grad-hof__` hasn't been preprocessed by the standard
    /// pipeline (inline_function_calls doesn't recurse into Lambdas). This method
    /// finds the Lambda, applies the same preprocessing (inline, lower_gets,
    /// resolve_static_constants), and puts it back.
    fn preprocess_vag_lambda(
        &self,
        body: &CompiledExpr,
        registry: &HashMap<String, FunctionDef>,
        constants: &HashMap<(String, Vec<usize>), f64>,
        param_shapes: &HashMap<String, Vec<i64>>,
        param_index_maps: &[(String, BTreeMap<Vec<String>, Vec<usize>>)],
        known_types: &[(String, StableHLOType)],
        arity_err: &std::cell::Cell<Option<SheafError>>,
    ) -> CompiledExpr {
        preprocess_vag_lambda_rec(
            body,
            registry,
            constants,
            param_shapes,
            param_index_maps,
            known_types,
            arity_err,
        )
    }

    /// JIT-compile a value-and-grad closure into a single VMFB.
    ///
    /// The closure `func` is `(fn [p] loss-body)` with captured values in its closure.
    /// We promote tensor captures to MLIR parameters, resolve scalar captures as constants,
    /// then generate forward + backward passes in a single MLIR function.
    ///
    /// Returns `(module_name, signature, param_order)`, where `param_order`
    /// lists function parameters followed by tensor captures.
    pub fn try_jit_value_and_grad(
        &mut self,
        func: &Value,
        wrt_arg: &Value,
        registry: &HashMap<String, FunctionDef>,
    ) -> Option<(String, FunctionSignature, Vec<String>)> {
        let iree_compile = self.iree_compile_path.clone()?;

        let transaction = match JIT_COMPILATION_LOCK.lock() {
            Ok(guard) => guard,
            Err(_) => {
                self.jit_fail("__vag_", "JIT compilation lock is poisoned");
                return None;
            }
        };

        let result = self.try_jit_value_and_grad_locked(iree_compile, func, wrt_arg, registry);
        drop(transaction);
        result
    }

    fn try_jit_value_and_grad_locked(
        &mut self,
        iree_compile: String,
        func: &Value,
        wrt_arg: &Value,
        registry: &HashMap<String, FunctionDef>,
    ) -> Option<(String, FunctionSignature, Vec<String>)> {
        let (fn_params, body, closure) = match func {
            Value::Function {
                params,
                body,
                closure,
                ..
            } => (params, body, closure),
            _ => return None,
        };

        // Derive a human-readable name from the outermost function call
        let vag_fn_name = outermost_call_name(body).unwrap_or("anonymous".to_string());

        // Build a stable key for the failure set and VMFB cache.
        // Include wrt_arg type so shape changes (e.g. after grow-hydra) cause a cache miss.
        let wrt_type_str = value_to_stablehlo_type(wrt_arg)
            .map(|t| t.to_mlir())
            .unwrap_or_default();
        // Same AST + different scalar values produces a different VMFB.
        let mut scalar_captures_key: Vec<(String, String)> = closure
            .iter()
            .filter(|(name, _)| !name.starts_with("__"))
            .filter_map(|(name, val)| match val {
                Value::Float(f) => Some((name.clone(), format!("f{}", f))),
                Value::Int(n) => Some((name.clone(), format!("i{}", n))),
                _ => None,
            })
            .collect();
        scalar_captures_key.sort();
        let scalars_suffix: String = scalar_captures_key
            .iter()
            .map(|(n, v)| format!("{}={}", n, v))
            .collect::<Vec<_>>()
            .join(",");
        let vag_key = format!("__vag_{:?}_{}_{}", body, wrt_type_str, scalars_suffix);
        let module_name = vag_module_name(&vag_key);
        if self.failed_vag.contains(&vag_key) {
            return None;
        }

        // Return cached session if already compiled
        if let Some(cached) = self.vag_cache.get(&vag_key) {
            return Some(cached.clone());
        }

        // Skip impure or HOF-containing bodies
        if !collect_effects(body).is_empty() {
            return None;
        }
        if !collect_hof_calls(body).is_empty() {
            return None;
        }

        // Build combined parameter list: fn params first, then tensor captures.
        // Scalar captures are substituted directly in the body.
        let mut all_param_names: Vec<String> = Vec::new();
        let mut all_arg_values: Vec<Value> = Vec::new();
        let mut scalar_substitutions: Vec<(String, f64)> = Vec::new();
        let wrt_indices: Vec<usize>;

        // fn params (the wrt parameters)
        for p in fn_params {
            all_param_names.push(p.clone());
            all_arg_values.push(wrt_arg.clone());
        }
        wrt_indices = (0..fn_params.len()).collect();

        // Build an augmented registry with closure-captured functions.
        // These will be inlined (not passed as IREE parameters).
        let mut aug_registry = registry.clone();

        // Classify and add captures
        for (cap_name, cap_val) in closure {
            // Skip the __vag_fn__ sentinel
            if cap_name.starts_with("__") {
                continue;
            }
            match cap_val {
                Value::Float(f) => {
                    scalar_substitutions.push((cap_name.clone(), *f as f64));
                }
                Value::Int(n) => {
                    scalar_substitutions.push((cap_name.clone(), *n as f64));
                }
                Value::Function { params: fp, body: fb, .. } => {
                    // Closure-captured function: add to registry for inlining
                    aug_registry.entry(cap_name.clone()).or_insert_with(|| {
                        FunctionDef {
                            name: cap_name.clone(),
                            params: fp.clone(),
                            body: crate::core::ast::SheafValue::Nil(
                                crate::core::error::SourceLocation::new(0, 0, "".into()),
                            ),
                            body_compiled: Some(fb.clone()),
                            signature: None,
                            vmfb_module_name: None,
                            known_param_types: Vec::new(),
                            compile_error: None,
                        }
                    });
                }
                _ => {
                    // Tensor, Dict, Tuple etc. -> promote to MLIR parameter
                    if value_to_stablehlo_type(cap_val).is_ok() {
                        all_param_names.push(cap_name.clone());
                        all_arg_values.push(cap_val.clone());
                    } else {
                        self.jit_fail(&vag_key, &format!("unsupported capture type for '{}'", cap_name));
                        return None;
                    }
                }
            }
        }

        // Substitute scalar captures in body
        let mut body = body.clone();
        for (name, val) in &scalar_substitutions {
            body = substitute_scalar_param(&body, name, *val);
        }

        // Type inference from runtime args
        let mut known_types: Vec<(String, StableHLOType)> = Vec::new();
        let mut param_index_maps: Vec<(String, BTreeMap<Vec<String>, Vec<usize>>)> = Vec::new();
        let mut constants: HashMap<(String, Vec<usize>), f64> = HashMap::new();

        for (param_name, arg_val) in all_param_names.iter().zip(all_arg_values.iter()) {
            let ty = match value_to_stablehlo_type(arg_val) {
                Ok(ty) => ty,
                Err(e) => {
                    self.jit_fail(&vag_key, &format!("arg '{}' has {}", param_name, e.short_message()));
                    return None;
                }
            };

            if let Some(layout) = value_to_param_layout(param_name, arg_val) {
                let imap = layout_to_index_map(&layout);
                extract_scalar_constants(arg_val, param_name, &imap, &mut constants);
                body = lower_get_calls(&body, param_name, &imap);
                param_index_maps.push((param_name.clone(), imap));
            }

            known_types.push((param_name.clone(), ty));
        }

        // Signature inference
        let dummy_compiler = crate::core::expr::CompilerContext::new();
        let mut sig = match infer_function_signature_with_known(
            &dummy_compiler,
            &all_param_names,
            &body,
            &known_types,
        ) {
            Ok(s) => s,
            Err(e) => {
                self.jit_fail(&vag_key, &format!("signature inference: {}", e));
                return None;
            }
        };

        // Override param types from runtime values
        for (param_name, ty) in &known_types {
            if let Some(idx) = all_param_names.iter().position(|p| p == param_name) {
                sig.param_types[idx] = ty.clone();
            }
        }

        // -vv: log VAG signature and captures
        if crate::core::config::verbosity() >= 2 {
            for (pname, pty) in all_param_names.iter().zip(sig.param_types.iter()) {
                if let Some((_, imap)) = param_index_maps.iter().find(|(n, _)| n == pname) {
                    let mut top_fields: BTreeMap<usize, (String, String)> = BTreeMap::new();
                    for (path, indices) in imap.iter() {
                        if path.len() == 1 {
                            let ty_str = if let StableHLOType::Tuple(elems, _) = pty {
                                if let Some(t) = elems.get(indices[0]) {
                                    t.to_mlir()
                                } else {
                                    "?".to_string()
                                }
                            } else {
                                "?".to_string()
                            };
                            top_fields.entry(indices[0])
                                .or_insert((path[0].clone(), ty_str));
                        } else if !top_fields.contains_key(&indices[0]) {
                            let ty_str = if let StableHLOType::Tuple(elems, _) = pty {
                                if let Some(StableHLOType::Tuple(sub, _)) = elems.get(indices[0]) {
                                    format!("tuple<...> ({} fields)", sub.len())
                                } else {
                                    "tuple<...>".to_string()
                                }
                            } else {
                                "tuple<...>".to_string()
                            };
                            top_fields.insert(indices[0], (path[0].clone(), ty_str));
                        }
                    }
                    for (_, (key, ty_str)) in &top_fields {
                        sheaf_msg!("jit: value-and-grad | {}.{}: {}", pname, key, ty_str);
                    }
                } else {
                    sheaf_msg!("jit: value-and-grad | {}: {}", pname, pty.to_mlir());
                }
            }
            if !scalar_substitutions.is_empty() {
                sheaf_msg!("jit: value-and-grad | {} scalar captures", scalar_substitutions.len());
            }
        }

        // Inline user-defined function calls (including closure-captured lambdas)
        body = crate::autodiff::inline_function_calls(&body, &aug_registry);

        // Bail out if inlining produced a graph that is too large
        let node_count = expr_node_count(&body);
        if node_count > MAX_VAG_GRAPH_NODES {
            self.jit_fail(
                &vag_key,
                &format!(
                    "graph too large after inlining ({} nodes, limit {})",
                    node_count, MAX_VAG_GRAPH_NODES
                ),
            );
            return None;
        }

        // Fold dict literal gets: (get {:gamma g :beta b} :gamma) -> g
        body = crate::autodiff::fold_dict_gets(&body);

        // Post-inline: re-lower dict access from inlined bodies
        for (param_name, index_map) in &param_index_maps {
            body = lower_get_calls(&body, param_name, index_map);
        }
        body = lower_inlined_gets(&body, &param_index_maps);

        // Unroll reduces so reverse_grad can differentiate through them
        let known_types_vec: Vec<(String, StableHLOType)> = sig
            .param_types
            .iter()
            .enumerate()
            .map(|(i, ty)| (all_param_names[i].clone(), ty.clone()))
            .collect();
        // Debug: log remaining reduces before unrolling
        if crate::core::config::verbosity() >= 2 {
            sheaf_msg!(
                "jit: [vag] param_index_maps keys: {:?}",
                param_index_maps
                    .iter()
                    .map(|(n, _)| n.as_str())
                    .collect::<Vec<_>>()
            );
            log_remaining_reduces(&body, "before unroll");
        }
        body = unroll_reduces(&body, &known_types_vec);
        if crate::core::config::verbosity() >= 2 {
            log_remaining_reduces(&body, "after unroll");
        }

        // Bail out if unrolling produced a graph that is too large
        let node_count = expr_node_count(&body);
        if node_count > MAX_VAG_GRAPH_NODES {
            self.jit_fail(
                &vag_key,
                &format!(
                    "graph too large after unrolling ({} nodes, limit {})",
                    node_count, MAX_VAG_GRAPH_NODES
                ),
            );
            return None;
        }

        // Re-lower dict access introduced by unrolling (get on GetTupleElement elements)
        for (param_name, index_map) in &param_index_maps {
            body = lower_get_calls(&body, param_name, index_map);
        }
        body = lower_inlined_gets(&body, &param_index_maps);

        // Filter constants: only bake scalars used in shape-bearing positions
        // (see the main JIT path above for rationale).
        let filtered_vag_constants =
            filter_constants_for_shape_positions(&constants, &body);

        // Resolve static constants
        let mut param_shapes: HashMap<String, Vec<i64>> = all_param_names
            .iter()
            .zip(sig.param_types.iter())
            .filter_map(|(p, ty)| {
                let shape = ty.shape();
                if shape.is_empty() {
                    None
                } else {
                    Some((p.clone(), shape.to_vec()))
                }
            })
            .collect();
        // Inject shapes of GetTupleElement leaves for shape inference
        for (param_name, param_ty) in all_param_names.iter().zip(sig.param_types.iter()) {
            inject_tuple_shapes(param_name, param_ty, &[], &mut param_shapes);
        }
        body = resolve_static_constants(&body, &filtered_vag_constants, &param_shapes, false);

        body = match crate::lowering::transforms::lower_tuples_and_destructuring(
            body, &param_shapes,
        ) {
            Ok(b) => b,
            Err(e) => {
                self.jit_fail("value_and_grad", &e.short_message());
                return None;
            }
        };

        // Build key layouts for codegen
        let mut tuple_key_layouts: HashMap<String, BTreeMap<String, usize>> = HashMap::new();
        let mut idx_to_key: HashMap<(String, usize), String> = HashMap::new();
        for (param_name, index_map) in &param_index_maps {
            for (key_path, indices) in index_map {
                for depth in 0..key_path.len() {
                    let parent = if depth == 0 {
                        param_name.clone()
                    } else {
                        key_path[depth - 1].clone()
                    };
                    let child = &key_path[depth];
                    let idx = indices[depth];
                    tuple_key_layouts
                        .entry(parent.clone())
                        .or_default()
                        .entry(child.clone())
                        .or_insert(idx);
                    idx_to_key
                        .entry((parent, idx))
                        .or_insert_with(|| child.clone());
                }
            }
        }
        propagate_let_layouts(&body, &idx_to_key, &mut tuple_key_layouts);

        // Debug: log unresolved shapes
        if crate::core::config::verbosity() >= 2 {
            log_unresolved_shapes(&body, "before codegen");
        }

        // Value-and-grad codegen
        let backend = self.target_backend.clone();
        let codegen_result = {
            let registry_clone = registry.clone();
            let param_names = all_param_names.clone();
            let param_types = sig.param_types.clone();
            let body_clone = body.clone();
            let wrt_idx = wrt_indices.clone();
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
                let mut codegen = CodeGenerator::with_function_params(
                    registry_clone,
                    &param_names,
                    &param_types,
                );
                codegen.set_tuple_key_layouts(tuple_key_layouts);
                codegen.set_idx_to_key(idx_to_key);

                let inlined_body = body_clone;

                // Expand tuple params to synthetic leaf symbols
                let mut expanded_body = inlined_body.clone();
                let mut all_leaves: Vec<(usize, Vec<crate::lowering::codegen::TupleLeaf>)> = Vec::new();
                let mut all_wrt_symbols: Vec<String> = Vec::new();

                for &idx in &wrt_idx {
                    let param_name = &param_names[idx];
                    let param_ty = &param_types[idx];
                    match param_ty {
                        StableHLOType::Tuple(..) => {
                            let leaves = collect_tuple_leaves(&expanded_body, param_name);
                            if crate::core::config::verbosity() >= 2 {
                                sheaf_msg!("jit: [vag] param '{}': {} tuple leaves", param_name, leaves.len());
                            }
                            expanded_body = expand_tuple_to_symbols(&expanded_body, param_name);
                            for leaf in &leaves {
                                all_wrt_symbols.push(leaf.symbol.clone());
                            }
                            all_leaves.push((idx, leaves));
                        }
                        _ => {
                            all_wrt_symbols.push(param_name.clone());
                        }
                    }
                }

                // Convert to ANF
                let anf_expr = to_anf(&expanded_body);
                let (anf_bindings, anf_body) = match &anf_expr {
                    CompiledExpr::Let { bindings, body } => {
                        (bindings.clone(), body.as_ref().clone())
                    }
                    other => (vec![], other.clone()),
                };

                // Bind tuple leaves for codegen (use generate to handle both
                // leaf lookups and sub-tuple reconstruction from flat args)
                for &(idx, ref leaves) in &all_leaves {
                    let param_name = &param_names[idx];
                    for leaf in leaves {
                        let gte = CompiledExpr::GetTupleElement {
                            param: param_name.clone(),
                            indices: leaf.indices.clone(),
                        };
                        let (reg, ty) = codegen.generate(&gte)?;
                        codegen.bind_symbol(&leaf.symbol, reg, ty);
                    }
                }

                // Generate forward bindings (flat scope, no Let scoping).
                // Use generate_binding to handle Lambda, destructuring, layouts.
                for (name, value_expr) in &anf_bindings {
                    let name_str = name.as_simple().expect("Expected simple binding pattern in ANF");
                    codegen.generate_binding(name_str, value_expr)?;
                }

                // Generate the ANF body (loss value)
                let (loss_reg, loss_ty) = codegen.generate(&anf_body)?;

                // Build shape map from forward codegen for reverse-mode AD
                let shape_map: HashMap<String, Vec<i64>> = codegen.binding_shapes();

                // Convert anf_bindings to use String names for reverse_grad
                let anf_bindings_str: Vec<(String, CompiledExpr)> = anf_bindings
                    .iter()
                    .map(|(name, expr)| (name.as_simple().expect("Expected simple binding pattern in ANF").to_string(), expr.clone()))
                    .collect();
                // Run reverse-mode AD on ANF with shape info
                let (backward_bindings, grad_sym_map) =
                    reverse_grad(&anf_bindings_str, &anf_body, &all_wrt_symbols, &shape_map);

                // Apply AST-level optimizations to backward bindings
                let backward_bindings: Vec<(String, CompiledExpr)> = backward_bindings
                    .into_iter()
                    .map(|(name, expr)| (name, cse(simplify(expr))))
                    .collect();

                if crate::core::config::verbosity() >= 2 {
                    sheaf_msg!("jit: [vag] ANF: {} fwd bindings, {} bwd bindings, {} wrt symbols",
                        anf_bindings.len(), backward_bindings.len(), all_wrt_symbols.len());
                }

                if std::env::var("SHEAF_DEBUG_GRAD").is_ok() {
                    eprintln!("--- Forward ANF bindings ---");
                    for (name, val) in &anf_bindings_str {
                        eprintln!("  {} = {:?}  [shape: {:?}]", name, val, shape_map.get(name));
                    }
                    eprintln!("  body = {:?}", anf_body);
                    eprintln!("--- Backward bindings ---");
                    for (name, val) in &backward_bindings {
                        eprintln!("  {} = {:?}", name, val);
                    }
                    eprintln!("--- Grad map ---");
                    for (sym, grad_name) in &grad_sym_map {
                        eprintln!("  {} -> {}", sym, grad_name);
                    }
                    eprintln!("--- wrt symbols ---");
                    for s in &all_wrt_symbols {
                        eprintln!("  {}", s);
                    }
                }

        // Generate backward bindings.
        for (name, value_expr) in &backward_bindings {
    let (reg, ty) = codegen.generate(value_expr)?;
        codegen.bind_symbol(name, reg, ty);
    }

                // Collect gradient registers for each wrt param.
                let mut grad_regs: Vec<Register> = Vec::new();
                let mut grad_tys: Vec<StableHLOType> = Vec::new();

                for &idx in &wrt_idx {
                    let param_name = &param_names[idx];
                    let param_ty = &param_types[idx];
                    match param_ty {
                        StableHLOType::Tuple(..) => {
                            let leaves = all_leaves
                                .iter()
                                .find(|(i, _)| *i == idx)
                                .map(|(_, l)| l)
                                .unwrap();

                            // Each leaf gradient is a Symbol name from grad_sym_map
                            let leaf_grad_map: std::collections::HashMap<String, CompiledExpr> =
                                leaves.iter().map(|leaf| {
                                    let grad_expr = grad_sym_map
                                        .get(&leaf.symbol)
                                        .map(|sym_name| CompiledExpr::Symbol(sym_name.clone()))
                                        .unwrap_or_else(|| {
                                            // Generate zeros of the correct shape for missing grads
                                            let leaf_ty = resolve_leaf_type(param_ty, &leaf.indices);
                                            let shape = leaf_ty.shape();
                                            if shape.is_empty() {
                                                CompiledExpr::Float(0.0)
                                            } else {
                                                CompiledExpr::FunctionCall {
                                                    name: "zeros".to_string(),
                                                    args: vec![CompiledExpr::Vector(
                                                        shape.iter().map(|&d| CompiledExpr::Integer(d)).collect()
                                                    )],
                                                    loc: None,
                                                }
                                            }
                                        });
                                    (leaf.symbol.clone(), grad_expr)
                                }).collect();

                            let (grad_reg, grad_ty) =
                                codegen.build_grad_tuple_from_map(leaves, param_ty, &leaf_grad_map)?;
                            grad_regs.push(grad_reg);
                            grad_tys.push(grad_ty);
                        }
                        _ => {
                            let grad_expr = grad_sym_map
                                .get(param_name)
                                .map(|sym_name| CompiledExpr::Symbol(sym_name.clone()))
                                .unwrap_or(CompiledExpr::Float(0.0));
                            let (grad_reg, grad_ty) = codegen.generate(&grad_expr)?;
                            let (grad_reg, grad_ty) =
                                codegen.reduce_broadcast_grad(grad_reg, &grad_ty, param_ty)?;
                            grad_regs.push(grad_reg);
                            grad_tys.push(grad_ty);
                        }
                    }
                }

                // Pack result: (loss, grad0, grad1, ...)
                let mut all_regs = vec![loss_reg];
                all_regs.extend(grad_regs);
                let mut all_tys = vec![loss_ty.clone()];
                all_tys.extend(grad_tys.clone());

                let decl = codegen.finish_multi(
                    "value_and_grad",
                    &param_types,
                    &all_regs,
                    &all_tys,
                );
                let return_type = StableHLOType::Tuple(all_tys, None);
                Ok::<_, crate::core::error::SheafError>((decl, return_type))
            }))
        };

        let (mlir_decl, return_type) = match codegen_result {
            Ok(Ok(result)) => result,
            Ok(Err(e)) => {
                self.jit_fail(&vag_key, &format!("codegen: {}", e));
                return None;
            }
            Err(_) => {
                self.jit_fail(&vag_key, "codegen: internal panic");
                return None;
            }
        };

        sig.return_type = return_type;
        sig.return_dict_keys = None;

        let mlir = StableHLOEmitter::emit_module_named(Some(&module_name), &[mlir_decl]);

        if crate::core::config::verbosity() >= 2 {
            sheaf_msg!("jit: value-and-grad | MLIR {} lines", mlir.lines().count());
        }

        // Content hash for staleness check
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        mlir.hash(&mut hasher);
        backend.hash(&mut hasher);
        IREE_COMPILER_VERSION.hash(&mut hasher);
        let content_hash = format!("{:016x}", hasher.finish());

        let cache_dir = PathBuf::from("__sheaf__");
        let vag_cache_name = format!("{}-vag", vag_fn_name);
        let backend_suffix = backend.replace('-', "_");
        let cached_vmfb = cache_dir.join(format!("{}.{}.vmfb", vag_cache_name, backend_suffix));

        let force_recompile = crate::core::config::verbosity() >= 2;
        let vmfb_data = if !force_recompile
            && cached_vmfb.exists()
            && manifest_hash_matches(&cache_dir, &vag_cache_name, &content_hash)
        {
            match std::fs::read(&cached_vmfb) {
                Ok(d) => {
                    if crate::core::config::verbosity() >= 2 {
                        sheaf_msg!(
                            "jit: value_and_grad (cached, {}KB, {})",
                            d.len() / 1024,
                            backend
                        );
                    } else if crate::core::config::verbosity() >= 1 {
                        sheaf_msg!("jit: value_and_grad (cached)");
                    }
                    d
                }
                Err(_) => {
                    self.jit_fail(&vag_key, "failed to read cached compilation");
                    return None;
                }
            }
        } else {
            // Debug: save a copy for inspection
            if crate::core::config::verbosity() >= 2 {
                let debug_path = cache_dir.join(format!("{}-vag-debug.mlir", vag_fn_name));
                let _ = std::fs::write(&debug_path, &mlir);
                sheaf_msg!("jit: value-and-grad | saved {}", debug_path.display());
            }

            let data = match self.run_iree_compile(&iree_compile, "value_and_grad", &mlir, &backend)
            {
                Some(d) => d,
                None => {
                    self.jit_fail(&vag_key, "compilation failed on all backends");
                    return None;
                }
            };

            let _ = std::fs::create_dir_all(&cache_dir);
            let _ = std::fs::write(&cached_vmfb, &data);
            update_manifest(&cache_dir, &vag_cache_name, &content_hash);

            data
        };

        // Load into the process-wide IREE session.
        let session = match crate::runtime::iree_session::shared_session() {
            Ok(s) => s,
            Err(e) => {
                self.jit_fail(&vag_key, &format!("JIT engine init: {}", e));
                return None;
            }
        };
        if let Err(e) = session.load_vmfb(vmfb_data) {
            let _ = std::fs::remove_file(&cached_vmfb);
            self.jit_fail(&vag_key, &format!("JIT load: {}", e));
            return None;
        }

        let result = (module_name, sig, all_param_names);
        self.vag_cache.insert(vag_key, result.clone());
        Some(result)
    }

    /// Run iree-compile on MLIR source for the given backend.
    /// Returns compiled VMFB bytes, or None if compilation fails.
    fn run_iree_compile(
        &self,
        iree_compile: &str,
        name: &str,
        mlir: &str,
        backend: &str,
    ) -> Option<Vec<u8>> {
        if crate::core::config::verbosity() >= 1 {
            sheaf_msg!("jit: compiling {} [{}]...", name, backend);
        }

        let safe_name = name.replace('?', "_q").replace('!', "_b");
        let stamp = std::process::id();
        let cache_dir = std::path::PathBuf::from("__sheaf__");
        let _ = std::fs::create_dir_all(&cache_dir);
        let mlir_path = cache_dir.join(format!("{}-{}.mlir", safe_name, stamp));
        let vmfb_path = cache_dir.join(format!("{}-{}.vmfb", safe_name, stamp));

        if std::fs::write(&mlir_path, mlir).is_err() {
            eprintln!("sheaf: failed to write temp MLIR, aborting");
            std::process::exit(1);
        }

        let stderr_cfg = if crate::core::config::verbosity() >= 2 {
            std::process::Stdio::inherit()
        } else {
            std::process::Stdio::null()
        };

        let mut cmd = std::process::Command::new(iree_compile);
        cmd.arg(&mlir_path)
            .arg(format!("--iree-hal-target-backends={}", backend))
            .arg("-o")
            .arg(&vmfb_path)
            .stderr(stderr_cfg);
        if backend == "metal-spirv" {
            cmd.arg("--iree-metal-compile-to-metallib=false");
        }
        if backend == "cuda" {
            if let Some(target) = detect_cuda_target() {
                cmd.arg(format!("--iree-cuda-target={}", target));
            }
        }
        if backend == "llvm-cpu" {
            cmd.arg("--iree-llvmcpu-target-cpu=host");
            cmd.arg("--iree-llvmcpu-enable-ukernels=all");
            cmd.arg("--iree-opt-data-tiling");
        }
        let status = cmd.status();

        if crate::core::config::verbosity() >= 2 {
            let debug_mlir = cache_dir.join(format!("{}-debug.mlir", safe_name));
            let _ = std::fs::rename(&mlir_path, &debug_mlir);
            sheaf_msg!("jit: {} | saved {}", name, debug_mlir.display());
        } else {
            let _ = std::fs::remove_file(&mlir_path);
        }

        let ok = match &status {
            Ok(s) => s.success(),
            Err(_) => false,
        };

        if ok {
            if let Ok(data) = std::fs::read(&vmfb_path) {
                let _ = std::fs::remove_file(&vmfb_path);
                return Some(data);
            }
        }
        let _ = std::fs::remove_file(&vmfb_path);
        None
    }

    fn jit_fail(&mut self, name: &str, reason: &str) {
        // Ordinary failures are logged here; structural and per-variant
        // recording is done at the call site that owns the cache key.
        // Value-and-grad failures stay isolated in failed_vag and keep
        // their reason for classify_vag_skip.
        if name.starts_with("__vag_") {
            self.failed_vag.insert(name.to_string());
            self.last_vag_fail_reason = Some(reason.to_string());
        }
        let display_name = if name.starts_with("__vag_") {
            "value-and-grad"
        } else {
            name
        };
        if crate::core::config::verbosity() >= 1 {
            sheaf_msg!("jit: {} skipped ({})", display_name, reason);
        }
    }

    /// Classify the last VAG skip reason as legitimate (unsupported pattern)
    /// or a bug (JIT should handle but failed).
    pub fn classify_vag_skip(&self) -> JitVagOutcome {
        match &self.last_vag_fail_reason {
            None => JitVagOutcome::Unsupported,
            Some(reason) => {
                let legit = reason.starts_with("has HOF calls")
                    || reason.starts_with("scalar-only")
                    || reason.starts_with("unsupported capture type")
                    || reason.starts_with("graph too large")
                    || reason.contains("Function call not yet supported");
                if legit {
                    JitVagOutcome::Unsupported
                } else {
                    JitVagOutcome::Bug(reason.clone())
                }
            }
        }
    }
}

/// Why the JIT skipped a value-and-grad call.
#[derive(Debug)]
pub enum JitVagOutcome {
    /// JIT compiled and executed successfully, or had a runtime error.
    Success(Result<Value, SheafError>),
    /// JIT architecture doesn't support this pattern: interpreter fallback OK.
    Unsupported,
    /// JIT should handle this but failed: likely a Sheaf bug.
    Bug(String),
}

/// Navigate a tuple type tree using indices to find the leaf type.
fn resolve_leaf_type(ty: &StableHLOType, indices: &[usize]) -> StableHLOType {
    let mut current = ty.clone();
    for &idx in indices {
        if let StableHLOType::Tuple(elems, _) = &current {
            if idx < elems.len() {
                current = elems[idx].clone();
            } else {
                return StableHLOType::scalar_f32();
            }
        } else {
            return current;
        }
    }
    current
}


#[cfg(iree_runtime)]
fn inject_tuple_shapes(
    param_name: &str,
    ty: &StableHLOType,
    indices: &[usize],
    shapes: &mut HashMap<String, Vec<i64>>,
) {
    match ty {
        StableHLOType::Tuple(elems, keys) => {
            for (i, elem_ty) in elems.iter().enumerate() {
                let mut child_indices = indices.to_vec();
                child_indices.push(i);
                if let Some(key_names) = keys {
                    if let Some(key) = key_names.get(i) {
                        shapes.insert(key.clone(), elem_ty.shape().to_vec());
                        inject_tuple_shapes(key, elem_ty, &child_indices, shapes);
                        continue;
                    }
                }
                inject_tuple_shapes(param_name, elem_ty, &child_indices, shapes);
            }
        }
        _ => {
            let shape = ty.shape();
            if !shape.is_empty() && !indices.is_empty() {
                let key = format!("{}@{:?}", param_name, indices);
                shapes.insert(key, shape.to_vec());
            }
        }
    }
}

#[cfg(iree_runtime)]
fn log_remaining_reduces(expr: &CompiledExpr, label: &str) {
    match expr {
        CompiledExpr::FunctionCall { name, args, .. } if name == "reduce" => {
            let coll_desc = if let Some(coll) = args.get(2) {
                format!("{:?}", coll)
            } else {
                "missing".to_string()
            };
            sheaf_msg!("jit: [{}] reduce with coll={}", label, coll_desc);
        }
        CompiledExpr::FunctionCall { args, .. } => {
            for a in args {
                log_remaining_reduces(a, label);
            }
        }
        CompiledExpr::Let { bindings, body } => {
            for (_, v) in bindings {
                log_remaining_reduces(v, label);
            }
            log_remaining_reduces(body, label);
        }
        CompiledExpr::Lambda { body, .. } => log_remaining_reduces(body, label),
        CompiledExpr::Do(exprs) => {
            for e in exprs {
                log_remaining_reduces(e, label);
            }
        }
        _ => {}
    }
}

#[cfg(iree_runtime)]
fn log_unresolved_shapes(expr: &CompiledExpr, label: &str) {
    match expr {
        CompiledExpr::FunctionCall { name, args, .. } if name == "reshape" && args.len() == 2 => {
            match &args[1] {
                CompiledExpr::Vector(elems) => {
                    let unresolved: Vec<_> = elems
                        .iter()
                        .filter(|e| !matches!(e, CompiledExpr::Integer(_)))
                        .map(|e| format!("{:?}", e))
                        .collect();
                    if !unresolved.is_empty() {
                        sheaf_msg!(
                            "jit: [{}] reshape with unresolved shape elements: {:?}",
                            label,
                            unresolved
                        );
                    }
                }
                other => {
                    sheaf_msg!(
                        "jit: [{}] reshape with non-vector shape: {:?}",
                        label,
                        other
                    );
                }
            }
        }
        CompiledExpr::FunctionCall { args, .. } => {
            for a in args {
                log_unresolved_shapes(a, label);
            }
        }
        CompiledExpr::Let { bindings, body } => {
            for (_, v) in bindings {
                log_unresolved_shapes(v, label);
            }
            log_unresolved_shapes(body, label);
        }
        CompiledExpr::Lambda { body, .. } => log_unresolved_shapes(body, label),
        CompiledExpr::Do(exprs) => {
            for e in exprs {
                log_unresolved_shapes(e, label);
            }
        }
        _ => {}
    }
}

/// Extract the outermost function call name from an expression (for cache naming).
fn outermost_call_name(expr: &CompiledExpr) -> Option<String> {
    match expr {
        CompiledExpr::FunctionCall { name, .. } => Some(name.clone()),
        CompiledExpr::Let { body, .. } => outermost_call_name(body),
        CompiledExpr::Do(exprs) => exprs.last().and_then(outermost_call_name),
        _ => None,
    }
}

/// Read the manifest and check if the stored hash matches.
fn manifest_hash_matches(cache_dir: &std::path::Path, name: &str, expected_hash: &str) -> bool {
    let manifest_path = cache_dir.join("manifest.json");
    let data = match std::fs::read_to_string(&manifest_path) {
        Ok(d) => d,
        Err(_) => return false,
    };
    let manifest: serde_json::Value = match serde_json::from_str(&data) {
        Ok(v) => v,
        Err(_) => return false,
    };
    manifest.get(name).and_then(|v| v.as_str()) == Some(expected_hash)
}

/// Update the manifest with a new hash entry.
fn update_manifest(cache_dir: &std::path::Path, name: &str, hash: &str) {
    let manifest_path = cache_dir.join("manifest.json");
    let mut manifest: serde_json::Map<String, serde_json::Value> =
        std::fs::read_to_string(&manifest_path)
            .ok()
            .and_then(|d| serde_json::from_str(&d).ok())
            .unwrap_or_default();
    manifest.insert(
        name.to_string(),
        serde_json::Value::String(hash.to_string()),
    );
    let _ = std::fs::write(
        &manifest_path,
        serde_json::to_string_pretty(&manifest).unwrap(),
    );
}


fn preprocess_vag_lambda_rec(
    expr: &CompiledExpr,
    registry: &HashMap<String, FunctionDef>,
    constants: &HashMap<(String, Vec<usize>), f64>,
    param_shapes: &HashMap<String, Vec<i64>>,
    param_index_maps: &[(String, BTreeMap<Vec<String>, Vec<usize>>)],
    known_types: &[(String, StableHLOType)],
    arity_err: &std::cell::Cell<Option<SheafError>>,
) -> CompiledExpr {
    let recurse = |e: &CompiledExpr| {
        preprocess_vag_lambda_rec(
            e,
            registry,
            constants,
            param_shapes,
            param_index_maps,
            known_types,
            arity_err,
        )
    };

    match expr {
        CompiledExpr::LambdaCall { callee, args } => {
            let new_callee = recurse(callee);
            let new_args: Vec<CompiledExpr> = args.iter().map(&recurse).collect();

            if let CompiledExpr::FunctionCall { name, args: inner_args, .. } = &new_callee {
                if name == "__value-and-grad-hof__" && inner_args.len() == 1 {
                    let preprocessed_lambda = preprocess_one_vag_lambda(
                        &inner_args[0],
                        registry,
                        constants,
                        param_shapes,
                        param_index_maps,
                        known_types,
                        arity_err,
                    );
                    return CompiledExpr::LambdaCall {
                        callee: Box::new(CompiledExpr::FunctionCall {
                            name: "__value-and-grad-hof__".to_string(),
                            args: vec![preprocessed_lambda],
                            loc: None,
                        }),
                        args: new_args,
                    };
                }
            }
            CompiledExpr::LambdaCall { callee: Box::new(new_callee), args: new_args }
        }
        _ => expr.map_children(recurse),
    }
}

fn preprocess_one_vag_lambda(
    lambda_expr: &CompiledExpr,
    registry: &HashMap<String, FunctionDef>,
    constants: &HashMap<(String, Vec<usize>), f64>,
    param_shapes: &HashMap<String, Vec<i64>>,
    param_index_maps: &[(String, BTreeMap<Vec<String>, Vec<usize>>)],
    known_types: &[(String, StableHLOType)],
    arity_err: &std::cell::Cell<Option<SheafError>>,
) -> CompiledExpr {
    let (params, body) = match lambda_expr {
        CompiledExpr::Lambda { params, body, .. } => (params.clone(), body.as_ref().clone()),
        _ => return lambda_expr.clone(),
    };

    let mut lambda_param_shapes: HashMap<String, Vec<i64>> = param_shapes.clone();
    for (name, ty) in known_types {
        if params.contains(name) {
            inject_tuple_shapes(name, ty, &[], &mut lambda_param_shapes);
        }
    }

    // Surface the destructuring arity error to the caller via the Cell instead of
    // silently swallowing it; if no error occurs, this is a no-op.
    let body = match preprocess_body(&body, registry, param_index_maps, constants, &lambda_param_shapes, false) {
        Ok(b) => b,
        Err(err) => {
            // Fallback: return the un-preprocessed lambda. This is rarer than the
            // happy path and the clone cost is acceptable (a destructuring arity
            // error is a user error, not a performance-critical path).
            arity_err.set(Some(err));
            return lambda_expr.clone();
        }
    };

    CompiledExpr::Lambda {
        params,
        body: Box::new(body),
    }
}

fn preprocess_body(
    body: &CompiledExpr,
    registry: &HashMap<String, FunctionDef>,
    param_index_maps: &[(String, BTreeMap<Vec<String>, Vec<usize>>)],
    constants: &HashMap<(String, Vec<usize>), f64>,
    param_shapes: &HashMap<String, Vec<i64>>,
    skip_lambda: bool,
) -> Result<CompiledExpr, SheafError> {
    let mut body = crate::autodiff::inline_function_calls(body, registry);

    for (param_name, index_map) in param_index_maps {
        body = lower_get_calls(&body, param_name, index_map);
    }
    body = lower_inlined_gets(&body, param_index_maps);

    body = resolve_static_constants(&body, constants, param_shapes, skip_lambda);

    // Arity mismatch and other destructuring errors bubble up as SheafError::Compile.
    crate::lowering::transforms::lower_tuples_and_destructuring(body, param_shapes)
}

#[cfg(test)]
mod cache_key_tests {
    use super::{
        cache_key_for_function, function_definition_identity, module_name_for,
        CompiledModuleInfo, JitCompiler, JIT_COMPILATION_LOCK,
    };
    use crate::core::ast::SheafValue;
    use crate::core::inference::FunctionSignature;
    use crate::core::error::SourceLocation;
    use crate::core::expr::{CompiledExpr, FunctionDef};
    use crate::interpreter::value::{Dtype, Value};
    use ndarray::ArrayD;
    use std::collections::{BTreeMap, HashMap};
    use std::sync::Arc;

    fn tensor(shape: Vec<usize>, fill: f32, dtype: Dtype) -> Value {
        Value::Tensor {
            data: Arc::new(ArrayD::from_elem(shape, fill)),
            dtype,
        }
    }

    fn definition(name: &str, params: &[&str], source: &str, body: CompiledExpr) -> FunctionDef {
        FunctionDef {
            name: name.to_string(),
            params: params.iter().map(|param| (*param).to_string()).collect(),
            body: SheafValue::String(source.to_string(), SourceLocation::unknown()),
            body_compiled: Some(body),
            signature: None,
            vmfb_module_name: None,
            known_param_types: Vec::new(),
            compile_error: None,
        }
    }

    fn passthrough() -> CompiledExpr {
        CompiledExpr::Symbol("x".to_string())
    }

    fn reshape_body() -> CompiledExpr {
        CompiledExpr::FunctionCall {
            name: "reshape".to_string(),
            args: vec![
                CompiledExpr::Symbol("x".to_string()),
                CompiledExpr::Vector(vec![CompiledExpr::Symbol("n".to_string())]),
            ],
            loc: None,
        }
    }

    fn key(definition: &FunctionDef, args: &[Value], registry: &HashMap<String, FunctionDef>) -> super::JitCacheKey {
        cache_key_for_function(definition, args, registry).unwrap()
    }

    fn jit_compiler() -> JitCompiler {
        JitCompiler {
            iree_compile_path: None,
            target_backend: "llvm-cpu".to_string(),
            failed_definitions: std::collections::HashSet::new(),
            failed_keys: std::collections::HashSet::new(),
            failed_vag: std::collections::HashSet::new(),
            variants: HashMap::new(),
            last_vag_fail_reason: None,
            vag_cache: HashMap::new(),
        }
    }

    fn signature() -> FunctionSignature {
        FunctionSignature {
            param_types: vec![crate::StableHLOType::scalar_f32()],
            return_type: crate::StableHLOType::scalar_f32(),
            return_dict_keys: None,
            arg_type_layouts: Vec::new(),
            captured_scalars: HashMap::new(),
        }
    }

    fn module(name: &str) -> CompiledModuleInfo {
        CompiledModuleInfo {
            module_name: name.to_string(),
            signature: signature(),
        }
    }

    #[test]
    fn cache_key_distinguishes_shapes_but_not_tensor_data() {
        let definition = definition("f", &["x"], "f", passthrough());
        let registry = HashMap::from([("f".to_string(), definition.clone())]);
        let first = key(&definition, &[tensor(vec![2, 3], 0.0, Dtype::F32)], &registry);
        let same_shape = key(&definition, &[tensor(vec![2, 3], 9.0, Dtype::F32)], &registry);
        let different_shape = key(&definition, &[tensor(vec![3, 2], 0.0, Dtype::F32)], &registry);
        assert_eq!(first, same_shape);
        assert_ne!(first, different_shape);
    }

    #[test]
    fn cache_key_distinguishes_dtype_and_nested_layout() {
        let definition = definition("f", &["x"], "f", passthrough());
        let registry = HashMap::from([("f".to_string(), definition.clone())]);
        let f32_key = key(&definition, &[tensor(vec![2], 0.0, Dtype::F32)], &registry);
        let bf16_key = key(&definition, &[tensor(vec![2], 0.0, Dtype::BF16)], &registry);
        let nested_key = key(
            &definition,
            &[Value::Tuple(vec![tensor(vec![2], 0.0, Dtype::F32)])],
            &registry,
        );
        assert_ne!(f32_key, bf16_key);
        assert_ne!(f32_key, nested_key);
    }

    #[test]
    fn cache_key_tracks_only_shape_scalars() {
        let reshape = definition("reshape-it", &["x", "n"], "reshape", reshape_body());
        let registry = HashMap::from([("reshape-it".to_string(), reshape.clone())]);
        let first = key(&reshape, &[tensor(vec![4], 0.0, Dtype::F32), Value::Int(2)], &registry);
        let second = key(&reshape, &[tensor(vec![4], 0.0, Dtype::F32), Value::Int(4)], &registry);
        assert_ne!(first, second);

        let scale = definition(
            "scale",
            &["x", "n"],
            "scale",
            CompiledExpr::FunctionCall {
                name: "multiply".to_string(),
                args: vec![CompiledExpr::Symbol("x".to_string()), CompiledExpr::Symbol("n".to_string())],
                loc: None,
            },
        );
        let registry = HashMap::from([("scale".to_string(), scale.clone())]);
        let first = key(&scale, &[tensor(vec![4], 0.0, Dtype::F32), Value::Int(2)], &registry);
        let second = key(&scale, &[tensor(vec![4], 0.0, Dtype::F32), Value::Int(4)], &registry);
        assert_eq!(first, second);
    }

    #[test]
    fn direct_recursion_is_rejected_before_cache_key_analysis() {
        let recursive = definition(
            "recursive",
            &["x"],
            "recursive",
            CompiledExpr::FunctionCall {
                name: "recursive".to_string(),
                args: vec![CompiledExpr::Symbol("x".to_string())],
                loc: None,
            },
        );
        let registry = HashMap::from([("recursive".to_string(), recursive.clone())]);
        let identity = function_definition_identity(&recursive, &registry);
        let mut compiler = jit_compiler();
        compiler.iree_compile_path = Some("iree-compile".to_string());

        assert!(compiler
            .try_jit_compile(&recursive, &[tensor(vec![2], 0.0, Dtype::F32)], &registry)
            .is_none());
        assert!(compiler.failed_definitions.contains(&identity));

        assert!(compiler
            .try_jit_compile(&recursive, &[tensor(vec![2], 0.0, Dtype::F32)], &registry)
            .is_none());
        assert_eq!(compiler.failed_definitions.len(), 1);
    }

    #[test]
    fn cache_key_tracks_shape_scalar_used_by_inlined_function() {
        let callee = definition("reshape-it", &["x", "n"], "reshape", reshape_body());
        let caller = definition(
            "caller",
            &["x", "n"],
            "caller",
            CompiledExpr::FunctionCall {
                name: "reshape-it".to_string(),
                args: vec![
                    CompiledExpr::Symbol("x".to_string()),
                    CompiledExpr::Symbol("n".to_string()),
                ],
                loc: None,
            },
        );
        let registry = HashMap::from([
            ("reshape-it".to_string(), callee),
            ("caller".to_string(), caller.clone()),
        ]);
        let first = key(&caller, &[tensor(vec![4], 0.0, Dtype::F32), Value::Int(2)], &registry);
        let second = key(&caller, &[tensor(vec![4], 0.0, Dtype::F32), Value::Int(4)], &registry);
        assert_ne!(first, second);
    }

    #[test]
    fn cache_key_tracks_transitive_callee_identity() {
        let callee = definition("callee", &["x"], "one", passthrough());
        let caller = definition(
            "caller",
            &["x"],
            "caller",
            CompiledExpr::FunctionCall {
                name: "callee".to_string(),
                args: vec![CompiledExpr::Symbol("x".to_string())],
                loc: None,
            },
        );
        let mut registry = HashMap::from([
            ("callee".to_string(), callee),
            ("caller".to_string(), caller.clone()),
        ]);
        let first = key(&caller, &[tensor(vec![2], 0.0, Dtype::F32)], &registry);
        registry.get_mut("callee").unwrap().body = SheafValue::String("two".to_string(), SourceLocation::unknown());
        let second = key(&caller, &[tensor(vec![2], 0.0, Dtype::F32)], &registry);
        assert_ne!(first, second);
        assert_ne!(module_name_for(&first), module_name_for(&second));
    }

    #[test]
    fn cache_key_preserves_dict_field_order() {
        let definition = definition("f", &["x"], "f", passthrough());
        let registry = HashMap::from([("f".to_string(), definition.clone())]);
        let mut first = BTreeMap::new();
        first.insert("a".to_string(), tensor(vec![2], 0.0, Dtype::F32));
        let mut second = BTreeMap::new();
        second.insert("b".to_string(), tensor(vec![2], 0.0, Dtype::F32));
        assert_ne!(key(&definition, &[Value::Dict(first)], &registry), key(&definition, &[Value::Dict(second)], &registry));
    }

    #[test]
    fn module_catalog_publishes_and_finds_variant_by_key() {
        let definition = definition("catalog-publish", &["x"], "f", passthrough());
        let registry = HashMap::from([("catalog-publish".to_string(), definition.clone())]);
        let key = key(&definition, &[tensor(vec![2], 0.0, Dtype::F32)], &registry);
        let mut compiler = jit_compiler();

        assert!(compiler
            .register_module(key.clone(), module("jit_catalog_publish"))
            .is_ok());
        let other_compiler = jit_compiler();

        assert!(matches!(
            other_compiler.module_for_key(&key),
            Ok(Some(metadata)) if metadata == module("jit_catalog_publish")
        ));
    }

    #[test]
    fn module_catalog_returns_cloned_metadata() {
        let definition = definition("catalog-clone", &["x"], "f", passthrough());
        let registry = HashMap::from([("catalog-clone".to_string(), definition.clone())]);
        let key = key(&definition, &[tensor(vec![2], 0.0, Dtype::F32)], &registry);
        let mut compiler = jit_compiler();
        assert!(compiler
            .register_module(key.clone(), module("jit_catalog_clone"))
            .is_ok());

        let result = compiler.module_for_key(&key);
        assert!(matches!(&result, Ok(Some(_))));
        let Some(mut metadata) = result.ok().flatten() else {
            return;
        };
        metadata.module_name.push_str("_modified");
        metadata.signature.param_types.clear();

        assert!(matches!(
            compiler.module_for_key(&key),
            Ok(Some(metadata)) if metadata == module("jit_catalog_clone")
        ));
    }

    #[test]
    fn module_catalog_keeps_distinct_keys_separate() {
        let definition = definition("catalog-distinct", &["x"], "f", passthrough());
        let registry = HashMap::from([("catalog-distinct".to_string(), definition.clone())]);
        let first = key(&definition, &[tensor(vec![2], 0.0, Dtype::F32)], &registry);
        let second = key(&definition, &[tensor(vec![3], 0.0, Dtype::F32)], &registry);
        let mut compiler = jit_compiler();
        assert!(compiler
            .register_module(first.clone(), module("jit_catalog_first"))
            .is_ok());
        assert!(compiler
            .register_module(second.clone(), module("jit_catalog_second"))
            .is_ok());

        assert!(matches!(
            compiler.module_for_key(&first),
            Ok(Some(metadata)) if metadata == module("jit_catalog_first")
        ));
        assert!(matches!(
            compiler.module_for_key(&second),
            Ok(Some(metadata)) if metadata == module("jit_catalog_second")
        ));
    }

    #[test]
    fn module_catalog_publication_preserves_local_variant() {
        let definition = definition("catalog-local", &["x"], "f", passthrough());
        let registry = HashMap::from([("catalog-local".to_string(), definition.clone())]);
        let key = key(&definition, &[tensor(vec![2], 0.0, Dtype::F32)], &registry);
        let mut compiler = jit_compiler();
        assert!(compiler
            .register_module(key.clone(), module("jit_catalog_local"))
            .is_ok());

        assert!(matches!(
            compiler.variant_for(&key),
            Some((module_name, module_signature))
                if module_name == "jit_catalog_local" && module_signature == &signature()
        ));
    }

    #[test]
    fn compilation_transaction_excludes_another_compiler() {
        let first = JIT_COMPILATION_LOCK.lock().unwrap();
        assert!(JIT_COMPILATION_LOCK.try_lock().is_err());
        drop(first);
        assert!(JIT_COMPILATION_LOCK.try_lock().is_ok());
    }

    #[test]
    fn jit_fail_does_not_blacklist_regular_definition() {
        let mut compiler = jit_compiler();
        // An ordinary JIT failure must be logged without poisoning any
        // failure set: structural and per-variant recording happens at the
        // call site that owns the cache key.
        compiler.jit_fail("regular-fn", "scalar-only args");
        assert!(compiler.failed_definitions.is_empty());
        assert!(compiler.failed_keys.is_empty());
        assert!(compiler.failed_vag.is_empty());
    }

    #[test]
    fn variant_failure_cache_keeps_distinct_shapes_independent() {
        let definition = definition("variant-fn", &["x"], "f", passthrough());
        let registry = HashMap::from([("variant-fn".to_string(), definition.clone())]);
        let small = key(&definition, &[tensor(vec![2], 0.0, Dtype::F32)], &registry);
        let large = key(&definition, &[tensor(vec![3], 0.0, Dtype::F32)], &registry);
        assert_ne!(small, large);

        let mut compiler = jit_compiler();
        compiler.failed_keys.insert(small.clone());

        assert!(compiler.failed_keys.contains(&small));
        // A distinct runtime shape is its own variant: a failed sibling
        // with a different key must not blacklist it.
        assert!(!compiler.failed_keys.contains(&large));
    }

    #[test]
    fn vag_failure_recording_isolates_from_ordinary_sets() {
        let mut compiler = jit_compiler();
        let vag_key = "__vag_isolated".to_string();
        compiler.jit_fail(&vag_key, "compilation failed on all backends");

        assert!(compiler.failed_vag.contains(&vag_key));
        assert_eq!(
            compiler.last_vag_fail_reason.as_deref(),
            Some("compilation failed on all backends")
        );
        // Value-and-grad failures stay isolated from the ordinary failure sets.
        assert!(compiler.failed_definitions.is_empty());
        assert!(compiler.failed_keys.is_empty());
    }
}
