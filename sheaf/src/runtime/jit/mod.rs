// Copyright (c) 2026 Damien Boureille
// Licensed under the MIT License.

//! JIT auto-compilation: transparently compile pure functions on first call.
//!
//! When the interpreter calls a function that has no pre-compiled VMFB,
//! the JIT attempts to compile it on the fly via the following pipeline:
//! type inference -> dict lowering -> inlining -> codegen -> MLIR
//! -> iree-compile -> load VMFB. On success, subsequent calls dispatch via IREE.
//! On failure, the function is added to a blocklist and the interpreter
//! handles it normally.

use crate::SheafError;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Mutex, OnceLock};

/// Cached CUDA target architecture (e.g. "sm_75"), detected once via NVML.
static CUDA_TARGET: OnceLock<Option<String>> = OnceLock::new();

/// Process-wide JIT events, primarily for deterministic integration tests.
pub static JIT_CATALOG_HITS: AtomicUsize = AtomicUsize::new(0);
pub static JIT_CATALOG_MISSES: AtomicUsize = AtomicUsize::new(0);
pub static JIT_MODULE_LOADS: AtomicUsize = AtomicUsize::new(0);
pub static JIT_EXTERNAL_COMPILATIONS: AtomicUsize = AtomicUsize::new(0);

pub fn detect_cuda_target() -> Option<String> {
    CUDA_TARGET
        .get_or_init(|| {
            let nvml = nvml_wrapper::Nvml::init().or_else(|_| {
                nvml_wrapper::Nvml::builder()
                    .lib_path(std::ffi::OsStr::new("libnvidia-ml.so.1"))
                    .init()
            });
            let nvml = match nvml {
                Ok(n) => n,
                Err(e) => {
                    sheaf_msg!("nvml: init failed: {}", e);
                    return None;
                }
            };
            let device = match nvml.device_by_index(0) {
                Ok(d) => d,
                Err(e) => {
                    sheaf_msg!("nvml: device_by_index failed: {}", e);
                    return None;
                }
            };
            let cc = match device.cuda_compute_capability() {
                Ok(c) => c,
                Err(e) => {
                    sheaf_msg!("nvml: compute_capability failed: {}", e);
                    return None;
                }
            };
            let target = format!("sm_{}{}", cc.major, cc.minor);
            Some(target)
        })
        .clone()
}

use super::toolchain::{ensure_toolchain, find_iree_compile, IREE_COMPILER_VERSION};

use crate::autodiff::{
    reverse::{reverse_grad, to_anf},
    simplify,
    transforms::cse,
};
use crate::core::expr::{CompiledExpr, FunctionDef};
use crate::lowering::codegen::{collect_tuple_leaves, expand_tuple_to_symbols, CodeGenerator};
use crate::lowering::config::{layout_to_index_map, lower_get_calls};
use crate::lowering::effects::{collect_effects, collect_hof_calls};
use crate::lowering::jit_eligibility;
use crate::lowering::stablehlo::{Register, StableHLOEmitter};
use crate::lowering::transforms::{
    extract_scalar_constants, filter_constants_for_shape_positions, lower_inlined_gets,
    propagate_let_layouts, resolve_static_constants, substitute_scalar_param, unroll_reduces,
};
use crate::sheaf_msg;

/// Identity of a compiled JIT variant.
///
/// The key contains the transitive function definition, argument types and
/// layouts, and runtime scalars consumed in shape expressions. Tensor contents
/// and non-constant scalar values are excluded. Unlike the disk content hash,
/// this key can be computed without emitting MLIR.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct JitCacheKey {
    /// Definition and transitive inlining identity, computed before lowering.
    definition_hash: String,
    function_name: String,
    argument_types: Vec<String>,
    argument_layouts: Vec<ArgumentLayout>,
    captured_scalars: Vec<(String, Vec<usize>, u64)>,
}

type ArgumentLayout = (String, Vec<(Vec<String>, Vec<i64>, Vec<usize>)>);

fn runtime_argument_layouts(args: &[Value]) -> Vec<ArgumentLayout> {
    args.iter()
        .filter_map(|arg| value_to_param_layout("", arg))
        .map(|layout| {
            let fields = layout
                .fields
                .into_iter()
                .map(|field| (field.path, field.shape, field.tuple_index))
                .collect();
            (layout.name, fields)
        })
        .collect()
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

fn function_definition_identity(
    func_def: &FunctionDef,
    registry: &HashMap<String, FunctionDef>,
) -> String {
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

impl JitCacheKey {
    fn fingerprint(&self) -> String {
        use std::hash::{Hash, Hasher};
        let mut first = StableHasher(0xcbf29ce484222325);
        let mut second = StableHasher(0x84222325cbf29ce4);
        self.hash(&mut first);
        self.hash(&mut second);
        format!("{:016x}{:016x}", first.finish(), second.finish())
    }
}

#[cfg(test)]
fn cache_key_for(
    fn_name: &str,
    params: &[String],
    args: &[Value],
    body_compiled: &CompiledExpr,
) -> JitCacheKey {
    cache_key_for_with_identity(fn_name, params, args, body_compiled, String::new())
}

/// Builds the cache key used by both lookup and compilation.
pub fn cache_key_for_function(
    func_def: &FunctionDef,
    args: &[Value],
    registry: &HashMap<String, FunctionDef>,
) -> Option<JitCacheKey> {
    let body = func_def.body_compiled.as_ref()?;
    let mut key = cache_key_for_with_identity(
        &func_def.name,
        &func_def.params,
        args,
        body,
        function_definition_identity(func_def, registry),
    );
    key.captured_scalars = shape_bearing_scalars(func_def, args, registry)?;
    Some(key)
}

/// Returns shape-dependent scalar constants after call inlining and get lowering.
fn shape_bearing_scalars(
    func_def: &FunctionDef,
    args: &[Value],
    registry: &HashMap<String, FunctionDef>,
) -> Option<Vec<(String, Vec<usize>, u64)>> {
    let mut body = func_def.body_compiled.clone()?;
    let mut constants = HashMap::new();
    let mut index_maps = Vec::new();
    let structural_signature = args
        .iter()
        .map(|arg| {
            value_to_stablehlo_type(arg)
                .map(|ty| ty.to_mlir())
                .unwrap_or_else(|_| "<untypeable>".into())
        })
        .collect::<Vec<_>>();
    let analysis_key = (
        function_definition_identity(func_def, registry),
        structural_signature,
        runtime_argument_layouts(args),
    );

    for (index, arg) in args.iter().enumerate() {
        let param = func_def
            .params
            .get(index)
            .cloned()
            .unwrap_or_else(|| format!("__a{index}"));
        let index_map = if let Some(layout) = value_to_param_layout(&param, arg) {
            layout_to_index_map(&layout)
        } else {
            std::iter::once((Vec::new(), vec![])).collect()
        };
        extract_scalar_constants(arg, &param, &index_map, &mut constants);
        body = lower_get_calls(&body, &param, &index_map);
        index_maps.push((param, index_map));
    }

    let positions = if let Some(positions) = shape_analysis_cache()
        .lock()
        .expect("shape analysis cache lock poisoned")
        .get(&analysis_key)
        .cloned()
    {
        positions
    } else {
        body = crate::autodiff::inline_function_calls(&body, registry);
        for (param, index_map) in &index_maps {
            body = lower_get_calls(&body, param, index_map);
        }
        body = lower_inlined_gets(&body, &index_maps);
        let mut positions = crate::lowering::transforms::collect_shape_gtes(&body)
            .into_iter()
            .collect::<Vec<_>>();
        positions.sort();
        shape_analysis_cache()
            .lock()
            .expect("shape analysis cache lock poisoned")
            .insert(analysis_key, positions.clone());
        positions
    };

    Some(
        constants
            .into_iter()
            .filter(|(path, _)| positions.contains(path))
            .collect::<BTreeMap<_, _>>()
            .into_iter()
            .map(|((param, indices), value)| (param, indices, value.to_bits()))
            .collect(),
    )
}

fn cache_key_for_with_identity(
    fn_name: &str,
    params: &[String],
    args: &[Value],
    body_compiled: &CompiledExpr,
    definition_hash: String,
) -> JitCacheKey {
    let argument_types = args
        .iter()
        .map(|arg| {
            value_to_stablehlo_type(arg)
                .map(|ty| ty.to_mlir())
                .unwrap_or_else(|_| "<untypeable>".to_string())
        })
        .collect();
    let argument_layouts = runtime_argument_layouts(args);

    let mut constants: HashMap<(String, Vec<usize>), f64> = HashMap::new();
    for (i, arg) in args.iter().enumerate() {
        let param_name = params
            .get(i)
            .cloned()
            .unwrap_or_else(|| format!("__a{}", i));
        if let Some(layout) = value_to_param_layout(&param_name, arg) {
            let imap = layout_to_index_map(&layout);
            extract_scalar_constants(arg, &param_name, &imap, &mut constants);
        } else {
            let fake_imap: BTreeMap<Vec<String>, Vec<usize>> =
                std::iter::once((Vec::new(), vec![])).collect();
            extract_scalar_constants(arg, &param_name, &fake_imap, &mut constants);
        }
    }
    let captured_scalars = filter_constants_for_shape_positions(&constants, body_compiled)
        .into_iter()
        .collect::<BTreeMap<_, _>>()
        .into_iter()
        .map(|((param, indices), value)| (param, indices, value.to_bits()))
        .collect();

    JitCacheKey {
        definition_hash,
        function_name: fn_name.to_string(),
        argument_types,
        argument_layouts,
        captured_scalars,
    }
}

/// Derives a unique IREE module name for a compiled variant.
pub fn module_name_for(fn_name: &str, cache_key: &JitCacheKey) -> String {
    let safe = fn_name
        .replace('-', "_")
        .replace('?', "_q")
        .replace('!', "_b");
    format!("{}__{}", safe, cache_key.fingerprint())
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

/// Process-wide metadata for modules loaded in the shared IREE session.
/// The mutex must not be held during compilation, module loading, or dispatch.
#[derive(Default)]
struct SharedModuleCatalog {
    modules: HashMap<JitCacheKey, CompiledModuleInfo>,
    identities: HashMap<String, JitCacheKey>,
    compiling: HashSet<JitCacheKey>,
    growth_warned: bool,
}

static SHARED_MODULE_CATALOG: OnceLock<Mutex<SharedModuleCatalog>> = OnceLock::new();

struct CompilationReservation {
    key: Option<JitCacheKey>,
}

impl CompilationReservation {
    fn new(key: JitCacheKey) -> Self {
        Self { key: Some(key) }
    }

    fn finish(&mut self) {
        self.key = None;
    }
}

impl Drop for CompilationReservation {
    fn drop(&mut self) {
        if let Some(key) = self.key.take() {
            shared_module_catalog()
                .lock()
                .expect("JIT catalogue lock poisoned")
                .compiling
                .remove(&key);
        }
    }
}

/// Cached shape-position analyses. Scalar values are projected afterwards.
static SHAPE_ANALYSIS_CACHE: OnceLock<
    Mutex<HashMap<(String, Vec<String>, Vec<ArgumentLayout>), Vec<(String, Vec<usize>)>>>,
> = OnceLock::new();

fn shared_module_catalog() -> &'static Mutex<SharedModuleCatalog> {
    SHARED_MODULE_CATALOG.get_or_init(|| Mutex::new(SharedModuleCatalog::default()))
}

fn shape_analysis_cache(
) -> &'static Mutex<HashMap<(String, Vec<String>, Vec<ArgumentLayout>), Vec<(String, Vec<usize>)>>>
{
    SHAPE_ANALYSIS_CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

pub struct JitCompiler {
    iree_compile_path: Option<String>,
    target_backend: String,
    /// Definitions rejected independently of runtime argument shapes.
    failed_definitions: HashSet<String>,
    failed_vag: HashSet<String>,
    /// Variants that failed compilation.
    failed_keys: HashSet<JitCacheKey>,
    last_vag_fail_reason: Option<String>,
    /// Cache compiled VAG sessions: vag_key -> (session_idx, signature, param_names)
    vag_cache: HashMap<String, (String, FunctionSignature, Vec<String>)>,
}

/// Dispatch metadata for a module loaded in the shared IREE session.
#[derive(Debug, Clone)]
pub struct CompiledModuleInfo {
    pub function_name: String,
    pub module_name: String,
    pub sig: FunctionSignature,
}

fn module_growth_warning(
    total: usize,
    threshold: usize,
    already_warned: bool,
    variants: &BTreeMap<&str, usize>,
) -> Option<String> {
    if threshold == 0 || already_warned || total <= threshold {
        return None;
    }
    let breakdown = variants
        .iter()
        .map(|(name, count)| format!("{}={}", name, count))
        .collect::<Vec<_>>()
        .join(", ");
    Some(format!(
        "WARN jit: {} distinct VMFB modules loaded (variants: {}).",
        total, breakdown
    ))
}

impl Default for JitCompiler {
    fn default() -> Self {
        Self::new()
    }
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
                        sheaf_msg!("error: Could not download the compiler toolchain and no cached compilation found.");
                        sheaf_msg!("       Reason: {}", e);
                        sheaf_msg!("       Install curl and unzip, or set IREE_COMPILE to a local iree-compile binary.");
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
            failed_vag: HashSet::new(),
            failed_keys: HashSet::new(),
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
            }
            .to_string();
        }
        // Avoid creating a probe session solely for backend detection.
        if let Some(backend) = crate::runtime::iree_session::IreeSession::cached_target_backend() {
            return backend.to_string();
        }
        // The active session backend is refreshed before compilation.
        "llvm-cpu".to_string()
    }

    /// Number of distinct VMFB modules loaded into the shared session.
    pub fn loaded_module_count(&self) -> usize {
        shared_module_catalog()
            .lock()
            .expect("JIT catalogue lock poisoned")
            .modules
            .len()
    }

    /// Obtain shared module metadata without exposing the catalogue lock to
    /// dispatch callers.
    pub(crate) fn module_for_key(&self, key: &JitCacheKey) -> Option<CompiledModuleInfo> {
        shared_module_catalog()
            .lock()
            .expect("JIT catalogue lock poisoned")
            .modules
            .get(key)
            .cloned()
    }
}

/// Why the JIT skipped a value-and-grad call.
#[derive(Debug)]
pub enum JitVagOutcome {
    /// JIT compiled and executed successfully, or had a runtime error.
    Success(Result<Value, SheafError>),
    /// The pattern is unsupported by JIT; use the interpreter fallback.
    Unsupported,
    /// JIT failed for a supported pattern.
    Bug(String),
}

mod compile;
mod preprocess;
mod support;
mod vag;

#[cfg(test)]
mod cache_key_tests {
    use super::{cache_key_for, cache_key_for_function, module_growth_warning, JitCompiler};
    use crate::core::ast::SheafValue;
    use crate::core::error::SourceLocation;
    use crate::core::expr::{CompiledExpr, FunctionDef};
    use crate::interpreter::value::{Dtype, Value};
    use ndarray::ArrayD;
    use std::collections::HashMap;
    use std::sync::Arc;

    fn tensor_f32(shape: Vec<usize>, fill: f32) -> Value {
        Value::Tensor {
            data: Arc::new(ArrayD::from_elem(shape, fill)),
            dtype: Dtype::F32,
        }
    }

    /// Body without shape-bearing scalar inputs.
    fn body_no_shape_scalar() -> CompiledExpr {
        CompiledExpr::FunctionCall {
            name: "first".to_string(),
            args: vec![CompiledExpr::Symbol("x".to_string())],
            loc: None,
        }
    }

    /// Body whose reshape dimensions read the scalar at `cfg[0]`.
    fn body_with_shape_scalar_on_cfg() -> CompiledExpr {
        let cfg_shape = CompiledExpr::Vector(vec![
            CompiledExpr::GetTupleElement {
                param: "cfg".into(),
                indices: vec![0],
            },
            CompiledExpr::GetTupleElement {
                param: "cfg".into(),
                indices: vec![0],
            },
        ]);
        CompiledExpr::FunctionCall {
            name: "reshape".to_string(),
            args: vec![CompiledExpr::Symbol("x".to_string()), cfg_shape],
            loc: None,
        }
    }

    fn function(name: &str, body: CompiledExpr) -> FunctionDef {
        function_with_params(name, &["x"], body)
    }

    fn function_with_params(name: &str, params: &[&str], body: CompiledExpr) -> FunctionDef {
        FunctionDef {
            name: name.into(),
            params: params.iter().map(|param| (*param).to_string()).collect(),
            body: SheafValue::Nil(SourceLocation::unknown()),
            body_compiled: Some(body),
            signature: None,
            vmfb_module_name: None,
            known_param_types: Vec::new(),
            compile_error: None,
        }
    }

    fn dict_cfg(n: f32) -> Value {
        let mut map = std::collections::BTreeMap::new();
        map.insert("n".to_string(), Value::Float(n));
        Value::Dict(map)
    }

    fn tensor_with_dtype(shape: Vec<usize>, dtype: Dtype) -> Value {
        Value::Tensor {
            data: Arc::new(ArrayD::zeros(shape)),
            dtype,
        }
    }

    /// Distinct runtime shapes produce distinct variants.
    #[test]
    fn cache_key_for_distinguishes_consecutive_distinct_shapes() {
        let body = body_no_shape_scalar();
        let params = vec!["x".to_string()];
        let args_a = vec![tensor_f32(vec![1, 5], 0.0)];
        let args_b = vec![tensor_f32(vec![1, 5, 1], 0.0)];
        let key_a = cache_key_for("sigmoid", &params, &args_a, &body);
        let key_b = cache_key_for("sigmoid", &params, &args_b, &body);
        assert_ne!(
            key_a, key_b,
            "two distinct shapes must yield distinct cache keys (CLEVR pattern)"
        );
        assert_eq!(key_a, cache_key_for("sigmoid", &params, &args_a, &body));
        assert_eq!(key_b, cache_key_for("sigmoid", &params, &args_b, &body));
    }

    /// Tensor contents do not affect the cache key.
    #[test]
    fn cache_key_for_collision_free_across_data() {
        let body = body_no_shape_scalar();
        let params = vec!["x".to_string()];
        let args_a = vec![tensor_f32(vec![1, 5], 0.0)];
        let args_b = vec![tensor_f32(vec![1, 5], 1.0)];
        let key_a = cache_key_for("sigmoid", &params, &args_a, &body);
        let key_b = cache_key_for("sigmoid", &params, &args_b, &body);
        assert_eq!(
            key_a, key_b,
            "two tensors with same shape/dtype but different data must hash the same (no tensor bytes hashed)"
        );
    }

    /// Shape-bearing scalar values affect the cache key.
    #[test]
    fn cache_key_for_distinguishes_direct_shape_scalars() {
        let body = CompiledExpr::FunctionCall {
            name: "reshape".into(),
            args: vec![
                CompiledExpr::Symbol("x".into()),
                CompiledExpr::Vector(vec![CompiledExpr::Symbol("n".into())]),
            ],
            loc: None,
        };
        let params = vec!["x".to_string(), "n".to_string()];
        let a = cache_key_for(
            "reshape",
            &params,
            &[tensor_f32(vec![4], 0.0), Value::Int(2)],
            &body,
        );
        let b = cache_key_for(
            "reshape",
            &params,
            &[tensor_f32(vec![4], 0.0), Value::Int(4)],
            &body,
        );
        assert_ne!(a, b);
    }

    #[test]
    fn cache_key_for_distinguishes_captured_shape_scalars() {
        let body = body_with_shape_scalar_on_cfg();
        let params = vec!["x".to_string(), "cfg".to_string()];
        // Two cfg dicts with different `:n` -- the shape-bearing scalar of `cfg`.
        let args_a = vec![tensor_f32(vec![4, 4], 0.0), dict_cfg(4.0)];
        let args_b = vec![tensor_f32(vec![4, 4], 0.0), dict_cfg(8.0)];
        let key_a = cache_key_for("my_reshape", &params, &args_a, &body);
        let key_b = cache_key_for("my_reshape", &params, &args_b, &body);
        assert_ne!(
            key_a, key_b,
            "shape-dependent scalar values must produce distinct cache keys"
        );
    }

    #[test]
    fn cache_key_for_function_includes_definition_and_inlined_callees() {
        let args = vec![tensor_f32(vec![2, 3], 1.0)];
        let callee = function("callee", body_no_shape_scalar());
        let caller = function(
            "caller",
            CompiledExpr::FunctionCall {
                name: "callee".into(),
                args: vec![CompiledExpr::Symbol("x".into())],
                loc: None,
            },
        );
        let mut registry = std::collections::HashMap::from([
            ("callee".to_string(), callee),
            ("caller".to_string(), caller.clone()),
        ]);
        let first = cache_key_for_function(&caller, &args, &registry).unwrap();
        let callee = registry.get_mut("callee").unwrap();
        callee.body = SheafValue::String("(relu x)".to_string(), SourceLocation::unknown());
        callee.body_compiled = Some(CompiledExpr::FunctionCall {
            name: "relu".into(),
            args: vec![CompiledExpr::Symbol("x".into())],
            loc: None,
        });
        let second = cache_key_for_function(&caller, &args, &registry).unwrap();
        assert_ne!(first, second);
    }

    #[test]
    fn cache_key_tracks_shape_scalars_used_by_inlined_callees() {
        let callee = function_with_params(
            "reshape-with-config",
            &["x", "cfg"],
            body_with_shape_scalar_on_cfg(),
        );
        let caller = function_with_params(
            "caller",
            &["x", "cfg"],
            CompiledExpr::FunctionCall {
                name: "reshape-with-config".into(),
                args: vec![
                    CompiledExpr::Symbol("x".into()),
                    CompiledExpr::Symbol("cfg".into()),
                ],
                loc: None,
            },
        );
        let registry = HashMap::from([
            (callee.name.clone(), callee),
            (caller.name.clone(), caller.clone()),
        ]);
        let tensor = tensor_f32(vec![4, 4], 0.0);
        let first =
            cache_key_for_function(&caller, &[tensor.clone(), dict_cfg(4.0)], &registry).unwrap();
        let second = cache_key_for_function(&caller, &[tensor, dict_cfg(8.0)], &registry).unwrap();
        assert_ne!(first, second);
    }

    #[test]
    fn variant_failure_does_not_blacklist_a_definition() {
        let mut compiler = JitCompiler::new();
        compiler.jit_fail("polymorphic", "shape-specific codegen failure");
        assert!(compiler.failed_definitions.is_empty());
        assert!(compiler.failed_keys.is_empty());
    }

    #[test]
    fn cache_key_ignores_non_shape_bearing_scalar_values() {
        let body = body_no_shape_scalar();
        let params = vec!["x".to_string(), "scale".to_string()];
        let tensor = tensor_f32(vec![2, 3], 1.0);
        let a = cache_key_for(
            "scale",
            &params,
            &[tensor.clone(), Value::Float(2.0)],
            &body,
        );
        let b = cache_key_for("scale", &params, &[tensor, Value::Float(9.0)], &body);
        assert_eq!(a, b);
    }

    #[test]
    fn cache_key_distinguishes_structure_and_dtype() {
        let body = body_no_shape_scalar();
        let params = vec!["x".to_string()];
        let f32_key = cache_key_for(
            "typed",
            &params,
            &[tensor_with_dtype(vec![2], Dtype::F32)],
            &body,
        );
        let bf16_key = cache_key_for(
            "typed",
            &params,
            &[tensor_with_dtype(vec![2], Dtype::BF16)],
            &body,
        );
        let tuple_key = cache_key_for(
            "typed",
            &params,
            &[Value::Tuple(vec![tensor_with_dtype(vec![2], Dtype::F32)])],
            &body,
        );
        assert_ne!(f32_key, bf16_key);
        assert_ne!(f32_key, tuple_key);
    }

    #[test]
    fn module_warning_fires_once_after_crossing_configured_threshold() {
        let variants = std::collections::BTreeMap::from([("alpha", 2), ("beta", 1)]);
        assert!(module_growth_warning(2, 2, false, &variants).is_none());
        let warning = module_growth_warning(3, 2, false, &variants).unwrap();
        assert!(warning.contains("3 distinct VMFB modules"));
        assert!(warning.contains("alpha=2"));
        assert!(warning.contains("beta=1"));
        assert!(module_growth_warning(4, 2, true, &variants).is_none());
        assert!(module_growth_warning(3, 0, false, &variants).is_none());
    }
}
