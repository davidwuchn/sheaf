// Copyright (c) 2026 Damien Boureille
// Licensed under the MIT License.

//! Call validation and interpreter/JIT conformance tests.

use sheaf_compiler::core::{
    config, expr::CompiledExpr, inference::{FunctionSignature, reconstruct_jit_result},
};
use sheaf_compiler::interpreter::eval::Interpreter;
use sheaf_compiler::interpreter::value::{Dtype, Value};
use sheaf_compiler::runtime::iree_session::{IreeSession, shared_session};
use sheaf_compiler::runtime::jit::{
    JitCompileOutcome, JitCompiler, cache_key_for_function, module_name_for,
};
use std::collections::BTreeMap;
use std::sync::Arc;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FailureKind {
    InvalidCallAccepted,
    WrongDiagnostic,
    Compilation,
    Infrastructure,
    Execution,
    Result,
}

#[derive(Debug)]
struct Failure {
    kind: FailureKind,
    detail: String,
}

impl Failure {
    fn new(kind: FailureKind, detail: impl Into<String>) -> Self {
        Self { kind, detail: detail.into() }
    }
}

#[derive(Clone, Copy)]
enum Expectation {
    Pass,
    KnownFailure {
        kind: FailureKind,
        reason: &'static str,
        detail_contains: &'static str,
    },
}

fn check_expectation(
    expectation: Expectation,
    observed: Result<(), Failure>,
) -> Result<String, String> {
    match (expectation, observed) {
        (Expectation::Pass, Ok(())) => Ok("pass".to_string()),
        (Expectation::Pass, Err(failure)) => Err(format!("{failure:?}")),
        (Expectation::KnownFailure { reason, .. }, Ok(())) => Err(format!(
            "resolved known failure: {reason}; remove its marker and require success"
        )),
        (Expectation::KnownFailure { kind, reason, detail_contains }, Err(failure)) => {
            if failure.kind != kind || !failure.detail.contains(detail_contains) {
                return Err(format!(
                    "expected {kind:?} ({reason}), observed {failure:?}"
                ));
            }
            Ok(format!("known failure: {reason}: {}", failure.detail))
        }
    }
}

fn record(
    errors: &mut Vec<String>,
    name: &str,
    phase: &str,
    expectation: Expectation,
    observed: Result<(), Failure>,
) {
    match check_expectation(expectation, observed) {
        Ok(status) => println!("{name} / {phase}: {status}"),
        Err(error) => errors.push(format!("{name} / {phase}: {error}")),
    }
}

fn initialize_device() {
    let device = std::env::var("SHEAF_SIGNATURE_DEVICE").unwrap_or_else(|_| "cpu".to_string());
    config::init(0, Some(device), false);
}

fn interpreter_only() -> Interpreter {
    let mut interpreter = Interpreter::new();
    interpreter.env_mut().jit_compiler = None;
    interpreter
}

struct InvalidCall {
    name: &'static str,
    source: &'static str,
    diagnostic: &'static str,
    expectation: Expectation,
}

const ACCEPTED_INVALID: Expectation = Expectation::KnownFailure {
    kind: FailureKind::InvalidCallAccepted,
    reason: "call validation is missing",
    detail_contains: "returned",
};

const fn apply_options_gap() -> Expectation {
    Expectation::KnownFailure {
        kind: FailureKind::WrongDiagnostic,
        reason: "apply does not accept an options dictionary yet",
        detail_contains: "apply requires 2 arguments",
    }
}

#[test]
fn invalid_call_signatures() {
    initialize_device();
    let cases = [
        InvalidCall {
            name: "unknown-keyword", source: "(sum [[1 2] [3 4]] :axiss 0)",
            diagnostic: "axiss", expectation: ACCEPTED_INVALID,
        },
        InvalidCall {
            name: "axis-type", source: "(sum [[1 2] [3 4]] :axis \"bad\")",
            diagnostic: "axis", expectation: ACCEPTED_INVALID,
        },
        InvalidCall {
            name: "duplicate-keyword", source: "(sum [[1 2] [3 4]] :axis 0 :axis 1)",
            diagnostic: "axis", expectation: ACCEPTED_INVALID,
        },
        InvalidCall {
            name: "axis-without-value", source: "(sum [[1 2] [3 4]] :axis)",
            diagnostic: "axis", expectation: ACCEPTED_INVALID,
        },
        InvalidCall {
            name: "builtin-extra", source: "(zeros '[2] 999)",
            diagnostic: "zeros", expectation: ACCEPTED_INVALID,
        },
        InvalidCall {
            name: "lambda-extra", source: "((fn [x] x) 1 2)",
            diagnostic: "argument", expectation: ACCEPTED_INVALID,
        },
        InvalidCall {
            name: "apply-lambda-extra", source: "(apply (fn [x] x) '[1 2])",
            diagnostic: "argument", expectation: ACCEPTED_INVALID,
        },
        InvalidCall {
            name: "named-function-extra",
            source: "(defn signature_arity [x] x) (signature_arity 1 2)",
            diagnostic: "expects 1 arguments", expectation: Expectation::Pass,
        },
        InvalidCall {
            name: "named-function-missing",
            source: "(defn signature_arity [x] x) (signature_arity)",
            diagnostic: "expects 1 arguments", expectation: Expectation::Pass,
        },
        InvalidCall {
            name: "cast-type", source: "(cast [1 2] \"f32\")",
            diagnostic: "keyword", expectation: Expectation::Pass,
        },
        InvalidCall {
            name: "random-normal-arity", source: "(random-normal (random-key 42))",
            diagnostic: "random-normal", expectation: Expectation::Pass,
        },
        InvalidCall {
            name: "computed-keyword-is-positional",
            source: "(let [k :axis] (sum [1.0 2.0] k 0))",
            diagnostic: "argument", expectation: ACCEPTED_INVALID,
        },
        InvalidCall {
            name: "indirect-keywords-are-data",
            source: "(apply sum [[1.0 2.0] :axis 0 :axis 0])",
            diagnostic: "argument", expectation: ACCEPTED_INVALID,
        },
        InvalidCall {
            name: "indirect-unknown-keyword",
            source: "(apply sum [[1.0 2.0]] {:axiss 0})",
            diagnostic: "axiss", expectation: apply_options_gap(),
        },
        InvalidCall {
            name: "apply-named-type",
            source: "(apply sum [[1.0 2.0]] {:axis \"bad\"})",
            diagnostic: "axis", expectation: apply_options_gap(),
        },
        InvalidCall {
            name: "apply-named-dict",
            source: "(apply sum [[1.0 2.0]] '[1 2])",
            diagnostic: "must be a dict", expectation: apply_options_gap(),
        },
        InvalidCall {
            name: "fractional-axis", source: "(sum [1 2] :axis 0.5)",
            diagnostic: "axis", expectation: ACCEPTED_INVALID,
        },
        InvalidCall {
            name: "non-boolean-flag", source: "(sum [1 2] :keepdims 1)",
            diagnostic: "boolean", expectation: ACCEPTED_INVALID,
        },
        InvalidCall {
            name: "conflicting-dtypes", source: "(sum [1.5 2.0] :f32 :i32)",
            diagnostic: "conflicting keywords", expectation: ACCEPTED_INVALID,
        },
        InvalidCall {
            name: "indirect-conflicting-dtypes",
            source: "(apply sum [[1.5 2.0]] {:f32 true :i32 true})",
            diagnostic: "conflicting keywords", expectation: apply_options_gap(),
        },
        InvalidCall {
            name: "private-matmul-gradient",
            source: "(@-grad-lhs [[1.0]] [[2.0]] [[3.0]])",
            diagnostic: "internal operation", expectation: ACCEPTED_INVALID,
        },
        InvalidCall {
            name: "private-sum-to-shape", source: "(sum_to_shape 1 '[1])",
            diagnostic: "internal operation",
            expectation: Expectation::KnownFailure {
                kind: FailureKind::WrongDiagnostic,
                reason: "sum_to_shape is rejected as an unknown function, not an internal operation",
                detail_contains: "Unknown function: sum_to_shape",
            },
        },
        InvalidCall {
            name: "private-scan-gradient", source: "(__scan_vjp__ 1 2 3)",
            diagnostic: "internal operation",
            expectation: Expectation::KnownFailure {
                kind: FailureKind::WrongDiagnostic,
                reason: "__scan_vjp__ is rejected as an unknown function, not an internal operation",
                detail_contains: "Unknown function: __scan_vjp__",
            },
        },
        InvalidCall {
            name: "user-function-redefinition",
            source: "(defn signature_target [x] x) (defn signature_target [x] (+ x 1))",
            diagnostic: "Redefinition is not allowed", expectation: Expectation::Pass,
        },
        InvalidCall {
            name: "builtin-redefinition",
            source: "(defn max [a b] (+ a b)) (max 1 2)",
            diagnostic: "Redefinition is not allowed", expectation: ACCEPTED_INVALID,
        },
        InvalidCall {
            name: "builtin-redefinition-after-use",
            source: "(defn signature_before [x] (max x 2)) (defn max [a b] (+ a b)) (signature_before 1)",
            diagnostic: "Redefinition is not allowed", expectation: ACCEPTED_INVALID,
        },
    ];
    let mut errors = Vec::new();
    for case in cases {
        let result = interpreter_only().eval(case.source);
        let observed = match result {
            Ok(value) => Err(Failure::new(
                FailureKind::InvalidCallAccepted, format!("returned {value:?}"),
            )),
            Err(error) => {
                let message = error.to_string();
                if message.contains(case.diagnostic) {
                    Ok(())
                } else {
                    Err(Failure::new(FailureKind::WrongDiagnostic, message))
                }
            }
        };
        record(&mut errors, case.name, "validation", case.expectation, observed);
    }
    assert!(errors.is_empty(), "{}", errors.join("\n"));
}

#[test]
fn valid_call_signatures() {
    let cases = [
        ("keyword-as-data", "(get {:axis 3} :axis)", "3", Expectation::Pass),
        (
            "local-shadows-registry",
            "(defn signature_target [a] (+ a 10)) (let [signature_target (fn [a] (+ a 1))] (signature_target 2))",
            "3", Expectation::Pass,
        ),
        ("local-shadows-builtin", "(let [count (fn [x] (+ x 1))] (count 2))", "3", Expectation::Pass),
        ("indirect-alias", "(apply count '[[:a :b]])", "2", Expectation::Pass),
        (
            "computed-keyword-value",
            "(let [a 0] (sum [[1.0 2.0] [3.0 4.0]] :axis a))",
            "[4.0 6.0]", Expectation::Pass,
        ),
        (
            "indirect-keywords",
            "(let [x [[1.0 2.0] [3.0 4.0]]] (apply sum [x] {:axis 0 :keepdims true}))",
            "[[4.0 6.0]]",
            Expectation::KnownFailure {
                kind: FailureKind::Execution,
                reason: "apply does not accept an options dictionary yet",
                detail_contains: "apply requires 2 arguments",
            },
        ),
        (
            "computed-keyword-indirect",
            "(let [k :axis x [[1.0 2.0] [3.0 4.0]]] (apply sum [x] {k 0}))",
            "[4.0 6.0]",
            Expectation::KnownFailure {
                kind: FailureKind::Execution,
                reason: "apply does not accept an options dictionary yet",
                detail_contains: "apply requires 2 arguments",
            },
        ),
        ("indirect-builtin", "(apply + '[1.0 2.0])", "3.0", Expectation::Pass),
        ("and-short-circuit", "(and false (zeros))", "false", Expectation::Pass),
        ("or-short-circuit", "(or true (zeros))", "true", Expectation::Pass),
        ("split-default", "(len (random-split (random-key 42)))", "2", Expectation::Pass),
        ("eye-default", "(eye 2)", "[[1.0 0.0] [0.0 1.0]]", Expectation::Pass),
        (
            "first-class-random-normal",
            "(apply random-normal [(random-key 42) '[2]])",
            "(random-normal (random-key 42) '[2])",
            Expectation::KnownFailure {
                kind: FailureKind::Execution,
                reason: "registered random-normal builtin is not a resolvable function value",
                detail_contains: "Undefined symbol: random-normal",
            },
        ),
    ];
    let mut errors = Vec::new();
    for (name, source, expected, expectation) in cases {
        let mut interpreter = interpreter_only();
        let expected = interpreter.eval(expected).expect("valid reference result");
        let result = interpreter.eval(source)
            .map_err(|error| Failure::new(FailureKind::Execution, error.to_string()))
            .and_then(|value| compare_values(&value, &expected, "result")
                .map_err(|error| Failure::new(FailureKind::Result, error)));
        record(&mut errors, name, "valid call", expectation, result);
    }
    assert!(errors.is_empty(), "{}", errors.join("\n"));
}

#[test]
fn print_alias_signatures() {
    let binary = std::env::var_os("SHEAF_TEST_BINARY").expect("declared Bazel binary");
    let mut errors = Vec::new();
    for (name, expectation) in [
        ("print", Expectation::Pass),
        ("println", Expectation::KnownFailure {
            kind: FailureKind::Result,
            reason: "println does not use the keyword binding of its print alias",
            detail_contains: ":end",
        }),
    ] {
        let output = std::process::Command::new(&binary)
            .args(["--device", "cpu", "-c", &format!("({name} \"hello\" :end \"!\")")])
            .output().expect("execute declared Bazel binary");
        let observed = if !output.status.success() {
            Err(Failure::new(
                FailureKind::Execution, String::from_utf8_lossy(&output.stderr).into_owned(),
            ))
        } else if output.stdout == b"hello!" {
            Ok(())
        } else {
            Err(Failure::new(
                FailureKind::Result,
                format!("stdout {:?}, expected hello!", String::from_utf8_lossy(&output.stdout)),
            ))
        };
        record(&mut errors, name, "stdout", expectation, observed);
    }
    assert!(errors.is_empty(), "{}", errors.join("\n"));
}

fn compare_values(actual: &Value, expected: &Value, path: &str) -> Result<(), String> {
    let actual = actual.ensure_host().map_err(|error| error.to_string())?;
    let expected = expected.ensure_host().map_err(|error| error.to_string())?;
    let scalar = |value: &Value| match value {
        Value::Int(n) => Some((Dtype::I32, *n as f32)),
        Value::Float(n) => Some((Dtype::F32, *n)),
        Value::Tensor { data, dtype } if data.ndim() == 0 => {
            Some((*dtype, *data.iter().next().expect("scalar tensor element")))
        }
        _ => None,
    };
    if let (Some((actual_dtype, actual_value)), Some((expected_dtype, expected_value))) =
        (scalar(&actual), scalar(&expected))
    {
        if actual_dtype != expected_dtype {
            return Err(format!("{path}: dtype {actual_dtype:?}, expected {expected_dtype:?}"));
        }
        return compare_number(actual_value, expected_value, path);
    }
    match (&actual, &expected) {
        (
            Value::Tensor { data: a, dtype: ad },
            Value::Tensor { data: e, dtype: ed },
        ) => {
            if ad != ed || a.shape() != e.shape() {
                return Err(format!(
                    "{path}: {ad:?}{:?}, expected {ed:?}{:?}", a.shape(), e.shape(),
                ));
            }
            for (index, (&a, &e)) in a.iter().zip(e.iter()).enumerate() {
                compare_number(a, e, &format!("{path}[{index}]"))?;
            }
            Ok(())
        }
        (Value::List(a), Value::List(e)) | (Value::Tuple(a), Value::Tuple(e)) => {
            if a.len() != e.len() {
                return Err(format!("{path}: length {}, expected {}", a.len(), e.len()));
            }
            for (index, (a, e)) in a.iter().zip(e.iter()).enumerate() {
                compare_values(a, e, &format!("{path}[{index}]"))?;
            }
            Ok(())
        }
        (Value::Dict(a), Value::Dict(e)) => {
            if !a.keys().eq(e.keys()) {
                return Err(format!("{path}: dict keys differ"));
            }
            for (key, e) in e {
                compare_values(&a[key], e, &format!("{path}.{key}"))?;
            }
            Ok(())
        }
        (Value::Bool(a), Value::Bool(e)) if a == e => Ok(()),
        (Value::Nil, Value::Nil) => Ok(()),
        (Value::String(a), Value::String(e)) | (Value::Keyword(a), Value::Keyword(e))
            if a == e => Ok(()),
        _ => Err(format!("{path}: {actual:?}, expected {expected:?}")),
    }
}

fn compare_number(actual: f32, expected: f32, path: &str) -> Result<(), String> {
    if actual == expected || (actual.is_nan() && expected.is_nan()) {
        return Ok(());
    }
    if actual.is_finite() && expected.is_finite()
        && (actual - expected).abs() <= 1e-5 + 1e-5 * expected.abs()
    {
        return Ok(());
    }
    Err(format!("{path}: {actual}, expected {expected}"))
}

struct TensorCase {
    name: &'static str,
    params: &'static str,
    args: &'static [&'static str],
    body: &'static str,
    expected: &'static str,
    interpreted: Expectation,
    compilation: Expectation,
    compiled: Expectation,
}

const fn result_gap(detail_contains: &'static str) -> Expectation {
    Expectation::KnownFailure {
        kind: FailureKind::Result,
        reason: "operation result differs from its signature",
        detail_contains,
    }
}

struct CompiledCall {
    session: Arc<IreeSession>,
    qualified_name: String,
    signature: FunctionSignature,
}

fn compile_call(
    interpreter: &Interpreter,
    name: &str,
    args: &[Value],
) -> Result<CompiledCall, Failure> {
    let func = interpreter.registry_get(name).expect("test function registered");
    let registry = &interpreter.env().registry;
    let session = shared_session()
        .map_err(|error| Failure::new(FailureKind::Infrastructure, error.to_string()))?;
    let mut jit = JitCompiler::new();
    let outcome = jit.try_jit_compile(func, args, registry, &session);
    let sig = match outcome {
        JitCompileOutcome::Compiled(signature) => signature,
        JitCompileOutcome::Failed(reason) => {
            return Err(Failure::new(FailureKind::Compilation, format!("Failed: {reason}")));
        }
        JitCompileOutcome::Unsupported(reason) => {
            return Err(Failure::new(FailureKind::Compilation, format!("Unsupported: {reason}")));
        }
        JitCompileOutcome::InProgress => {
            return Err(Failure::new(FailureKind::Infrastructure, "variant still compiling"));
        }
    };
    let key = cache_key_for_function(func, args, registry)
        .ok_or_else(|| Failure::new(FailureKind::Compilation, "missing compiled cache key"))?;
    let module = module_name_for(name, &key);
    Ok(CompiledCall {
        session, qualified_name: format!("{module}.{name}"), signature: sig,
    })
}

impl CompiledCall {
    fn execute(self, args: &[Value]) -> Result<Value, Failure> {
        let sig = self.signature;
        // Invoke the loaded VMFB directly. This path cannot execute an interpreter fallback.
        let mut result = self.session.call_typed_device(&self.qualified_name, args, &sig.return_type)
            .map_err(|error| Failure::new(FailureKind::Execution, error.to_string()))?;
        if !sig.arg_type_layouts.is_empty() {
            result = reconstruct_jit_result(result, &sig.return_type, &sig.arg_type_layouts);
        }
        if let Some(layout) = &sig.return_layout {
            result = layout.reconstruct(result);
        }
        if let (Some(keys), Value::Tuple(items)) = (&sig.return_dict_keys, &result)
            && keys.len() == items.len()
        {
            result = Value::Dict(keys.iter().cloned().zip(items.iter().cloned()).collect());
        }
        Ok(result)
    }
}

#[test]
fn tensor_operation_signatures() {
    initialize_device();
    let cases = [
        TensorCase {
            name: "broadcast", params: "x y", args: &["[[1.0] [2.0]]", "[[3.0 4.0]]"],
            body: "(+ x y)", expected: "[[4.0 5.0] [5.0 6.0]]",
            interpreted: Expectation::Pass, compilation: Expectation::Pass,
            compiled: Expectation::Pass,
        },
        TensorCase {
            name: "sum_axis", params: "x", args: &["[[1.0 2.0] [3.0 4.0]]"],
            body: "(sum x :axis 0 :keepdims)", expected: "[[4.0 6.0]]",
            interpreted: Expectation::Pass, compilation: Expectation::Pass,
            compiled: Expectation::Pass,
        },
        TensorCase {
            name: "product_keepdims", params: "x", args: &["[[1.0 2.0] [3.0 4.0]]"],
            body: "(product x :axis 0 :keepdims)", expected: "[[3.0 8.0]]",
            interpreted: result_gap("result: F32[2], expected F32[1, 2]"), compilation: Expectation::Pass,
            compiled: Expectation::Pass,
        },
        TensorCase {
            name: "transpose_axes", params: "x", args: &["[[1.0 2.0] [3.0 4.0]]"],
            body: "(transpose x '[0 1])", expected: "[[1.0 2.0] [3.0 4.0]]",
            interpreted: result_gap("result[1]: 3, expected 2"), compilation: Expectation::Pass,
            compiled: Expectation::Pass,
        },
        TensorCase {
            name: "normalize_global", params: "x", args: &["[[1.0 9.0] [8.0 2.0]]"],
            body: "(normalize x)", expected: "[[0.05 0.45] [0.4 0.1]]",
            interpreted: Expectation::Pass, compilation: Expectation::Pass,
            compiled: result_gap("result[0]: 0.1, expected 0.05"),
        },
        TensorCase {
            name: "softmax_axis", params: "x", args: &["[[1.0 2.0] [3.0 4.0]]"],
            body: "(softmax x :axis 0)",
            expected: "[[0.11920292 0.11920292] [0.88079708 0.88079708]]",
            interpreted: result_gap("result[0]: 0.268941"), compilation: Expectation::Pass,
            compiled: result_gap("result[0]: 0.268941"),
        },
        TensorCase {
            name: "log_softmax_axis", params: "x", args: &["[[1.0 2.0] [3.0 4.0]]"],
            body: "(log-softmax x :axis 0)",
            expected: "[[-2.126928 -2.126928] [-0.126928 -0.126928]]",
            interpreted: result_gap("result[0]: -1.313261"), compilation: Expectation::Pass,
            compiled: result_gap("result[0]: -1.313261"),
        },
        TensorCase {
            name: "argmax_global", params: "x", args: &["[[1.0 9.0] [8.0 2.0]]"],
            body: "(argmax x)", expected: "1",
            interpreted: Expectation::Pass, compilation: Expectation::Pass,
            compiled: result_gap("shape=[2]"),
        },
        TensorCase {
            name: "indexed_shape", params: "p", args: &["{:W (zeros '[2 1])}"],
            body: "(let [n (get (shape (get p :W)) 0)] (zeros [n 3]))",
            expected: "(zeros '[2 3])",
            interpreted: Expectation::Pass, compilation: Expectation::Pass,
            compiled: Expectation::Pass,
        },
        TensorCase {
            name: "keyword_data", params: "p", args: &["{:axis [1.0 2.0]}"],
            body: "(get p :axis)", expected: "[1.0 2.0]",
            interpreted: Expectation::Pass, compilation: Expectation::Pass,
            compiled: Expectation::Pass,
        },
        TensorCase {
            name: "nested_list", params: "x", args: &["[1.0 2.0]"],
            body: "(append '[] [x (+ x 1.0)])",
            expected: "(append '[] [(tensor '[1.0 2.0]) (tensor '[2.0 3.0])])",
            interpreted: Expectation::Pass, compilation: Expectation::Pass,
            compiled: Expectation::Pass,
        },
        TensorCase {
            name: "xavier_shape", params: "key dims", args: &["(random-key 42)", "'[2 3]"],
            body: "(xavier-normal key dims)", expected: "(xavier-normal (random-key 42) '[2 3])",
            interpreted: Expectation::Pass,
            compilation: Expectation::KnownFailure {
                kind: FailureKind::Compilation,
                reason: "random-normal cannot resolve a shape parameter",
                detail_contains: "random-normal expects a vector shape argument",
            },
            compiled: Expectation::Pass,
        },
        TensorCase {
            name: "split_destructure", params: "key", args: &["(random-key 42)"],
            body: "(let [[k1 k2] (random-split key)] (zeros '[2]))",
            expected: "(zeros '[2])",
            interpreted: Expectation::Pass,
            compilation: Expectation::KnownFailure {
                kind: FailureKind::Compilation,
                reason: "random-split result length is lost before destructuring",
                detail_contains: "Destructuring source has a dynamic length",
            },
            compiled: Expectation::Pass,
        },
    ];
    let mut errors = Vec::new();
    for case in cases {
        let mut interpreter = interpreter_only();
        let name = format!("signature_{}", case.name);
        interpreter.eval(&format!("(defn {name} [{}] {})", case.params, case.body))
            .expect("valid test definition");
        let args = case.args.iter().map(|source| {
            interpreter.eval(source).expect("valid test argument")
        }).collect::<Vec<_>>();
        let expected = interpreter.eval(case.expected).expect("valid reference result");
        let mut call_args = Vec::new();
        for (index, value) in args.iter().enumerate() {
            let param = format!("signature_arg_{index}");
            interpreter.env_mut().set_global(&param, value.clone());
            call_args.push(CompiledExpr::Symbol(param));
        }
        let call = CompiledExpr::FunctionCall {
            name: name.clone(), args: call_args, loc: None,
        };
        let interpreted = sheaf_compiler::interpreter::eval(&call, interpreter.env_mut())
            .map_err(|error| Failure::new(FailureKind::Execution, error.to_string()))
            .and_then(|result| compare_values(&result, &expected, "result")
                .map_err(|error| Failure::new(FailureKind::Result, error)));
        record(&mut errors, case.name, "interpreter", case.interpreted, interpreted);
        match compile_call(&interpreter, &name, &args) {
            Err(failure) => {
                record(&mut errors, case.name, "compilation", case.compilation, Err(failure));
                println!("{} / compiled result: blocked by compilation", case.name);
            }
            Ok(compiled) => {
                record(&mut errors, case.name, "compilation", case.compilation, Ok(()));
                let compared = compiled.execute(&args)
                    .and_then(|value| compare_values(&value, &expected, "result")
                    .map_err(|error| Failure::new(FailureKind::Result, error)));
                record(&mut errors, case.name, "VMFB", case.compiled, compared);
            }
        }
    }
    assert!(errors.is_empty(), "{}", errors.join("\n"));
}

#[test]
fn known_failure_markers_reject_unexpected_success_and_unrelated_failures() {
    const RESULT_GAP: Expectation = result_gap("result: wrong shape");
    assert!(check_expectation(RESULT_GAP, Ok(())).is_err());
    assert!(check_expectation(
        RESULT_GAP, Err(Failure::new(FailureKind::Execution, "device error")),
    ).is_err());
    assert!(check_expectation(
        RESULT_GAP, Err(Failure::new(FailureKind::Result, "result: wrong shape")),
    ).is_ok());
    assert!(check_expectation(
        RESULT_GAP, Err(Failure::new(FailureKind::Result, "result: wrong dtype")),
    ).is_err());
    assert!(check_expectation(
        ACCEPTED_INVALID, Err(Failure::new(FailureKind::InvalidCallAccepted, "unrelated")),
    ).is_err());
}

#[test]
fn comparator_checks_structure_dtype_shape_and_values() {
    let mut interpreter = interpreter_only();
    for (actual, expected) in [
        ("(quote [[1 2]])", "(quote [1 2])"),
        ("[1.0 2.0]", "(cast (tensor '[1.0 2.0]) :i32)"),
        ("[[1.0 2.0]]", "[1.0 2.0]"),
        ("[1.0 2.0]", "[1.0 3.0]"),
        ("{:a [1.0]}", "{:b [1.0]}"),
    ] {
        let a = interpreter.eval(actual).unwrap();
        let e = interpreter.eval(expected).unwrap();
        assert!(compare_values(&a, &e, "result").is_err(), "{actual} vs {expected}");
    }
    let list = Value::List(vec![Value::Int(1)]);
    let tuple = Value::Tuple(vec![Value::Int(1)]);
    assert!(compare_values(&list, &tuple, "result").is_err());
    let map = Value::Dict(BTreeMap::from([("a".to_string(), Value::Float(1.0))]));
    assert!(compare_values(&map, &map, "result").is_ok());
    assert!(compare_number(f32::INFINITY, 1.0, "result").is_err());
}
