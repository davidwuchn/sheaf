// Copyright (c) 2025-2026 Damien Boureille
// Licensed under the MIT License.

//! Interpreter regressions.

use sheaf_compiler::interpreter::eval_exprs;
use std::path::Path;

struct TestCase {
    name: String,
    expr: String,
    expected: String,
    line: usize,
}

fn parse_test_yaml(path: &Path) -> Vec<TestCase> {
    let content = std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("Failed to read {}: {}", path.display(), e));

    let mut cases = Vec::new();
    let mut name = String::new();
    let mut expr = String::new();
    let mut expected = String::new();
    let mut entry_line = 0;

    for (i, line) in content.lines().enumerate() {
        let l = line.trim();

        if l.is_empty() || l.starts_with('#') {
            continue;
        }

        if l.starts_with("- name:") {
            if !expr.is_empty() {
                cases.push(TestCase {
                    name: name.clone(),
                    expr: expr.trim().to_string(),
                    expected: expected.trim().to_string(),
                    line: entry_line,
                });
            }
            name = l.trim_start_matches("- name:").trim().trim_matches('\'').trim_matches('"').to_string();
            expr.clear();
            expected.clear();
            entry_line = i + 1;
        } else if l.starts_with("test:") {
            let val = l.trim_start_matches("test:").trim();
            if val == ">" || val == "|" || val.is_empty() {
                expr.clear();
            } else {
                expr = val.to_string();
            }
        } else if l.starts_with("expected:") {
            let val = l.trim_start_matches("expected:").trim().trim_matches('\'').trim_matches('"').to_string();
            expected = val;
        } else if !expected.is_empty() {
        } else {
            if !expr.is_empty() {
                expr.push(' ');
            }
            expr.push_str(l);
        }
    }

    if !expr.is_empty() {
        cases.push(TestCase {
            name,
            expr: expr.trim().to_string(),
            expected: expected.trim().to_string(),
            line: entry_line,
        });
    }

    cases
}

fn eval(src: &str) -> String {
    match eval_exprs(src) {
        Ok(val) => format!("{}", val),
        Err(e) => format!("ERROR: {}", e),
    }
}

fn normalize(s: &str) -> String {
    let s = s.replace("\\n", " ");
    let s = s.split_whitespace().collect::<Vec<_>>().join(" ");
    s.replace("[ ", "[").replace(" ]", "]")
}

fn compare(actual: &str, expected: &str, name: &str) -> bool {
    if actual == expected {
        return true;
    }
    if normalize(actual) == normalize(expected) {
        return true;
    }
    if name.contains("gensym") {
        let prefix = expected.split(|c: char| c.is_ascii_hexdigit()).next().unwrap_or(expected);
        return actual.starts_with(prefix);
    }
    if let (Ok(a), Ok(e)) = (actual.parse::<f64>(), expected.parse::<f64>()) {
        return (a - e).abs() < 1e-4;
    }
    false
}

#[test]
fn test_all_yaml() {
    let base = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests");

    let mut cases = parse_test_yaml(&base.join("interpreter_tests.yaml"));
    let vag_cases = parse_test_yaml(&base.join("vag_tests.yaml"));
    let macro_cases = parse_test_yaml(&base.join("macro_tests.yaml"));
    let vag_count = vag_cases.len();
    let macro_count = macro_cases.len();
    cases.extend(vag_cases);
    cases.extend(macro_cases);

    assert!(!cases.is_empty(), "No test cases loaded");

    let mut failures = Vec::new();

    for case in &cases {
        let actual = eval(&case.expr);
        if !compare(&actual, &case.expected, &case.name) {
            failures.push(format!(
                "  [line {}] {}: expr: {}\n    expected: {}\n    got:      {}",
                case.line, case.name, case.expr, case.expected, actual
            ));
        }
    }

    if !failures.is_empty() {
        panic!(
            "\n{} test(s) failed out of {}:\n\n{}\n",
            failures.len(),
            cases.len(),
            failures.join("\n\n")
        );
    }

    eprintln!("{} tests passed ({} interpreter + {} VAG + {} macro)", cases.len(), cases.len() - vag_count - macro_count, vag_count, macro_count);
}

#[test]
fn test_guard_no_nan() {
    use sheaf_compiler::interpreter::apply_guard_check;
    use sheaf_compiler::interpreter::value::Value;
    use sheaf_compiler::core::expr::GuardCheck;
    use ndarray::{ArrayD, IxDyn};
    use std::sync::Arc;
    use std::collections::BTreeMap;

    let check = GuardCheck::NoNan;

    assert!(apply_guard_check(&check, &Value::Float(1.0)).is_ok());

    assert!(apply_guard_check(&check, &Value::Float(f32::NAN)).is_err());

    assert!(apply_guard_check(&check, &Value::Float(f32::INFINITY)).is_err());

    let clean = Value::Tensor {
        data: Arc::new(ArrayD::from_shape_vec(IxDyn(&[3]), vec![1.0, 2.0, 3.0]).unwrap()),
        dtype: sheaf_compiler::interpreter::value::Dtype::F32,
    };
    assert!(apply_guard_check(&check, &clean).is_ok());

    let nan_tensor = Value::Tensor {
        data: Arc::new(ArrayD::from_shape_vec(IxDyn(&[3]), vec![1.0, f32::NAN, 3.0]).unwrap()),
        dtype: sheaf_compiler::interpreter::value::Dtype::F32,
    };
    assert!(apply_guard_check(&check, &nan_tensor).is_err());

    let mut map = BTreeMap::new();
    map.insert("clean".to_string(), Value::Float(1.0));
    map.insert("bad".to_string(), Value::Float(f32::NAN));
    assert!(apply_guard_check(&check, &Value::Dict(map)).is_err());

    let list = Value::List(vec![Value::Float(1.0), Value::Float(f32::NAN)]);
    assert!(apply_guard_check(&check, &list).is_err());
}

#[test]
fn test_io_safetensors_roundtrip() {
    use sheaf_compiler::interpreter::value::Value;

    let path = std::env::temp_dir().join(format!(
        "sheaf_io_roundtrip_{}.safetensors",
        std::process::id()
    ));

    struct Guard<'a>(&'a std::path::Path);
    impl Drop for Guard<'_> {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(self.0);
        }
    }
    let _cleanup = Guard(&path);

    let src = format!(
        r#"
        (let [w {{:blocks [{{:w (reshape (arange 6) '[2 3])}}
                         {{:w (reshape (arange 4) '[2 2])}}]
                   :head   {{:b (reshape (arange 8) '[2 4])}}}}
              _   (io "save" "{path}" w)
              r   (io "load" "{path}")]
          (float
            (+ (sum (abs (- (get-in r [:blocks 0 :w]) (get-in w [:blocks 0 :w]))))
               (sum (abs (- (get-in r [:blocks 1 :w]) (get-in w [:blocks 1 :w]))))
               (sum (abs (- (get-in r [:head :b])   (get-in w [:head :b])))))))
        "#,
        path = path.display()
    );

    let val = eval_exprs(&src).expect("io roundtrip eval failed");
    let diff = match val {
        Value::Float(f) => f,
        Value::Tensor { data, .. } => *data.first().expect("expected scalar"),
        other => panic!("expected scalar, got {}", other.type_name()),
    };
    assert!(
        diff.abs() < 1e-6,
        "safetensors roundtrip mismatch: total abs diff = {}",
        diff
    );
}

#[test]
fn arithmetic_uses_weak_scalar_dtypes() {
    use sheaf_compiler::core::dtype::ElementType;
    use sheaf_compiler::interpreter::value::Value;

    for (dtype, literal, expected) in [
        (ElementType::F16, "1.0006", 1.000_976_6),
        (ElementType::BF16, "1.005", 1.0078125),
    ] {
        for scalar_first in [false, true] {
            let expression = if scalar_first {
                format!("(+ {literal} (cast (tensor [0.0]) :{}))", dtype.name())
            } else {
                format!("(+ (cast (tensor [0.0]) :{}) {literal})", dtype.name())
            };
            let Value::Tensor { data, dtype: result_dtype } = eval_exprs(&expression).unwrap()
            else {
                panic!("expected a tensor");
            };
            assert_eq!(result_dtype, dtype);
            assert_eq!(data.as_slice().unwrap(), &[expected]);
        }
    }

    for name in ["-", "*", "/"] {
        for dtype in [ElementType::F16, ElementType::BF16] {
            for scalar_first in [false, true] {
                let expression = if scalar_first {
                    format!("({name} 1.5 (cast (tensor [2.0]) :{}))", dtype.name())
                } else {
                    format!("({name} (cast (tensor [2.0]) :{}) 1.5)", dtype.name())
                };
                let Value::Tensor { dtype: result_dtype, .. } =
                    eval_exprs(&expression).unwrap()
                else {
                    panic!("expected a tensor");
                };
                assert_eq!(result_dtype, dtype);
            }
        }
    }

    for expression in [
        "(+ (reshape (cast (tensor [1.0]) :f16) (quote [])) 1.0)",
        "(- (reshape (cast (tensor [1.0]) :f16) (quote [])))",
    ] {
        let Value::Tensor { data, dtype } = eval_exprs(expression).unwrap() else {
            panic!("expected a tensor");
        };
        assert!(data.shape().is_empty());
        assert_eq!(dtype, ElementType::F16);
    }

    for name in ["+", "-", "*", "/"] {
        let expression = format!(
            "({name} (cast (tensor [1.0]) :f16) (cast (tensor [1.0]) :bf16))",
        );
        let error = eval_exprs(&expression).unwrap_err();
        assert!(error.to_string().contains("dtype mismatch: f16 and bf16"));
    }
}
