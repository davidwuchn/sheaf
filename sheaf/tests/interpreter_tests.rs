// Copyright (c) 2025-2026 Damien Boureille
// Licensed under the MIT License.

//! Data-driven regression tests for Sheaf.
//! Test cases are defined in tests.yaml and executed against the interpreter.

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

        // Skip comments and blank lines
        if l.is_empty() || l.starts_with('#') {
            continue;
        }

        if l.starts_with("- name:") {
            // Save previous entry
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
            if val == ">" || val == "|" {
                // Multi-line YAML scalar follows
                expr.clear();
            } else {
                expr = val.to_string();
            }
        } else if l.starts_with("expected:") {
            let val = l.trim_start_matches("expected:").trim().trim_matches('\'').trim_matches('"').to_string();
            expected = val;
        } else if expr.is_empty() || !expected.is_empty() {
            // Continuation line for expected (rare)
        } else {
            // Continuation line for multi-line test expr
            if !expr.is_empty() {
                expr.push(' ');
            }
            expr.push_str(l);
        }
    }

    // Last entry
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
    // Unescape \n from YAML, collapse whitespace, trim bracket padding
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
    // gensym: non-deterministic, check prefix only
    if name.contains("gensym") {
        let prefix = expected.split(|c: char| c.is_ascii_hexdigit()).next().unwrap_or(expected);
        return actual.starts_with(prefix);
    }
    // Scalar numeric tolerance (f32)
    if let (Ok(a), Ok(e)) = (actual.parse::<f64>(), expected.parse::<f64>()) {
        return (a - e).abs() < 1e-4;
    }
    false
}

#[test]
fn test_all_yaml() {
    let yaml_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/interpreter_tests.yaml");
    let cases = parse_test_yaml(&yaml_path);

    assert!(!cases.is_empty(), "No test cases loaded from tests.yaml");

    let mut failures = Vec::new();

    for case in &cases {
        let actual = eval(&case.expr);
        if !compare(&actual, &case.expected, &case.name) {
            failures.push(format!(
                "  [line {}] {} — expr: {}\n    expected: {}\n    got:      {}",
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

    eprintln!("{} tests passed", cases.len());
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

    // Clean scalar: should pass
    assert!(apply_guard_check(&check, &Value::Float(1.0)).is_ok());

    // NaN scalar: should fail
    assert!(apply_guard_check(&check, &Value::Float(f32::NAN)).is_err());

    // Inf scalar: should fail
    assert!(apply_guard_check(&check, &Value::Float(f32::INFINITY)).is_err());

    // Clean tensor: should pass
    let clean = Value::Tensor {
        data: Arc::new(ArrayD::from_shape_vec(IxDyn(&[3]), vec![1.0, 2.0, 3.0]).unwrap()),
        dtype: sheaf_compiler::interpreter::value::Dtype::F32,
    };
    assert!(apply_guard_check(&check, &clean).is_ok());

    // NaN tensor: should fail
    let nan_tensor = Value::Tensor {
        data: Arc::new(ArrayD::from_shape_vec(IxDyn(&[3]), vec![1.0, f32::NAN, 3.0]).unwrap()),
        dtype: sheaf_compiler::interpreter::value::Dtype::F32,
    };
    assert!(apply_guard_check(&check, &nan_tensor).is_err());

    // Dict with NaN nested inside: should fail
    let mut map = BTreeMap::new();
    map.insert("clean".to_string(), Value::Float(1.0));
    map.insert("bad".to_string(), Value::Float(f32::NAN));
    assert!(apply_guard_check(&check, &Value::Dict(map)).is_err());

    // List with NaN: should fail
    let list = Value::List(vec![Value::Float(1.0), Value::Float(f32::NAN)]);
    assert!(apply_guard_check(&check, &list).is_err());
}
