use super::*;

#[test]
fn test_generate_constant() {
    let mut codegen = CodeGenerator::new();
    let expr = CompiledExpr::Integer(42);
    let result = codegen.generate(&expr);
    assert!(result.is_ok());
}

#[test]
fn test_generate_binop() {
    let mut codegen = CodeGenerator::new();
    let expr = CompiledExpr::FunctionCall {
        name: "+".to_string(),
        args: vec![CompiledExpr::Integer(1), CompiledExpr::Integer(2)],
        loc: None,
    };
    let result = codegen.generate(&expr);
    assert!(result.is_ok());
}

#[test]
fn test_emit_function() {
    let codegen = CodeGenerator::new();
    let expr = CompiledExpr::FunctionCall {
        name: "+".to_string(),
        args: vec![CompiledExpr::Integer(1), CompiledExpr::Integer(2)],
        loc: None,
    };
    let mlir = codegen.emit_function("test", &expr);
    assert!(mlir.is_ok());
    let mlir_str = mlir.unwrap();
    assert!(mlir_str.contains("stablehlo.add"));
    assert!(mlir_str.contains("@test"));
}

#[test]
fn test_generate_compare() {
    let mut codegen = CodeGenerator::new();
    let expr = CompiledExpr::FunctionCall {
        name: ">".to_string(),
        args: vec![CompiledExpr::Float(5.0), CompiledExpr::Float(2.0)],
        loc: None,
    };
    let result = codegen.generate(&expr);
    assert!(result.is_ok());
    let (_, ty) = result.unwrap();
    assert!(matches!(ty, StableHLOType::ScalarF32));
}

#[test]
fn test_emit_compare() {
    let codegen = CodeGenerator::new();
    let expr = CompiledExpr::FunctionCall {
        name: "=".to_string(),
        args: vec![CompiledExpr::Integer(1), CompiledExpr::Integer(1)],
        loc: None,
    };
    let mlir = codegen.emit_function("test_eq", &expr);
    assert!(mlir.is_ok());
    let mlir_str = mlir.unwrap();
    assert!(mlir_str.contains("stablehlo.abs"));
    assert!(mlir_str.contains("stablehlo.minimum"));
}

#[test]
fn test_emit_boolean_and() {
    let codegen = CodeGenerator::new();
    let expr = CompiledExpr::FunctionCall {
        name: "and".to_string(),
        args: vec![
            CompiledExpr::FunctionCall {
                name: ">".to_string(),
                args: vec![CompiledExpr::Float(5.0), CompiledExpr::Float(2.0)],
                loc: None,
            },
            CompiledExpr::FunctionCall {
                name: "<".to_string(),
                args: vec![CompiledExpr::Float(1.0), CompiledExpr::Float(3.0)],
                loc: None,
            },
        ],
        loc: None,
    };
    let mlir = codegen.emit_function("test_and", &expr);
    assert!(mlir.is_ok());
    let mlir_str = mlir.unwrap();
    assert!(mlir_str.contains("stablehlo.sign"));
    assert!(mlir_str.contains("stablehlo.multiply"));
}

#[test]
fn test_emit_boolean_not() {
    let codegen = CodeGenerator::new();
    let expr = CompiledExpr::FunctionCall {
        name: "not".to_string(),
        args: vec![CompiledExpr::FunctionCall {
            name: ">".to_string(),
            args: vec![CompiledExpr::Float(5.0), CompiledExpr::Float(10.0)],
            loc: None,
        }],
        loc: None,
    };
    let mlir = codegen.emit_function("test_not", &expr);
    assert!(mlir.is_ok());
    let mlir_str = mlir.unwrap();
    assert!(mlir_str.contains("stablehlo.sign"));
    assert!(mlir_str.contains("stablehlo.subtract"));
}

#[test]
fn test_reduction_preserves_dtype() {
    use crate::core::dtype::ElementType;

    let registry = HashMap::new();
    let param_type = StableHLOType::tensor(vec![2, 2], ElementType::F16);
    let result_type = StableHLOType::tensor(vec![2, 1], ElementType::F16);
    let codegen = CodeGenerator::with_function_params(
        &registry,
        &["x".to_string()],
        std::slice::from_ref(&param_type),
    );
    let body = CompiledExpr::FunctionCall {
        name: "mean".to_string(),
        args: vec![
            CompiledExpr::Symbol("x".to_string()),
            CompiledExpr::Keyword("axis".to_string()),
            CompiledExpr::Integer(-1),
            CompiledExpr::Keyword("keepdims".to_string()),
        ],
        loc: None,
    };
    let (mlir, actual_type) = codegen
        .emit_func_declaration(
            "mean",
            &body,
            std::slice::from_ref(&param_type),
            &result_type,
        )
        .unwrap();
    assert!(mlir.contains("stablehlo.reduce"));
    assert!(mlir.contains("stablehlo.constant dense<0.0> : tensor<f16>"));
    assert_eq!(actual_type, result_type);
}

#[test]
fn test_arithmetic_converts_weak_scalars() {
    use crate::core::dtype::ElementType;

    for (name, stablehlo_name) in [
        ("+", "add"),
        ("-", "subtract"),
        ("*", "multiply"),
        ("/", "divide"),
    ] {
        for dtype in [ElementType::F16, ElementType::BF16] {
            for shape in [vec![2], Vec::new()] {
                for scalar_first in [false, true] {
                    let registry = HashMap::new();
                    let param_type = StableHLOType::tensor(shape.clone(), dtype);
                    let codegen = CodeGenerator::with_function_params(
                        &registry,
                        &["x".to_string()],
                        std::slice::from_ref(&param_type),
                    );
                    let tensor = CompiledExpr::Symbol("x".to_string());
                    let scalar = CompiledExpr::Symbol("one".to_string());
                    let args = if scalar_first {
                        vec![scalar, tensor]
                    } else {
                        vec![tensor, scalar]
                    };
                    let body = CompiledExpr::Let {
                        bindings: vec![(
                            BindingPattern::Simple("one".to_string()),
                            CompiledExpr::Float(1.0),
                        )],
                        body: Box::new(CompiledExpr::FunctionCall {
                            name: name.to_string(),
                            args,
                            loc: None,
                        }),
                    };
                    let (mlir, result_type) = codegen
                        .emit_func_declaration(
                            "arithmetic",
                            &body,
                            std::slice::from_ref(&param_type),
                            &param_type,
                        )
                        .unwrap();
                    assert!(mlir.contains("stablehlo.convert"));
                    assert!(mlir.contains(&format!("-> tensor<{}>", dtype.to_mlir_str())));
                    assert!(mlir.contains(&format!("stablehlo.{}", stablehlo_name)));
                    assert_eq!(result_type, param_type);
                }
            }
        }
    }
}

#[test]
fn test_dot_operations_preserve_dtypes() {
    use crate::core::dtype::ElementType;

    for name in ["@", "einsum"] {
        for dtype in [ElementType::F16, ElementType::BF16] {
            let registry = HashMap::new();
            let param_types = vec![
                StableHLOType::tensor(vec![1, 2], dtype),
                StableHLOType::tensor(vec![2, 1], dtype),
            ];
            let codegen = CodeGenerator::with_function_params(
                &registry,
                &["x".to_string(), "y".to_string()],
                &param_types,
            );
            let mut args = vec![
                CompiledExpr::Symbol("x".to_string()),
                CompiledExpr::Symbol("y".to_string()),
            ];
            if name == "einsum" {
                args.insert(0, CompiledExpr::String("ij,jk->ik".to_string()));
            }
            let body = CompiledExpr::FunctionCall {
                name: name.to_string(),
                args,
                loc: None,
            };
            let result_type = StableHLOType::tensor(vec![1, 1], dtype);
            let (mlir, actual_type) = codegen
                .emit_func_declaration("dot", &body, &param_types, &result_type)
                .unwrap();
            assert!(mlir.contains("stablehlo.dot_general"));
            assert!(mlir.contains(&format!("tensor<1x1x{}>", dtype.to_mlir_str())));
            assert_eq!(actual_type, result_type);
        }
    }
}

#[test]
fn test_arithmetic_rejects_strong_dtype_mismatches() {
    use crate::core::dtype::ElementType;

    let registry = HashMap::new();
    let param_types = vec![
        StableHLOType::tensor(vec![2], ElementType::F16),
        StableHLOType::tensor(vec![2], ElementType::BF16),
    ];
    for name in ["+", "-", "*", "/"] {
        let codegen = CodeGenerator::with_function_params(
            &registry,
            &["x".to_string(), "y".to_string()],
            &param_types,
        );
        let body = CompiledExpr::FunctionCall {
            name: name.to_string(),
            args: vec![
                CompiledExpr::Symbol("x".to_string()),
                CompiledExpr::Symbol("y".to_string()),
            ],
            loc: None,
        };
        let error = codegen
            .emit_func_declaration("arithmetic", &body, &param_types, &param_types[0])
            .unwrap_err();
        assert!(error.to_string().contains("dtype mismatch: f16 and bf16"));
    }
}
