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
    };
    let result = codegen.generate(&expr);
    assert!(result.is_ok());
    let (_, ty) = result.unwrap();
    // Result should be i1 type (boolean results)
    assert!(matches!(ty, StableHLOType::ScalarI1));
}

#[test]
fn test_emit_compare() {
    let codegen = CodeGenerator::new();
    let expr = CompiledExpr::FunctionCall {
        name: "=".to_string(),
        args: vec![CompiledExpr::Integer(1), CompiledExpr::Integer(1)],
    };
    let mlir = codegen.emit_function("test_eq", &expr);
    assert!(mlir.is_ok());
    let mlir_str = mlir.unwrap();
    assert!(mlir_str.contains("stablehlo.compare"));
    assert!(mlir_str.contains("comparison_direction = #stablehlo<comparison_direction EQ>"));
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
            },
            CompiledExpr::FunctionCall {
                name: "<".to_string(),
                args: vec![CompiledExpr::Float(1.0), CompiledExpr::Float(3.0)],
            },
        ],
    };
    let mlir = codegen.emit_function("test_and", &expr);
    assert!(mlir.is_ok());
    let mlir_str = mlir.unwrap();
    assert!(mlir_str.contains("stablehlo.compare"));
    assert!(mlir_str.contains("stablehlo.and"));
}

#[test]
fn test_emit_boolean_not() {
    let codegen = CodeGenerator::new();
    let expr = CompiledExpr::FunctionCall {
        name: "not".to_string(),
        args: vec![CompiledExpr::FunctionCall {
            name: ">".to_string(),
            args: vec![CompiledExpr::Float(5.0), CompiledExpr::Float(10.0)],
        }],
    };
    let mlir = codegen.emit_function("test_not", &expr);
    assert!(mlir.is_ok());
    let mlir_str = mlir.unwrap();
    assert!(mlir_str.contains("stablehlo.compare"));
    assert!(mlir_str.contains("stablehlo.not"));
}
