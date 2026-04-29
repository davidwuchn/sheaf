// Copyright (c) 2026 Damien Boureille
// Licensed under the MIT License.

//! StableHLO emitter - generates MLIR StableHLO from Sheaf AST

mod broadcast;
mod compare_ops;
mod constants;
mod dot_ops;
mod emit;
mod indexing_ops;
mod operations;
mod random_ops;
mod reduce_ops;
mod shape_ops;
mod tensor_ops;

use crate::core::ast::SheafValue;
use std::collections::HashMap;

/// StableHLO type representation
#[derive(Debug, Clone, PartialEq)]
pub enum StableHLOType {
    /// Scalar tensor: tensor<f32>
    ScalarF32,
    /// Scalar tensor: tensor<bf16>
    ScalarBF16,
    /// Scalar tensor: tensor<f64>
    ScalarF64,
    /// Scalar tensor: tensor<i64>
    ScalarI64,
    /// Scalar tensor: tensor<i1> (boolean)
    ScalarI1,
    /// Tensor with shape: tensor<2x3xf32>
    Tensor { shape: Vec<i64>, dtype: String },
    /// Tuple of types: tuple<tensor<2x3xf32>, tensor<8xf32>>
    /// When keys is Some, this represents a dict (reconstructed as Value::Dict on output).
    Tuple(Vec<StableHLOType>, Option<Vec<String>>),
}

impl StableHLOType {
    pub fn scalar_f32() -> Self {
        Self::ScalarF32
    }

    pub fn scalar_bf16() -> Self {
        Self::ScalarBF16
    }

    pub fn scalar_i64() -> Self {
        Self::ScalarI64
    }

    pub fn f32_tensor(shape: impl Into<Vec<i64>>) -> Self {
        Self::Tensor {
            shape: shape.into(),
            dtype: "f32".to_string(),
        }
    }

    pub fn bf16_tensor(shape: impl Into<Vec<i64>>) -> Self {
        Self::Tensor {
            shape: shape.into(),
            dtype: "bf16".to_string(),
        }
    }

    /// Create a tensor with the given dtype string ("f32", "bf16", "i32", etc.)
    pub fn typed_tensor(shape: impl Into<Vec<i64>>, dtype: &str) -> Self {
        Self::Tensor {
            shape: shape.into(),
            dtype: dtype.to_string(),
        }
    }

    pub fn i32_tensor(shape: impl Into<Vec<i64>>) -> Self {
        Self::Tensor {
            shape: shape.into(),
            dtype: "i32".to_string(),
        }
    }

    pub fn i64_tensor(shape: impl Into<Vec<i64>>) -> Self {
        Self::Tensor {
            shape: shape.into(),
            dtype: "i64".to_string(),
        }
    }

    pub fn i1_tensor(shape: impl Into<Vec<i64>>) -> Self {
        Self::Tensor {
            shape: shape.into(),
            dtype: "i1".to_string(),
        }
    }

    /// Get the shape of this type, or empty slice for scalars/tuples
    pub fn shape(&self) -> &[i64] {
        match self {
            Self::ScalarF32
            | Self::ScalarBF16
            | Self::ScalarF64
            | Self::ScalarI64
            | Self::ScalarI1
            | Self::Tuple(..) => &[],
            Self::Tensor { shape, .. } => shape,
        }
    }

    /// Get the dtype string
    pub fn dtype(&self) -> &str {
        match self {
            Self::ScalarF32 => "f32",
            Self::ScalarBF16 => "bf16",
            Self::ScalarF64 => "f64",
            Self::ScalarI64 => "i64",
            Self::ScalarI1 => "i1",
            Self::Tensor { dtype, .. } => dtype,
            Self::Tuple(..) => "tuple",
        }
    }

    /// Is this a float type (f32 or bf16)?
    pub fn is_float(&self) -> bool {
        matches!(self.dtype(), "f32" | "bf16")
    }

    /// Check if two types have the same tuple nesting structure.
    /// Leaf types (tensors, scalars) are considered structurally equivalent.
    pub fn tuple_structure_matches(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Tuple(a, _), Self::Tuple(b, _)) => {
                a.len() == b.len()
                    && a.iter()
                        .zip(b.iter())
                        .all(|(x, y)| x.tuple_structure_matches(y))
            }
            (Self::Tuple(..), _) | (_, Self::Tuple(..)) => false,
            _ => true, // both are leaf types
        }
    }

    pub fn to_mlir(&self) -> String {
        match self {
            Self::ScalarF32 => "tensor<f32>".to_string(),
            Self::ScalarBF16 => "tensor<bf16>".to_string(),
            Self::ScalarF64 => "tensor<f64>".to_string(),
            Self::ScalarI64 => "tensor<i64>".to_string(),
            Self::ScalarI1 => "tensor<i1>".to_string(),
            Self::Tensor { shape, dtype } => {
                if shape.is_empty() {
                    format!("tensor<{}>", dtype)
                } else {
                    let shape_str = shape
                        .iter()
                        .map(|d| d.to_string())
                        .collect::<Vec<_>>()
                        .join("x");
                    format!("tensor<{}x{}>", shape_str, dtype)
                }
            }
            Self::Tuple(elems, _) => {
                let elems_str = elems
                    .iter()
                    .map(|t| t.to_mlir())
                    .collect::<Vec<_>>()
                    .join(", ");
                format!("tuple<{}>", elems_str)
            }
        }
    }

    /// Parse an MLIR type string back into a StableHLOType.
    /// Accepts: "tensor<f32>", "tensor<2x3xf32>", "tuple<tensor<2xf32>, tensor<f32>>".
    pub fn parse(s: &str) -> Option<Self> {
        let s = s.trim();
        if s.starts_with("tuple<") && s.ends_with('>') {
            let inner = &s[6..s.len() - 1];
            let elems = split_tuple_args(inner);
            let parsed: Option<Vec<Self>> = elems.iter().map(|e| Self::parse(e)).collect();
            return parsed.map(|elems| Self::Tuple(elems, None));
        }
        if s.starts_with("tensor<") && s.ends_with('>') {
            let inner = &s[7..s.len() - 1]; // e.g. "2x3xf32" or "f32"
            let parts: Vec<&str> = inner.split('x').collect();
            if parts.len() == 1 {
                return match parts[0] {
                    "f32" => Some(Self::ScalarF32),
                    "bf16" => Some(Self::ScalarBF16),
                    "f64" => Some(Self::ScalarF64),
                    "i64" => Some(Self::ScalarI64),
                    "i1" => Some(Self::ScalarI1),
                    _ => None,
                };
            }
            let dtype = parts.last()?.to_string();
            let shape: Option<Vec<i64>> = parts[..parts.len() - 1]
                .iter()
                .map(|d| d.parse::<i64>().ok())
                .collect();
            return shape.map(|s| Self::Tensor { shape: s, dtype });
        }
        None
    }
}

/// Split top-level comma-separated args in a tuple, respecting nesting.
fn split_tuple_args(s: &str) -> Vec<&str> {
    let mut result = Vec::new();
    let mut depth = 0;
    let mut start = 0;
    for (i, c) in s.char_indices() {
        match c {
            '<' => depth += 1,
            '>' => depth -= 1,
            ',' if depth == 0 => {
                result.push(s[start..i].trim());
                start = i + 1;
            }
            _ => {}
        }
    }
    let last = s[start..].trim();
    if !last.is_empty() {
        result.push(last);
    }
    result
}

/// Register name in SSA form: %0, %1, etc. or %arg0, %arg1, etc.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Register {
    /// Regular SSA register: %0, %1, etc.
    Reg(usize),
    /// Function argument: %arg0, %arg1, etc.
    Arg(usize),
}

impl Register {
    pub fn new(id: usize) -> Self {
        Self::Reg(id)
    }

    pub fn arg(id: usize) -> Self {
        Self::Arg(id)
    }

    pub fn to_mlir(&self) -> String {
        match self {
            Self::Reg(id) => format!("%{}", id),
            Self::Arg(id) => format!("%arg{}", id),
        }
    }
}

/// MLIR StableHLO emitter
pub struct StableHLOEmitter {
    counter: usize,
    pub(crate) body: Vec<String>,
    /// Cache for reduce_mean results: (input_register, normalized_axis, keepdims) -> (result_reg, result_type).
    /// Avoids recomputing mean(x, axis) when var(x, axis) also needs it internally.
    reduce_mean_cache: HashMap<(Register, usize, bool), (Register, StableHLOType)>,
    /// Virtual tuple registers: maps a register to its constituent (register, type) pairs.
    /// Tuples are never materialized as MLIR ops: they exist only as register groupings.
    virtual_tuples: HashMap<Register, Vec<(Register, StableHLOType)>>,
    /// Compile-time known scalar values for constant propagation.
    /// Populated from constants and scalar function params; used to resolve
    /// shape-critical values (e.g. K in top_k) at codegen time.
    known_scalars: HashMap<Register, f64>,
    known_tensors: HashMap<Register, Vec<f64>>,
}

impl StableHLOEmitter {
    pub fn new() -> Self {
        Self {
            counter: 0,
            body: Vec::new(),
            reduce_mean_cache: HashMap::new(),
            virtual_tuples: HashMap::new(),
            known_scalars: HashMap::new(),
            known_tensors: HashMap::new(),
        }
    }

    /// Record a known scalar value for a register (constant propagation).
    pub fn set_known_scalar(&mut self, reg: Register, value: f64) {
        self.known_scalars.insert(reg, value);
    }

    /// Look up a register's known compile-time scalar value.
    pub fn known_scalar_value(&self, reg: &Register) -> Option<f64> {
        self.known_scalars.get(reg).copied()
    }

    pub fn set_known_tensor(&mut self, reg: Register, values: Vec<f64>) {
        self.known_tensors.insert(reg, values);
    }

    pub fn known_tensor_values(&self, reg: &Register) -> Option<&Vec<f64>> {
        self.known_tensors.get(reg)
    }

    /// Generate a fresh register name
    pub fn fresh_register(&mut self) -> Register {
        let reg = Register::new(self.counter);
        self.counter += 1;
        reg
    }

    /// Add an instruction to the body
    pub fn emit_instruction(&mut self, instruction: String) {
        self.body.push(instruction);
    }

    /// Register a tuple-typed function parameter as a virtual tuple.
    /// Leaf types map to Register::arg(flat_idx); sub-tuples map to fresh virtual registers.
    pub fn register_virtual_param(&mut self, ty: &StableHLOType, flat_idx: &mut usize) -> Register {
        match ty {
            StableHLOType::Tuple(elems, _) => {
                let vreg = self.fresh_register();
                let constituents: Vec<_> = elems.iter()
                    .map(|elem_ty| {
                        let sub_reg = self.register_virtual_param(elem_ty, flat_idx);
                        (sub_reg, elem_ty.clone())
                    })
                    .collect();
                self.virtual_tuples.insert(vreg, constituents);
                vreg
            }
            _ => {
                let reg = Register::arg(*flat_idx);
                *flat_idx += 1;
                reg
            }
        }
    }

    /// Recursively collect leaf registers from a (possibly virtual) tuple.
    pub fn collect_virtual_leaves(&self, reg: Register, ty: &StableHLOType) -> Vec<(Register, StableHLOType)> {
        if let Some(constituents) = self.virtual_tuples.get(&reg) {
            constituents.iter()
                .flat_map(|(r, t)| self.collect_virtual_leaves(*r, t))
                .collect()
        } else {
            vec![(reg, ty.clone())]
        }
    }

    /// Compile an expression to a register
    pub fn compile_expr(&mut self, expr: &SheafValue) -> (Register, StableHLOType) {
        match expr {
            SheafValue::Float(x, _) => {
                let reg = self.emit_constant_f32(*x);
                (reg, StableHLOType::scalar_f32())
            }
            SheafValue::Integer(n, _) => {
                let reg = self.emit_constant_f32(*n as f64);
                (reg, StableHLOType::scalar_f32())
            }

            SheafValue::Vector(elems, _) => {
                if !elems.is_empty() && matches!(elems[0], SheafValue::Vector(_, _)) {
                    let rows: Vec<Vec<f64>> = elems
                        .iter()
                        .map(|row| {
                            if let SheafValue::Vector(row_elems, _) = row {
                                row_elems
                                    .iter()
                                    .map(|e| match e {
                                        SheafValue::Float(x, _) => *x,
                                        SheafValue::Integer(n, _) => *n as f64,
                                        _ => panic!("Matrix element must be number"),
                                    })
                                    .collect()
                            } else {
                                panic!("Matrix rows must be vectors")
                            }
                        })
                        .collect();
                    self.emit_tensor_constant(&rows)
                } else {
                    let values: Vec<f64> = elems
                        .iter()
                        .map(|e| match e {
                            SheafValue::Float(x, _) => *x,
                            SheafValue::Integer(n, _) => *n as f64,
                            _ => panic!("Vector element must be number"),
                        })
                        .collect();
                    self.emit_tensor_constant(&vec![values])
                }
            }

            SheafValue::List(elems, _) if !elems.is_empty() => {
                if let Some(op) = elems[0].as_symbol() {
                    match op {
                        "+" | "-" | "*" | "/" if elems.len() == 3 => {
                            let (lhs_reg, lhs_ty) = self.compile_expr(&elems[1]);
                            let (rhs_reg, rhs_ty) = self.compile_expr(&elems[2]);
                            self.emit_binop(op, &lhs_reg, &rhs_reg, &lhs_ty, &rhs_ty)
                        }

                        "@" if elems.len() == 3 => {
                            let (lhs_reg, lhs_ty) = self.compile_expr(&elems[1]);
                            let (rhs_reg, rhs_ty) = self.compile_expr(&elems[2]);
                            self.emit_matmul(&lhs_reg, &rhs_reg, &lhs_ty, &rhs_ty)
                        }

                        "relu" | "sigmoid" | "tanh" | "sqrt" | "exp" | "log"
                            if elems.len() == 2 =>
                        {
                            let (operand_reg, operand_ty) = self.compile_expr(&elems[1]);
                            let result_reg = self.emit_unary(op, &operand_reg, &operand_ty);
                            (result_reg, operand_ty)
                        }

                        "zeros" if elems.len() == 2 => {
                            let shape = self.parse_shape_vector(&elems[1]);
                            self.emit_zeros(&shape)
                        }

                        "random-normal" if elems.len() == 3 => {
                            let (key_reg, key_ty) = self.compile_expr(&elems[1]);
                            let shape = self.parse_shape_vector(&elems[2]);
                            self.emit_random_normal(&key_reg, &key_ty, &shape)
                        }

                        _ => panic!("Unsupported operation: {}", op),
                    }
                } else {
                    panic!("First element of list must be a symbol: {}", expr)
                }
            }

            _ => panic!("Unsupported expression: {}", expr),
        }
    }

    /// Parse a shape vector like [2 8] into vec![2, 8]
    fn parse_shape_vector(&self, expr: &SheafValue) -> Vec<i64> {
        if let SheafValue::Vector(elems, _) = expr {
            elems
                .iter()
                .map(|e| match e {
                    SheafValue::Integer(n, _) => *n,
                    _ => panic!("Shape element must be integer"),
                })
                .collect()
        } else {
            panic!("Shape must be a vector")
        }
    }
}

impl Default for StableHLOEmitter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::error::SourceLocation;

    fn make_int(n: i64) -> SheafValue {
        SheafValue::Integer(n, SourceLocation::unknown())
    }

    fn make_float(x: f64) -> SheafValue {
        SheafValue::Float(x, SourceLocation::unknown())
    }

    fn make_symbol(s: &str) -> SheafValue {
        SheafValue::Symbol(s.to_string(), SourceLocation::unknown())
    }

    fn make_list(elems: Vec<SheafValue>) -> SheafValue {
        SheafValue::List(elems, SourceLocation::unknown())
    }

    #[test]
    fn test_emit_constant() {
        let mut emitter = StableHLOEmitter::new();
        let reg = emitter.emit_constant_f32(42.0);
        assert_eq!(reg.to_mlir(), "%0");
        assert_eq!(emitter.body.len(), 1);
        assert!(emitter.body[0].contains("dense<42.0>"));
    }

    #[test]
    fn test_emit_add() {
        let mut emitter = StableHLOEmitter::new();
        let expr = make_list(vec![make_symbol("+"), make_int(1), make_int(2)]);
        let mlir = emitter.emit_function("add", &expr);

        assert!(mlir.contains("stablehlo.constant"));
        assert!(mlir.contains("stablehlo.add"));
        assert!(mlir.contains("@add"));
    }

    #[test]
    fn test_emit_nested() {
        let mut emitter = StableHLOEmitter::new();
        let expr = make_list(vec![
            make_symbol("*"),
            make_list(vec![make_symbol("+"), make_int(1), make_int(2)]),
            make_int(4),
        ]);
        let mlir = emitter.emit_function("nested", &expr);

        assert!(mlir.contains("stablehlo.add"));
        assert!(mlir.contains("stablehlo.multiply"));
        assert!(mlir.contains("@nested"));
    }

    #[test]
    fn test_emit_float() {
        let mut emitter = StableHLOEmitter::new();
        let expr = make_float(3.14);
        let mlir = emitter.emit_function("pi", &expr);

        assert!(mlir.contains("dense<3.14>"));
    }
}
