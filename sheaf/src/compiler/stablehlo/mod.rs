// Copyright (c) 2026 Damien Boureille
// Licensed under the MIT License.

//! StableHLO emitter - generates MLIR StableHLO from Sheaf AST

mod compare_ops;
mod dot_ops;
mod reduce_ops;
mod tensor_ops;

use crate::ast::SheafValue;
use std::collections::HashMap;
use std::fmt::Write;

/// StableHLO type representation
#[derive(Debug, Clone, PartialEq)]
pub enum StableHLOType {
    /// Scalar tensor: tensor<f32>
    ScalarF32,
    /// Scalar tensor: tensor<f64>
    ScalarF64,
    /// Scalar tensor: tensor<i64>
    ScalarI64,
    /// Scalar tensor: tensor<i1> (boolean)
    ScalarI1,
    /// Tensor with shape: tensor<2x3xf32>
    Tensor { shape: Vec<i64>, dtype: String },
    /// Tuple of types: tuple<tensor<2x3xf32>, tensor<8xf32>>
    Tuple(Vec<StableHLOType>),
}

impl StableHLOType {
    pub fn scalar_f32() -> Self {
        Self::ScalarF32
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
            | Self::ScalarF64
            | Self::ScalarI64
            | Self::ScalarI1
            | Self::Tuple(_) => &[],
            Self::Tensor { shape, .. } => shape,
        }
    }

    /// Get the dtype string
    pub fn dtype(&self) -> &str {
        match self {
            Self::ScalarF32 => "f32",
            Self::ScalarF64 => "f64",
            Self::ScalarI64 => "i64",
            Self::ScalarI1 => "i1",
            Self::Tensor { dtype, .. } => dtype,
            Self::Tuple(_) => "tuple",
        }
    }

    /// Check if two types have the same tuple nesting structure.
    /// Leaf types (tensors, scalars) are considered structurally equivalent.
    pub fn tuple_structure_matches(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Tuple(a), Self::Tuple(b)) => {
                a.len() == b.len()
                    && a.iter()
                        .zip(b.iter())
                        .all(|(x, y)| x.tuple_structure_matches(y))
            }
            (Self::Tuple(_), _) | (_, Self::Tuple(_)) => false,
            _ => true, // both are leaf types
        }
    }

    pub fn to_mlir(&self) -> String {
        match self {
            Self::ScalarF32 => "tensor<f32>".to_string(),
            Self::ScalarF64 => "tensor<f64>".to_string(),
            Self::ScalarI64 => "tensor<i64>".to_string(),
            Self::ScalarI1 => "tensor<i1>".to_string(),
            Self::Tensor { shape, dtype } => {
                let shape_str = shape
                    .iter()
                    .map(|d| d.to_string())
                    .collect::<Vec<_>>()
                    .join("x");
                format!("tensor<{}x{}>", shape_str, dtype)
            }
            Self::Tuple(elems) => {
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
            return parsed.map(Self::Tuple);
        }
        if s.starts_with("tensor<") && s.ends_with('>') {
            let inner = &s[7..s.len() - 1]; // e.g. "2x3xf32" or "f32"
            let parts: Vec<&str> = inner.split('x').collect();
            if parts.len() == 1 {
                return match parts[0] {
                    "f32" => Some(Self::ScalarF32),
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
    /// Cache for reduce_mean results: (input_register, normalized_axis, keepdims) → (result_reg, result_type).
    /// Avoids recomputing mean(x, axis) when var(x, axis) also needs it internally.
    reduce_mean_cache: HashMap<(Register, usize, bool), (Register, StableHLOType)>,
    /// Virtual tuple registers: maps a register to its constituent (register, type) pairs.
    /// Tuples are never materialized as MLIR ops — they exist only as register groupings.
    virtual_tuples: HashMap<Register, Vec<(Register, StableHLOType)>>,
}

impl StableHLOEmitter {
    pub fn new() -> Self {
        Self {
            counter: 0,
            body: Vec::new(),
            reduce_mean_cache: HashMap::new(),
            virtual_tuples: HashMap::new(),
        }
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
            StableHLOType::Tuple(elems) => {
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

    /// Emit a constant scalar
    pub fn emit_constant_f32(&mut self, value: f64) -> Register {
        let reg = self.fresh_register();
        let ty = StableHLOType::scalar_f32();
        // Format with .0 if integer value to satisfy IREE
        let value_str = if value.fract() == 0.0 && value.is_finite() {
            format!("{:.1}", value)
        } else {
            format!("{}", value)
        };
        self.body.push(format!(
            "    {} = stablehlo.constant dense<{}> : {}",
            reg.to_mlir(),
            value_str,
            ty.to_mlir()
        ));
        reg
    }

    /// Emit a constant i32 scalar
    pub fn emit_constant_i32(&mut self, value: i64) -> Register {
        let reg = self.fresh_register();
        let ty = StableHLOType::Tensor { shape: vec![], dtype: "i32".to_string() };
        self.body.push(format!(
            "    {} = stablehlo.constant dense<{}> : {}",
            reg.to_mlir(),
            value,
            ty.to_mlir()
        ));
        reg
    }

    /// Emit a constant integer (i64)
    pub fn emit_constant_i64(&mut self, value: i64) -> Register {
        let reg = self.fresh_register();
        let ty = StableHLOType::ScalarI64;
        self.body.push(format!(
            "    {} = stablehlo.constant dense<{}> : {}",
            reg.to_mlir(),
            value,
            ty.to_mlir()
        ));
        reg
    }

    /// Convert a tensor from one type to another (e.g. i1→f32, f32→i32)
    pub fn emit_convert(&mut self, reg: &Register, from_ty: &StableHLOType, to_ty: &StableHLOType) -> Register {
        let result = self.fresh_register();
        self.body.push(format!(
            "    {} = stablehlo.convert {} : ({}) -> {}",
            result.to_mlir(),
            reg.to_mlir(),
            from_ty.to_mlir(),
            to_ty.to_mlir(),
        ));
        result
    }

    /// Emit a tensor constant from a nested vector
    /// For example: [[1.0, 2.0], [3.0, 4.0]] -> tensor<2x2xf32>
    pub fn emit_tensor_constant(&mut self, values: &[Vec<f64>]) -> (Register, StableHLOType) {
        let reg = self.fresh_register();

        // Infer shape from nested structure
        let rows = values.len();
        let cols = if rows > 0 { values[0].len() } else { 0 };
        let shape = vec![rows as i64, cols as i64];
        let ty = StableHLOType::f32_tensor(shape);

        // Build nested structure for dense representation
        let rows_str: Vec<String> = values
            .iter()
            .map(|row| {
                let row_values: Vec<String> = row
                    .iter()
                    .map(|&v| {
                        if v.fract() == 0.0 && v.is_finite() {
                            format!("{:.1}", v)
                        } else {
                            format!("{}", v)
                        }
                    })
                    .collect();
                format!("[{}]", row_values.join(", "))
            })
            .collect();

        let values_str = rows_str.join(", ");

        self.body.push(format!(
            "    {} = stablehlo.constant dense<[{}]> : {}",
            reg.to_mlir(),
            values_str,
            ty.to_mlir()
        ));

        (reg, ty)
    }

    /// Emit an N-dimensional tensor constant from flat data and a shape.
    pub fn emit_nd_tensor_constant(
        &mut self,
        data: &[f64],
        shape: &[i64],
    ) -> (Register, StableHLOType) {
        let reg = self.fresh_register();
        let ty = StableHLOType::f32_tensor(shape.to_vec());

        let values_str = if shape.is_empty() {
            format_f64(data[0])
        } else {
            format_dense_attr(data, shape, 0)
        };

        self.body.push(format!(
            "    {} = stablehlo.constant dense<{}> : {}",
            reg.to_mlir(),
            values_str,
            ty.to_mlir()
        ));

        (reg, ty)
    }

    /// Emit a binary operation with broadcasting support
    pub fn emit_binop(
        &mut self,
        op: &str,
        lhs: &Register,
        rhs: &Register,
        lhs_ty: &StableHLOType,
        rhs_ty: &StableHLOType,
    ) -> (Register, StableHLOType) {
        let stablehlo_op = match op {
            "+" => "stablehlo.add",
            "-" => "stablehlo.subtract",
            "*" => "stablehlo.multiply",
            "/" => "stablehlo.divide",
            "**" => "stablehlo.power",
            "//" => "stablehlo.floor_divide",
            "%" | "mod" => "stablehlo.remainder",
            "min" => "stablehlo.minimum",
            "max" => "stablehlo.maximum",
            _ => panic!("Unsupported binop: {}", op),
        };

        // Determine result type (broadcast if needed)
        let result_ty = self.broadcast_types(lhs_ty, rhs_ty);

        // Check if we need to broadcast operands
        let (actual_lhs, actual_rhs) =
            self.maybe_broadcast_operands(lhs, rhs, lhs_ty, rhs_ty, &result_ty);

        let reg = self.fresh_register();
        self.body.push(format!(
            "    {} = {} {}, {} : {}",
            reg.to_mlir(),
            stablehlo_op,
            actual_lhs.to_mlir(),
            actual_rhs.to_mlir(),
            result_ty.to_mlir()
        ));
        (reg, result_ty)
    }

    /// Maybe broadcast operands to match result shape
    pub(crate) fn maybe_broadcast_operands(
        &mut self,
        lhs: &Register,
        rhs: &Register,
        lhs_ty: &StableHLOType,
        rhs_ty: &StableHLOType,
        result_ty: &StableHLOType,
    ) -> (Register, Register) {
        let lhs_shape = lhs_ty.shape();
        let rhs_shape = rhs_ty.shape();
        let result_shape = result_ty.shape();

        let actual_lhs = if lhs_shape != result_shape && !result_shape.is_empty() {
            self.emit_broadcast(lhs, lhs_ty, result_ty)
        } else {
            lhs.clone()
        };

        let actual_rhs = if rhs_shape != result_shape && !result_shape.is_empty() {
            self.emit_broadcast(rhs, rhs_ty, result_ty)
        } else {
            rhs.clone()
        };

        (actual_lhs, actual_rhs)
    }

    /// Emit broadcast_in_dim to convert from_ty to to_ty
    pub fn emit_broadcast(
        &mut self,
        operand: &Register,
        from_ty: &StableHLOType,
        to_ty: &StableHLOType,
    ) -> Register {
        let from_shape = from_ty.shape();
        let to_shape = to_ty.shape();

        let dims = if from_shape.is_empty() {
            vec![]
        } else {
            // Try numpy-style right-alignment first:
            // [4] → [4, 4] maps to dims=[1], [1, 256] → [4, 256, 384] maps to dims=[1, 2]
            let offset = to_shape.len() - from_shape.len();
            let right_aligned: Vec<usize> = (offset..to_shape.len()).collect();
            let valid = from_shape.iter().enumerate().all(|(i, &src)| {
                src == to_shape[right_aligned[i]] || src == 1
            });
            if valid {
                right_aligned
            } else {
                // Fallback: greedy left-to-right size matching
                // [1024] → [1024, 65] maps to dims=[0]
                let mut mapping = Vec::with_capacity(from_shape.len());
                let mut search_start = 0;
                for &src_dim in from_shape {
                    for j in search_start..to_shape.len() {
                        if to_shape[j] == src_dim || src_dim == 1 {
                            mapping.push(j);
                            search_start = j + 1;
                            break;
                        }
                    }
                }
                if mapping.len() != from_shape.len() {
                    // Last resort: right-align anyway (let StableHLO report the error)
                    right_aligned
                } else {
                    mapping
                }
            }
        };

        let reg = self.fresh_register();
        if dims.is_empty() {
            self.body.push(format!(
                "    {} = stablehlo.broadcast_in_dim {}, dims = [] : ({}) -> {}",
                reg.to_mlir(),
                operand.to_mlir(),
                from_ty.to_mlir(),
                to_ty.to_mlir()
            ));
        } else {
            let dims_str = dims
                .iter()
                .map(|d| d.to_string())
                .collect::<Vec<_>>()
                .join(", ");
            self.body.push(format!(
                "    {} = stablehlo.broadcast_in_dim {}, dims = [{}] : ({}) -> {}",
                reg.to_mlir(),
                operand.to_mlir(),
                dims_str,
                from_ty.to_mlir(),
                to_ty.to_mlir()
            ));
        }

        reg
    }

    /// Broadcast types: choose result type for binary op
    pub(crate) fn broadcast_types(&self, lhs: &StableHLOType, rhs: &StableHLOType) -> StableHLOType {
        let lhs_shape = lhs.shape();
        let rhs_shape = rhs.shape();

        if lhs_shape.is_empty() && rhs_shape.is_empty() {
            return lhs.clone();
        }
        if lhs_shape.is_empty() {
            return rhs.clone();
        }
        if rhs_shape.is_empty() {
            return lhs.clone();
        }

        // Numpy-style broadcasting: align from trailing dims, take max of each
        let max_ndim = lhs_shape.len().max(rhs_shape.len());
        let mut result_shape = Vec::with_capacity(max_ndim);
        for i in 0..max_ndim {
            let l = if i < max_ndim - lhs_shape.len() {
                1
            } else {
                lhs_shape[i - (max_ndim - lhs_shape.len())]
            };
            let r = if i < max_ndim - rhs_shape.len() {
                1
            } else {
                rhs_shape[i - (max_ndim - rhs_shape.len())]
            };
            result_shape.push(l.max(r));
        }
        StableHLOType::f32_tensor(result_shape)
    }

    /// Emit unary operation (relu, sigmoid, tanh, etc.)
    pub fn emit_unary(&mut self, op: &str, operand: &Register, ty: &StableHLOType) -> Register {
        let reg = self.fresh_register();

        match op {
            "tanh" => {
                self.body.push(format!(
                    "    {} = stablehlo.tanh {} : {}",
                    reg.to_mlir(),
                    operand.to_mlir(),
                    ty.to_mlir()
                ));
            }
            "sqrt" => {
                self.body.push(format!(
                    "    {} = stablehlo.sqrt {} : {}",
                    reg.to_mlir(),
                    operand.to_mlir(),
                    ty.to_mlir()
                ));
            }
            "exp" => {
                self.body.push(format!(
                    "    {} = stablehlo.exponential {} : {}",
                    reg.to_mlir(),
                    operand.to_mlir(),
                    ty.to_mlir()
                ));
            }
            "log" => {
                self.body.push(format!(
                    "    {} = stablehlo.log {} : {}",
                    reg.to_mlir(),
                    operand.to_mlir(),
                    ty.to_mlir()
                ));
            }
            "not" => {
                self.body.push(format!(
                    "    {} = stablehlo.not {} : {}",
                    reg.to_mlir(),
                    operand.to_mlir(),
                    ty.to_mlir()
                ));
            }
            "abs" => {
                self.body.push(format!(
                    "    {} = stablehlo.abs {} : {}",
                    reg.to_mlir(),
                    operand.to_mlir(),
                    ty.to_mlir()
                ));
            }
            _ => panic!("Unsupported unary op: {}", op),
        }

        reg
    }

    /// Extract field from a tuple at a given index.
    /// Virtual tuples resolve without emitting MLIR.
    pub fn emit_get_tuple_element(
        &mut self,
        tuple_reg: &Register,
        tuple_ty: &StableHLOType,
        index: usize,
        result_ty: &StableHLOType,
    ) -> Register {
        if let Some(constituents) = self.virtual_tuples.get(tuple_reg) {
            return constituents[index].0;
        }
        // Fallback: real tuple (shouldn't happen with virtual-only approach)
        let reg = self.fresh_register();
        self.body.push(format!(
            "    {} = stablehlo.get_tuple_element {} [{index}] : ({}) -> {}",
            reg.to_mlir(),
            tuple_reg.to_mlir(),
            tuple_ty.to_mlir(),
            result_ty.to_mlir(),
        ));
        reg
    }

    /// Pack registers into a virtual tuple (no MLIR emitted).
    pub fn emit_tuple(
        &mut self,
        regs: &[Register],
        types: &[StableHLOType],
    ) -> (Register, StableHLOType) {
        let vreg = self.fresh_register();
        let constituents: Vec<_> = regs.iter().zip(types.iter())
            .map(|(r, t)| (*r, t.clone()))
            .collect();
        self.virtual_tuples.insert(vreg, constituents);
        (vreg, StableHLOType::Tuple(types.to_vec()))
    }

    /// Emit a return statement
    pub fn emit_return(&mut self, reg: &Register, ty: &StableHLOType) {
        self.body
            .push(format!("    return {} : {}", reg.to_mlir(), ty.to_mlir()));
    }

    /// Emit a multi-value return: `return %0, %1 : type0, type1`
    pub fn emit_return_multi(&mut self, regs: &[Register], types: &[StableHLOType]) {
        let vals = regs
            .iter()
            .map(|r| r.to_mlir())
            .collect::<Vec<_>>()
            .join(", ");
        let tys = types
            .iter()
            .map(|t| t.to_mlir())
            .collect::<Vec<_>>()
            .join(", ");
        self.body.push(format!("    return {} : {}", vals, tys));
    }

    /// Emit a function with multiple return values.
    pub fn emit_func_declaration_multi(
        &mut self,
        name: &str,
        param_types: &[StableHLOType],
        return_types: &[StableHLOType],
        body_instructions: &[String],
    ) -> String {
        let sanitized_name = Self::sanitize_func_name(name);
        let mut output = String::new();

        let params: Vec<String> = param_types
            .iter()
            .enumerate()
            .map(|(i, ty)| format!("%arg{}: {}", i, ty.to_mlir()))
            .collect();
        let params_str = params.join(", ");

        let ret_str = if return_types.len() == 1 {
            return_types[0].to_mlir()
        } else {
            format!(
                "({})",
                return_types
                    .iter()
                    .map(|t| t.to_mlir())
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        };

        writeln!(
            output,
            "  func.func @{}({}) -> {} {{",
            sanitized_name, params_str, ret_str
        )
        .unwrap();

        for line in body_instructions {
            writeln!(output, "{}", line).unwrap();
        }

        writeln!(output, "  }}").unwrap();
        output
    }

    /// Emit a function declaration (func.func)
    pub fn emit_func_declaration(
        &mut self,
        name: &str,
        param_types: &[StableHLOType],
        return_type: &StableHLOType,
        body_instructions: &[String],
    ) -> String {
        let sanitized_name = Self::sanitize_func_name(name);
        let mut output = String::new();

        let params: Vec<String> = param_types
            .iter()
            .enumerate()
            .map(|(i, ty)| format!("%arg{}: {}", i, ty.to_mlir()))
            .collect();
        let params_str = params.join(", ");

        writeln!(
            output,
            "  func.func @{}({}) -> {} {{",
            sanitized_name,
            params_str,
            return_type.to_mlir()
        )
        .unwrap();

        for line in body_instructions {
            writeln!(output, "{}", line).unwrap();
        }

        writeln!(output, "  }}").unwrap();
        output
    }

    /// Emit a function call (func.call)
    pub fn emit_func_call(
        &mut self,
        name: &str,
        arg_registers: &[Register],
        arg_types: &[StableHLOType],
        return_type: &StableHLOType,
    ) -> Register {
        let sanitized_name = Self::sanitize_func_name(name);
        let reg = self.fresh_register();

        let args_str = arg_registers
            .iter()
            .map(|r| r.to_mlir())
            .collect::<Vec<_>>()
            .join(", ");

        let arg_types_str = arg_types
            .iter()
            .map(|ty| ty.to_mlir())
            .collect::<Vec<_>>()
            .join(", ");

        self.body.push(format!(
            "    {} = func.call @{}({}) : ({}) -> {}",
            reg.to_mlir(),
            sanitized_name,
            args_str,
            arg_types_str,
            return_type.to_mlir()
        ));

        reg
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
                            let shape = self.parse_shape_vector(&elems[2]);
                            self.emit_random_normal(&shape)
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

    /// Sanitize function name for MLIR (replace dashes with underscores)
    fn sanitize_func_name(name: &str) -> String {
        name.replace('-', "_")
    }

    /// Generate a complete MLIR module with a function body already emitted
    pub fn emit_function_body(&self, name: &str, result_ty: &StableHLOType) -> String {
        let sanitized_name = Self::sanitize_func_name(name);
        let mut output = String::new();
        writeln!(output, "// Generated by Sheaf Rust compiler").unwrap();
        writeln!(output, "//").unwrap();
        writeln!(output).unwrap();
        writeln!(output, "module {{").unwrap();
        writeln!(
            output,
            "  func.func @{}() -> {} {{",
            sanitized_name,
            result_ty.to_mlir()
        )
        .unwrap();

        for line in &self.body {
            writeln!(output, "{}", line).unwrap();
        }

        writeln!(output, "  }}").unwrap();
        writeln!(output, "}}").unwrap();

        output
    }

    /// Generate a complete MLIR module with a function
    pub fn emit_function(&mut self, name: &str, expr: &SheafValue) -> String {
        let (result_reg, result_ty) = self.compile_expr(expr);
        self.emit_return(&result_reg, &result_ty);

        let sanitized_name = Self::sanitize_func_name(name);
        let mut output = String::new();
        writeln!(output, "// Generated by Sheaf Rust compiler").unwrap();
        writeln!(output, "//").unwrap();
        writeln!(output, "// Source: {}", expr).unwrap();
        writeln!(output).unwrap();
        writeln!(output, "module {{").unwrap();
        writeln!(
            output,
            "  func.func @{}() -> {} {{",
            sanitized_name,
            result_ty.to_mlir()
        )
        .unwrap();

        for line in &self.body {
            writeln!(output, "{}", line).unwrap();
        }

        writeln!(output, "  }}").unwrap();
        writeln!(output, "}}").unwrap();

        output
    }

    /// Generate a complete MLIR module with multiple function declarations
    pub fn emit_module(func_declarations: &[String]) -> String {
        let mut output = String::new();
        writeln!(output, "// Generated by Sheaf Rust compiler").unwrap();
        writeln!(output, "//").unwrap();
        writeln!(output).unwrap();
        writeln!(output, "module {{").unwrap();

        for func_decl in func_declarations {
            write!(output, "{}", func_decl).unwrap();
        }

        writeln!(output, "}}").unwrap();
        output
    }
}

impl Default for StableHLOEmitter {
    fn default() -> Self {
        Self::new()
    }
}

fn format_f64(v: f64) -> String {
    if v.fract() == 0.0 && v.is_finite() {
        format!("{:.1}", v)
    } else {
        format!("{}", v)
    }
}

/// Recursively format flat data into MLIR dense attribute nesting.
fn format_dense_attr(data: &[f64], shape: &[i64], dim: usize) -> String {
    if dim == shape.len() - 1 {
        let n = shape[dim] as usize;
        let vals: Vec<String> = data[..n].iter().map(|&v| format_f64(v)).collect();
        format!("[{}]", vals.join(", "))
    } else {
        let stride: usize = shape[dim + 1..].iter().map(|&d| d as usize).product();
        let n = shape[dim] as usize;
        let subs: Vec<String> = (0..n)
            .map(|i| format_dense_attr(&data[i * stride..], shape, dim + 1))
            .collect();
        format!("[{}]", subs.join(", "))
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
