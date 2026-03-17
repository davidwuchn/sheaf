// Copyright (c) 2026 Damien Boureille
// Licensed under the MIT License.

//! StableHLO type system: StableHLOType, Register, and parsing utilities.

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
pub(super) fn split_tuple_args(s: &str) -> Vec<&str> {
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
