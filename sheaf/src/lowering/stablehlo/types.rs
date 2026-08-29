// Copyright (c) 2026 Damien Boureille
// Licensed under the MIT License.

//! StableHLO type system: StableHLOType, Register, and parsing utilities.

use crate::core::dtype::ElementType;

/// StableHLO type representation
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum StableHLOType {
    /// Scalar tensor: tensor<f32>
    ScalarF32,
    /// Scalar tensor: tensor<f16>
    ScalarF16,
    /// Scalar tensor: tensor<bf16>
    ScalarBF16,
    /// Scalar tensor: tensor<f64>
    ScalarF64,
    /// Scalar tensor: tensor<i64>
    ScalarI64,
    /// Scalar tensor: tensor<i1> (boolean)
    ScalarI1,
    /// Tensor with shape: tensor<2x3xf32>
    Tensor { shape: Vec<i64>, dtype: ElementType },
    /// Tuple of types: tuple<tensor<2x3xf32>, tensor<8xf32>>
    /// When keys is Some, this represents a dict (reconstructed as Value::Dict on output).
    Tuple(Vec<StableHLOType>, Option<Vec<String>>),
}

impl StableHLOType {
    pub fn scalar_f32() -> Self {
        Self::ScalarF32
    }

    pub fn scalar_f16() -> Self {
        Self::ScalarF16
    }

    pub fn scalar_bf16() -> Self {
        Self::ScalarBF16
    }

    pub fn scalar_i64() -> Self {
        Self::ScalarI64
    }

    pub fn f32_tensor(shape: impl Into<Vec<i64>>) -> Self {
        Self::tensor(shape, ElementType::F32)
    }

    pub fn f16_tensor(shape: impl Into<Vec<i64>>) -> Self {
        Self::tensor(shape, ElementType::F16)
    }

    pub fn bf16_tensor(shape: impl Into<Vec<i64>>) -> Self {
        Self::tensor(shape, ElementType::BF16)
    }

    /// Create a tensor with the given StableHLO element type.
    pub fn tensor(shape: impl Into<Vec<i64>>, dtype: ElementType) -> Self {
        Self::Tensor { shape: shape.into(), dtype }
    }

    /// Create a tensor from a StableHLO element type string.
    pub fn typed_tensor(shape: impl Into<Vec<i64>>, dtype: &str) -> Self {
        let dtype = ElementType::from_mlir_str(dtype)
            .unwrap_or_else(|| panic!("unsupported StableHLO element type: {}", dtype));
        Self::tensor(shape, dtype)
    }

    pub fn i32_tensor(shape: impl Into<Vec<i64>>) -> Self {
        Self::tensor(shape, ElementType::I32)
    }

    pub fn i64_tensor(shape: impl Into<Vec<i64>>) -> Self {
        Self::tensor(shape, ElementType::I64)
    }

    pub fn i1_tensor(shape: impl Into<Vec<i64>>) -> Self {
        Self::tensor(shape, ElementType::Bool)
    }

    /// Get the shape of this type, or empty slice for scalars/tuples
    pub fn shape(&self) -> &[i64] {
        match self {
            Self::ScalarF32
            | Self::ScalarF16
            | Self::ScalarBF16
            | Self::ScalarF64
            | Self::ScalarI64
            | Self::ScalarI1
            | Self::Tuple(..) => &[],
            Self::Tensor { shape, .. } => shape,
        }
    }

    /// Get the element type of a tensor leaf.
    pub fn element_type(&self) -> Option<ElementType> {
        match self {
            Self::ScalarF32 => Some(ElementType::F32),
            Self::ScalarF16 => Some(ElementType::F16),
            Self::ScalarBF16 => Some(ElementType::BF16),
            Self::ScalarF64 => Some(ElementType::F64),
            Self::ScalarI64 => Some(ElementType::I64),
            Self::ScalarI1 => Some(ElementType::Bool),
            Self::Tensor { dtype, .. } => Some(*dtype),
            Self::Tuple(..) => None,
        }
    }

    /// Get the StableHLO dtype string.
    pub fn dtype(&self) -> &str {
        self.element_type()
            .map(ElementType::to_mlir_str)
            .unwrap_or("tuple")
    }

    pub fn is_float(&self) -> bool {
        self.element_type().is_some_and(ElementType::is_float)
    }

    pub fn with_element_type(&self, dtype: ElementType) -> Option<Self> {
        match self {
            Self::Tensor { shape, .. } => Some(Self::tensor(shape.clone(), dtype)),
            Self::Tuple(..) => None,
            _ => Some(match dtype {
                ElementType::F16 => Self::ScalarF16,
                ElementType::BF16 => Self::ScalarBF16,
                ElementType::F32 => Self::ScalarF32,
                ElementType::F64 => Self::ScalarF64,
                ElementType::I64 => Self::ScalarI64,
                ElementType::Bool => Self::ScalarI1,
                ElementType::I32 => Self::tensor(Vec::new(), ElementType::I32),
            }),
        }
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
            Self::ScalarF16 => "tensor<f16>".to_string(),
            Self::ScalarBF16 => "tensor<bf16>".to_string(),
            Self::ScalarF64 => "tensor<f64>".to_string(),
            Self::ScalarI64 => "tensor<i64>".to_string(),
            Self::ScalarI1 => "tensor<i1>".to_string(),
            Self::Tensor { shape, dtype } => {
                if shape.is_empty() {
                    format!("tensor<{}>", dtype.to_mlir_str())
                } else {
                    let shape_str = shape
                        .iter()
                        .map(|d| d.to_string())
                        .collect::<Vec<_>>()
                        .join("x");
                    format!("tensor<{}x{}>", shape_str, dtype.to_mlir_str())
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
                    "f16" => Some(Self::ScalarF16),
                    "bf16" => Some(Self::ScalarBF16),
                    "f64" => Some(Self::ScalarF64),
                    "i64" => Some(Self::ScalarI64),
                    "i1" => Some(Self::ScalarI1),
                    _ => None,
                };
            }
            let dtype = ElementType::from_mlir_str(parts.last()?)?;
            let shape: Option<Vec<i64>> = parts[..parts.len() - 1]
                .iter()
                .map(|d| d.parse::<i64>().ok())
                .collect();
            return shape.map(|shape| Self::Tensor { shape, dtype });
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

#[cfg(test)]
mod tests {
    use super::StableHLOType;
    use crate::core::dtype::ElementType;

    #[test]
    fn tensor_serde_remains_compatible_with_string_dtypes() {
        let ty = StableHLOType::tensor(vec![2, 3], ElementType::BF16);
        let json = serde_json::to_string(&ty).unwrap();
        assert_eq!(json, r#"{"Tensor":{"shape":[2,3],"dtype":"bf16"}}"#);
        assert_eq!(serde_json::from_str::<StableHLOType>(&json).unwrap(), ty);
    }

    #[test]
    fn parses_all_supported_tensor_element_types() {
        for dtype in ["f16", "bf16", "f32", "f64", "i1", "i32", "i64"] {
            let mlir = format!("tensor<2x{}>", dtype);
            let ty = StableHLOType::parse(&mlir).unwrap();
            assert_eq!(ty.to_mlir(), mlir);
        }
    }

    #[test]
    fn rejects_unknown_tensor_element_types() {
        assert!(StableHLOType::parse("tensor<2xcomplex<f32>>").is_none());
    }
}
