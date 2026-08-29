// Copyright (c) 2026 Damien Boureille
// Licensed under the MIT License.

//! Tensor dtypes.

use std::fmt;

#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    Hash,
    serde::Serialize,
    serde::Deserialize,
)]
pub enum ElementType {
    #[serde(rename = "f16")]
    F16,
    #[serde(rename = "bf16")]
    BF16,
    #[serde(rename = "f32")]
    F32,
    #[serde(rename = "f64")]
    F64,
    #[serde(rename = "i1")]
    Bool,
    #[serde(rename = "i32")]
    I32,
    #[serde(rename = "i64")]
    I64,
}

impl ElementType {
    pub fn from_mlir_str(value: &str) -> Option<Self> {
        match value {
            "f16" => Some(Self::F16),
            "bf16" => Some(Self::BF16),
            "f32" => Some(Self::F32),
            "f64" => Some(Self::F64),
            "i1" => Some(Self::Bool),
            "i32" => Some(Self::I32),
            "i64" => Some(Self::I64),
            _ => None,
        }
    }

    pub fn from_keyword(value: &str) -> Option<Self> {
        match value {
            "f16" => Some(Self::F16),
            "bf16" => Some(Self::BF16),
            "f32" => Some(Self::F32),
            "i32" => Some(Self::I32),
            "bool" => Some(Self::Bool),
            _ => None,
        }
    }

    pub const fn to_mlir_str(self) -> &'static str {
        match self {
            Self::F16 => "f16",
            Self::BF16 => "bf16",
            Self::F32 => "f32",
            Self::F64 => "f64",
            Self::Bool => "i1",
            Self::I32 => "i32",
            Self::I64 => "i64",
        }
    }

    pub const fn name(self) -> &'static str {
        match self {
            Self::Bool => "bool",
            other => other.to_mlir_str(),
        }
    }

    pub const fn element_size(self) -> usize {
        match self {
            Self::Bool => 1,
            Self::F16 | Self::BF16 => 2,
            Self::F32 | Self::I32 => 4,
            Self::F64 | Self::I64 => 8,
        }
    }

    pub const fn is_float(self) -> bool {
        matches!(self, Self::F16 | Self::BF16 | Self::F32 | Self::F64)
    }

    pub const fn is_integer(self) -> bool {
        matches!(self, Self::I32 | Self::I64)
    }
}

impl fmt::Display for ElementType {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.name())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DtypeStrength {
    Weak,
    Strong,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DtypeOperand {
    pub dtype: ElementType,
    pub strength: DtypeStrength,
}

impl DtypeOperand {
    pub const fn weak(dtype: ElementType) -> Self {
        Self { dtype, strength: DtypeStrength::Weak }
    }

    pub const fn strong(dtype: ElementType) -> Self {
        Self { dtype, strength: DtypeStrength::Strong }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DtypeMismatch {
    pub lhs: ElementType,
    pub rhs: ElementType,
}

impl fmt::Display for DtypeMismatch {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "dtype mismatch: {} and {}", self.lhs, self.rhs)
    }
}

pub fn resolve_arithmetic_dtype(
    lhs: DtypeOperand,
    rhs: DtypeOperand,
) -> Result<ElementType, DtypeMismatch> {
    if lhs.dtype == rhs.dtype {
        return Ok(lhs.dtype);
    }

    let dtype = match (lhs.strength, rhs.strength) {
        (DtypeStrength::Strong, DtypeStrength::Strong) => None,
        (DtypeStrength::Weak, DtypeStrength::Strong) => {
            resolve_weak_dtype(lhs.dtype, rhs.dtype)
        }
        (DtypeStrength::Strong, DtypeStrength::Weak) => {
            resolve_weak_dtype(rhs.dtype, lhs.dtype)
        }
        (DtypeStrength::Weak, DtypeStrength::Weak) => {
            resolve_weak_pair(lhs.dtype, rhs.dtype)
        }
    };
    dtype.ok_or(DtypeMismatch {
        lhs: lhs.dtype,
        rhs: rhs.dtype,
    })
}

fn resolve_weak_dtype(weak: ElementType, strong: ElementType) -> Option<ElementType> {
    if strong.is_float() {
        (weak.is_float() || weak.is_integer()).then_some(strong)
    } else if strong.is_integer() {
        weak.is_integer().then_some(strong)
    } else {
        None
    }
}

fn resolve_weak_pair(lhs: ElementType, rhs: ElementType) -> Option<ElementType> {
    if lhs.is_float() || rhs.is_float() {
        Some(if lhs == ElementType::F64 || rhs == ElementType::F64 {
            ElementType::F64
        } else {
            ElementType::F32
        })
    } else if lhs.is_integer() && rhs.is_integer() {
        Some(if lhs == ElementType::I64 || rhs == ElementType::I64 {
            ElementType::I64
        } else {
            ElementType::I32
        })
    } else {
        None
    }
}

#[cfg(not(sheaf_frontend))]
pub(crate) fn quantize_f32(value: f32, dtype: ElementType) -> f32 {
    match dtype {
        ElementType::F16 => f16_bits_to_f32(f32_to_f16_bits(value)),
        ElementType::BF16 => bf16_bits_to_f32(f32_to_bf16_bits(value)),
        ElementType::I32 => (value as i32) as f32,
        ElementType::I64 => (value as i64) as f32,
        ElementType::Bool => (value != 0.0) as u8 as f32,
        ElementType::F32 | ElementType::F64 => value,
    }
}

#[cfg(not(sheaf_frontend))]
pub(crate) fn f32_to_bf16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    if value.is_nan() {
        return ((bits >> 16) as u16) | 0x0040;
    }
    let retained_lsb = (bits >> 16) & 1;
    (bits.wrapping_add(0x7fff + retained_lsb) >> 16) as u16
}

#[cfg(not(sheaf_frontend))]
pub(crate) fn bf16_bits_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

#[cfg(not(sheaf_frontend))]
pub(crate) fn f32_to_f16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exponent = ((bits >> 23) & 0xff) as i32;
    let mantissa = bits & 0x7f_ffff;

    if exponent == 0xff {
        return if mantissa == 0 {
            sign | 0x7c00
        } else {
            sign | 0x7c00 | ((mantissa >> 13) as u16).max(1)
        };
    }

    let half_exponent = exponent - 127 + 15;
    if half_exponent >= 31 {
        return sign | 0x7c00;
    }
    if half_exponent <= 0 {
        if half_exponent < -10 {
            return sign;
        }
        let significand = mantissa | 0x80_0000;
        let shift = (14 - half_exponent) as u32;
        let mut half_mantissa = significand >> shift;
        let remainder = significand & ((1 << shift) - 1);
        let halfway = 1 << (shift - 1);
        if remainder > halfway || (remainder == halfway && half_mantissa & 1 != 0) {
            half_mantissa += 1;
        }
        return sign | half_mantissa as u16;
    }

    let mut result = sign | ((half_exponent as u16) << 10) | (mantissa >> 13) as u16;
    let remainder = mantissa & 0x1fff;
    if remainder > 0x1000 || (remainder == 0x1000 && result & 1 != 0) {
        result += 1;
    }
    result
}

#[cfg(not(sheaf_frontend))]
pub(crate) fn f16_bits_to_f32(bits: u16) -> f32 {
    let sign = (bits as u32 & 0x8000) << 16;
    let exponent = ((bits >> 10) & 0x1f) as i32;
    let mantissa = (bits & 0x03ff) as u32;
    let result = match exponent {
        0 if mantissa == 0 => sign,
        0 => {
            let highest_bit = 31 - mantissa.leading_zeros();
            let exponent_bits = 103 + highest_bit;
            let fraction = (mantissa << (10 - highest_bit)) & 0x03ff;
            sign | (exponent_bits << 23) | (fraction << 13)
        }
        31 => sign | 0x7f80_0000 | (mantissa << 13),
        _ => sign | ((exponent as u32 + 112) << 23) | (mantissa << 13),
    };
    f32::from_bits(result)
}

#[cfg(all(test, not(sheaf_frontend)))]
mod tests {
    use super::{
        DtypeOperand, ElementType, bf16_bits_to_f32, f16_bits_to_f32, f32_to_bf16_bits,
        f32_to_f16_bits, resolve_arithmetic_dtype,
    };

    #[test]
    fn serde_uses_stable_mlir_spellings() {
        assert_eq!(serde_json::to_string(&ElementType::BF16).unwrap(), "\"bf16\"");
        assert_eq!(serde_json::to_string(&ElementType::Bool).unwrap(), "\"i1\"");
        assert_eq!(
            serde_json::from_str::<ElementType>("\"f16\"").unwrap(),
            ElementType::F16,
        );
    }

    #[test]
    fn bool_has_distinct_user_and_mlir_names() {
        assert_eq!(ElementType::Bool.name(), "bool");
        assert_eq!(ElementType::Bool.to_mlir_str(), "i1");
    }

    #[test]
    fn user_keywords_only_accept_runtime_tensor_types() {
        for dtype in ["f16", "bf16", "f32", "i32", "bool"] {
            assert!(ElementType::from_keyword(dtype).is_some(), "{dtype}");
        }
        for dtype in ["f64", "i1", "i64"] {
            assert_eq!(ElementType::from_keyword(dtype), None, "{dtype}");
        }
    }

    #[test]
    fn weak_scalars_adopt_strong_float_dtypes() {
        for dtype in [ElementType::F16, ElementType::BF16, ElementType::F32] {
            assert_eq!(
                resolve_arithmetic_dtype(
                    DtypeOperand::strong(dtype),
                    DtypeOperand::weak(ElementType::F32),
                ),
                Ok(dtype),
            );
            assert_eq!(
                resolve_arithmetic_dtype(
                    DtypeOperand::weak(ElementType::I32),
                    DtypeOperand::strong(dtype),
                ),
                Ok(dtype),
            );
        }
    }

    #[test]
    fn strong_dtypes_must_match() {
        for rhs in [ElementType::BF16, ElementType::F32] {
            assert!(resolve_arithmetic_dtype(
                DtypeOperand::strong(ElementType::F16),
                DtypeOperand::strong(rhs),
            )
            .is_err());
        }
    }

    #[test]
    fn weak_float_does_not_narrow_to_integer() {
        assert!(resolve_arithmetic_dtype(
            DtypeOperand::strong(ElementType::I32),
            DtypeOperand::weak(ElementType::F32),
        )
        .is_err());
    }

    #[test]
    fn weak_dtypes_use_default_widths() {
        assert_eq!(
            resolve_arithmetic_dtype(
                DtypeOperand::weak(ElementType::I32),
                DtypeOperand::weak(ElementType::F32),
            ),
            Ok(ElementType::F32),
        );
        assert_eq!(
            resolve_arithmetic_dtype(
                DtypeOperand::weak(ElementType::I32),
                DtypeOperand::weak(ElementType::I64),
            ),
            Ok(ElementType::I64),
        );
    }

    #[test]
    fn bf16_conversion_handles_nan_and_ties_to_even() {
        assert!(bf16_bits_to_f32(f32_to_bf16_bits(f32::NAN)).is_nan());
        assert_eq!(f32_to_bf16_bits(1.0 + 2.0f32.powi(-8)), 0x3f80);
        assert_eq!(f32_to_bf16_bits(1.0 + 3.0 * 2.0f32.powi(-8)), 0x3f82);
    }

    #[test]
    fn f16_conversion_handles_subnormal_nan_and_ties_to_even() {
        assert_eq!(f32_to_f16_bits(2.0f32.powi(-15)), 0x0200);
        assert!(f16_bits_to_f32(f32_to_f16_bits(f32::NAN)).is_nan());
        assert_eq!(f32_to_f16_bits(1.0 + 2.0f32.powi(-11)), 0x3c00);
        assert_eq!(f32_to_f16_bits(1.0 + 3.0 * 2.0f32.powi(-11)), 0x3c02);
    }
}
