// Copyright (c) 2026 Damien Boureille
// Licensed under the MIT License.

//! Neural network operations runtime module
//!
//! Activation functions: relu, sigmoid, tanh, gelu, softmax
//!
//! This module provides runtime emission helpers for StableHLO neural network operations.

use crate::compiler::stablehlo::{Register, StableHLOEmitter, StableHLOType};

/// Emit ReLU activation: max(x, 0)
pub fn emit_relu(
    emitter: &mut StableHLOEmitter,
    operand: &Register,
    ty: &StableHLOType,
) -> Register {
    emitter.emit_unary("relu", operand, ty)
}

/// Emit sigmoid activation: 1 / (1 + exp(-x))
pub fn emit_sigmoid(
    emitter: &mut StableHLOEmitter,
    operand: &Register,
    ty: &StableHLOType,
) -> Register {
    emitter.emit_unary("sigmoid", operand, ty)
}

/// Emit tanh activation
pub fn emit_tanh(
    emitter: &mut StableHLOEmitter,
    operand: &Register,
    ty: &StableHLOType,
) -> Register {
    emitter.emit_unary("tanh", operand, ty)
}

/// Emit GELU activation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
pub fn emit_gelu(
    emitter: &mut StableHLOEmitter,
    operand: &Register,
    ty: &StableHLOType,
) -> (Register, StableHLOType) {
    let c_half = emitter.emit_constant_f32(0.5);
    let c_coeff = emitter.emit_constant_f32(0.044715);
    let c_sqrt2pi = emitter.emit_constant_f32(0.7978845608028654); // sqrt(2/pi)
    let scalar_ty = StableHLOType::scalar_f32();

    // Broadcast constants to operand shape if needed
    let (half, half_ty, coeff, coeff_ty, sqrt2pi, sqrt2pi_ty) = if !ty.shape().is_empty() {
        (
            emitter.emit_broadcast(&c_half, &scalar_ty, ty), ty.clone(),
            emitter.emit_broadcast(&c_coeff, &scalar_ty, ty), ty.clone(),
            emitter.emit_broadcast(&c_sqrt2pi, &scalar_ty, ty), ty.clone(),
        )
    } else {
        (c_half, scalar_ty.clone(), c_coeff, scalar_ty.clone(), c_sqrt2pi, scalar_ty.clone())
    };

    // x^3
    let (x2, x2_ty) = emitter.emit_binop("*", operand, operand, ty, ty);
    let (x3, x3_ty) = emitter.emit_binop("*", &x2, operand, &x2_ty, ty);

    // 0.044715 * x^3
    let (cx3, cx3_ty) = emitter.emit_binop("*", &coeff, &x3, &coeff_ty, &x3_ty);

    // x + 0.044715 * x^3
    let (inner, inner_ty) = emitter.emit_binop("+", operand, &cx3, ty, &cx3_ty);

    // sqrt(2/pi) * (x + 0.044715 * x^3)
    let (scaled, _scaled_ty) = emitter.emit_binop("*", &sqrt2pi, &inner, &sqrt2pi_ty, &inner_ty);

    // tanh(...)
    let tanh_reg = emitter.emit_unary("tanh", &scaled, ty);

    // 1 + tanh(...)
    let one = emitter.emit_constant_f32(1.0);
    let (one_b, one_b_ty) = if !ty.shape().is_empty() {
        (emitter.emit_broadcast(&one, &scalar_ty, ty), ty.clone())
    } else {
        (one, scalar_ty.clone())
    };
    let (one_plus, one_plus_ty) = emitter.emit_binop("+", &one_b, &tanh_reg, &one_b_ty, ty);

    // 0.5 * x
    let (half_x, half_x_ty) = emitter.emit_binop("*", &half, operand, &half_ty, ty);

    // 0.5 * x * (1 + tanh(...))
    emitter.emit_binop("*", &half_x, &one_plus, &half_x_ty, &one_plus_ty)
}

/// Emit softmax along a given axis: exp(x - max(x)) / sum(exp(x - max(x)))
pub fn emit_softmax(
    emitter: &mut StableHLOEmitter,
    operand: &Register,
    ty: &StableHLOType,
    axis: i64,
) -> (Register, StableHLOType) {
    // Step 1: max along axis with keepdims (for numerical stability)
    let (max_reg, max_ty) = emitter.emit_reduce_max(operand, ty, axis, true);

    // Step 2: x - max (broadcast handled by emit_binop)
    let (shifted, shifted_ty) = emitter.emit_binop("-", operand, &max_reg, ty, &max_ty);

    // Step 3: exp(x - max)
    let exp_reg = emitter.emit_unary("exp", &shifted, &shifted_ty);

    // Step 4: sum(exp) along axis, keepdims=true
    let (sum_reg, sum_ty) = emitter.emit_reduce_sum(&exp_reg, &shifted_ty, axis, true);

    // Step 5: exp / sum (broadcast handled by emit_binop)
    emitter.emit_binop("/", &exp_reg, &sum_reg, &shifted_ty, &sum_ty)
}
