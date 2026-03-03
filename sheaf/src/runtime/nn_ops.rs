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

/// Emit log-softmax: x - log(sum(exp(x - max(x))))
pub fn emit_log_softmax(
    emitter: &mut StableHLOEmitter,
    operand: &Register,
    ty: &StableHLOType,
    axis: i64,
) -> (Register, StableHLOType) {
    let (max_reg, max_ty) = emitter.emit_reduce_max(operand, ty, axis, true);
    let (shifted, shifted_ty) = emitter.emit_binop("-", operand, &max_reg, ty, &max_ty);
    let exp_reg = emitter.emit_unary("exp", &shifted, &shifted_ty);
    let (sum_reg, sum_ty) = emitter.emit_reduce_sum(&exp_reg, &shifted_ty, axis, true);
    let log_sum = emitter.emit_unary("log", &sum_reg, &sum_ty);
    emitter.emit_binop("-", &shifted, &log_sum, &shifted_ty, &sum_ty)
}

/// Emit SiLU (swish): x * sigmoid(x)
pub fn emit_silu(
    emitter: &mut StableHLOEmitter,
    operand: &Register,
    ty: &StableHLOType,
) -> (Register, StableHLOType) {
    let sig = emitter.emit_unary("sigmoid", operand, ty);
    emitter.emit_binop("*", operand, &sig, ty, ty)
}

/// Emit leaky-ReLU: where(x > 0, x, alpha * x)
pub fn emit_leaky_relu(
    emitter: &mut StableHLOEmitter,
    operand: &Register,
    ty: &StableHLOType,
    alpha: f64,
) -> (Register, StableHLOType) {
    let zero = emitter.emit_constant_f32(0.0);
    let zero_ty = StableHLOType::scalar_f32();
    let (cond, cond_ty) = emitter.emit_compare(">", operand, &zero, ty, &zero_ty);
    let alpha_reg = emitter.emit_constant_f32(alpha);
    let (ax, ax_ty) = emitter.emit_binop("*", &alpha_reg, operand, &zero_ty, ty);
    emitter.emit_select(&cond, operand, &ax, &cond_ty, ty, &ax_ty)
}

/// Emit SELU: scale * where(x > 0, x, alpha * (exp(x) - 1))
pub fn emit_selu(
    emitter: &mut StableHLOEmitter,
    operand: &Register,
    ty: &StableHLOType,
) -> (Register, StableHLOType) {
    let alpha = 1.6732632423543772;
    let scale = 1.0507009873554805;
    let zero = emitter.emit_constant_f32(0.0);
    let zero_ty = StableHLOType::scalar_f32();
    let (cond, cond_ty) = emitter.emit_compare(">", operand, &zero, ty, &zero_ty);
    let exp_x = emitter.emit_unary("exp", operand, ty);
    let one = emitter.emit_constant_f32(1.0);
    let (exp_minus_1, em1_ty) = emitter.emit_binop("-", &exp_x, &one, ty, &zero_ty);
    let alpha_reg = emitter.emit_constant_f32(alpha);
    let (alpha_em1, alpha_em1_ty) = emitter.emit_binop("*", &alpha_reg, &exp_minus_1, &zero_ty, &em1_ty);
    let (inner, inner_ty) = emitter.emit_select(&cond, operand, &alpha_em1, &cond_ty, ty, &alpha_em1_ty);
    let scale_reg = emitter.emit_constant_f32(scale);
    emitter.emit_binop("*", &scale_reg, &inner, &zero_ty, &inner_ty)
}

/// Emit CELU: where(x > 0, x, alpha * (exp(x/alpha) - 1))
pub fn emit_celu(
    emitter: &mut StableHLOEmitter,
    operand: &Register,
    ty: &StableHLOType,
    alpha: f64,
) -> (Register, StableHLOType) {
    let zero = emitter.emit_constant_f32(0.0);
    let zero_ty = StableHLOType::scalar_f32();
    let (cond, cond_ty) = emitter.emit_compare(">", operand, &zero, ty, &zero_ty);
    let alpha_reg = emitter.emit_constant_f32(alpha);
    let (x_over_a, xoa_ty) = emitter.emit_binop("/", operand, &alpha_reg, ty, &zero_ty);
    let exp_xoa = emitter.emit_unary("exp", &x_over_a, &xoa_ty);
    let one = emitter.emit_constant_f32(1.0);
    let (exp_minus_1, em1_ty) = emitter.emit_binop("-", &exp_xoa, &one, &xoa_ty, &zero_ty);
    let (alpha_em1, alpha_em1_ty) = emitter.emit_binop("*", &alpha_reg, &exp_minus_1, &zero_ty, &em1_ty);
    emitter.emit_select(&cond, operand, &alpha_em1, &cond_ty, ty, &alpha_em1_ty)
}

/// Emit MSE loss: mean((pred - target)^2)
pub fn emit_mse_loss(
    emitter: &mut StableHLOEmitter,
    pred: &Register,
    target: &Register,
    pred_ty: &StableHLOType,
    target_ty: &StableHLOType,
) -> (Register, StableHLOType) {
    let (diff, diff_ty) = emitter.emit_binop("-", pred, target, pred_ty, target_ty);
    let (sq, sq_ty) = emitter.emit_binop("*", &diff, &diff, &diff_ty, &diff_ty);
    // Reduce all dimensions to scalar via mean
    let ndim = sq_ty.shape().len();
    let mut cur_reg = sq;
    let mut cur_ty = sq_ty;
    for _ in (0..ndim).rev() {
        let (r, t) = emitter.emit_reduce_mean(&cur_reg, &cur_ty, -1, false);
        cur_reg = r;
        cur_ty = t;
    }
    (cur_reg, cur_ty)
}

/// Emit MAE loss: mean(|pred - target|)
pub fn emit_mae_loss(
    emitter: &mut StableHLOEmitter,
    pred: &Register,
    target: &Register,
    pred_ty: &StableHLOType,
    target_ty: &StableHLOType,
) -> (Register, StableHLOType) {
    let (diff, diff_ty) = emitter.emit_binop("-", pred, target, pred_ty, target_ty);
    let abs_diff = emitter.emit_unary("abs", &diff, &diff_ty);
    let abs_ty = diff_ty;
    let ndim = abs_ty.shape().len();
    let mut cur_reg = abs_diff;
    let mut cur_ty = abs_ty;
    for _ in (0..ndim).rev() {
        let (r, t) = emitter.emit_reduce_mean(&cur_reg, &cur_ty, -1, false);
        cur_reg = r;
        cur_ty = t;
    }
    (cur_reg, cur_ty)
}

/// Emit sparse cross-entropy loss: -mean(sum(one_hot(labels, C) * log_softmax(logits), axis=-1))
/// logits: [batch, classes], labels: [batch] (integer indices as f32)
pub fn emit_sparse_cross_entropy(
    emitter: &mut StableHLOEmitter,
    logits: &Register,
    labels: &Register,
    logits_ty: &StableHLOType,
    labels_ty: &StableHLOType,
) -> (Register, StableHLOType) {
    let shape = logits_ty.shape();
    let num_classes = *shape.last().unwrap();

    // log-softmax along last axis
    let (log_sm, log_sm_ty) = emit_log_softmax(emitter, logits, logits_ty, -1);

    // one-hot encoding of labels → [batch, classes]
    let (oh, oh_ty) = emitter.emit_one_hot(labels, labels_ty, num_classes);

    // Multiply: pick correct class log-probs
    let (prod, prod_ty) = emitter.emit_binop("*", &oh, &log_sm, &oh_ty, &log_sm_ty);

    // Sum along class axis (-1) → [batch]
    let (batch_loss, batch_loss_ty) = emitter.emit_reduce_sum(&prod, &prod_ty, -1, false);

    // Negate
    let neg_one = emitter.emit_constant_f32(-1.0);
    let (neg_loss, neg_loss_ty) = emitter.emit_binop("*", &neg_one, &batch_loss, &StableHLOType::scalar_f32(), &batch_loss_ty);

    // Mean over batch
    emitter.emit_reduce_mean(&neg_loss, &neg_loss_ty, 0, false)
}
