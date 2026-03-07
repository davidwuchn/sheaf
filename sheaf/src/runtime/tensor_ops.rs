// Copyright (c) 2026 Damien Boureille
// Licensed under the MIT License.

//! Tensor operations runtime module
//!
//! Contains:
//! - Tensor creation: zeros, ones, random-normal, etc.
//! - Tensor manipulation: reshape, transpose, slice, concat, etc.
//! - Tensor reduction: sum, mean, min, max, etc.
//!
//! This module provides runtime emission helpers for StableHLO tensor operations.

use crate::compiler::stablehlo::{Register, StableHLOEmitter, StableHLOType};

/// Emit zeros tensor: (zeros [M N]) -> tensor<MxNxf32>
pub fn emit_zeros(emitter: &mut StableHLOEmitter, shape: &[i64]) -> (Register, StableHLOType) {
    emitter.emit_zeros(shape)
}

/// Emit random-normal tensor: (random-normal key [M N])
/// For now, we emit a constant with small values (placeholder)
/// TODO: Proper RNG with seed/key
pub fn emit_random_normal(
    emitter: &mut StableHLOEmitter,
    shape: &[i64],
) -> (Register, StableHLOType) {
    emitter.emit_random_normal(shape)
}

/// Emit ones tensor: (ones [M N]) -> tensor<MxNxf32>
pub fn emit_ones(emitter: &mut StableHLOEmitter, shape: &[i64]) -> (Register, StableHLOType) {
    emitter.emit_ones(shape)
}

/// Emit reshape: (reshape tensor [M N]) -> tensor<MxNxf32>
pub fn emit_reshape(
    emitter: &mut StableHLOEmitter,
    operand: &Register,
    operand_ty: &StableHLOType,
    new_shape: &[i64],
) -> (Register, StableHLOType) {
    emitter.emit_reshape(operand, operand_ty, new_shape)
}

/// Emit transpose: (transpose tensor [1 0]) -> permutes dimensions
pub fn emit_transpose(
    emitter: &mut StableHLOEmitter,
    operand: &Register,
    operand_ty: &StableHLOType,
    permutation: &[i64],
) -> (Register, StableHLOType) {
    emitter.emit_transpose(operand, operand_ty, permutation)
}

/// Emit iota (arange): (arange [N]) -> tensor<Nxf32> with values [0, 1, 2, ..., N-1]
pub fn emit_arange(
    emitter: &mut StableHLOEmitter,
    shape: &[i64],
    dimension: i64,
) -> (Register, StableHLOType) {
    emitter.emit_iota(shape, dimension)
}

/// Emit concatenate: (concat [tensor1 tensor2 ...] axis)
pub fn emit_concatenate(
    emitter: &mut StableHLOEmitter,
    operands: &[Register],
    operand_types: &[StableHLOType],
    dimension: i64,
) -> (Register, StableHLOType) {
    emitter.emit_concatenate(operands, operand_types, dimension)
}

/// Emit where (conditional selection): (where condition x y)
pub fn emit_where(
    emitter: &mut StableHLOEmitter,
    condition: &Register,
    x: &Register,
    y: &Register,
    condition_ty: &StableHLOType,
    x_ty: &StableHLOType,
    y_ty: &StableHLOType,
) -> (Register, StableHLOType) {
    emitter.emit_where(condition, x, y, condition_ty, x_ty, y_ty)
}

/// Emit swapaxes: (swapaxes x axis1 axis2)
pub fn emit_swapaxes(
    emitter: &mut StableHLOEmitter,
    operand: &Register,
    operand_ty: &StableHLOType,
    axis1: i64,
    axis2: i64,
) -> (Register, StableHLOType) {
    emitter.emit_swapaxes(operand, operand_ty, axis1, axis2)
}

/// Emit tril (lower triangular): (tril x)
pub fn emit_tril(
    emitter: &mut StableHLOEmitter,
    operand: &Register,
    operand_ty: &StableHLOType,
) -> (Register, StableHLOType) {
    emitter.emit_tril(operand, operand_ty)
}

/// Emit sum reduction: (sum x :axis 1) or (sum x :axis 1 :keepdims true)
pub fn emit_sum(
    emitter: &mut StableHLOEmitter,
    operand: &Register,
    operand_ty: &StableHLOType,
    axis: i64,
    keepdims: bool,
) -> (Register, StableHLOType) {
    emitter.emit_reduce_sum(operand, operand_ty, axis, keepdims)
}

/// Emit mean reduction: (mean x :axis 1) or (mean x :axis 1 :keepdims true)
pub fn emit_mean(
    emitter: &mut StableHLOEmitter,
    operand: &Register,
    operand_ty: &StableHLOType,
    axis: i64,
    keepdims: bool,
) -> (Register, StableHLOType) {
    emitter.emit_reduce_mean(operand, operand_ty, axis, keepdims)
}

/// Emit product reduction: (product x :axis 1)
pub fn emit_product(
    emitter: &mut StableHLOEmitter,
    operand: &Register,
    operand_ty: &StableHLOType,
    axis: i64,
    keepdims: bool,
) -> (Register, StableHLOType) {
    emitter.emit_reduce_product(operand, operand_ty, axis, keepdims)
}

/// Emit min reduction: (min x :axis 1)
pub fn emit_min_reduce(
    emitter: &mut StableHLOEmitter,
    operand: &Register,
    operand_ty: &StableHLOType,
    axis: i64,
    keepdims: bool,
) -> (Register, StableHLOType) {
    emitter.emit_reduce_min(operand, operand_ty, axis, keepdims)
}

/// Emit max reduction: (max x :axis 1)
pub fn emit_max_reduce(
    emitter: &mut StableHLOEmitter,
    operand: &Register,
    operand_ty: &StableHLOType,
    axis: i64,
    keepdims: bool,
) -> (Register, StableHLOType) {
    emitter.emit_reduce_max(operand, operand_ty, axis, keepdims)
}

/// Emit argmax: (argmax x :axis 1)
pub fn emit_argmax(
    emitter: &mut StableHLOEmitter,
    operand: &Register,
    operand_ty: &StableHLOType,
    axis: i64,
) -> (Register, StableHLOType) {
    emitter.emit_argmax(operand, operand_ty, axis)
}

/// Emit argmin: (argmin x :axis 1)
pub fn emit_argmin(
    emitter: &mut StableHLOEmitter,
    operand: &Register,
    operand_ty: &StableHLOType,
    axis: i64,
) -> (Register, StableHLOType) {
    emitter.emit_argmin(operand, operand_ty, axis)
}

/// Emit identity matrix: (eye N) or (eye N M)
pub fn emit_eye(emitter: &mut StableHLOEmitter, n: i64, m: i64) -> (Register, StableHLOType) {
    emitter.emit_eye(n, m)
}

/// Emit one-hot encoding: (one-hot indices num_classes)
pub fn emit_one_hot(
    emitter: &mut StableHLOEmitter,
    indices: &Register,
    indices_ty: &StableHLOType,
    num_classes: i64,
) -> (Register, StableHLOType) {
    emitter.emit_one_hot(indices, indices_ty, num_classes)
}

/// Emit variance: var(x, axis) = mean((x - mean(x, axis, keepdims=true))^2, axis)
pub fn emit_var(
    emitter: &mut StableHLOEmitter,
    operand: &Register,
    operand_ty: &StableHLOType,
    axis: i64,
    keepdims: bool,
) -> (Register, StableHLOType) {
    // mean(x, axis, keepdims=true) for broadcasting
    let (mean_reg, mean_ty) = emitter.emit_reduce_mean(operand, operand_ty, axis, true);
    // x - mean
    let (diff, diff_ty) = emitter.emit_binop("-", operand, &mean_reg, operand_ty, &mean_ty);
    // (x - mean)^2
    let (sq, sq_ty) = emitter.emit_binop("*", &diff, &diff, &diff_ty, &diff_ty);
    // mean of squared diffs
    emitter.emit_reduce_mean(&sq, &sq_ty, axis, keepdims)
}

/// Emit normalize: normalize(x, axis) = x / sum(x, axis, keepdims=true)
pub fn emit_normalize(
    emitter: &mut StableHLOEmitter,
    operand: &Register,
    operand_ty: &StableHLOType,
    axis: i64,
) -> (Register, StableHLOType) {
    let (sum_reg, sum_ty) = emitter.emit_reduce_sum(operand, operand_ty, axis, true);
    emitter.emit_binop("/", operand, &sum_reg, operand_ty, &sum_ty)
}

/// Emit dynamic-slice: (dynamic-slice tensor start end) — 1D slice with inclusive end
pub fn emit_dynamic_slice(
    emitter: &mut StableHLOEmitter,
    operand: &Register,
    operand_ty: &StableHLOType,
    start: i64,
    end: i64,
) -> (Register, StableHLOType) {
    emitter.emit_slice_range(operand, operand_ty, start, end)
}

/// Emit roll: (roll tensor shift) — circular shift along flat dimension
pub fn emit_roll(
    emitter: &mut StableHLOEmitter,
    operand: &Register,
    operand_ty: &StableHLOType,
    shift: i64,
) -> (Register, StableHLOType) {
    emitter.emit_roll(operand, operand_ty, shift)
}

/// Emit index-update: (index-update tensor idx new-value)
pub fn emit_index_update(
    emitter: &mut StableHLOEmitter,
    operand: &Register,
    operand_ty: &StableHLOType,
    index: i64,
    value: &Register,
    value_ty: &StableHLOType,
) -> (Register, StableHLOType) {
    emitter.emit_index_update(operand, operand_ty, index, value, value_ty)
}

/// Emit append-and-roll: shift 1D tensor left by 1, append new value at end.
/// (append-and-roll [a b c d] x) → [b c d x]
pub fn emit_append_and_roll(
    emitter: &mut StableHLOEmitter,
    operand: &Register,
    operand_ty: &StableHLOType,
    value: &Register,
    value_ty: &StableHLOType,
) -> (Register, StableHLOType) {
    let n = operand_ty.shape()[0];
    // slice [1:n-1] (inclusive end) = elements 1..n-1
    let (tail, tail_ty) = emitter.emit_slice_range(operand, operand_ty, 1, n - 1);
    // Reshape scalar value to [1] tensor
    let val_1d_ty = StableHLOType::f32_tensor(vec![1]);
    let val_1d = emitter.fresh_register();
    emitter.body.push(format!(
        "    {} = stablehlo.reshape {} : ({}) -> {}",
        val_1d.to_mlir(),
        value.to_mlir(),
        value_ty.to_mlir(),
        val_1d_ty.to_mlir(),
    ));
    // concat tail ++ [value]
    emitter.emit_concatenate(&[tail, val_1d], &[tail_ty, val_1d_ty], 0)
}

/// Emit slice: (slice tensor start end :axis N) — start inclusive, end exclusive
pub fn emit_slice(
    emitter: &mut StableHLOEmitter,
    operand: &Register,
    operand_ty: &StableHLOType,
    start: i64,
    end: i64,
    axis: usize,
) -> (Register, StableHLOType) {
    emitter.emit_slice_axis(operand, operand_ty, start, end, axis)
}

/// Emit tensor-split: (tensor-split tensor num-sections)
pub fn emit_tensor_split(
    emitter: &mut StableHLOEmitter,
    operand: &Register,
    operand_ty: &StableHLOType,
    num_sections: i64,
) -> (Register, StableHLOType) {
    emitter.emit_tensor_split(operand, operand_ty, num_sections)
}

/// Emit gather along axis 0: (get operand indices) where indices is a tensor.
pub fn emit_gather_axis0(
    emitter: &mut StableHLOEmitter,
    operand: &Register,
    operand_ty: &StableHLOType,
    indices: &Register,
    indices_ty: &StableHLOType,
) -> (Register, StableHLOType) {
    emitter.emit_gather_axis0(operand, operand_ty, indices, indices_ty)
}
