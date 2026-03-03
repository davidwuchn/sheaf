// Copyright (c) 2026 Damien Boureille
// Licensed under the MIT License.

//! Reduction operations (sum, mean, max, min, product, argmax, argmin) for StableHLO.

use super::{Register, StableHLOEmitter, StableHLOType};

impl StableHLOEmitter {
    /// Emit stablehlo.reduce to compute sum along one axis.
    ///
    /// input: tensor<...xf32>, axis: which dimension to reduce
    /// returns: tensor with that dimension removed (or size 1 if keepdims)
    pub fn emit_reduce_sum(
        &mut self,
        input: &Register,
        input_ty: &StableHLOType,
        axis: i64,
        keepdims: bool,
    ) -> (Register, StableHLOType) {
        let shape = input_ty.shape();
        // Scalar reduce is a no-op: nothing to sum over
        if shape.is_empty() {
            return (input.clone(), input_ty.clone());
        }
        let ndim = shape.len();
        let axis_usize = if axis < 0 {
            (ndim as i64 + axis) as usize
        } else {
            axis as usize
        };
        let axis_usize = axis_usize.min(ndim.saturating_sub(1));

        // Result shape: remove the reduced axis
        let reduced_shape: Vec<i64> = shape
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != axis_usize)
            .map(|(_, &d)| d)
            .collect();

        let result_ty = if reduced_shape.is_empty() {
            StableHLOType::scalar_f32()
        } else {
            StableHLOType::f32_tensor(reduced_shape)
        };

        // Zero initializer (scalar)
        let zero_reg = self.emit_constant_f32(0.0);
        let result_reg = self.fresh_register();

        self.body.push(format!(
            "    {} = stablehlo.reduce({} init: {}) applies stablehlo.add across dimensions = [{}] : ({}, {}) -> {}",
            result_reg.to_mlir(),
            input.to_mlir(),
            zero_reg.to_mlir(),
            axis_usize,
            input_ty.to_mlir(),
            StableHLOType::scalar_f32().to_mlir(),
            result_ty.to_mlir(),
        ));

        if keepdims {
            // Re-insert the reduced dimension as size 1
            let mut keepdim_shape = shape.to_vec();
            keepdim_shape[axis_usize] = 1;
            let keepdim_ty = StableHLOType::f32_tensor(keepdim_shape);

            let keepdim_reg = self.fresh_register();
            self.body.push(format!(
                "    {} = stablehlo.reshape {} : ({}) -> {}",
                keepdim_reg.to_mlir(),
                result_reg.to_mlir(),
                result_ty.to_mlir(),
                keepdim_ty.to_mlir(),
            ));
            (keepdim_reg, keepdim_ty)
        } else {
            (result_reg, result_ty)
        }
    }

    /// Emit stablehlo.reduce to compute mean along one axis.
    pub fn emit_reduce_mean(
        &mut self,
        input: &Register,
        input_ty: &StableHLOType,
        axis: i64,
        keepdims: bool,
    ) -> (Register, StableHLOType) {
        let shape = input_ty.shape();
        let ndim = shape.len();

        // mean of a scalar is identity
        if ndim == 0 {
            return (input.clone(), input_ty.clone());
        }

        let axis_usize = if axis < 0 {
            (ndim as i64 + axis) as usize
        } else {
            axis as usize
        };
        let n = shape[axis_usize] as f64;

        let (sum_reg, sum_ty) = self.emit_reduce_sum(input, input_ty, axis, keepdims);

        // Divide by n
        let n_reg = self.emit_constant_f32(n);

        // Broadcast n to match sum shape if needed
        let result_reg = self.fresh_register();
        if sum_ty == StableHLOType::scalar_f32() {
            self.body.push(format!(
                "    {} = stablehlo.divide {}, {} : {}",
                result_reg.to_mlir(),
                sum_reg.to_mlir(),
                n_reg.to_mlir(),
                sum_ty.to_mlir(),
            ));
        } else {
            let n_broadcast_reg = self.fresh_register();
            self.body.push(format!(
                "    {} = stablehlo.broadcast_in_dim {}, dims = [] : ({}) -> {}",
                n_broadcast_reg.to_mlir(),
                n_reg.to_mlir(),
                StableHLOType::scalar_f32().to_mlir(),
                sum_ty.to_mlir(),
            ));
            self.body.push(format!(
                "    {} = stablehlo.divide {}, {} : {}",
                result_reg.to_mlir(),
                sum_reg.to_mlir(),
                n_broadcast_reg.to_mlir(),
                sum_ty.to_mlir(),
            ));
        }
        (result_reg, sum_ty)
    }

    /// Emit stablehlo.reduce to compute max along one axis.
    pub fn emit_reduce_max(
        &mut self,
        input: &Register,
        input_ty: &StableHLOType,
        axis: i64,
        keepdims: bool,
    ) -> (Register, StableHLOType) {
        let shape = input_ty.shape();
        if shape.is_empty() {
            return (input.clone(), input_ty.clone());
        }
        let ndim = shape.len();
        let axis_usize = if axis < 0 {
            (ndim as i64 + axis) as usize
        } else {
            axis as usize
        };
        let axis_usize = axis_usize.min(ndim.saturating_sub(1));

        let reduced_shape: Vec<i64> = shape
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != axis_usize)
            .map(|(_, &d)| d)
            .collect();

        let result_ty = if reduced_shape.is_empty() {
            StableHLOType::scalar_f32()
        } else {
            StableHLOType::f32_tensor(reduced_shape)
        };

        // -inf initializer for max reduction
        let init_reg = self.fresh_register();
        self.body.push(format!(
            "    {} = stablehlo.constant dense<0xFF800000> : {}",
            init_reg.to_mlir(),
            StableHLOType::scalar_f32().to_mlir(),
        ));

        let result_reg = self.fresh_register();
        self.body.push(format!(
            "    {} = stablehlo.reduce({} init: {}) applies stablehlo.maximum across dimensions = [{}] : ({}, {}) -> {}",
            result_reg.to_mlir(),
            input.to_mlir(),
            init_reg.to_mlir(),
            axis_usize,
            input_ty.to_mlir(),
            StableHLOType::scalar_f32().to_mlir(),
            result_ty.to_mlir(),
        ));

        if keepdims {
            let mut keepdim_shape = shape.to_vec();
            keepdim_shape[axis_usize] = 1;
            let keepdim_ty = StableHLOType::f32_tensor(keepdim_shape);

            let keepdim_reg = self.fresh_register();
            self.body.push(format!(
                "    {} = stablehlo.reshape {} : ({}) -> {}",
                keepdim_reg.to_mlir(),
                result_reg.to_mlir(),
                result_ty.to_mlir(),
                keepdim_ty.to_mlir(),
            ));
            (keepdim_reg, keepdim_ty)
        } else {
            (result_reg, result_ty)
        }
    }
}
