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
    /// Results are cached: calling with the same (input, axis, keepdims) returns
    /// the previously computed register, avoiding duplicate reductions (e.g. var
    /// internally calling mean on the same operand that mean already computed).
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

        let cache_key = (*input, axis_usize, keepdims);
        if let Some(cached) = self.reduce_mean_cache.get(&cache_key) {
            return cached.clone();
        }
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
        self.reduce_mean_cache.insert(cache_key, (result_reg, sum_ty.clone()));
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

    /// Emit stablehlo.reduce to compute min along one axis.
    pub fn emit_reduce_min(
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

        // +inf initializer for min reduction
        let init_reg = self.fresh_register();
        self.body.push(format!(
            "    {} = stablehlo.constant dense<0x7F800000> : {}",
            init_reg.to_mlir(),
            StableHLOType::scalar_f32().to_mlir(),
        ));

        let result_reg = self.fresh_register();
        self.body.push(format!(
            "    {} = stablehlo.reduce({} init: {}) applies stablehlo.minimum across dimensions = [{}] : ({}, {}) -> {}",
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

    /// Emit stablehlo.reduce to compute product along one axis.
    pub fn emit_reduce_product(
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

        // 1.0 initializer for product
        let one_reg = self.emit_constant_f32(1.0);
        let result_reg = self.fresh_register();

        self.body.push(format!(
            "    {} = stablehlo.reduce({} init: {}) applies stablehlo.multiply across dimensions = [{}] : ({}, {}) -> {}",
            result_reg.to_mlir(),
            input.to_mlir(),
            one_reg.to_mlir(),
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

    /// Emit argmax: index of the maximum value along an axis.
    /// Returns f32 indices (matching our tensor type system).
    ///
    /// Strategy: max_val → compare → iota → where(mask, iota, +inf) → reduce_min
    pub fn emit_argmax(
        &mut self,
        input: &Register,
        input_ty: &StableHLOType,
        axis: i64,
    ) -> (Register, StableHLOType) {
        let shape = input_ty.shape();
        let ndim = shape.len();
        let axis_usize = if axis < 0 {
            (ndim as i64 + axis) as usize
        } else {
            axis as usize
        };

        // 1. Get max value along axis (keepdims=true for broadcasting)
        let (max_reg, max_ty) = self.emit_reduce_max(input, input_ty, axis, true);

        // 2. Compare input == max_val (bool mask)
        let (mask_reg, mask_ty) = self.emit_compare("==", input, &max_reg, input_ty, &max_ty);

        // 3. Create iota indices along axis
        let (iota_reg, iota_ty) = self.emit_iota(&shape, axis_usize as i64);

        // 4. +inf constant, broadcast to input shape
        let inf_scalar = self.fresh_register();
        self.body.push(format!(
            "    {} = stablehlo.constant dense<0x7F800000> : {}",
            inf_scalar.to_mlir(),
            StableHLOType::scalar_f32().to_mlir(),
        ));
        let inf_reg = self.emit_broadcast(&inf_scalar, &StableHLOType::scalar_f32(), input_ty);

        // 5. Where mask is true, take iota index; else +inf
        let (masked_reg, _masked_ty) = self.emit_select(&mask_reg, &iota_reg, &inf_reg, &mask_ty, &iota_ty, input_ty);

        // 6. Reduce min along axis → first index of maximum value
        self.emit_reduce_min(&masked_reg, input_ty, axis, false)
    }

    /// Emit argmin: index of the minimum value along an axis.
    /// Returns f32 indices (matching our tensor type system).
    pub fn emit_argmin(
        &mut self,
        input: &Register,
        input_ty: &StableHLOType,
        axis: i64,
    ) -> (Register, StableHLOType) {
        let shape = input_ty.shape();
        let ndim = shape.len();
        let axis_usize = if axis < 0 {
            (ndim as i64 + axis) as usize
        } else {
            axis as usize
        };

        // 1. Get min value along axis (keepdims=true for broadcasting)
        let (min_reg, min_ty) = self.emit_reduce_min(input, input_ty, axis, true);

        // 2. Compare input == min_val
        let (mask_reg, mask_ty) = self.emit_compare("==", input, &min_reg, input_ty, &min_ty);

        // 3. Create iota indices along axis
        let (iota_reg, iota_ty) = self.emit_iota(&shape, axis_usize as i64);

        // 4. +inf constant, broadcast to input shape
        let inf_scalar = self.fresh_register();
        self.body.push(format!(
            "    {} = stablehlo.constant dense<0x7F800000> : {}",
            inf_scalar.to_mlir(),
            StableHLOType::scalar_f32().to_mlir(),
        ));
        let inf_reg = self.emit_broadcast(&inf_scalar, &StableHLOType::scalar_f32(), input_ty);

        // 5. Where mask is true, take iota index; else +inf
        let (masked_reg, _masked_ty) = self.emit_select(&mask_reg, &iota_reg, &inf_reg, &mask_ty, &iota_ty, input_ty);

        // 6. Reduce min along axis → first index of minimum value
        self.emit_reduce_min(&masked_reg, input_ty, axis, false)
    }
}
