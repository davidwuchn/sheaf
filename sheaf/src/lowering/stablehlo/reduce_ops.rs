// Copyright (c) 2026 Damien Boureille
// Licensed under the MIT License.

//! Reduction operations (sum, mean, max, min, product, argmax, argmin) for StableHLO.

use super::{Register, StableHLOEmitter, StableHLOType};

impl StableHLOEmitter {
    pub fn emit_reduce_sum(
        &mut self,
        input: &Register,
        input_ty: &StableHLOType,
        axis: i64,
        keepdims: bool,
    ) -> (Register, StableHLOType) {
        let shape = input_ty.shape();
        if shape.is_empty() {
            return (*input, input_ty.clone());
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

    pub fn emit_reduce_mean(
        &mut self,
        input: &Register,
        input_ty: &StableHLOType,
        axis: i64,
        keepdims: bool,
    ) -> (Register, StableHLOType) {
        let shape = input_ty.shape();
        let ndim = shape.len();

        if ndim == 0 {
            return (*input, input_ty.clone());
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

        let n_reg = self.emit_constant_f32(n);

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

    pub fn emit_reduce_max(
        &mut self,
        input: &Register,
        input_ty: &StableHLOType,
        axis: i64,
        keepdims: bool,
    ) -> (Register, StableHLOType) {
        let shape = input_ty.shape();
        if shape.is_empty() {
            return (*input, input_ty.clone());
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

    pub fn emit_reduce_min(
        &mut self,
        input: &Register,
        input_ty: &StableHLOType,
        axis: i64,
        keepdims: bool,
    ) -> (Register, StableHLOType) {
        let shape = input_ty.shape();
        if shape.is_empty() {
            return (*input, input_ty.clone());
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

    pub fn emit_reduce_product(
        &mut self,
        input: &Register,
        input_ty: &StableHLOType,
        axis: i64,
        keepdims: bool,
    ) -> (Register, StableHLOType) {
        let shape = input_ty.shape();
        if shape.is_empty() {
            return (*input, input_ty.clone());
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

    // Select the first index equal to the reduced extremum.
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

        let (max_reg, max_ty) = self.emit_reduce_max(input, input_ty, axis, true);

        let (mask_reg, mask_ty) = self.emit_compare("==", input, &max_reg, input_ty, &max_ty);

        // Create iota indices along axis
        let (iota_reg, iota_ty) = self.emit_iota(shape, axis_usize as i64);

        // +inf constant, broadcast to input shape
        let inf_scalar = self.fresh_register();
        self.body.push(format!(
            "    {} = stablehlo.constant dense<0x7F800000> : {}",
            inf_scalar.to_mlir(),
            StableHLOType::scalar_f32().to_mlir(),
        ));
        let inf_reg = self.emit_broadcast(&inf_scalar, &StableHLOType::scalar_f32(), input_ty);

        // Where mask is true, take iota index; else +inf
        let (masked_reg, _masked_ty) = self.emit_select(&mask_reg, &iota_reg, &inf_reg, &mask_ty, &iota_ty, input_ty);

        let (result_f32, result_f32_ty) = self.emit_reduce_min(&masked_reg, input_ty, axis, false);

        let result_i32_ty = StableHLOType::i32_tensor(result_f32_ty.shape());
        let result_i32 = self.emit_convert(&result_f32, &result_f32_ty, &result_i32_ty);
        (result_i32, result_i32_ty)
    }

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

        let (min_reg, min_ty) = self.emit_reduce_min(input, input_ty, axis, true);

        let (mask_reg, mask_ty) = self.emit_compare("==", input, &min_reg, input_ty, &min_ty);

        // Create iota indices along axis
        let (iota_reg, iota_ty) = self.emit_iota(shape, axis_usize as i64);

        // +inf constant, broadcast to input shape
        let inf_scalar = self.fresh_register();
        self.body.push(format!(
            "    {} = stablehlo.constant dense<0x7F800000> : {}",
            inf_scalar.to_mlir(),
            StableHLOType::scalar_f32().to_mlir(),
        ));
        let inf_reg = self.emit_broadcast(&inf_scalar, &StableHLOType::scalar_f32(), input_ty);

        // Where mask is true, take iota index; else +inf
        let (masked_reg, _masked_ty) = self.emit_select(&mask_reg, &iota_reg, &inf_reg, &mask_ty, &iota_ty, input_ty);

        let (result_f32, result_f32_ty) = self.emit_reduce_min(&masked_reg, input_ty, axis, false);

        let result_i32_ty = StableHLOType::i32_tensor(result_f32_ty.shape());
        let result_i32 = self.emit_convert(&result_f32, &result_f32_ty, &result_i32_ty);
        (result_i32, result_i32_ty)
    }
}
