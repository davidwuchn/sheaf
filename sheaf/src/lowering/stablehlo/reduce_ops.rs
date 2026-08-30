// Copyright (c) 2026 Damien Boureille
// Licensed under the MIT License.

//! Reductions.

use crate::core::dtype::ElementType;

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

        let result_ty = reduction_type(input_ty, reduced_shape);
        let dtype = input_ty.element_type().unwrap_or(ElementType::F32);
        let (zero_reg, zero_ty) = self.emit_typed_scalar_constant(0.0, dtype);
        let result_reg = self.fresh_register();

        self.body.push(format!(
            "    {} = stablehlo.reduce({} init: {}) applies stablehlo.add across dimensions = [{}] : ({}, {}) -> {}",
            result_reg.to_mlir(),
            input.to_mlir(),
            zero_reg.to_mlir(),
            axis_usize,
            input_ty.to_mlir(),
            zero_ty.to_mlir(),
            result_ty.to_mlir(),
        ));

        if keepdims {
            let mut keepdim_shape = shape.to_vec();
            keepdim_shape[axis_usize] = 1;
            let keepdim_ty = reduction_type(input_ty, keepdim_shape);

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

        let dtype = input_ty.element_type().unwrap_or(ElementType::F32);
        let (n_reg, n_ty) = self.emit_typed_scalar_constant(n, dtype);
        let (result_reg, result_ty) =
            self.emit_binop("/", &sum_reg, &n_reg, &sum_ty, &n_ty);
        self.reduce_mean_cache
            .insert(cache_key, (result_reg, result_ty.clone()));
        (result_reg, result_ty)
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

        let result_ty = reduction_type(input_ty, reduced_shape);

        let dtype = input_ty.element_type().unwrap_or(ElementType::F32);
        let (init_reg, init_ty) =
            self.emit_typed_scalar_constant(f64::NEG_INFINITY, dtype);

        let result_reg = self.fresh_register();
        self.body.push(format!(
            "    {} = stablehlo.reduce({} init: {}) applies stablehlo.maximum across dimensions = [{}] : ({}, {}) -> {}",
            result_reg.to_mlir(),
            input.to_mlir(),
            init_reg.to_mlir(),
            axis_usize,
            input_ty.to_mlir(),
            init_ty.to_mlir(),
            result_ty.to_mlir(),
        ));

        if keepdims {
            let mut keepdim_shape = shape.to_vec();
            keepdim_shape[axis_usize] = 1;
            let keepdim_ty = reduction_type(input_ty, keepdim_shape);

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

        let result_ty = reduction_type(input_ty, reduced_shape);

        let dtype = input_ty.element_type().unwrap_or(ElementType::F32);
        let (init_reg, init_ty) =
            self.emit_typed_scalar_constant(f64::INFINITY, dtype);

        let result_reg = self.fresh_register();
        self.body.push(format!(
            "    {} = stablehlo.reduce({} init: {}) applies stablehlo.minimum across dimensions = [{}] : ({}, {}) -> {}",
            result_reg.to_mlir(),
            input.to_mlir(),
            init_reg.to_mlir(),
            axis_usize,
            input_ty.to_mlir(),
            init_ty.to_mlir(),
            result_ty.to_mlir(),
        ));

        if keepdims {
            let mut keepdim_shape = shape.to_vec();
            keepdim_shape[axis_usize] = 1;
            let keepdim_ty = reduction_type(input_ty, keepdim_shape);

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

        let result_ty = reduction_type(input_ty, reduced_shape);

        let dtype = input_ty.element_type().unwrap_or(ElementType::F32);
        let (one_reg, one_ty) = self.emit_typed_scalar_constant(1.0, dtype);
        let result_reg = self.fresh_register();

        self.body.push(format!(
            "    {} = stablehlo.reduce({} init: {}) applies stablehlo.multiply across dimensions = [{}] : ({}, {}) -> {}",
            result_reg.to_mlir(),
            input.to_mlir(),
            one_reg.to_mlir(),
            axis_usize,
            input_ty.to_mlir(),
            one_ty.to_mlir(),
            result_ty.to_mlir(),
        ));

        if keepdims {
            let mut keepdim_shape = shape.to_vec();
            keepdim_shape[axis_usize] = 1;
            let keepdim_ty = reduction_type(input_ty, keepdim_shape);

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

        let (iota_reg, iota_ty) = self.emit_iota(shape, axis_usize as i64);

        let inf_scalar = self.fresh_register();
        self.body.push(format!(
            "    {} = stablehlo.constant dense<0x7F800000> : {}",
            inf_scalar.to_mlir(),
            StableHLOType::scalar_f32().to_mlir(),
        ));
        let inf_reg = self.emit_broadcast(
            &inf_scalar,
            &StableHLOType::scalar_f32(),
            &iota_ty,
        );
        let (masked_reg, masked_ty) = self.emit_select(
            &mask_reg,
            &iota_reg,
            &inf_reg,
            &mask_ty,
            &iota_ty,
            &iota_ty,
        );
        let (result_f32, result_f32_ty) =
            self.emit_reduce_min(&masked_reg, &masked_ty, axis, false);

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

        let (iota_reg, iota_ty) = self.emit_iota(shape, axis_usize as i64);

        let inf_scalar = self.fresh_register();
        self.body.push(format!(
            "    {} = stablehlo.constant dense<0x7F800000> : {}",
            inf_scalar.to_mlir(),
            StableHLOType::scalar_f32().to_mlir(),
        ));
        let inf_reg = self.emit_broadcast(
            &inf_scalar,
            &StableHLOType::scalar_f32(),
            &iota_ty,
        );
        let (masked_reg, masked_ty) = self.emit_select(
            &mask_reg,
            &iota_reg,
            &inf_reg,
            &mask_ty,
            &iota_ty,
            &iota_ty,
        );
        let (result_f32, result_f32_ty) =
            self.emit_reduce_min(&masked_reg, &masked_ty, axis, false);

        let result_i32_ty = StableHLOType::i32_tensor(result_f32_ty.shape());
        let result_i32 = self.emit_convert(&result_f32, &result_f32_ty, &result_i32_ty);
        (result_i32, result_i32_ty)
    }
}

fn reduction_type(input: &StableHLOType, shape: Vec<i64>) -> StableHLOType {
    StableHLOType::tensor(
        shape,
        input.element_type().unwrap_or(ElementType::F32),
    )
}
