// Copyright (c) 2026 Damien Boureille
// Licensed under the MIT License.

//! Indexing, slicing, and gathering operations for StableHLO.

use super::{Register, StableHLOEmitter, StableHLOType};

impl StableHLOEmitter {
    /// Emit slice along axis 0 at a given index, then reshape to remove that axis.
    /// E.g. tensor<5x9xf32> at index 0 -> tensor<9xf32>
    pub fn emit_index_axis0(
        &mut self,
        input: &Register,
        input_ty: &StableHLOType,
        index: i64,
    ) -> (Register, StableHLOType) {
        let shape = input_ty.shape();
        assert!(!shape.is_empty(), "Cannot index a scalar");

        let ndim = shape.len();

        // Build start_indices and limit_indices
        let mut start = vec![0i64; ndim];
        let mut limit = shape.to_vec();
        let strides = vec![1i64; ndim];
        start[0] = index;
        limit[0] = index + 1;

        let dims_str = format_slice_dims(&start, &limit, &strides);

        // Slice to [1, d1, d2, ...]
        let slice_reg = self.fresh_register();
        let mut slice_shape = shape.to_vec();
        slice_shape[0] = 1;
        let slice_ty = StableHLOType::f32_tensor(slice_shape.clone());
        self.body.push(format!(
            "    {} = stablehlo.slice {} [{}] : ({}) -> {}",
            slice_reg.to_mlir(),
            input.to_mlir(),
            dims_str,
            input_ty.to_mlir(),
            slice_ty.to_mlir(),
        ));

        // Reshape to remove axis 0: [1, d1, d2, ...] -> [d1, d2, ...]
        let result_shape: Vec<i64> = shape[1..].to_vec();
        if result_shape.is_empty() {
            let result_ty = StableHLOType::scalar_f32();
            let result_reg = self.fresh_register();
            self.body.push(format!(
                "    {} = stablehlo.reshape {} : ({}) -> {}",
                result_reg.to_mlir(),
                slice_reg.to_mlir(),
                slice_ty.to_mlir(),
                result_ty.to_mlir(),
            ));
            (result_reg, result_ty)
        } else {
            let result_ty = StableHLOType::f32_tensor(result_shape);
            let result_reg = self.fresh_register();
            self.body.push(format!(
                "    {} = stablehlo.reshape {} : ({}) -> {}",
                result_reg.to_mlir(),
                slice_reg.to_mlir(),
                slice_ty.to_mlir(),
                result_ty.to_mlir(),
            ));
            (result_reg, result_ty)
        }
    }

    /// Emit slice on the last axis, then squeeze that dimension if size 1.
    /// E.g. tensor<5x9xf32> slice [3:4] on last axis -> tensor<5xf32>
    /// E.g. tensor<5x9xf32> slice [2:5] on last axis -> tensor<5x3xf32>
    pub fn emit_slice_last_axis(
        &mut self,
        input: &Register,
        input_ty: &StableHLOType,
        start: i64,
        end: i64,
    ) -> (Register, StableHLOType) {
        let shape = input_ty.shape();
        let ndim = shape.len();

        let mut start_indices = vec![0i64; ndim];
        let mut limit_indices = shape.to_vec();
        let strides = vec![1i64; ndim];
        start_indices[ndim - 1] = start;
        limit_indices[ndim - 1] = end;

        let dims_str = format_slice_dims(&start_indices, &limit_indices, &strides);

        let mut slice_shape = shape.to_vec();
        slice_shape[ndim - 1] = end - start;
        let slice_ty = StableHLOType::f32_tensor(slice_shape.clone());

        let slice_reg = self.fresh_register();
        self.body.push(format!(
            "    {} = stablehlo.slice {} [{}] : ({}) -> {}",
            slice_reg.to_mlir(),
            input.to_mlir(),
            dims_str,
            input_ty.to_mlir(),
            slice_ty.to_mlir(),
        ));

        // If slice size is 1 on last axis, squeeze it
        if end - start == 1 {
            let result_shape: Vec<i64> = shape[..ndim - 1].to_vec();
            if result_shape.is_empty() {
                let result_ty = StableHLOType::scalar_f32();
                let result_reg = self.fresh_register();
                self.body.push(format!(
                    "    {} = stablehlo.reshape {} : ({}) -> {}",
                    result_reg.to_mlir(),
                    slice_reg.to_mlir(),
                    slice_ty.to_mlir(),
                    result_ty.to_mlir(),
                ));
                (result_reg, result_ty)
            } else {
                let result_ty = StableHLOType::f32_tensor(result_shape);
                let result_reg = self.fresh_register();
                self.body.push(format!(
                    "    {} = stablehlo.reshape {} : ({}) -> {}",
                    result_reg.to_mlir(),
                    slice_reg.to_mlir(),
                    slice_ty.to_mlir(),
                    result_ty.to_mlir(),
                ));
                (result_reg, result_ty)
            }
        } else {
            (slice_reg, slice_ty)
        }
    }

    /// Emit slice along axis 0: (dynamic-slice tensor start end)
    /// start is inclusive, end is inclusive (matches interpreter semantics)
    pub fn emit_slice_range(
        &mut self,
        input: &Register,
        input_ty: &StableHLOType,
        start: i64,
        end: i64,
    ) -> (Register, StableHLOType) {
        let shape = input_ty.shape();
        let ndim = shape.len();

        let mut start_indices = vec![0i64; ndim];
        let mut limit_indices = shape.to_vec();
        let strides = vec![1i64; ndim];
        start_indices[0] = start;
        limit_indices[0] = end + 1; // inclusive end

        let dims_str = format_slice_dims(&start_indices, &limit_indices, &strides);

        let mut result_shape = shape.to_vec();
        result_shape[0] = end + 1 - start;
        let result_ty = StableHLOType::f32_tensor(result_shape);

        let result_reg = self.fresh_register();
        self.body.push(format!(
            "    {} = stablehlo.slice {} [{}] : ({}) -> {}",
            result_reg.to_mlir(),
            input.to_mlir(),
            dims_str,
            input_ty.to_mlir(),
            result_ty.to_mlir(),
        ));
        (result_reg, result_ty)
    }

    /// Emit roll (circular shift): (roll tensor shift)
    /// Positive shift moves elements forward (right), wrapping around.
    /// Implemented as: concat(slice[n-shift:], slice[:n-shift])
    pub fn emit_roll(
        &mut self,
        input: &Register,
        input_ty: &StableHLOType,
        shift: i64,
    ) -> (Register, StableHLOType) {
        let shape = input_ty.shape();
        let n = shape[0];
        let shift = ((shift % n) + n) % n; // normalize to [0, n)

        if shift == 0 {
            return (*input, input_ty.clone());
        }

        // Split point: elements from [n-shift..n-1] go first, then [0..n-shift-1]
        let split = n - shift;

        // slice [split:n-1] (inclusive end)
        let (tail, tail_ty) = self.emit_slice_range(input, input_ty, split, n - 1);
        // slice [0:split-1] (inclusive end)
        let (head, _head_ty) = self.emit_slice_range(input, input_ty, 0, split - 1);

        // concat tail + head along axis 0
        self.emit_concatenate(&[tail, head], &[tail_ty.clone(), input_ty.clone()], 0)
    }

    /// Emit reverse (flip): (flip tensor [:axis N])
    /// Reverses elements along the specified axis using stablehlo.reverse.
    pub fn emit_reverse(
        &mut self,
        input: &Register,
        input_ty: &StableHLOType,
        axis: i64,
    ) -> (Register, StableHLOType) {
        let shape = input_ty.shape();
        let ndim = shape.len() as i64;
        let ax = if axis < 0 { (ndim + axis) as usize } else { axis as usize };

        let reg = self.fresh_register();
        let dims_str = format!("{}", ax);
        self.body.push(format!(
            " {} = stablehlo.reverse {}, dims = [{}] : ({}) -> {}",
            reg.to_mlir(),
            input.to_mlir(),
            dims_str,
            input_ty.to_mlir(),
            input_ty.to_mlir(),
        ));
        (reg, input_ty.clone())
    }

    /// Emit index-update: (index-update tensor idx new-value)
    /// Returns a new tensor with tensor[idx] replaced by new-value.
    /// Uses stablehlo.dynamic_update_slice.
    pub fn emit_index_update(
        &mut self,
        input: &Register,
        input_ty: &StableHLOType,
        index: i64,
        value: &Register,
        value_ty: &StableHLOType,
    ) -> (Register, StableHLOType) {
        let shape = input_ty.shape();
        let ndim = shape.len();

        // Reshape value to have leading dim of 1 for the update slice
        let mut update_shape = vec![1i64];
        update_shape.extend_from_slice(value_ty.shape());
        // If value is scalar and tensor is 1D, update_shape = [1]
        if value_ty.shape().is_empty() && ndim == 1 {
            // scalar update into 1D tensor
            let update_ty = StableHLOType::f32_tensor(vec![1]);
            let update_reg = self.fresh_register();
            self.body.push(format!(
                "    {} = stablehlo.reshape {} : ({}) -> {}",
                update_reg.to_mlir(),
                value.to_mlir(),
                value_ty.to_mlir(),
                update_ty.to_mlir(),
            ));

            let idx_reg = self.emit_constant_i32(index);
            let scalar_i32 = StableHLOType::Tensor { shape: vec![], dtype: "i32".to_string() };
            let result_reg = self.fresh_register();
            self.body.push(format!(
                "    {} = stablehlo.dynamic_update_slice {}, {}, {} : ({}, {}, {}) -> {}",
                result_reg.to_mlir(),
                input.to_mlir(),
                update_reg.to_mlir(),
                idx_reg.to_mlir(),
                input_ty.to_mlir(),
                update_ty.to_mlir(),
                scalar_i32.to_mlir(),
                input_ty.to_mlir(),
            ));
            (result_reg, input_ty.clone())
        } else {
            // N-D: value has shape [D1, D2, ...], update slice has shape [1, D1, D2, ...]
            let update_ty = StableHLOType::f32_tensor(update_shape);
            let update_reg = self.fresh_register();
            self.body.push(format!(
                "    {} = stablehlo.reshape {} : ({}) -> {}",
                update_reg.to_mlir(),
                value.to_mlir(),
                value_ty.to_mlir(),
                update_ty.to_mlir(),
            ));

            // Start indices: [index, 0, 0, ...]
            let idx_reg = self.emit_constant_i32(index);
            let zero_idx = self.emit_constant_i32(0);
            let scalar_i32 = StableHLOType::Tensor { shape: vec![], dtype: "i32".to_string() };
            let mut start_regs = vec![idx_reg.to_mlir()];
            for _ in 1..ndim {
                start_regs.push(zero_idx.to_mlir());
            }
            let start_str = start_regs.join(", ");

            let mut start_types = vec![scalar_i32.to_mlir()];
            for _ in 1..ndim {
                start_types.push(scalar_i32.to_mlir());
            }
            let start_types_str = start_types.join(", ");

            let result_reg = self.fresh_register();
            self.body.push(format!(
                "    {} = stablehlo.dynamic_update_slice {}, {}, {} : ({}, {}, {}) -> {}",
                result_reg.to_mlir(),
                input.to_mlir(),
                update_reg.to_mlir(),
                start_str,
                input_ty.to_mlir(),
                update_ty.to_mlir(),
                start_types_str,
                input_ty.to_mlir(),
            ));
            (result_reg, input_ty.clone())
        }
    }

    /// Emit slice along axis 0 with exclusive end: (slice tensor start end)
    /// start inclusive, end exclusive: matches standard Python/NumPy semantics
    pub fn emit_slice_exclusive(
        &mut self,
        input: &Register,
        input_ty: &StableHLOType,
        start: i64,
        end: i64,
    ) -> (Register, StableHLOType) {
        self.emit_slice_axis(input, input_ty, start, end, 0)
    }

    /// Emit slice along a given axis: (slice tensor start end :axis N)
    pub fn emit_slice_axis(
        &mut self,
        input: &Register,
        input_ty: &StableHLOType,
        start: i64,
        end: i64,
        axis: usize,
    ) -> (Register, StableHLOType) {
        let shape = input_ty.shape();
        let ndim = shape.len();

        let mut start_indices = vec![0i64; ndim];
        let mut limit_indices = shape.to_vec();
        let strides = vec![1i64; ndim];
        start_indices[axis] = start;
        limit_indices[axis] = end;

        let dims_str = format_slice_dims(&start_indices, &limit_indices, &strides);

        let mut result_shape = shape.to_vec();
        result_shape[axis] = end - start;
        let dtype = match input_ty {
            StableHLOType::Tensor { dtype, .. } => dtype.clone(),
            _ => "f32".to_string(),
        };
        let result_ty = StableHLOType::Tensor { shape: result_shape, dtype };

        let result_reg = self.fresh_register();
        self.body.push(format!(
            "    {} = stablehlo.slice {} [{}] : ({}) -> {}",
            result_reg.to_mlir(),
            input.to_mlir(),
            dims_str,
            input_ty.to_mlir(),
            result_ty.to_mlir(),
        ));
        (result_reg, result_ty)
    }

    /// Emit tensor-split: split tensor into N equal sections along axis 0
    /// Returns a tuple of N tensors
    pub fn emit_tensor_split(
        &mut self,
        input: &Register,
        input_ty: &StableHLOType,
        num_sections: i64,
    ) -> (Register, StableHLOType) {
        let shape = input_ty.shape();
        let total = shape[0];
        let section_size = total / num_sections;

        let mut section_regs = Vec::new();
        let mut section_types = Vec::new();

        for i in 0..num_sections {
            let start = i * section_size;
            let end = start + section_size;
            let (reg, ty) = self.emit_slice_exclusive(input, input_ty, start, end);
            section_regs.push(reg);
            section_types.push(ty);
        }

        // Pack into a tuple
        self.emit_tuple(&section_regs, &section_types)
    }

    /// Emit gather along axis 0: (get operand indices) where indices is a tensor.
    /// operand shape [N, D1, D2, ...], indices shape [I1, I2, ...]
    /// result shape [I1, I2, ..., D1, D2, ...]
    pub fn emit_gather_axis0(
        &mut self,
        operand: &Register,
        operand_ty: &StableHLOType,
        indices: &Register,
        indices_ty: &StableHLOType,
    ) -> (Register, StableHLOType) {
        let operand_shape = operand_ty.shape();
        let indices_shape = indices_ty.shape();

        // Convert indices to i32 (Sheaf tensors are f32; i32 avoids SPIRV crash)
        let indices_int_ty = StableHLOType::i32_tensor(indices_shape.to_vec());
        let indices_int_reg = self.emit_convert(indices, indices_ty, &indices_int_ty);

        // Reshape indices to add trailing index_vector_dim: [I1, I2, ...] -> [I1, I2, ..., 1]
        let mut reshaped_shape: Vec<i64> = indices_shape.to_vec();
        reshaped_shape.push(1);
        let indices_3d_reg = self.fresh_register();
        let indices_3d_ty = StableHLOType::i32_tensor(reshaped_shape.clone());
        self.body.push(format!(
            "    {} = stablehlo.reshape {} : ({}) -> {}",
            indices_3d_reg.to_mlir(),
            indices_int_reg.to_mlir(),
            indices_int_ty.to_mlir(),
            indices_3d_ty.to_mlir(),
        ));

        // Compute result shape: indices_shape + operand_shape[1:]
        let row_shape = &operand_shape[1..];
        let mut result_shape: Vec<i64> = indices_shape.to_vec();
        result_shape.extend_from_slice(row_shape);
        let result_ty = StableHLOType::f32_tensor(result_shape);

        // offset_dims = [rank(indices), rank(indices)+1, ..., rank(result)-1]
        let idx_rank = indices_shape.len();
        let offset_dims: Vec<i64> = (idx_rank..idx_rank + row_shape.len())
            .map(|d| d as i64)
            .collect();

        // slice_sizes = [1, D1, D2, ...]
        let mut slice_sizes: Vec<i64> = vec![1];
        slice_sizes.extend_from_slice(row_shape);

        let index_vector_dim = reshaped_shape.len() - 1;

        let result_reg = self.fresh_register();
        let offset_dims_str = offset_dims.iter().map(|d| d.to_string()).collect::<Vec<_>>().join(", ");
        let slice_sizes_str = slice_sizes.iter().map(|d| d.to_string()).collect::<Vec<_>>().join(", ");

        self.body.push(format!(
            "    {} = \"stablehlo.gather\"({}, {}) {{\n\
             \x20     dimension_numbers = #stablehlo.gather<\n\
             \x20       offset_dims = [{}],\n\
             \x20       collapsed_slice_dims = [0],\n\
             \x20       operand_batching_dims = [],\n\
             \x20       start_indices_batching_dims = [],\n\
             \x20       start_index_map = [0],\n\
             \x20       index_vector_dim = {}>,\n\
             \x20     slice_sizes = array<i64: {}>,\n\
             \x20     indices_are_sorted = false\n\
             \x20   }} : ({}, {}) -> {}",
            result_reg.to_mlir(),
            operand.to_mlir(),
            indices_3d_reg.to_mlir(),
            offset_dims_str,
            index_vector_dim,
            slice_sizes_str,
            operand_ty.to_mlir(),
            indices_3d_ty.to_mlir(),
            result_ty.to_mlir(),
        ));

        (result_reg, result_ty)
    }

    /// Emit top_k: sort descending + slice first K elements.
    /// Returns (values: tensor<Kxf32>, indices: tensor<Kxf32>).
    pub fn emit_top_k(
        &mut self,
        input: &Register,
        input_ty: &StableHLOType,
        k: i64,
    ) -> (Register, StableHLOType) {
        let shape = input_ty.shape();
        assert!(!shape.is_empty(), "top_k requires at least 1D input");
        let last_axis = shape.len() - 1;

        // Create iota indices [0, 1, ..., N-1] as i32 (required by chlo.top_k pattern)
        let iota_reg = self.fresh_register();
        let iota_ty = StableHLOType::i32_tensor(shape.to_vec());
        self.body.push(format!(
            "    {} = stablehlo.iota dim = {} : {}",
            iota_reg.to_mlir(),
            last_axis,
            iota_ty.to_mlir()
        ));

        // Sort descending along last axis: returns (sorted_values, sorted_indices)
        let sorted_vals = self.fresh_register();
        let sorted_idxs = self.fresh_register();

        // Use fresh registers for comparator block args to avoid name collisions
        let cmp_lhs = self.fresh_register();
        let cmp_rhs = self.fresh_register();
        let cmp_lhs_idx = self.fresh_register();
        let cmp_rhs_idx = self.fresh_register();
        let cmp_pred = self.fresh_register();

        self.body.push(format!(
            "    {}, {} = \"stablehlo.sort\"({}, {}) ({{\n\
             \x20   ^bb0({}: tensor<f32>, {}: tensor<f32>, {}: tensor<i32>, {}: tensor<i32>):\n\
             \x20     {} = \"stablehlo.compare\"({}, {}) {{comparison_direction = #stablehlo<comparison_direction GT>}} : (tensor<f32>, tensor<f32>) -> tensor<i1>\n\
             \x20     \"stablehlo.return\"({}) : (tensor<i1>) -> ()\n\
             \x20   }}) {{dimension = {} : i64, is_stable = true}} : ({}, {}) -> ({}, {})",
            sorted_vals.to_mlir(),
            sorted_idxs.to_mlir(),
            input.to_mlir(),
            iota_reg.to_mlir(),
            cmp_lhs.to_mlir(),
            cmp_rhs.to_mlir(),
            cmp_lhs_idx.to_mlir(),
            cmp_rhs_idx.to_mlir(),
            cmp_pred.to_mlir(),
            cmp_lhs.to_mlir(),
            cmp_rhs.to_mlir(),
            cmp_pred.to_mlir(),
            last_axis,
            input_ty.to_mlir(),
            iota_ty.to_mlir(),
            input_ty.to_mlir(),
            iota_ty.to_mlir(),
        ));

        // Slice first K elements along last axis
        let (top_vals, top_vals_ty) = self.emit_slice_axis(
            &sorted_vals, input_ty, 0, k, last_axis,
        );
        let (top_idxs_i32, top_idxs_i32_ty) = self.emit_slice_axis(
            &sorted_idxs, &iota_ty, 0, k, last_axis,
        );

        // Convert indices i32 -> f32 for f32-only codegen consistency
        let top_idxs_f32_ty = StableHLOType::f32_tensor(top_idxs_i32_ty.shape().to_vec());
        let top_idxs = self.emit_convert(&top_idxs_i32, &top_idxs_i32_ty, &top_idxs_f32_ty);

        // Pack into virtual tuple
        self.emit_tuple(
            &[top_vals, top_idxs],
            &[top_vals_ty, top_idxs_f32_ty],
        )
    }
}

/// Format slice dimensions as `start:limit:stride, ...` for StableHLO assembly.
pub(super) fn format_slice_dims(starts: &[i64], limits: &[i64], strides: &[i64]) -> String {
    starts
        .iter()
        .zip(limits.iter())
        .zip(strides.iter())
        .map(|((s, l), st)| format!("{}:{}:{}", s, l, st))
        .collect::<Vec<_>>()
        .join(", ")
}
