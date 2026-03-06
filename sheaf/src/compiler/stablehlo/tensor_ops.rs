// Copyright (c) 2026 Damien Boureille
// Licensed under the MIT License.

//! Tensor creation and manipulation operations for StableHLO.

use super::{Register, StableHLOEmitter, StableHLOType};

impl StableHLOEmitter {
    /// Emit zeros tensor: (zeros [M N]) -> tensor<MxNxf32>
    pub fn emit_zeros(&mut self, shape: &[i64]) -> (Register, StableHLOType) {
        let reg = self.fresh_register();
        let ty = StableHLOType::f32_tensor(shape.to_vec());

        self.body.push(format!(
            "    {} = stablehlo.constant dense<0.0> : {}",
            reg.to_mlir(),
            ty.to_mlir()
        ));

        (reg, ty)
    }

    /// Emit random-normal tensor: (random-normal key [M N])
    /// For now, we emit a constant with small values (placeholder)
    /// TODO: Proper RNG with seed/key
    pub fn emit_random_normal(&mut self, shape: &[i64]) -> (Register, StableHLOType) {
        let reg = self.fresh_register();
        let ty = StableHLOType::f32_tensor(shape.to_vec());

        // Placeholder: emit constant with 0.01 (will need proper RNG later)
        self.body.push(format!(
            "    {} = stablehlo.constant dense<0.01> : {}",
            reg.to_mlir(),
            ty.to_mlir()
        ));

        (reg, ty)
    }

    /// Emit ones tensor: (ones [M N]) -> tensor<MxNxf32>
    pub fn emit_ones(&mut self, shape: &[i64]) -> (Register, StableHLOType) {
        let reg = self.fresh_register();
        let ty = StableHLOType::f32_tensor(shape.to_vec());

        self.body.push(format!(
            "    {} = stablehlo.constant dense<1.0> : {}",
            reg.to_mlir(),
            ty.to_mlir()
        ));

        (reg, ty)
    }

    /// Emit reshape: (reshape tensor [M N]) -> tensor<MxNxf32>
    pub fn emit_reshape(
        &mut self,
        operand: &Register,
        operand_ty: &StableHLOType,
        new_shape: &[i64],
    ) -> (Register, StableHLOType) {
        let reg = self.fresh_register();
        let result_ty = StableHLOType::f32_tensor(new_shape.to_vec());

        self.body.push(format!(
            "    {} = stablehlo.reshape {} : ({}) -> {}",
            reg.to_mlir(),
            operand.to_mlir(),
            operand_ty.to_mlir(),
            result_ty.to_mlir()
        ));

        (reg, result_ty)
    }

    /// Emit transpose: (transpose tensor [1 0]) -> permutes dimensions
    pub fn emit_transpose(
        &mut self,
        operand: &Register,
        operand_ty: &StableHLOType,
        permutation: &[i64],
    ) -> (Register, StableHLOType) {
        let reg = self.fresh_register();

        // Compute result shape by applying permutation
        let operand_shape = operand_ty.shape();

        // Transpose of a scalar or 1D is identity
        if operand_shape.len() <= 1 {
            return (operand.clone(), operand_ty.clone());
        }

        let result_shape: Vec<i64> = permutation
            .iter()
            .map(|&i| operand_shape[i as usize])
            .collect();

        let result_ty = StableHLOType::f32_tensor(result_shape);

        // Format permutation as [0, 1, 2]
        let perm_str = permutation
            .iter()
            .map(|d| d.to_string())
            .collect::<Vec<_>>()
            .join(", ");

        self.body.push(format!(
            "    {} = stablehlo.transpose {}, dims = [{}] : ({}) -> {}",
            reg.to_mlir(),
            operand.to_mlir(),
            perm_str,
            operand_ty.to_mlir(),
            result_ty.to_mlir()
        ));

        (reg, result_ty)
    }

    /// Emit iota (arange): (arange N) -> tensor<Nxf32> with values [0, 1, 2, ..., N-1]
    pub fn emit_iota(&mut self, shape: &[i64], dimension: i64) -> (Register, StableHLOType) {
        let reg = self.fresh_register();
        let ty = StableHLOType::f32_tensor(shape.to_vec());

        self.body.push(format!(
            "    {} = stablehlo.iota dim = {} : {}",
            reg.to_mlir(),
            dimension,
            ty.to_mlir()
        ));

        (reg, ty)
    }

    /// Emit concatenate: (concat [tensor1 tensor2 ...] axis)
    pub fn emit_concatenate(
        &mut self,
        operands: &[Register],
        operand_types: &[StableHLOType],
        dimension: i64,
    ) -> (Register, StableHLOType) {
        let reg = self.fresh_register();

        // Compute result shape: same as first operand except for concat dimension
        let first_shape = operand_types[0].shape();
        let mut result_shape = first_shape.to_vec();

        // Sum the sizes along the concatenation dimension
        let concat_dim_size: i64 = operand_types
            .iter()
            .map(|ty| ty.shape()[dimension as usize])
            .sum();

        result_shape[dimension as usize] = concat_dim_size;
        let result_ty = StableHLOType::f32_tensor(result_shape);

        // Format operands as %0, %1, %2
        let operands_str = operands
            .iter()
            .map(|r| r.to_mlir())
            .collect::<Vec<_>>()
            .join(", ");

        // Format types as (tensor<2x3xf32>, tensor<2x3xf32>)
        let types_str = operand_types
            .iter()
            .map(|ty| ty.to_mlir())
            .collect::<Vec<_>>()
            .join(", ");

        self.body.push(format!(
            "    {} = stablehlo.concatenate {}, dim = {} : ({}) -> {}",
            reg.to_mlir(),
            operands_str,
            dimension,
            types_str,
            result_ty.to_mlir()
        ));

        (reg, result_ty)
    }

    /// Emit where (conditional selection): (where condition x y)
    /// Selects elements from x when condition is true, from y when false.
    /// Broadcasts condition and y to match x's shape (NumPy semantics).
    pub fn emit_where(
        &mut self,
        condition: &Register,
        x: &Register,
        y: &Register,
        condition_ty: &StableHLOType,
        x_ty: &StableHLOType,
        y_ty: &StableHLOType,
    ) -> (Register, StableHLOType) {
        let result_shape = x_ty.shape().to_vec();

        // Broadcast condition to i1 tensor matching result shape if needed
        let pred_shape = condition_ty.shape();
        let (actual_cond, actual_cond_ty) = if pred_shape != result_shape.as_slice() {
            let target = StableHLOType::i1_tensor(result_shape.clone());
            let r = self.emit_broadcast(condition, condition_ty, &target);
            (r, target)
        } else {
            (condition.clone(), condition_ty.clone())
        };

        // Broadcast y (on_false) to result shape if needed
        let y_shape = y_ty.shape();
        let (actual_y, actual_y_ty) = if y_shape != result_shape.as_slice() {
            let target = StableHLOType::f32_tensor(result_shape);
            let r = self.emit_broadcast(y, y_ty, &target);
            (r, target)
        } else {
            (y.clone(), y_ty.clone())
        };

        self.emit_select(&actual_cond, x, &actual_y, &actual_cond_ty, x_ty, &actual_y_ty)
    }

    /// Emit swapaxes: (swapaxes x axis1 axis2)
    /// Interchanges two axes using transpose
    pub fn emit_swapaxes(
        &mut self,
        operand: &Register,
        operand_ty: &StableHLOType,
        axis1: i64,
        axis2: i64,
    ) -> (Register, StableHLOType) {
        let operand_shape = operand_ty.shape();
        let rank = operand_shape.len() as i64;

        // Normalize negative indices
        let a1 = if axis1 < 0 { (rank + axis1) as usize } else { axis1 as usize };
        let a2 = if axis2 < 0 { (rank + axis2) as usize } else { axis2 as usize };

        // Build permutation that swaps axis1 and axis2
        let mut permutation: Vec<i64> = (0..rank).collect();
        permutation[a1] = a2 as i64;
        permutation[a2] = a1 as i64;

        // Use transpose with the permutation
        self.emit_transpose(operand, operand_ty, &permutation)
    }

    /// Emit tril (lower triangular): (tril x)
    /// Returns the lower triangular part of a matrix, zeros above diagonal
    pub fn emit_tril(
        &mut self,
        operand: &Register,
        operand_ty: &StableHLOType,
    ) -> (Register, StableHLOType) {
        let reg = self.fresh_register();
        let result_ty = operand_ty.clone();

        let shape = operand_ty.shape();
        if shape.len() != 2 {
            panic!("tril requires a 2D tensor");
        }

        let m = shape[0];
        let n = shape[1];

        // Create row indices: iota dim=0
        let row_iota = self.fresh_register();
        let row_iota_ty = StableHLOType::f32_tensor(vec![m, n]);
        self.body.push(format!(
            "    {} = stablehlo.iota dim = 0 : {}",
            row_iota.to_mlir(),
            row_iota_ty.to_mlir()
        ));

        // Create col indices: iota dim=1
        let col_iota = self.fresh_register();
        let col_iota_ty = StableHLOType::f32_tensor(vec![m, n]);
        self.body.push(format!(
            "    {} = stablehlo.iota dim = 1 : {}",
            col_iota.to_mlir(),
            col_iota_ty.to_mlir()
        ));

        // Compare: row_idx >= col_idx (i >= j for lower triangle)
        let mask = self.fresh_register();
        let mask_ty = StableHLOType::i1_tensor(vec![m, n]);
        self.body.push(format!(
            "    {} = \"stablehlo.compare\"({}, {}) {{",
            mask.to_mlir(),
            row_iota.to_mlir(),
            col_iota.to_mlir()
        ));
        self.body
            .push("      comparison_direction = #stablehlo<comparison_direction GE>,".to_string());
        self.body
            .push("      compare_type = #stablehlo<comparison_type FLOAT>".to_string());
        self.body.push(format!(
            "    }} : ({}, {}) -> {}",
            row_iota_ty.to_mlir(),
            col_iota_ty.to_mlir(),
            mask_ty.to_mlir()
        ));

        // Create zero tensor for the false branch
        let zero_reg = self.fresh_register();
        self.body.push(format!(
            "    {} = stablehlo.constant dense<0.0> : {}",
            zero_reg.to_mlir(),
            result_ty.to_mlir()
        ));

        // Select: where mask is true, use operand, else use zero
        self.body.push(format!(
            "    {} = stablehlo.select {}, {}, {} : {}, {}",
            reg.to_mlir(),
            mask.to_mlir(),
            operand.to_mlir(),
            zero_reg.to_mlir(),
            mask_ty.to_mlir(),
            result_ty.to_mlir()
        ));

        (reg, result_ty)
    }

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
    /// E.g. tensor<5x9xf32> slice [3:4] on last axis → tensor<5xf32>
    /// E.g. tensor<5x9xf32> slice [2:5] on last axis → tensor<5x3xf32>
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

    /// Emit identity matrix: (eye N) or (eye N M)
    /// Strategy: iota(dim=0) == iota(dim=1) → select(mask, 1.0, 0.0)
    pub fn emit_eye(&mut self, n: i64, m: i64) -> (Register, StableHLOType) {
        let shape = vec![n, m];
        let result_ty = StableHLOType::f32_tensor(shape.clone());

        // Row indices: [0,0,...; 1,1,...; 2,2,...] shape [N,M]
        let (row_iota, _) = self.emit_iota(&shape, 0);
        // Col indices: [0,1,2...; 0,1,2...; ...] shape [N,M]
        let (col_iota, iota_ty) = self.emit_iota(&shape, 1);

        // Compare row == col → bool mask [N, M]
        let (mask_reg, mask_ty) = self.emit_compare("==", &row_iota, &col_iota, &iota_ty, &iota_ty);

        // ones and zeros tensors
        let one_scalar = self.emit_constant_f32(1.0);
        let zero_scalar = self.emit_constant_f32(0.0);
        let ones_reg = self.emit_broadcast(&one_scalar, &StableHLOType::scalar_f32(), &result_ty);
        let zeros_reg = self.emit_broadcast(&zero_scalar, &StableHLOType::scalar_f32(), &result_ty);

        // select(mask, 1.0, 0.0)
        self.emit_select(&mask_reg, &ones_reg, &zeros_reg, &mask_ty, &result_ty, &result_ty)
    }

    /// Emit one-hot encoding: (one-hot indices num_classes)
    /// indices: tensor<Nxf32> (integer values as f32), num_classes: static int
    /// Returns tensor<NxCxf32>
    pub fn emit_one_hot(
        &mut self,
        indices: &Register,
        indices_ty: &StableHLOType,
        num_classes: i64,
    ) -> (Register, StableHLOType) {
        let indices_shape = indices_ty.shape();

        if indices_shape.is_empty() {
            // Scalar index → output [C]
            let out_shape = vec![num_classes];
            let out_ty = StableHLOType::f32_tensor(out_shape.clone());

            // iota [C] along dim 0
            let (class_iota, iota_ty) = self.emit_iota(&out_shape, 0);

            // Broadcast scalar index to [C]
            let idx_broadcast = self.emit_broadcast(indices, indices_ty, &iota_ty);

            // Compare indices == iota
            let (mask_reg, mask_ty) = self.emit_compare("==", &idx_broadcast, &class_iota, &iota_ty, &iota_ty);

            let one_scalar = self.emit_constant_f32(1.0);
            let zero_scalar = self.emit_constant_f32(0.0);
            let ones_reg = self.emit_broadcast(&one_scalar, &StableHLOType::scalar_f32(), &out_ty);
            let zeros_reg = self.emit_broadcast(&zero_scalar, &StableHLOType::scalar_f32(), &out_ty);

            self.emit_select(&mask_reg, &ones_reg, &zeros_reg, &mask_ty, &out_ty, &out_ty)
        } else {
            // Tensor indices [N] → output [N, C]
            let n = indices_shape[0];
            let out_shape = vec![n, num_classes];
            let out_ty = StableHLOType::f32_tensor(out_shape.clone());

            // iota [N, C] along dim 1 → class indices
            let (class_iota, iota_ty) = self.emit_iota(&out_shape, 1);

            // Reshape indices [N] → [N, 1] then broadcast to [N, C]
            let idx_2d_shape = vec![n, 1];
            let idx_2d_ty = StableHLOType::f32_tensor(idx_2d_shape);
            let idx_2d = self.fresh_register();
            self.body.push(format!(
                "    {} = stablehlo.reshape {} : ({}) -> {}",
                idx_2d.to_mlir(),
                indices.to_mlir(),
                indices_ty.to_mlir(),
                idx_2d_ty.to_mlir(),
            ));
            let idx_broadcast = self.emit_broadcast(&idx_2d, &idx_2d_ty, &iota_ty);

            // Compare indices == iota
            let (mask_reg, mask_ty) = self.emit_compare("==", &idx_broadcast, &class_iota, &iota_ty, &iota_ty);

            let one_scalar = self.emit_constant_f32(1.0);
            let zero_scalar = self.emit_constant_f32(0.0);
            let ones_reg = self.emit_broadcast(&one_scalar, &StableHLOType::scalar_f32(), &out_ty);
            let zeros_reg = self.emit_broadcast(&zero_scalar, &StableHLOType::scalar_f32(), &out_ty);

            self.emit_select(&mask_reg, &ones_reg, &zeros_reg, &mask_ty, &out_ty, &out_ty)
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
            return (input.clone(), input_ty.clone());
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
        update_shape.extend_from_slice(&value_ty.shape());
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

            let idx_reg = self.emit_constant_i64(index);
            let result_reg = self.fresh_register();
            self.body.push(format!(
                "    {} = stablehlo.dynamic_update_slice {}, {}, {} : ({}, {}, {}) -> {}",
                result_reg.to_mlir(),
                input.to_mlir(),
                update_reg.to_mlir(),
                idx_reg.to_mlir(),
                input_ty.to_mlir(),
                update_ty.to_mlir(),
                StableHLOType::ScalarI64.to_mlir(),
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
            let idx_reg = self.emit_constant_i64(index);
            let zero_idx = self.emit_constant_i64(0);
            let mut start_regs = vec![idx_reg.to_mlir()];
            for _ in 1..ndim {
                start_regs.push(zero_idx.to_mlir());
            }
            let start_str = start_regs.join(", ");

            let mut start_types = vec![StableHLOType::ScalarI64.to_mlir()];
            for _ in 1..ndim {
                start_types.push(StableHLOType::ScalarI64.to_mlir());
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
    /// start inclusive, end exclusive — matches standard Python/NumPy semantics
    pub fn emit_slice_exclusive(
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
        limit_indices[0] = end;

        let dims_str = format_slice_dims(&start_indices, &limit_indices, &strides);

        let mut result_shape = shape.to_vec();
        result_shape[0] = end - start;
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
}

/// Format slice dimensions as `start:limit:stride, ...` for StableHLO assembly.
fn format_slice_dims(starts: &[i64], limits: &[i64], strides: &[i64]) -> String {
    starts
        .iter()
        .zip(limits.iter())
        .zip(strides.iter())
        .map(|((s, l), st)| format!("{}:{}:{}", s, l, st))
        .collect::<Vec<_>>()
        .join(", ")
}
