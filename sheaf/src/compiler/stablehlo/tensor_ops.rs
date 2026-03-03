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
    /// Selects elements from x when condition is true, from y when false
    pub fn emit_where(
        &mut self,
        condition: &Register,
        x: &Register,
        y: &Register,
        condition_ty: &StableHLOType,
        x_ty: &StableHLOType,
        y_ty: &StableHLOType,
    ) -> (Register, StableHLOType) {
        // Use stablehlo.select: select(pred, on_true, on_false)
        self.emit_select(condition, x, y, condition_ty, x_ty, y_ty)
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
        let rank = operand_shape.len();

        // Build permutation that swaps axis1 and axis2
        let mut permutation: Vec<i64> = (0..rank as i64).collect();
        permutation[axis1 as usize] = axis2;
        permutation[axis2 as usize] = axis1;

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

        let start_str = start.iter().map(|v| v.to_string()).collect::<Vec<_>>().join(", ");
        let limit_str = limit.iter().map(|v| v.to_string()).collect::<Vec<_>>().join(", ");
        let strides_str = strides.iter().map(|v| v.to_string()).collect::<Vec<_>>().join(", ");

        // Slice to [1, d1, d2, ...]
        let slice_reg = self.fresh_register();
        let mut slice_shape = shape.to_vec();
        slice_shape[0] = 1;
        let slice_ty = StableHLOType::f32_tensor(slice_shape.clone());
        self.body.push(format!(
            "    {} = stablehlo.slice {} [{}] to [{}] step [{}] : ({}) -> {}",
            slice_reg.to_mlir(),
            input.to_mlir(),
            start_str,
            limit_str,
            strides_str,
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

        let start_str = start_indices.iter().map(|v| v.to_string()).collect::<Vec<_>>().join(", ");
        let limit_str = limit_indices.iter().map(|v| v.to_string()).collect::<Vec<_>>().join(", ");
        let strides_str = strides.iter().map(|v| v.to_string()).collect::<Vec<_>>().join(", ");

        let mut slice_shape = shape.to_vec();
        slice_shape[ndim - 1] = end - start;
        let slice_ty = StableHLOType::f32_tensor(slice_shape.clone());

        let slice_reg = self.fresh_register();
        self.body.push(format!(
            "    {} = stablehlo.slice {} [{}] to [{}] step [{}] : ({}) -> {}",
            slice_reg.to_mlir(),
            input.to_mlir(),
            start_str,
            limit_str,
            strides_str,
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
}
