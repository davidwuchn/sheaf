// Copyright (c) 2026 Damien Boureille
// Licensed under the MIT License.

//! Tensor creation, selection, and broadcast operations for StableHLO.

use super::{Register, StableHLOEmitter, StableHLOType};

/// Compute the broadcast result shape from up to 3 input shapes.
/// Follows numpy broadcasting rules: align from the right, each dim is max.
fn broadcast_result_shape(a: &[i64], b: &[i64], c: &[i64]) -> Vec<i64> {
    let ab = broadcast_two(a, b);
    broadcast_two(&ab, c)
}

fn broadcast_two(a: &[i64], b: &[i64]) -> Vec<i64> {
    let len = a.len().max(b.len());
    let mut result = vec![1i64; len];
    for i in 0..len {
        let da = if i < len - a.len() { 1 } else { a[i - (len - a.len())] };
        let db = if i < len - b.len() { 1 } else { b[i - (len - b.len())] };
        result[i] = da.max(db);
    }
    result
}

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
        // Result shape = broadcast of x, y, and condition shapes
        let x_shape = x_ty.shape();
        let y_shape = y_ty.shape();
        let cond_shape = condition_ty.shape();
        let result_shape = broadcast_result_shape(x_shape, y_shape, cond_shape);

        // Broadcast condition in f32 (comparisons return f32 0.0/1.0)
        let (actual_cond, actual_cond_ty) = if cond_shape != result_shape.as_slice() {
            let target = StableHLOType::f32_tensor(result_shape.clone());
            let r = self.emit_broadcast(condition, condition_ty, &target);
            (r, target)
        } else {
            (condition.clone(), condition_ty.clone())
        };

        // Broadcast x (on_true) to result shape if needed
        let (actual_x, actual_x_ty) = if x_shape != result_shape.as_slice() {
            let target = StableHLOType::f32_tensor(result_shape.clone());
            let r = self.emit_broadcast(x, x_ty, &target);
            (r, target)
        } else {
            (x.clone(), x_ty.clone())
        };

        // Broadcast y (on_false) to result shape if needed
        let (actual_y, actual_y_ty) = if y_shape != result_shape.as_slice() {
            let target = StableHLOType::f32_tensor(result_shape);
            let r = self.emit_broadcast(y, y_ty, &target);
            (r, target)
        } else {
            (y.clone(), y_ty.clone())
        };

        self.emit_select(&actual_cond, &actual_x, &actual_y, &actual_cond_ty, &actual_x_ty, &actual_y_ty)
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
        // emit_compare returns f32 (0.0/1.0)
        let (mask, _mask_ty) = self.emit_compare(">=", &row_iota, &col_iota, &row_iota_ty, &col_iota_ty);

        // Multiply: operand * mask_f32 (zeros out upper triangle)
        self.body.push(format!(
            "    {} = stablehlo.multiply {}, {} : {}",
            reg.to_mlir(),
            operand.to_mlir(),
            mask.to_mlir(),
            result_ty.to_mlir()
        ));

        (reg, result_ty)
    }

    /// Emit identity matrix: (eye N) or (eye N M)
    /// Strategy: iota(dim=0) == iota(dim=1) -> select(mask, 1.0, 0.0)
    pub fn emit_eye(&mut self, n: i64, m: i64) -> (Register, StableHLOType) {
        let shape = vec![n, m];
        let result_ty = StableHLOType::f32_tensor(shape.clone());

        // Row indices: [0,0,...; 1,1,...; 2,2,...] shape [N,M]
        let (row_iota, _) = self.emit_iota(&shape, 0);
        // Col indices: [0,1,2...; 0,1,2...; ...] shape [N,M]
        let (col_iota, iota_ty) = self.emit_iota(&shape, 1);

        // Compare row == col -> bool mask [N, M]
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
            // Scalar index -> output [C]
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
            // Tensor indices [...] -> output [..., C]
            let mut out_shape: Vec<i64> = indices_shape.to_vec();
            out_shape.push(num_classes);
            let out_ty = StableHLOType::f32_tensor(out_shape.clone());

            // iota [..., C] along last dim -> class indices
            let last_dim = (out_shape.len() - 1) as i64;
            let (class_iota, iota_ty) = self.emit_iota(&out_shape, last_dim);

            // Reshape indices [...] -> [..., 1] then broadcast to [..., C]
            let mut idx_expanded_shape: Vec<i64> = indices_shape.to_vec();
            idx_expanded_shape.push(1);
            let idx_2d_ty = StableHLOType::f32_tensor(idx_expanded_shape);
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
}
