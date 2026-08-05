// Copyright (c) 2026 Damien Boureille
// Licensed under the MIT License.

//! Shape manipulation operations for StableHLO (reshape, transpose, concat, etc.).

use super::{Register, StableHLOEmitter, StableHLOType};

impl StableHLOEmitter {
    /// Emit reshape: (reshape tensor [M N]) -> tensor<MxNxf32>
    pub fn emit_reshape(
        &mut self,
        operand: &Register,
        operand_ty: &StableHLOType,
        new_shape: &[i64],
    ) -> (Register, StableHLOType) {
        let reg = self.fresh_register();
        // Preserve dtype from operand
        let dtype = match operand_ty {
            StableHLOType::Tensor { dtype, .. } => dtype.clone(),
            _ => "f32".to_string(),
        };
        let result_ty = StableHLOType::Tensor { shape: new_shape.to_vec(), dtype };

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
            return (*operand, operand_ty.clone());
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
}
