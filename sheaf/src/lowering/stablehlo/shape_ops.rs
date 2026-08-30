// Copyright (c) 2026 Damien Boureille
// Licensed under the MIT License.

//! Shape operations.

use super::{Register, StableHLOEmitter, StableHLOType};

impl StableHLOEmitter {
    pub fn emit_reshape(
        &mut self,
        operand: &Register,
        operand_ty: &StableHLOType,
        new_shape: &[i64],
    ) -> (Register, StableHLOType) {
        let reg = self.fresh_register();
        let result_ty = StableHLOType::typed_tensor(new_shape.to_vec(), operand_ty.dtype());

        self.body.push(format!(
            "    {} = stablehlo.reshape {} : ({}) -> {}",
            reg.to_mlir(),
            operand.to_mlir(),
            operand_ty.to_mlir(),
            result_ty.to_mlir()
        ));

        (reg, result_ty)
    }

    pub fn emit_transpose(
        &mut self,
        operand: &Register,
        operand_ty: &StableHLOType,
        permutation: &[i64],
    ) -> (Register, StableHLOType) {
        let reg = self.fresh_register();

        let operand_shape = operand_ty.shape();

        if operand_shape.len() <= 1 {
            return (*operand, operand_ty.clone());
        }

        let result_shape: Vec<i64> = permutation
            .iter()
            .map(|&i| operand_shape[i as usize])
            .collect();
        let result_ty = StableHLOType::tensor(
            result_shape,
            operand_ty.element_type().unwrap(),
        );

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

    pub fn emit_concatenate(
        &mut self,
        operands: &[Register],
        operand_types: &[StableHLOType],
        dimension: i64,
    ) -> (Register, StableHLOType) {
        let reg = self.fresh_register();

        let first_shape = operand_types[0].shape();
        let mut result_shape = first_shape.to_vec();

        let concat_dim_size: i64 = operand_types
            .iter()
            .map(|ty| ty.shape()[dimension as usize])
            .sum();

        result_shape[dimension as usize] = concat_dim_size;
        let result_ty = StableHLOType::tensor(
            result_shape,
            operand_types[0].element_type().unwrap(),
        );

        let operands_str = operands
            .iter()
            .map(|r| r.to_mlir())
            .collect::<Vec<_>>()
            .join(", ");

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

    pub fn emit_swapaxes(
        &mut self,
        operand: &Register,
        operand_ty: &StableHLOType,
        axis1: i64,
        axis2: i64,
    ) -> (Register, StableHLOType) {
        let operand_shape = operand_ty.shape();
        let rank = operand_shape.len() as i64;

        let a1 = if axis1 < 0 { (rank + axis1) as usize } else { axis1 as usize };
        let a2 = if axis2 < 0 { (rank + axis2) as usize } else { axis2 as usize };

        let mut permutation: Vec<i64> = (0..rank).collect();
        permutation[a1] = a2 as i64;
        permutation[a2] = a1 as i64;

        self.emit_transpose(operand, operand_ty, &permutation)
    }
}
