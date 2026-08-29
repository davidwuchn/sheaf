// Copyright (c) 2026 Damien Boureille
// Licensed under the MIT License.

//! Broadcasting: shape inference and broadcast_in_dim emission.

use super::{Register, StableHLOEmitter, StableHLOType};

impl StableHLOEmitter {
    pub(crate) fn maybe_broadcast_operands(
        &mut self,
        lhs: &Register,
        rhs: &Register,
        lhs_ty: &StableHLOType,
        rhs_ty: &StableHLOType,
        result_ty: &StableHLOType,
    ) -> (Register, Register) {
        let lhs_shape = lhs_ty.shape();
        let rhs_shape = rhs_ty.shape();
        let result_shape = result_ty.shape();

        let actual_lhs = if lhs_shape != result_shape && !result_shape.is_empty() {
            self.emit_broadcast(lhs, lhs_ty, result_ty)
        } else {
            *lhs
        };

        let actual_rhs = if rhs_shape != result_shape && !result_shape.is_empty() {
            self.emit_broadcast(rhs, rhs_ty, result_ty)
        } else {
            *rhs
        };

        (actual_lhs, actual_rhs)
    }

    pub fn emit_broadcast(
        &mut self,
        operand: &Register,
        from_ty: &StableHLOType,
        to_ty: &StableHLOType,
    ) -> Register {
        let from_shape = from_ty.shape();
        let to_shape = to_ty.shape();

        let dims = if from_shape.is_empty() {
            vec![]
        } else {
            let offset = to_shape.len() - from_shape.len();
            let right_aligned: Vec<usize> = (offset..to_shape.len()).collect();
            let valid = from_shape.iter().enumerate().all(|(i, &src)| {
                src == to_shape[right_aligned[i]] || src == 1
            });
            if valid {
                right_aligned
            } else {
                let mut mapping = Vec::with_capacity(from_shape.len());
                let mut search_start = 0;
                for &src_dim in from_shape {
                    for (j, &target_dim) in to_shape.iter().enumerate().skip(search_start) {
                        if target_dim == src_dim || src_dim == 1 {
                            mapping.push(j);
                            search_start = j + 1;
                            break;
                        }
                    }
                }
                if mapping.len() != from_shape.len() {
                    right_aligned
                } else {
                    mapping
                }
            }
        };

        let reg = self.fresh_register();
        if dims.is_empty() {
            self.body.push(format!(
                "    {} = stablehlo.broadcast_in_dim {}, dims = [] : ({}) -> {}",
                reg.to_mlir(),
                operand.to_mlir(),
                from_ty.to_mlir(),
                to_ty.to_mlir()
            ));
        } else {
            let dims_str = dims
                .iter()
                .map(|d| d.to_string())
                .collect::<Vec<_>>()
                .join(", ");
            self.body.push(format!(
                "    {} = stablehlo.broadcast_in_dim {}, dims = [{}] : ({}) -> {}",
                reg.to_mlir(),
                operand.to_mlir(),
                dims_str,
                from_ty.to_mlir(),
                to_ty.to_mlir()
            ));
        }

        reg
    }

    pub(crate) fn broadcast_types(&self, lhs: &StableHLOType, rhs: &StableHLOType) -> StableHLOType {
        let lhs_shape = lhs.shape();
        let rhs_shape = rhs.shape();

        if lhs_shape.is_empty() && rhs_shape.is_empty() {
            return match (lhs, rhs) {
                (StableHLOType::Tensor { .. }, _) => lhs.clone(),
                (_, StableHLOType::Tensor { .. }) => rhs.clone(),
                _ => lhs.clone(),
            };
        }
        if lhs_shape.is_empty() {
            return rhs.clone();
        }
        if rhs_shape.is_empty() {
            return lhs.clone();
        }

        let result_dtype = match (lhs.element_type(), rhs.element_type()) {
            (Some(lhs), Some(rhs)) if lhs == rhs => lhs,
            _ => crate::core::dtype::ElementType::F32,
        };
        let result_shape = crate::core::shape::broadcast_shapes(lhs_shape, rhs_shape)
            .expect("binary operand shapes must be validated before StableHLO lowering");
        StableHLOType::tensor(result_shape, result_dtype)
    }
}
