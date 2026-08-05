// Copyright (c) 2026 Damien Boureille
// Licensed under the MIT License.

//! Broadcasting: shape inference and broadcast_in_dim emission.

use super::{Register, StableHLOEmitter, StableHLOType};

impl StableHLOEmitter {
    /// Maybe broadcast operands to match result shape
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

    /// Emit broadcast_in_dim to convert from_ty to to_ty
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
            // Try numpy-style right-alignment first:
            // [4] -> [4, 4] maps to dims=[1], [1, 256] -> [4, 256, 384] maps to dims=[1, 2]
            let offset = to_shape.len() - from_shape.len();
            let right_aligned: Vec<usize> = (offset..to_shape.len()).collect();
            let valid = from_shape.iter().enumerate().all(|(i, &src)| {
                src == to_shape[right_aligned[i]] || src == 1
            });
            if valid {
                right_aligned
            } else {
                // Fallback: greedy left-to-right size matching
                // [1024] -> [1024, 65] maps to dims=[0]
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
                    // Last resort: right-align anyway (let StableHLO report the error)
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

    /// Broadcast types: choose result type for binary op.
    /// Preserves dtype: if both sides are bf16, result is bf16.
    /// If one side is f32 and the other bf16, result is f32 (widening).
    pub(crate) fn broadcast_types(&self, lhs: &StableHLOType, rhs: &StableHLOType) -> StableHLOType {
        let lhs_shape = lhs.shape();
        let rhs_shape = rhs.shape();

        if lhs_shape.is_empty() && rhs_shape.is_empty() {
            return lhs.clone();
        }
        if lhs_shape.is_empty() {
            return rhs.clone();
        }
        if rhs_shape.is_empty() {
            return lhs.clone();
        }

        // Resolve result dtype: both bf16 -> bf16, mixed -> f32
        let result_dtype = if lhs.dtype() == "bf16" && rhs.dtype() == "bf16" {
            "bf16"
        } else {
            "f32"
        };

        // Numpy-style broadcasting: align from trailing dims, take max of each
        let max_ndim = lhs_shape.len().max(rhs_shape.len());
        let mut result_shape = Vec::with_capacity(max_ndim);
        for i in 0..max_ndim {
            let l = if i < max_ndim - lhs_shape.len() {
                1
            } else {
                lhs_shape[i - (max_ndim - lhs_shape.len())]
            };
            let r = if i < max_ndim - rhs_shape.len() {
                1
            } else {
                rhs_shape[i - (max_ndim - rhs_shape.len())]
            };
            result_shape.push(l.max(r));
        }
        StableHLOType::typed_tensor(result_shape, result_dtype)
    }
}
