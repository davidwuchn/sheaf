// Copyright (c) 2026 Damien Boureille
// Licensed under the MIT License.

//! Comparison, selection, and boolean operations for StableHLO.

use super::{Register, StableHLOEmitter, StableHLOType};

impl StableHLOEmitter {
    pub fn emit_compare(
        &mut self,
        op: &str,
        lhs: &Register,
        rhs: &Register,
        lhs_ty: &StableHLOType,
        rhs_ty: &StableHLOType,
    ) -> (Register, StableHLOType) {
        let operand_ty = self.broadcast_types(lhs_ty, rhs_ty);
        let (a, b) = self.maybe_broadcast_operands(lhs, rhs, lhs_ty, rhs_ty, &operand_ty);
        let ty = &operand_ty;

        let dtype = ty.element_type().expect("comparison operand must have a dtype");
        let (zero, scalar_ty) = self.emit_typed_splat_constant(0.0, &[], dtype);
        let zero_bc = self.emit_broadcast(&zero, &scalar_ty, ty);
        let (one, scalar_ty) = self.emit_typed_splat_constant(1.0, &[], dtype);
        let one_bc = self.emit_broadcast(&one, &scalar_ty, ty);

        let result = match op {
            ">=" | ">" | "<=" | "<" => {
                let (first, second) = match op {
                    ">=" | ">" => (&a, &b),
                    _ => (&b, &a),
                };
                let diff = self.fresh_register();
                self.body.push(format!(
                    "    {} = stablehlo.subtract {}, {} : {}",
                    diff.to_mlir(), first.to_mlir(), second.to_mlir(), ty.to_mlir()
                ));
                let sgn = self.fresh_register();
                self.body.push(format!(
                    "    {} = stablehlo.sign {} : {}",
                    sgn.to_mlir(), diff.to_mlir(), ty.to_mlir()
                ));

                let needs_plus_one = matches!(op, ">=" | "<=");

                if needs_plus_one {
                    let shifted = self.fresh_register();
                    self.body.push(format!(
                        "    {} = stablehlo.add {}, {} : {}",
                        shifted.to_mlir(), sgn.to_mlir(), one_bc.to_mlir(), ty.to_mlir()
                    ));
                    let c = self.fresh_register();
                    self.body.push(format!(
                        "    {} = stablehlo.clamp {}, {}, {} : {}",
                        c.to_mlir(), zero_bc.to_mlir(), shifted.to_mlir(), one_bc.to_mlir(), ty.to_mlir()
                    ));
                    c
                } else {
                    let c = self.fresh_register();
                    self.body.push(format!(
                        "    {} = stablehlo.clamp {}, {}, {} : {}",
                        c.to_mlir(), zero_bc.to_mlir(), sgn.to_mlir(), one_bc.to_mlir(), ty.to_mlir()
                    ));
                    c
                }
            }
            "=" | "==" => {
                // 1 - min(abs(a-b), 1)
                let diff = self.fresh_register();
                self.body.push(format!(
                    "    {} = stablehlo.subtract {}, {} : {}",
                    diff.to_mlir(), a.to_mlir(), b.to_mlir(), ty.to_mlir()
                ));
                let adiff = self.fresh_register();
                self.body.push(format!(
                    "    {} = stablehlo.abs {} : {}",
                    adiff.to_mlir(), diff.to_mlir(), ty.to_mlir()
                ));
                let capped = self.fresh_register();
                self.body.push(format!(
                    "    {} = stablehlo.minimum {}, {} : {}",
                    capped.to_mlir(), adiff.to_mlir(), one_bc.to_mlir(), ty.to_mlir()
                ));
                let result = self.fresh_register();
                self.body.push(format!(
                    "    {} = stablehlo.subtract {}, {} : {}",
                    result.to_mlir(), one_bc.to_mlir(), capped.to_mlir(), ty.to_mlir()
                ));
                result
            }
            "!=" => {
                // min(abs(a-b), 1)
                let diff = self.fresh_register();
                self.body.push(format!(
                    "    {} = stablehlo.subtract {}, {} : {}",
                    diff.to_mlir(), a.to_mlir(), b.to_mlir(), ty.to_mlir()
                ));
                let adiff = self.fresh_register();
                self.body.push(format!(
                    "    {} = stablehlo.abs {} : {}",
                    adiff.to_mlir(), diff.to_mlir(), ty.to_mlir()
                ));
                let capped = self.fresh_register();
                self.body.push(format!(
                    "    {} = stablehlo.minimum {}, {} : {}",
                    capped.to_mlir(), adiff.to_mlir(), one_bc.to_mlir(), ty.to_mlir()
                ));
                capped
            }
            _ => panic!("Unsupported comparison: {}", op),
        };

        (result, operand_ty)
    }

    pub fn emit_select(
        &mut self,
        pred: &Register,
        on_true: &Register,
        on_false: &Register,
        pred_ty: &StableHLOType,
        on_true_ty: &StableHLOType,
        _on_false_ty: &StableHLOType,
    ) -> (Register, StableHLOType) {
        let result_ty = on_true_ty.clone();

        let dtype = pred_ty.element_type().expect("predicate must have a dtype");
        let (zero, zero_ty) = self.emit_typed_splat_constant(0.0, &[], dtype);
        let zero_bc = self.emit_broadcast(&zero, &zero_ty, pred_ty);

        let pred_i1_ty = StableHLOType::i1_tensor(pred_ty.shape().to_vec());
        let pred_i1 = self.fresh_register();
        self.body.push(format!(
            "    {} = stablehlo.compare NE, {}, {}, FLOAT : ({}, {}) -> {}",
            pred_i1.to_mlir(),
            pred.to_mlir(),
            zero_bc.to_mlir(),
            pred_ty.to_mlir(),
            pred_ty.to_mlir(),
            pred_i1_ty.to_mlir(),
        ));

        let reg = self.fresh_register();
        self.body.push(format!(
            "    {} = stablehlo.select {}, {}, {} : {}, {}",
            reg.to_mlir(),
            pred_i1.to_mlir(),
            on_true.to_mlir(),
            on_false.to_mlir(),
            pred_i1_ty.to_mlir(),
            result_ty.to_mlir(),
        ));

        (reg, result_ty)
    }

    pub fn emit_bool_binop(
        &mut self,
        op: &str,
        lhs: &Register,
        rhs: &Register,
        lhs_ty: &StableHLOType,
        rhs_ty: &StableHLOType,
    ) -> (Register, StableHLOType) {
        let stablehlo_op = match op {
            "and" => "stablehlo.multiply",
            "or" => "stablehlo.maximum",
            _ => panic!("Unsupported boolean binop: {}", op),
        };

        let result_ty = self.broadcast_types(lhs_ty, rhs_ty);
        let (actual_lhs, actual_rhs) =
            self.maybe_broadcast_operands(lhs, rhs, lhs_ty, rhs_ty, &result_ty);

        let reg = self.fresh_register();
        self.body.push(format!(
            "    {} = {} {}, {} : {}",
            reg.to_mlir(),
            stablehlo_op,
            actual_lhs.to_mlir(),
            actual_rhs.to_mlir(),
            result_ty.to_mlir()
        ));
        (reg, result_ty)
    }
}
