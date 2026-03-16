// Copyright (c) 2026 Damien Boureille
// Licensed under the MIT License.

//! Comparison, selection, and boolean operations for StableHLO.

use super::{Register, StableHLOEmitter, StableHLOType};

impl StableHLOEmitter {
    /// Emit a comparison operation using pure f32 arithmetic.
    /// Returns f32 tensor (0.0/1.0). Zero i1 types — avoids IREE SPIRV crash.
    ///
    /// GE: clamp(0, sign(a-b) + 1, 1)    GT: clamp(0, sign(a-b), 1)
    /// LE: clamp(0, sign(b-a) + 1, 1)    LT: clamp(0, sign(b-a), 1)
    /// EQ: 1 - min(abs(a-b), 1)           NE: min(abs(a-b), 1)
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

        let zero = self.emit_constant_f32(0.0);
        let zero_ty = StableHLOType::ScalarF32;
        let zero_bc = self.emit_broadcast(&zero, &zero_ty, ty);
        let one = self.emit_constant_f32(1.0);
        let one_bc = self.emit_broadcast(&one, &zero_ty, ty);

        let result = match op {
            ">=" | ">" | "<=" | "<" => {
                // diff = first - second (swap for LE/LT)
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
                let clamped = if needs_plus_one {
                    // clamp(0, sign(diff) + 1, 1)
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
                    // clamp(0, sign(diff), 1)
                    let c = self.fresh_register();
                    self.body.push(format!(
                        "    {} = stablehlo.clamp {}, {}, {} : {}",
                        c.to_mlir(), zero_bc.to_mlir(), sgn.to_mlir(), one_bc.to_mlir(), ty.to_mlir()
                    ));
                    c
                };
                clamped
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

    /// Emit select using native stablehlo.select with i1 predicate.
    /// Pred is f32 (0.0/1.0), converted to i1 via compare NE 0.0.
    ///
    /// Previous f32 arithmetic emulation (pred*a + (1-pred)*b) was buggy
    /// on IREE Metal SPIRV when broadcast dims=[2,3] combined with multiply
    /// corrupted values for tensors with batch dims > 1 (e.g. 1x6x256x256).
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

        // Convert f32 pred (0.0/1.0) to i1: compare NE pred, 0.0
        let zero = self.emit_constant_f32(0.0);
        let zero_ty = StableHLOType::ScalarF32;
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

        // Native select
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

    /// Emit boolean binary operation (and, or)
    /// Operands are f32 (0.0/1.0): AND = multiply, OR = maximum.
    pub fn emit_bool_binop(
        &mut self,
        op: &str,
        lhs: &Register,
        rhs: &Register,
        lhs_ty: &StableHLOType,
        rhs_ty: &StableHLOType,
    ) -> (Register, StableHLOType) {
        // f32 boolean ops: AND = multiply, OR = maximum
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
