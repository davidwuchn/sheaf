// Copyright (c) 2026 Damien Boureille
// Licensed under the MIT License.

//! Comparison, selection, and boolean operations for StableHLO.

use super::{Register, StableHLOEmitter, StableHLOType};

impl StableHLOEmitter {
    /// Emit a comparison operation
    /// Returns a tensor of i1 (boolean) values
    pub fn emit_compare(
        &mut self,
        op: &str,
        lhs: &Register,
        rhs: &Register,
        lhs_ty: &StableHLOType,
        rhs_ty: &StableHLOType,
    ) -> (Register, StableHLOType) {
        let comparison_direction = match op {
            "=" | "==" => "EQ",
            "!=" => "NE",
            "<" => "LT",
            "<=" => "LE",
            ">" => "GT",
            ">=" => "GE",
            _ => panic!("Unsupported comparison: {}", op),
        };

        // Determine result shape (broadcast if needed)
        let operand_ty = self.broadcast_types(lhs_ty, rhs_ty);

        // Check if we need to broadcast operands
        let (actual_lhs, actual_rhs) =
            self.maybe_broadcast_operands(lhs, rhs, lhs_ty, rhs_ty, &operand_ty);

        // Result type: same shape as operands but with i1 dtype
        let result_ty = if operand_ty.shape().is_empty() {
            // Scalar comparison returns scalar i1
            StableHLOType::ScalarI1
        } else {
            // Tensor comparison returns tensor of same shape with i1 elements
            StableHLOType::i1_tensor(operand_ty.shape())
        };

        let reg = self.fresh_register();
        // Use generic form with attributes in braces
        self.body.push(format!(
            "    {} = \"stablehlo.compare\"({}, {}) {{",
            reg.to_mlir(),
            actual_lhs.to_mlir(),
            actual_rhs.to_mlir()
        ));
        self.body.push(format!(
            "      comparison_direction = #stablehlo<comparison_direction {}>,",
            comparison_direction
        ));
        self.body.push(format!(
            "      compare_type = #stablehlo<comparison_type FLOAT>"
        ));
        self.body.push(format!(
            "    }} : ({}, {}) -> {}",
            operand_ty.to_mlir(),
            operand_ty.to_mlir(),
            result_ty.to_mlir()
        ));

        (reg, result_ty)
    }

    /// Emit select operation (conditional): select(pred, on_true, on_false)
    pub fn emit_select(
        &mut self,
        pred: &Register,
        on_true: &Register,
        on_false: &Register,
        pred_ty: &StableHLOType,
        on_true_ty: &StableHLOType,
        _on_false_ty: &StableHLOType,
    ) -> (Register, StableHLOType) {
        // Result type is the type of the branches (assume they match)
        let result_ty = on_true_ty.clone();

        let reg = self.fresh_register();
        self.body.push(format!(
            "    {} = stablehlo.select {}, {}, {} : {}, {}",
            reg.to_mlir(),
            pred.to_mlir(),
            on_true.to_mlir(),
            on_false.to_mlir(),
            pred_ty.to_mlir(),
            result_ty.to_mlir()
        ));

        (reg, result_ty)
    }

    /// Emit boolean binary operation (and, or)
    pub fn emit_bool_binop(
        &mut self,
        op: &str,
        lhs: &Register,
        rhs: &Register,
        lhs_ty: &StableHLOType,
        rhs_ty: &StableHLOType,
    ) -> (Register, StableHLOType) {
        let stablehlo_op = match op {
            "and" => "stablehlo.and",
            "or" => "stablehlo.or",
            _ => panic!("Unsupported boolean binop: {}", op),
        };

        // Determine result type (broadcast if needed)
        let result_ty = self.broadcast_types(lhs_ty, rhs_ty);

        // Check if we need to broadcast operands
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
