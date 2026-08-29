// Copyright (c) 2026 Damien Boureille
// Licensed under the MIT License.

//! Binary and unary operation emission.

use super::{Register, StableHLOEmitter, StableHLOType};

impl StableHLOEmitter {
    pub fn emit_binop(
        &mut self,
        op: &str,
        lhs: &Register,
        rhs: &Register,
        lhs_ty: &StableHLOType,
        rhs_ty: &StableHLOType,
    ) -> (Register, StableHLOType) {
        let stablehlo_op = match op {
            "+" => "stablehlo.add",
            "-" => "stablehlo.subtract",
            "*" => "stablehlo.multiply",
            "/" => "stablehlo.divide",
            "**" => "stablehlo.power",
            "//" => "stablehlo.floor_divide",
            "%" | "mod" => "stablehlo.remainder",
            "min" => "stablehlo.minimum",
            "max" => "stablehlo.maximum",
            _ => panic!("Unsupported binop: {}", op),
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

        if let (Some(lv), Some(rv)) = (
            self.known_scalars.get(&actual_lhs).copied(),
            self.known_scalars.get(&actual_rhs).copied(),
        ) {
            let result_val = match op {
                "+" => Some(lv + rv),
                "-" => Some(lv - rv),
                "*" => Some(lv * rv),
                "/" if rv != 0.0 => Some(lv / rv),
                "min" => Some(lv.min(rv)),
                "max" => Some(lv.max(rv)),
                _ => None,
            };
            if let Some(v) = result_val {
                self.known_scalars.insert(reg, v);
            }
        }

        (reg, result_ty)
    }

    pub fn emit_unary(&mut self, op: &str, operand: &Register, ty: &StableHLOType) -> Register {
        let reg = self.fresh_register();

        match op {
            "tanh" => {
                self.body.push(format!(
                    "    {} = stablehlo.tanh {} : {}",
                    reg.to_mlir(),
                    operand.to_mlir(),
                    ty.to_mlir()
                ));
            }
            "sqrt" => {
                self.body.push(format!(
                    "    {} = stablehlo.sqrt {} : {}",
                    reg.to_mlir(),
                    operand.to_mlir(),
                    ty.to_mlir()
                ));
            }
            "exp" => {
                self.body.push(format!(
                    "    {} = stablehlo.exponential {} : {}",
                    reg.to_mlir(),
                    operand.to_mlir(),
                    ty.to_mlir()
                ));
            }
            "log" => {
                self.body.push(format!(
                    "    {} = stablehlo.log {} : {}",
                    reg.to_mlir(),
                    operand.to_mlir(),
                    ty.to_mlir()
                ));
            }
            "not" => {
                self.body.push(format!(
                    "    {} = stablehlo.not {} : {}",
                    reg.to_mlir(),
                    operand.to_mlir(),
                    ty.to_mlir()
                ));
            }
            "abs" => {
                self.body.push(format!(
                    "    {} = stablehlo.abs {} : {}",
                    reg.to_mlir(),
                    operand.to_mlir(),
                    ty.to_mlir()
                ));
            }
            "neg" => {
                self.body.push(format!(
                    "    {} = stablehlo.negate {} : {}",
                    reg.to_mlir(),
                    operand.to_mlir(),
                    ty.to_mlir()
                ));
            }
            "sin" => {
                self.body.push(format!(
                    "    {} = stablehlo.sine {} : {}",
                    reg.to_mlir(),
                    operand.to_mlir(),
                    ty.to_mlir()
                ));
            }
            "cos" => {
                self.body.push(format!(
                    "    {} = stablehlo.cosine {} : {}",
                    reg.to_mlir(),
                    operand.to_mlir(),
                    ty.to_mlir()
                ));
            }
            _ => panic!("Unsupported unary op: {}", op),
        }

        reg
    }

    pub fn emit_convert(&mut self, reg: &Register, from_ty: &StableHLOType, to_ty: &StableHLOType) -> Register {
        let result = self.fresh_register();
        self.body.push(format!(
            "    {} = stablehlo.convert {} : ({}) -> {}",
            result.to_mlir(),
            reg.to_mlir(),
            from_ty.to_mlir(),
            to_ty.to_mlir(),
        ));
        result
    }
}
