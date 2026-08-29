// Copyright (c) 2025 Damien Boureille
// Licensed under the MIT License.

//! Math, comparison, and boolean builtin codegen.

use crate::lowering::stablehlo::{Register, StableHLOType};
use crate::core::dtype::{DtypeOperand, resolve_arithmetic_dtype};
use crate::core::expr::CompiledExpr;
use crate::core::error::{SheafError, SheafResult};
use super::CodeGenerator;

impl<'a> CodeGenerator<'a> {
    pub(super) fn generate_math_builtin(
        &mut self,
        name: &str,
        args: &[CompiledExpr],
    ) -> Option<SheafResult<(Register, StableHLOType)>> {
        if name == "+" && args.len() >= 2 {
            Some(self.gen_add(args))
        }
        else if matches!(name, "-" | "*" | "/") && args.len() >= 2 {
            Some(self.gen_arithmetic(name, args))
        }
        else if matches!(name, "**" | "//" | "%" | "mod") && args.len() == 2 {
            Some(self.gen_extended_arithmetic(name, args))
        }
        else if matches!(name, "min" | "max") && args.len() == 2 {
            Some(self.gen_minmax_binary(name, args))
        }
        else if matches!(name, "=" | "==" | "!=" | "<" | "<=" | ">" | ">=") && args.len() == 2 {
            Some(self.gen_comparison(name, args))
        }
        else if name == "@" && args.len() == 2 {
            Some(self.gen_matmul(args))
        }
        else if matches!(name, "and" | "or") && args.len() == 2 {
            Some(self.gen_boolean_binop(name, args))
        }
        else if name == "-" && args.len() == 1 {
            Some(self.gen_math_unary("neg", args))
        }
        else if matches!(name, "sqrt" | "exp" | "log" | "abs" | "neg" | "sin" | "cos") && args.len() == 1 {
            Some(self.gen_math_unary(name, args))
        }
        else if name == "tan" && args.len() == 1 {
            Some(self.gen_tan(args))
        }
        else if name == "not" && args.len() == 1 {
            Some(self.gen_not(args))
        }
        else if name == "tanh" && args.len() == 1 {
            Some(self.gen_tanh(args))
        }
        else if matches!(name, "minimum" | "maximum") && args.len() == 2 {
            Some(self.gen_minimum_maximum(name, args))
        }
        else {
            None
        }
    }

    fn gen_add(
        &mut self,
        args: &[CompiledExpr],
    ) -> SheafResult<(Register, StableHLOType)> {
        let (mut acc_reg, mut acc_ty) = self.generate(&args[0])?;
        let mut acc_is_weak = self.weak_scalars.contains(&acc_reg);

        for arg in &args[1..] {
            let (mut rhs_reg, mut rhs_ty) = self.generate(arg)?;
            let rhs_is_weak = self.weak_scalars.contains(&rhs_reg);
            let lhs_dtype = acc_ty.element_type().ok_or_else(add_type_error)?;
            let rhs_dtype = rhs_ty.element_type().ok_or_else(add_type_error)?;
            let dtype = resolve_arithmetic_dtype(
                if acc_is_weak {
                    DtypeOperand::weak(lhs_dtype)
                } else {
                    DtypeOperand::strong(lhs_dtype)
                },
                if rhs_is_weak {
                    DtypeOperand::weak(rhs_dtype)
                } else {
                    DtypeOperand::strong(rhs_dtype)
                },
            )
            .map_err(|error| SheafError::Compile {
                message: format!("+: {}", error),
                location: crate::core::error::SourceLocation::unknown(),
            })?;

            if lhs_dtype != dtype {
                let target = acc_ty.with_element_type(dtype).ok_or_else(add_type_error)?;
                acc_reg = self.emitter.emit_convert(&acc_reg, &acc_ty, &target);
                acc_ty = target;
            }
            if rhs_dtype != dtype {
                let target = rhs_ty.with_element_type(dtype).ok_or_else(add_type_error)?;
                rhs_reg = self.emitter.emit_convert(&rhs_reg, &rhs_ty, &target);
                rhs_ty = target;
            }

            (acc_reg, acc_ty) =
                self.emitter.emit_binop("+", &acc_reg, &rhs_reg, &acc_ty, &rhs_ty);
            acc_is_weak &= rhs_is_weak;
            if acc_is_weak {
                self.weak_scalars.insert(acc_reg);
            }
        }
        Ok((acc_reg, acc_ty))
    }

    fn gen_arithmetic(
        &mut self,
        name: &str,
        args: &[CompiledExpr],
    ) -> SheafResult<(Register, StableHLOType)> {
        let (mut acc_reg, mut acc_ty) = self.generate(&args[0])?;
        for arg in &args[1..] {
            let (rhs_reg, rhs_ty) = self.generate(arg)?;
            (acc_reg, acc_ty) = self.emitter.emit_binop(name, &acc_reg, &rhs_reg, &acc_ty, &rhs_ty);
        }
        Ok((acc_reg, acc_ty))
    }

    fn gen_extended_arithmetic(
        &mut self,
        name: &str,
        args: &[CompiledExpr],
    ) -> SheafResult<(Register, StableHLOType)> {
        let (lhs_reg, lhs_ty) = self.generate(&args[0])?;
        let (rhs_reg, rhs_ty) = self.generate(&args[1])?;
        let (result_reg, result_ty) = self.emitter.emit_binop(name, &lhs_reg, &rhs_reg, &lhs_ty, &rhs_ty);
        Ok((result_reg, result_ty))
    }

    fn gen_minmax_binary(
        &mut self,
        name: &str,
        args: &[CompiledExpr],
    ) -> SheafResult<(Register, StableHLOType)> {
        let (lhs_reg, lhs_ty) = self.generate(&args[0])?;
        let (rhs_reg, rhs_ty) = self.generate(&args[1])?;
        let (result_reg, result_ty) = self.emitter.emit_binop(name, &lhs_reg, &rhs_reg, &lhs_ty, &rhs_ty);
        Ok((result_reg, result_ty))
    }

    fn gen_comparison(
        &mut self,
        name: &str,
        args: &[CompiledExpr],
    ) -> SheafResult<(Register, StableHLOType)> {
        let (lhs_reg, lhs_ty) = self.generate(&args[0])?;
        let (rhs_reg, rhs_ty) = self.generate(&args[1])?;
        let (result_reg, result_ty) = self.emitter.emit_compare(name, &lhs_reg, &rhs_reg, &lhs_ty, &rhs_ty);
        Ok((result_reg, result_ty))
    }

    fn gen_matmul(
        &mut self,
        args: &[CompiledExpr],
    ) -> SheafResult<(Register, StableHLOType)> {
        let (lhs_reg, lhs_ty) = self.generate(&args[0])?;
        let (rhs_reg, rhs_ty) = self.generate(&args[1])?;
        let (result_reg, result_ty) =
            self.emitter.emit_matmul(&lhs_reg, &rhs_reg, &lhs_ty, &rhs_ty);
        Ok((result_reg, result_ty))
    }

    fn gen_boolean_binop(
        &mut self,
        name: &str,
        args: &[CompiledExpr],
    ) -> SheafResult<(Register, StableHLOType)> {
        let (lhs_reg, lhs_ty) = self.generate(&args[0])?;
        let (rhs_reg, rhs_ty) = self.generate(&args[1])?;
        let (result_reg, result_ty) = self.emitter.emit_bool_binop(name, &lhs_reg, &rhs_reg, &lhs_ty, &rhs_ty);
        Ok((result_reg, result_ty))
    }

    fn gen_math_unary(
        &mut self,
        name: &str,
        args: &[CompiledExpr],
    ) -> SheafResult<(Register, StableHLOType)> {
        let (operand_reg, operand_ty) = self.generate(&args[0])?;
        let result_reg = self.emitter.emit_unary(name, &operand_reg, &operand_ty);
        Ok((result_reg, operand_ty))
    }

    fn gen_tan(&mut self, args: &[CompiledExpr]) -> SheafResult<(Register, StableHLOType)> {
        // stablehlo.tangent is not supported by IREE 3.10, so emit tan = sin / cos.
        let (operand_reg, operand_ty) = self.generate(&args[0])?;
        let sin_reg = self.emitter.emit_unary("sin", &operand_reg, &operand_ty);
        let cos_reg = self.emitter.emit_unary("cos", &operand_reg, &operand_ty);
        let (result_reg, _) = self.emitter.emit_binop("/", &sin_reg, &cos_reg, &operand_ty, &operand_ty);
        Ok((result_reg, operand_ty))
    }

    fn gen_not(
        &mut self,
        args: &[CompiledExpr],
    ) -> SheafResult<(Register, StableHLOType)> {
        let (operand_reg, operand_ty) = self.generate(&args[0])?;
        let one = self.emitter.emit_constant_f32(1.0);
        let one_ty = StableHLOType::ScalarF32;
        let (result_reg, _) = self.emitter.emit_binop("-", &one, &operand_reg, &one_ty, &operand_ty);
        Ok((result_reg, operand_ty))
    }

    fn gen_tanh(
        &mut self,
        args: &[CompiledExpr],
    ) -> SheafResult<(Register, StableHLOType)> {
        let (operand_reg, operand_ty) = self.generate(&args[0])?;
        let result_reg = self.emitter.emit_unary("tanh", &operand_reg, &operand_ty);
        Ok((result_reg, operand_ty))
    }

    fn gen_minimum_maximum(
        &mut self,
        name: &str,
        args: &[CompiledExpr],
    ) -> SheafResult<(Register, StableHLOType)> {
        let (lhs_reg, lhs_ty) = self.generate(&args[0])?;
        let (rhs_reg, rhs_ty) = self.generate(&args[1])?;
        let op = if name == "minimum" { "min" } else { "max" };
        let (result_reg, result_ty) = self.emitter.emit_binop(op, &lhs_reg, &rhs_reg, &lhs_ty, &rhs_ty);
        Ok((result_reg, result_ty))
    }
}

fn add_type_error() -> SheafError {
    SheafError::Compile {
        message: "+: expected tensor or scalar operands".to_string(),
        location: crate::core::error::SourceLocation::unknown(),
    }
}
