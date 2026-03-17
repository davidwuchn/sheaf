// Copyright (c) 2025 Damien Boureille
// Licensed under the MIT License.

//! Math, comparison, and boolean builtin codegen.

use crate::lowering::stablehlo::{Register, StableHLOType};
use crate::core::expr::CompiledExpr;
use crate::core::error::SheafResult;
use crate::runtime::{math_ops, nn_ops};
use super::CodeGenerator;

impl CodeGenerator {
    pub(super) fn generate_math_builtin(
        &mut self,
        name: &str,
        args: &[CompiledExpr],
    ) -> Option<SheafResult<(Register, StableHLOType)>> {
        // Arithmetic operations (binary or n-ary fold-left)
        if matches!(name, "+" | "-" | "*" | "/") && args.len() >= 2 {
            Some(self.gen_arithmetic(name, args))
        }
        // Extended arithmetic: **, //, mod
        else if matches!(name, "**" | "//" | "%" | "mod") && args.len() == 2 {
            Some(self.gen_extended_arithmetic(name, args))
        }
        // Min/max operations (binary)
        else if matches!(name, "min" | "max") && args.len() == 2 {
            Some(self.gen_minmax_binary(name, args))
        }
        // Comparison operations
        else if matches!(name, "=" | "==" | "!=" | "<" | "<=" | ">" | ">=") && args.len() == 2 {
            Some(self.gen_comparison(name, args))
        }
        // Matrix multiply
        else if name == "@" && args.len() == 2 {
            Some(self.gen_matmul(args))
        }
        // Boolean binary operations
        else if matches!(name, "and" | "or") && args.len() == 2 {
            Some(self.gen_boolean_binop(name, args))
        }
        // Math unary operations: sqrt, exp, log, abs
        else if matches!(name, "sqrt" | "exp" | "log" | "abs") && args.len() == 1 {
            Some(self.gen_math_unary(name, args))
        }
        // Boolean not
        else if name == "not" && args.len() == 1 {
            Some(self.gen_not(args))
        }
        // tanh
        else if name == "tanh" && args.len() == 1 {
            Some(self.gen_tanh(args))
        }
        // minimum/maximum: element-wise min/max
        else if matches!(name, "minimum" | "maximum") && args.len() == 2 {
            Some(self.gen_minimum_maximum(name, args))
        }
        else {
            None
        }
    }

    fn gen_arithmetic(
        &mut self,
        name: &str,
        args: &[CompiledExpr],
    ) -> SheafResult<(Register, StableHLOType)> {
        let (mut acc_reg, mut acc_ty) = self.generate(&args[0])?;
        for arg in &args[1..] {
            let (rhs_reg, rhs_ty) = self.generate(arg)?;
            (acc_reg, acc_ty) = math_ops::emit_arithmetic_binop(
                &mut self.emitter,
                name,
                &acc_reg,
                &rhs_reg,
                &acc_ty,
                &rhs_ty,
            );
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
        let (result_reg, result_ty) = math_ops::emit_extended_arithmetic(
            &mut self.emitter,
            name,
            &lhs_reg,
            &rhs_reg,
            &lhs_ty,
            &rhs_ty,
        );
        Ok((result_reg, result_ty))
    }

    fn gen_minmax_binary(
        &mut self,
        name: &str,
        args: &[CompiledExpr],
    ) -> SheafResult<(Register, StableHLOType)> {
        let (lhs_reg, lhs_ty) = self.generate(&args[0])?;
        let (rhs_reg, rhs_ty) = self.generate(&args[1])?;
        let (result_reg, result_ty) = math_ops::emit_minmax(
            &mut self.emitter,
            name,
            &lhs_reg,
            &rhs_reg,
            &lhs_ty,
            &rhs_ty,
        );
        Ok((result_reg, result_ty))
    }

    fn gen_comparison(
        &mut self,
        name: &str,
        args: &[CompiledExpr],
    ) -> SheafResult<(Register, StableHLOType)> {
        let (lhs_reg, lhs_ty) = self.generate(&args[0])?;
        let (rhs_reg, rhs_ty) = self.generate(&args[1])?;
        let (result_reg, result_ty) = math_ops::emit_comparison(
            &mut self.emitter,
            name,
            &lhs_reg,
            &rhs_reg,
            &lhs_ty,
            &rhs_ty,
        );
        Ok((result_reg, result_ty))
    }

    fn gen_matmul(
        &mut self,
        args: &[CompiledExpr],
    ) -> SheafResult<(Register, StableHLOType)> {
        let (lhs_reg, lhs_ty) = self.generate(&args[0])?;
        let (rhs_reg, rhs_ty) = self.generate(&args[1])?;
        let (result_reg, result_ty) =
            math_ops::emit_matmul(&mut self.emitter, &lhs_reg, &rhs_reg, &lhs_ty, &rhs_ty);
        Ok((result_reg, result_ty))
    }

    fn gen_boolean_binop(
        &mut self,
        name: &str,
        args: &[CompiledExpr],
    ) -> SheafResult<(Register, StableHLOType)> {
        let (lhs_reg, lhs_ty) = self.generate(&args[0])?;
        let (rhs_reg, rhs_ty) = self.generate(&args[1])?;
        let (result_reg, result_ty) = math_ops::emit_boolean_binop(
            &mut self.emitter,
            name,
            &lhs_reg,
            &rhs_reg,
            &lhs_ty,
            &rhs_ty,
        );
        Ok((result_reg, result_ty))
    }

    fn gen_math_unary(
        &mut self,
        name: &str,
        args: &[CompiledExpr],
    ) -> SheafResult<(Register, StableHLOType)> {
        let (operand_reg, operand_ty) = self.generate(&args[0])?;
        let result_reg =
            math_ops::emit_math_unary(&mut self.emitter, name, &operand_reg, &operand_ty);
        Ok((result_reg, operand_ty))
    }

    fn gen_not(
        &mut self,
        args: &[CompiledExpr],
    ) -> SheafResult<(Register, StableHLOType)> {
        let (operand_reg, operand_ty) = self.generate(&args[0])?;
        let result_reg = math_ops::emit_not(&mut self.emitter, &operand_reg, &operand_ty);
        Ok((result_reg, operand_ty))
    }

    fn gen_tanh(
        &mut self,
        args: &[CompiledExpr],
    ) -> SheafResult<(Register, StableHLOType)> {
        let (operand_reg, operand_ty) = self.generate(&args[0])?;
        let result_reg = nn_ops::emit_tanh(&mut self.emitter, &operand_reg, &operand_ty);
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
        let (result_reg, result_ty) = math_ops::emit_minmax(
            &mut self.emitter, op, &lhs_reg, &rhs_reg, &lhs_ty, &rhs_ty,
        );
        Ok((result_reg, result_ty))
    }
}
