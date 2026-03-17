// Copyright (c) 2025 Damien Boureille
// Licensed under the MIT License.

//! Reduction builtin codegen: sum, mean, product, min/max reduce, argmax/argmin, var, normalize.

use crate::lowering::stablehlo::{Register, StableHLOType};
use crate::core::expr::CompiledExpr;
use crate::core::error::SheafResult;
use crate::runtime::tensor_ops;
use super::CodeGenerator;

impl CodeGenerator {
    pub(super) fn generate_reduction_builtin(
        &mut self,
        name: &str,
        args: &[CompiledExpr],
    ) -> Option<SheafResult<(Register, StableHLOType)>> {
        match name {
            "sum" | "mean" if !args.is_empty() => Some(self.gen_sum_mean(name, args)),
            "product" if !args.is_empty() => Some(self.gen_product(args)),
            "min" | "max"
                if !args.is_empty()
                    && (args.len() == 1
                        || matches!(&args[1], CompiledExpr::Keyword(_))) =>
            {
                Some(self.gen_minmax_reduce(name, args))
            }
            "argmax" | "argmin" if !args.is_empty() => Some(self.gen_argmax_argmin(name, args)),
            "var" if !args.is_empty() => Some(self.gen_var(args)),
            "normalize" if !args.is_empty() => Some(self.gen_normalize(args)),
            _ => None,
        }
    }

    fn gen_sum_mean(
        &mut self,
        name: &str,
        args: &[CompiledExpr],
    ) -> SheafResult<(Register, StableHLOType)> {
        let (operand_reg, operand_ty) = self.generate(&args[0])?;

        let mut axis: Option<i64> = None;
        let mut keepdims = false;
        let mut i = 1;
        while i < args.len() {
            match &args[i] {
                CompiledExpr::Keyword(k) if k == "axis" => {
                    if i + 1 < args.len() {
                        if let CompiledExpr::Integer(n) = &args[i + 1] {
                            axis = Some(*n);
                            i += 2;
                            continue;
                        }
                    }
                    i += 1;
                }
                CompiledExpr::Keyword(k) if k == "keepdims" => {
                    if i + 1 < args.len() {
                        if let CompiledExpr::Boolean(b) = &args[i + 1] {
                            keepdims = *b;
                            i += 2;
                            continue;
                        }
                    }
                    keepdims = true;
                    i += 1;
                }
                _ => { i += 1; }
            }
        }

        let (reg, ty) = match axis {
            Some(ax) => {
                if name == "sum" {
                    tensor_ops::emit_sum(&mut self.emitter, &operand_reg, &operand_ty, ax, keepdims)
                } else {
                    tensor_ops::emit_mean(&mut self.emitter, &operand_reg, &operand_ty, ax, keepdims)
                }
            }
            None => {
                let ndim = operand_ty.shape().len();
                if ndim == 0 {
                    (operand_reg, operand_ty)
                } else {
                    let mut cur_reg = operand_reg;
                    let mut cur_ty = operand_ty;
                    for _ in (0..ndim).rev() {
                        let (r, t) = if name == "sum" {
                            tensor_ops::emit_sum(&mut self.emitter, &cur_reg, &cur_ty, -1, false)
                        } else {
                            tensor_ops::emit_mean(&mut self.emitter, &cur_reg, &cur_ty, -1, false)
                        };
                        cur_reg = r;
                        cur_ty = t;
                    }
                    (cur_reg, cur_ty)
                }
            }
        };
        Ok((reg, ty))
    }

    fn gen_product(
        &mut self,
        args: &[CompiledExpr],
    ) -> SheafResult<(Register, StableHLOType)> {
        let (operand_reg, operand_ty) = self.generate(&args[0])?;

        let mut axis: Option<i64> = None;
        let mut keepdims = false;
        let mut i = 1;
        while i + 1 < args.len() {
            match &args[i] {
                CompiledExpr::Keyword(k) if k == "axis" => {
                    if let CompiledExpr::Integer(n) = &args[i + 1] {
                        axis = Some(*n);
                    }
                    i += 2;
                }
                CompiledExpr::Keyword(k) if k == "keepdims" => {
                    if let CompiledExpr::Boolean(b) = &args[i + 1] {
                        keepdims = *b;
                    }
                    i += 2;
                }
                _ => { i += 1; }
            }
        }

        let (reg, ty) = match axis {
            Some(ax) => {
                tensor_ops::emit_product(&mut self.emitter, &operand_reg, &operand_ty, ax, keepdims)
            }
            None => {
                let ndim = operand_ty.shape().len();
                if ndim == 0 {
                    (operand_reg, operand_ty)
                } else {
                    let mut cur_reg = operand_reg;
                    let mut cur_ty = operand_ty;
                    for _ in (0..ndim).rev() {
                        let (r, t) = tensor_ops::emit_product(&mut self.emitter, &cur_reg, &cur_ty, -1, false);
                        cur_reg = r;
                        cur_ty = t;
                    }
                    (cur_reg, cur_ty)
                }
            }
        };
        Ok((reg, ty))
    }

    fn gen_minmax_reduce(
        &mut self,
        name: &str,
        args: &[CompiledExpr],
    ) -> SheafResult<(Register, StableHLOType)> {
        let (operand_reg, operand_ty) = self.generate(&args[0])?;

        let mut axis: Option<i64> = None;
        let mut keepdims = false;
        let mut i = 1;
        while i + 1 < args.len() {
            match &args[i] {
                CompiledExpr::Keyword(k) if k == "axis" => {
                    if let CompiledExpr::Integer(n) = &args[i + 1] {
                        axis = Some(*n);
                    }
                    i += 2;
                }
                CompiledExpr::Keyword(k) if k == "keepdims" => {
                    if let CompiledExpr::Boolean(b) = &args[i + 1] {
                        keepdims = *b;
                    }
                    i += 2;
                }
                _ => { i += 1; }
            }
        }

        let emit_fn = if name == "min" {
            tensor_ops::emit_min_reduce
        } else {
            tensor_ops::emit_max_reduce
        };

        let (reg, ty) = match axis {
            Some(ax) => emit_fn(&mut self.emitter, &operand_reg, &operand_ty, ax, keepdims),
            None => {
                let ndim = operand_ty.shape().len();
                if ndim == 0 {
                    (operand_reg, operand_ty)
                } else {
                    let mut cur_reg = operand_reg;
                    let mut cur_ty = operand_ty;
                    for _ in (0..ndim).rev() {
                        let (r, t) = emit_fn(&mut self.emitter, &cur_reg, &cur_ty, -1, false);
                        cur_reg = r;
                        cur_ty = t;
                    }
                    (cur_reg, cur_ty)
                }
            }
        };
        Ok((reg, ty))
    }

    fn gen_argmax_argmin(
        &mut self,
        name: &str,
        args: &[CompiledExpr],
    ) -> SheafResult<(Register, StableHLOType)> {
        let (operand_reg, operand_ty) = self.generate(&args[0])?;

        let mut axis: i64 = -1;
        let mut i = 1;
        while i + 1 < args.len() {
            if let CompiledExpr::Keyword(k) = &args[i] {
                if k == "axis" {
                    if let CompiledExpr::Integer(n) = &args[i + 1] {
                        axis = *n;
                    }
                }
            }
            i += 2;
        }

        let (reg, ty) = if name == "argmax" {
            tensor_ops::emit_argmax(&mut self.emitter, &operand_reg, &operand_ty, axis)
        } else {
            tensor_ops::emit_argmin(&mut self.emitter, &operand_reg, &operand_ty, axis)
        };
        Ok((reg, ty))
    }

    fn gen_var(
        &mut self,
        args: &[CompiledExpr],
    ) -> SheafResult<(Register, StableHLOType)> {
        let (operand_reg, operand_ty) = self.generate(&args[0])?;

        let mut axis: Option<i64> = None;
        let mut keepdims = false;
        let mut i = 1;
        while i < args.len() {
            match &args[i] {
                CompiledExpr::Keyword(k) if k == "axis" => {
                    if i + 1 < args.len() {
                        if let CompiledExpr::Integer(n) = &args[i + 1] {
                            axis = Some(*n);
                            i += 2;
                            continue;
                        }
                    }
                    i += 1;
                }
                CompiledExpr::Keyword(k) if k == "keepdims" => {
                    if i + 1 < args.len() {
                        if let CompiledExpr::Boolean(b) = &args[i + 1] {
                            keepdims = *b;
                            i += 2;
                            continue;
                        }
                    }
                    keepdims = true;
                    i += 1;
                }
                _ => { i += 1; }
            }
        }

        let (reg, ty) = match axis {
            Some(ax) => {
                tensor_ops::emit_var(&mut self.emitter, &operand_reg, &operand_ty, ax, keepdims)
            }
            None => {
                let ndim = operand_ty.shape().len();
                if ndim == 0 {
                    let reg = self.emitter.emit_constant_f32(0.0);
                    (reg, StableHLOType::scalar_f32())
                } else {
                    let mut cur_reg = operand_reg;
                    let mut cur_ty = operand_ty;
                    for _ in (0..ndim).rev() {
                        let (r, t) = tensor_ops::emit_var(&mut self.emitter, &cur_reg, &cur_ty, -1, false);
                        cur_reg = r;
                        cur_ty = t;
                    }
                    (cur_reg, cur_ty)
                }
            }
        };
        Ok((reg, ty))
    }

    fn gen_normalize(
        &mut self,
        args: &[CompiledExpr],
    ) -> SheafResult<(Register, StableHLOType)> {
        let (operand_reg, operand_ty) = self.generate(&args[0])?;

        let mut axis: Option<i64> = None;
        let mut i = 1;
        while i + 1 < args.len() {
            match &args[i] {
                CompiledExpr::Keyword(k) if k == "axis" => {
                    if let CompiledExpr::Integer(n) = &args[i + 1] {
                        axis = Some(*n);
                    }
                    i += 2;
                }
                _ => { i += 1; }
            }
        }

        let ax = axis.unwrap_or(-1);
        let (reg, ty) = tensor_ops::emit_normalize(&mut self.emitter, &operand_reg, &operand_ty, ax);
        Ok((reg, ty))
    }
}
