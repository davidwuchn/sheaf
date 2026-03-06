// Copyright (c) 2025 Damien Boureille
// Licensed under the MIT License.

//! Autodiff-related codegen: inline value-and-grad, tuple gradients, broadcast reduction.

use crate::autodiff::{grad_simplified, inline_function_calls};
use crate::compiler::stablehlo::{Register, StableHLOType};
use crate::core::compiler::CompiledExpr;
use crate::core::error::{SheafError, SheafResult};
use super::helpers::{TupleLeaf, collect_tuple_leaves, expand_tuple_to_symbols};
use super::CodeGenerator;

impl CodeGenerator {
    /// Inline value-and-grad: forward pass + symbolic backward passes -> tuple.
    ///
    /// When a wrt parameter is a tuple (e.g. from defparams), the body contains
    /// `GetTupleElement { param, indices }` references to its leaves. We replace
    /// each leaf with a synthetic symbol, differentiate with respect to each one,
    /// and reassemble the gradients into a tuple matching the parameter structure.
    pub(super) fn generate_inline_value_and_grad(
        &mut self,
        lambda: &CompiledExpr,
        args: &[CompiledExpr],
        wrt_indices: &[usize],
    ) -> SheafResult<(Register, StableHLOType)> {
        let (params, body) = match lambda {
            CompiledExpr::Lambda { params, body } => (params, body),
            _ => {
                return Err(SheafError::Compile {
                    message: "InlineValueAndGrad: expected lambda".to_string(),
                    location: crate::core::error::SourceLocation::unknown(),
                })
            }
        };

        // Generate argument values
        let mut arg_regs = Vec::new();
        let mut arg_tys = Vec::new();
        for arg in args {
            let (reg, ty) = self.generate(arg)?;
            arg_regs.push(reg);
            arg_tys.push(ty);
        }

        // Bind lambda params -> arg registers
        let saved = self.bindings.clone();
        for (param, (reg, ty)) in params.iter().zip(arg_regs.iter().zip(arg_tys.iter())) {
            self.bindings
                .insert(param.clone(), (*reg, ty.clone()));
        }

        // Inline user-defined functions so autodiff can differentiate through them
        let inlined_body = inline_function_calls(body, &self.function_registry);

        // Forward pass
        let (loss_reg, loss_ty) = self.generate(&inlined_body)?;

        // Backward passes
        let mut grad_regs = Vec::new();
        let mut grad_tys = Vec::new();
        for &idx in wrt_indices {
            let param_ty = &arg_tys[idx];
            match param_ty {
                StableHLOType::Tuple(_) => {
                    // Tuple parameter: collect all GetTupleElement leaves,
                    // replace with synthetic symbols, differentiate each.
                    let param_name = &params[idx];
                    let leaves = collect_tuple_leaves(&inlined_body, param_name);
                    let expanded = expand_tuple_to_symbols(&inlined_body, param_name);

                    // Bind each synthetic leaf symbol to the corresponding register
                    for leaf in &leaves {
                        let mut current_reg = arg_regs[idx];
                        let mut current_ty = param_ty.clone();
                        for &i in &leaf.indices {
                            let elem_ty = match &current_ty {
                                StableHLOType::Tuple(elems) => elems[i].clone(),
                                _ => unreachable!(),
                            };
                            current_reg = self.emitter.emit_get_tuple_element(
                                &current_reg,
                                &current_ty,
                                i,
                                &elem_ty,
                            );
                            current_ty = elem_ty;
                        }
                        self.bindings
                            .insert(leaf.symbol.clone(), (current_reg, current_ty));
                    }

                    // Differentiate w.r.t. each leaf and build the gradient tuple
                    let (grad_reg, grad_ty) = self.generate_tuple_gradient(
                        &expanded, &leaves, param_ty,
                    )?;
                    grad_regs.push(grad_reg);
                    grad_tys.push(grad_ty);
                }
                _ => {
                    // Scalar/tensor parameter: differentiate directly
                    let grad_expr = grad_simplified(&inlined_body, &params[idx]);
                    let (grad_reg, grad_ty) = self.generate(&grad_expr)?;
                    grad_regs.push(grad_reg);
                    grad_tys.push(grad_ty);
                }
            }
        }

        // Restore bindings
        self.bindings = saved;

        // Pack into tuple: (loss, grad0, grad1, ...)
        let mut all_regs = vec![loss_reg];
        all_regs.extend(grad_regs);
        let mut all_tys = vec![loss_ty];
        all_tys.extend(grad_tys);

        Ok(self.emitter.emit_tuple(&all_regs, &all_tys))
    }

    /// Build the gradient tuple for a tuple parameter by differentiating
    /// w.r.t. each leaf symbol and reassembling into the original tuple structure.
    fn generate_tuple_gradient(
        &mut self,
        expanded_body: &CompiledExpr,
        leaves: &[TupleLeaf],
        param_ty: &StableHLOType,
    ) -> SheafResult<(Register, StableHLOType)> {
        self.build_grad_tuple(expanded_body, leaves, param_ty, &[])
    }

    fn build_grad_tuple(
        &mut self,
        body: &CompiledExpr,
        leaves: &[TupleLeaf],
        ty: &StableHLOType,
        prefix: &[usize],
    ) -> SheafResult<(Register, StableHLOType)> {
        match ty {
            StableHLOType::Tuple(elems) => {
                let mut sub_regs = Vec::new();
                let mut sub_tys = Vec::new();
                for (i, elem_ty) in elems.iter().enumerate() {
                    let mut child_prefix = prefix.to_vec();
                    child_prefix.push(i);
                    let (r, t) = self.build_grad_tuple(body, leaves, elem_ty, &child_prefix)?;
                    sub_regs.push(r);
                    sub_tys.push(t);
                }
                Ok(self.emitter.emit_tuple(&sub_regs, &sub_tys))
            }
            _leaf_ty => {
                // Find the leaf with matching indices
                let leaf = leaves
                    .iter()
                    .find(|l| l.indices == prefix)
                    .ok_or_else(|| SheafError::Compile {
                        message: format!(
                            "InlineValueAndGrad: no leaf found for indices {:?}",
                            prefix
                        ),
                        location: crate::core::error::SourceLocation::unknown(),
                    })?;
                let grad_expr = grad_simplified(body, &leaf.symbol);
                let (grad_reg, grad_ty) = self.generate(&grad_expr)?;
                // Reduce broadcast dims: if the gradient has more dims than the
                // parameter (e.g. batch dim from broadcasting b:[8] to [4,8]),
                // reduce_sum over the leading extra dimensions.
                self.reduce_broadcast_grad(grad_reg, &grad_ty, ty)
            }
        }
    }

    /// Reduce a gradient to match the parameter shape when broadcasting introduced
    /// extra leading dimensions. E.g. grad is [4,8] but param is [8] -> reduce_sum axis 0.
    fn reduce_broadcast_grad(
        &mut self,
        grad_reg: Register,
        grad_ty: &StableHLOType,
        param_ty: &StableHLOType,
    ) -> SheafResult<(Register, StableHLOType)> {
        let grad_shape = grad_ty.shape();
        let param_shape = param_ty.shape();

        if grad_shape == param_shape {
            return Ok((grad_reg, grad_ty.clone()));
        }

        let extra = grad_shape.len().saturating_sub(param_shape.len());
        if extra == 0 {
            return Ok((grad_reg, grad_ty.clone()));
        }

        let trailing = &grad_shape[extra..];
        if trailing != param_shape {
            return Ok((grad_reg, grad_ty.clone()));
        }

        let mut cur_reg = grad_reg;
        let mut cur_ty = grad_ty.clone();
        for _ in 0..extra {
            let (r, t) = self.emitter.emit_reduce_sum(&cur_reg, &cur_ty, 0, false);
            cur_reg = r;
            cur_ty = t;
        }
        Ok((cur_reg, cur_ty))
    }
}
