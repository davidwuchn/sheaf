// Copyright (c) 2025 Damien Boureille
// Licensed under the MIT License.

//! Autodiff-related codegen: tuple gradient assembly and broadcast reduction.
//! Used by the VAG (value-and-grad) trace-based path in jit.rs.

use crate::compiler::stablehlo::{Register, StableHLOType};
use crate::core::compiler::CompiledExpr;
use crate::core::error::SheafResult;
use super::helpers::TupleLeaf;
use super::CodeGenerator;

impl CodeGenerator {
    /// Build a gradient tuple using pre-computed gradient expressions (from reverse-mode AD).
    ///
    /// `grad_map`: maps leaf symbol name -> gradient CompiledExpr
    pub(crate) fn build_grad_tuple_from_map(
        &mut self,
        leaves: &[TupleLeaf],
        param_ty: &StableHLOType,
        grad_map: &std::collections::HashMap<String, CompiledExpr>,
    ) -> SheafResult<(Register, StableHLOType)> {
        self.build_grad_tuple_from_map_rec(leaves, param_ty, grad_map, &[])
    }

    fn build_grad_tuple_from_map_rec(
        &mut self,
        leaves: &[TupleLeaf],
        ty: &StableHLOType,
        grad_map: &std::collections::HashMap<String, CompiledExpr>,
        prefix: &[usize],
    ) -> SheafResult<(Register, StableHLOType)> {
        match ty {
            StableHLOType::Tuple(elems) => {
                let mut sub_regs = Vec::new();
                let mut sub_tys = Vec::new();
                for (i, elem_ty) in elems.iter().enumerate() {
                    let mut child_prefix = prefix.to_vec();
                    child_prefix.push(i);
                    let (r, t) = self.build_grad_tuple_from_map_rec(
                        leaves, elem_ty, grad_map, &child_prefix,
                    )?;
                    sub_regs.push(r);
                    sub_tys.push(t);
                }
                Ok(self.emitter.emit_tuple(&sub_regs, &sub_tys))
            }
            leaf_ty => {
                if let Some(leaf) = leaves.iter().find(|l| l.indices == prefix) {
                    let grad_expr = grad_map
                        .get(&leaf.symbol)
                        .cloned()
                        .unwrap_or(CompiledExpr::Float(0.0));
                    let (grad_reg, grad_ty) = self.generate(&grad_expr)?;
                    self.reduce_broadcast_grad(grad_reg, &grad_ty, ty)
                } else {
                    // No leaf found: parameter not directly accessed (e.g. passed
                    // to scan as a whole). Emit zeros of the correct shape.
                    let shape = leaf_ty.shape();
                    let zeros_expr = if shape.is_empty() {
                        CompiledExpr::Float(0.0)
                    } else {
                        CompiledExpr::FunctionCall {
                            name: "zeros".to_string(),
                            args: vec![CompiledExpr::Vector(
                                shape.iter().map(|&d| CompiledExpr::Integer(d)).collect()
                            )],
                            loc: None,
                        }
                    };
                    self.generate(&zeros_expr)
                }
            }
        }
    }

    /// Reduce a gradient to match the parameter shape when broadcasting introduced
    /// extra leading dimensions. E.g. grad is [4,8] but param is [8] -> reduce_sum axis 0.
    pub(crate) fn reduce_broadcast_grad(
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

        // Same number of elements but different shape → reshape (e.g. [32] → [32,1])
        let grad_elems: i64 = grad_shape.iter().product();
        let param_elems: i64 = param_shape.iter().product();
        if grad_elems == param_elems && grad_elems > 0 && grad_shape != param_shape {
            let (r, t) = self.emitter.emit_reshape(&grad_reg, grad_ty, &param_shape);
            return Ok((r, t));
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
