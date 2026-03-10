// Copyright (c) 2025 Damien Boureille
// Licensed under the MIT License.

//! Autodiff-related codegen: inline value-and-grad, tuple gradients, broadcast reduction.

use crate::autodiff::{find_undiffable_ops, grad_simplified, inline_function_calls};
use crate::autodiff::reverse::{to_anf, reverse_grad};
use crate::compiler::stablehlo::{Register, StableHLOType};
use crate::core::compiler::CompiledExpr;
use crate::core::error::{SheafError, SheafResult};
use super::helpers::{TupleLeaf, collect_tuple_leaves, expand_tuple_to_symbols};
use super::CodeGenerator;

impl CodeGenerator {
    /// Inline value-and-grad: forward pass + symbolic backward passes -> tuple.
    ///
    /// When a wrt parameter is a tuple (e.g. from traced dict layout), the body contains
    /// `GetTupleElement { param, indices }` references to its leaves. We replace
    /// each leaf with a synthetic symbol, differentiate with respect to each one,
    /// and reassemble the gradients into a tuple matching the parameter structure.
    pub(crate) fn generate_inline_value_and_grad(
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
        let mut inlined_body = inline_function_calls(body, &self.function_registry);

        // The lambda body was NOT processed by compile_function's lowering passes
        // (lower_get_calls, resolve_static_constants, unroll_reduces don't recurse
        // into InlineValueAndGrad). Replicate the full pipeline from
        // try_jit_value_and_grad here.
        {
            use crate::compiler::lower_get_calls;
            use crate::compiler::transforms::{lower_inlined_gets, resolve_static_constants, unroll_reduces};

            // Build index maps for lambda params (e.g. p -> params subtree)
            let mut all_index_maps = self.param_index_maps.clone();
            for (i, param_name) in params.iter().enumerate() {
                if matches!(&arg_tys[i], StableHLOType::Tuple(_)) {
                    if let Some(layout_key) = self.resolve_arg_layout_key(&args[i]) {
                        let index_map = self.build_index_map_from_key(&layout_key);
                        if !index_map.is_empty() {
                            all_index_maps.push((param_name.clone(), index_map));
                        }
                    }
                }
            }

            // Lower get calls for all params (outer + lambda)
            for (param_name, index_map) in &all_index_maps {
                inlined_body = lower_get_calls(&inlined_body, param_name, index_map);
            }
            inlined_body = lower_inlined_gets(&inlined_body, &all_index_maps);

            // Unroll reduces (e.g. transformer block iteration)
            let known_types: Vec<(String, StableHLOType)> = self.bindings
                .iter()
                .map(|(n, (_, t))| (n.clone(), t.clone()))
                .collect();
            inlined_body = unroll_reduces(&inlined_body, &known_types);

            // Re-lower get calls introduced by unrolling
            for (param_name, index_map) in &all_index_maps {
                inlined_body = lower_get_calls(&inlined_body, param_name, index_map);
            }
            inlined_body = lower_inlined_gets(&inlined_body, &all_index_maps);

            // Resolve static constants (e.g. (static (get config :n_embd)) -> 384)
            {
                let mut param_shapes = self.binding_shapes();
                // Inject shapes for tuple leaf nodes (GetTupleElement resolution
                // in try_infer_shape uses "param@[i,j,k]" keys)
                for (name, (_, ty)) in &self.bindings {
                    inject_tuple_leaf_shapes(name, ty, &[], &mut param_shapes);
                }
                inlined_body = resolve_static_constants(
                    &inlined_body, &self.scalar_constants, &param_shapes,
                );
            }
        }

        // Check if the body contains ops that symbolic AD can't handle
        let undiffable = find_undiffable_ops(&inlined_body);
        if !undiffable.is_empty() {
            // Use reverse-mode AD (ANF-based) for complex bodies (e.g. GPT forward)
            let result = self.generate_reverse_mode_vag(
                &inlined_body, params, &arg_regs, &arg_tys, wrt_indices,
            );
            self.bindings = saved;
            return result;
        }

        // Forward pass (symbolic AD path)
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

    /// Reverse-mode AD path for InlineValueAndGrad.
    ///
    /// Used when the lambda body contains ops that symbolic AD cannot differentiate
    /// (get, reduce, LambdaCall, etc.). Mirrors the pipeline in try_jit_value_and_grad
    /// (jit.rs) but emits into the current codegen scope instead of a separate module.
    fn generate_reverse_mode_vag(
        &mut self,
        inlined_body: &CompiledExpr,
        params: &[String],
        arg_regs: &[Register],
        arg_tys: &[StableHLOType],
        wrt_indices: &[usize],
    ) -> SheafResult<(Register, StableHLOType)> {
        // Expand tuple params to synthetic leaf symbols
        let mut expanded_body = inlined_body.clone();
        let mut all_leaves: Vec<(usize, Vec<TupleLeaf>)> = Vec::new();
        let mut all_wrt_symbols: Vec<String> = Vec::new();

        for &idx in wrt_indices {
            let param_name = &params[idx];
            let param_ty = &arg_tys[idx];
            match param_ty {
                StableHLOType::Tuple(_) => {
                    let leaves = collect_tuple_leaves(&expanded_body, param_name);
                    expanded_body = expand_tuple_to_symbols(&expanded_body, param_name);
                    for leaf in &leaves {
                        all_wrt_symbols.push(leaf.symbol.clone());
                    }
                    all_leaves.push((idx, leaves));
                }
                _ => {
                    all_wrt_symbols.push(param_name.clone());
                }
            }
        }

        // Convert to ANF
        let anf_expr = to_anf(&expanded_body);
        let (anf_bindings, anf_body) = match &anf_expr {
            CompiledExpr::Let { bindings, body } => {
                (bindings.clone(), body.as_ref().clone())
            }
            other => (vec![], other.clone()),
        };

        // Bind tuple leaves to codegen registers (for GetTupleElement resolution)
        for &(idx, ref leaves) in &all_leaves {
            let param_ty = &arg_tys[idx];
            for leaf in leaves {
                let mut current_reg = arg_regs[idx];
                let mut current_ty = param_ty.clone();
                for &i in &leaf.indices {
                    let elem_ty = match &current_ty {
                        StableHLOType::Tuple(elems) => elems[i].clone(),
                        _ => {
                            return Err(SheafError::Compile {
                                message: format!(
                                    "reverse_mode_vag: expected tuple at index {} for param {}",
                                    i, &params[idx]
                                ),
                                location: crate::core::error::SourceLocation::unknown(),
                            })
                        }
                    };
                    current_reg = self.emitter.emit_get_tuple_element(
                        &current_reg, &current_ty, i, &elem_ty,
                    );
                    current_ty = elem_ty;
                }
                self.bind_symbol(&leaf.symbol, current_reg, current_ty);
            }
        }

        // Generate forward bindings
        for (name, value_expr) in &anf_bindings {
            self.generate_binding(name, value_expr)?;
        }

        // Generate the ANF body (loss value)
        let (loss_reg, loss_ty) = self.generate(&anf_body)?;

        // Build shape map from forward codegen for reverse-mode AD
        let shape_map = self.binding_shapes();

        // Run reverse-mode AD on ANF
        let (backward_bindings, grad_sym_map) =
            reverse_grad(&anf_bindings, &anf_body, &all_wrt_symbols, &shape_map);

        // Generate backward bindings (adjoint computations)
        for (name, value_expr) in &backward_bindings {
            let (reg, ty) = self.generate(value_expr)?;
            self.bind_symbol(name, reg, ty);
        }

        // Collect gradient registers for each wrt param
        let mut grad_regs: Vec<Register> = Vec::new();
        let mut grad_tys: Vec<StableHLOType> = Vec::new();

        for &idx in wrt_indices {
            let param_name = &params[idx];
            let param_ty = &arg_tys[idx];
            match param_ty {
                StableHLOType::Tuple(_) => {
                    let leaves = all_leaves.iter()
                        .find(|(i, _)| *i == idx)
                        .map(|(_, l)| l)
                        .unwrap();

                    let leaf_grad_map: std::collections::HashMap<String, CompiledExpr> =
                        leaves.iter().map(|leaf| {
                            let grad_expr = grad_sym_map
                                .get(&leaf.symbol)
                                .map(|sym_name| CompiledExpr::Symbol(sym_name.clone()))
                                .unwrap_or_else(|| {
                                    let leaf_ty = resolve_leaf_type(param_ty, &leaf.indices);
                                    let shape = leaf_ty.shape();
                                    if shape.is_empty() {
                                        CompiledExpr::Float(0.0)
                                    } else {
                                        CompiledExpr::FunctionCall {
                                            name: "zeros".to_string(),
                                            args: vec![CompiledExpr::Vector(
                                                shape.iter().map(|&d| CompiledExpr::Integer(d)).collect()
                                            )],
                                        }
                                    }
                                });
                            (leaf.symbol.clone(), grad_expr)
                        }).collect();

                    let (grad_reg, grad_ty) =
                        self.build_grad_tuple_from_map(leaves, param_ty, &leaf_grad_map)?;
                    grad_regs.push(grad_reg);
                    grad_tys.push(grad_ty);
                }
                _ => {
                    let grad_expr = grad_sym_map
                        .get(param_name)
                        .map(|sym_name| CompiledExpr::Symbol(sym_name.clone()))
                        .unwrap_or(CompiledExpr::Float(0.0));
                    let (grad_reg, grad_ty) = self.generate(&grad_expr)?;
                    let (grad_reg, grad_ty) =
                        self.reduce_broadcast_grad(grad_reg, &grad_ty, param_ty)?;
                    grad_regs.push(grad_reg);
                    grad_tys.push(grad_ty);
                }
            }
        }

        // Pack into tuple: (loss, grad0, grad1, ...)
        let mut all_regs = vec![loss_reg];
        all_regs.extend(grad_regs);
        let mut all_tys = vec![loss_ty];
        all_tys.extend(grad_tys);

        Ok(self.emitter.emit_tuple(&all_regs, &all_tys))
    }

    /// Build the gradient tuple for a tuple parameter by differentiating
    /// w.r.t. each leaf symbol and reassembling into the original tuple structure.
    pub(crate) fn generate_tuple_gradient(
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
            _leaf_ty => {
                let leaf = leaves
                    .iter()
                    .find(|l| l.indices == prefix)
                    .ok_or_else(|| SheafError::Compile {
                        message: format!(
                            "build_grad_tuple_from_map: no leaf for {:?}",
                            prefix
                        ),
                        location: crate::core::error::SourceLocation::unknown(),
                    })?;
                let grad_expr = grad_map
                    .get(&leaf.symbol)
                    .cloned()
                    .unwrap_or(CompiledExpr::Float(0.0));
                let (grad_reg, grad_ty) = self.generate(&grad_expr)?;
                self.reduce_broadcast_grad(grad_reg, &grad_ty, ty)
            }
        }
    }

    /// Resolve the layout key name for an argument expression.
    /// Handles both lowered (`GetTupleElement`) and unlowered (`FunctionCall("get", ...)`)
    /// forms, since `lower_get_calls` doesn't recurse into `InlineValueAndGrad`.
    fn resolve_arg_layout_key(&self, arg: &CompiledExpr) -> Option<String> {
        match arg {
            CompiledExpr::GetTupleElement { param, indices } => {
                let mut key = param.clone();
                for &idx in indices {
                    key = self.idx_to_key.get(&(key, idx))?.clone();
                }
                Some(key)
            }
            CompiledExpr::Symbol(name) => Some(name.clone()),
            // Handle unlowered (get sym :key) chains
            CompiledExpr::FunctionCall { name, args } if name == "get" && args.len() >= 2 => {
                let base_key = self.resolve_arg_layout_key(&args[0])?;
                let mut current_key = base_key;
                for kw in &args[1..] {
                    let key_name = match kw {
                        CompiledExpr::Keyword(k) | CompiledExpr::String(k) => k.clone(),
                        _ => return None,
                    };
                    self.tuple_key_layouts.get(&current_key)?.get(&key_name)?;
                    current_key = key_name;
                }
                Some(current_key)
            }
            _ => None,
        }
    }

    /// Build a `BTreeMap<Vec<String>, Vec<usize>>` index map by recursively walking
    /// `tuple_key_layouts` from the given starting key.
    fn build_index_map_from_key(
        &self,
        key: &str,
    ) -> std::collections::BTreeMap<Vec<String>, Vec<usize>> {
        let mut map = std::collections::BTreeMap::new();
        self.build_index_map_rec(key, &[], &[], &mut map);
        map
    }

    fn build_index_map_rec(
        &self,
        key: &str,
        parent_path: &[String],
        parent_indices: &[usize],
        map: &mut std::collections::BTreeMap<Vec<String>, Vec<usize>>,
    ) {
        if let Some(layout) = self.tuple_key_layouts.get(key) {
            for (child_key, &child_idx) in layout {
                let mut path = parent_path.to_vec();
                path.push(child_key.clone());
                let mut indices = parent_indices.to_vec();
                indices.push(child_idx);
                map.insert(path.clone(), indices.clone());
                self.build_index_map_rec(child_key, &path, &indices, map);
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

/// Recursively inject shapes for tuple leaves into the shapes map.
/// Uses the "param@[i,j,k]" key format expected by `try_infer_shape`.
fn inject_tuple_leaf_shapes(
    param_name: &str,
    ty: &StableHLOType,
    indices: &[usize],
    shapes: &mut std::collections::HashMap<String, Vec<i64>>,
) {
    match ty {
        StableHLOType::Tuple(elems) => {
            for (i, elem_ty) in elems.iter().enumerate() {
                let mut child_indices = indices.to_vec();
                child_indices.push(i);
                inject_tuple_leaf_shapes(param_name, elem_ty, &child_indices, shapes);
            }
        }
        _ => {
            let shape = ty.shape();
            if !shape.is_empty() && !indices.is_empty() {
                let key = format!("{}@{:?}", param_name, indices);
                shapes.insert(key, shape.to_vec());
            }
        }
    }
}

fn resolve_leaf_type(ty: &StableHLOType, indices: &[usize]) -> StableHLOType {
    let mut current = ty.clone();
    for &idx in indices {
        if let StableHLOType::Tuple(elems) = &current {
            if idx < elems.len() {
                current = elems[idx].clone();
            } else {
                return StableHLOType::scalar_f32();
            }
        } else {
            return current;
        }
    }
    current
}
