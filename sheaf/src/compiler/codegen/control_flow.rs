// Copyright (c) 2025 Damien Boureille
// Licensed under the MIT License.

//! Control flow codegen: lambda inlining, reduce/scan unrolling, tree-map.

use crate::compiler::stablehlo::{Register, StableHLOType};
use crate::core::compiler::CompiledExpr;
use crate::core::error::{SheafError, SheafResult};
use super::CodeGenerator;

impl CodeGenerator {
    /// Inline a lambda call: generate args, bind params→registers, generate body.
    pub(super) fn inline_lambda_call(
        &mut self,
        callee: &CompiledExpr,
        args: &[CompiledExpr],
    ) -> SheafResult<(Register, StableHLOType)> {
        // Resolve callee — may be a Lambda directly, or a Symbol bound in lambda_bindings.
        let lambda = match callee {
            CompiledExpr::Lambda { .. } => callee.clone(),
            CompiledExpr::Symbol(name) => {
                self.lambda_bindings
                    .get(name)
                    .cloned()
                    .ok_or_else(|| SheafError::Compile {
                        message: format!("Undefined lambda: {}", name),
                        location: crate::core::error::SourceLocation::unknown(),
                    })?
            }
            other => {
                return Err(SheafError::Compile {
                    message: format!("Expected lambda at call site, got: {:?}", other),
                    location: crate::core::error::SourceLocation::unknown(),
                });
            }
        };

        let (params, body) = match lambda {
            CompiledExpr::Lambda { params, body } => (params, *body),
            _ => unreachable!(),
        };

        // Generate argument values.
        let mut arg_regs = Vec::new();
        let mut arg_tys = Vec::new();
        for arg in args {
            let (reg, ty) = self.generate(arg)?;
            arg_regs.push(reg);
            arg_tys.push(ty);
        }

        // Bind param names → (register, type) and generate body.
        let saved = self.bindings.clone();
        for (param, (reg, ty)) in params.iter().zip(arg_regs.iter().zip(arg_tys.iter())) {
            self.bindings
                .insert(param.clone(), (*reg, ty.clone()));
        }
        let result = self.generate(&body);
        self.bindings = saved;
        result
    }

    /// Static tree-map unrolling: apply lambda to each leaf of matching tuple trees.
    ///
    /// When the type is `Tuple(...)`, recursively extract elements from each tree,
    /// apply `generate_tree_map` on sub-elements, and reassemble into a new tuple.
    /// When the type is a leaf (tensor/scalar), inline the lambda with the leaf
    /// registers as arguments.
    ///
    /// `tree_tys` has one type per tree argument (may differ at leaves due to
    /// broadcast gradients having different shapes from params).
    pub(super) fn generate_tree_map(
        &mut self,
        lambda: &CompiledExpr,
        tree_regs: &[Register],
        tree_tys: &[StableHLOType],
    ) -> SheafResult<(Register, StableHLOType)> {
        // Use the first tree's type to drive the tuple structure
        match &tree_tys[0] {
            StableHLOType::Tuple(first_elem_tys, _) => {
                let mut result_regs = Vec::new();
                let mut result_tys = Vec::new();
                for (idx, _) in first_elem_tys.iter().enumerate() {
                    // Extract element `idx` from each tree using its own type
                    let mut sub_regs = Vec::new();
                    let mut sub_tys = Vec::new();
                    for (tree_reg, tree_ty) in tree_regs.iter().zip(tree_tys.iter()) {
                        let elem_ty = match tree_ty {
                            StableHLOType::Tuple(elems, _) => &elems[idx],
                            other => other,
                        };
                        let elem_reg = self.emitter.emit_get_tuple_element(
                            tree_reg,
                            tree_ty,
                            idx,
                            elem_ty,
                        );
                        sub_regs.push(elem_reg);
                        sub_tys.push(elem_ty.clone());
                    }
                    let (r, t) = self.generate_tree_map(lambda, &sub_regs, &sub_tys)?;
                    result_regs.push(r);
                    result_tys.push(t);
                }
                Ok(self.emitter.emit_tuple(&result_regs, &result_tys))
            }
            _ => {
                // Leaf: inline the lambda with tree_regs as arguments
                let (params, body) = match lambda {
                    CompiledExpr::Lambda { params, body } => (params, body),
                    _ => {
                        return Err(SheafError::Compile {
                            message: "tree-map: first argument must be a lambda".to_string(),
                            location: crate::core::error::SourceLocation::unknown(),
                        })
                    }
                };
                if params.len() != tree_regs.len() {
                    return Err(SheafError::Compile {
                        message: format!(
                            "tree-map: lambda has {} params but {} trees provided",
                            params.len(),
                            tree_regs.len()
                        ),
                        location: crate::core::error::SourceLocation::unknown(),
                    });
                }
                // Bind lambda params to the leaf registers with their actual types
                let saved = self.bindings.clone();
                for (param, (reg, ty)) in params.iter().zip(tree_regs.iter().zip(tree_tys.iter())) {
                    self.bindings
                        .insert(param.clone(), (*reg, ty.clone()));
                }
                let result = self.generate(body);
                self.bindings = saved;
                result
            }
        }
    }

    /// Static tree-reduce: fold a function over all leaves of a pytree.
    ///
    /// When the type is `Tuple(...)`, recursively descend into each element,
    /// threading the accumulator through. When the type is a leaf (tensor/scalar),
    /// inline the lambda with `(acc, leaf)` arguments.
    pub(super) fn generate_tree_reduce(
        &mut self,
        lambda: &CompiledExpr,
        tree_reg: Register,
        tree_ty: &StableHLOType,
        acc_reg: Register,
        acc_ty: &StableHLOType,
    ) -> SheafResult<(Register, StableHLOType)> {
        match tree_ty {
            StableHLOType::Tuple(elem_tys, _) => {
                let mut cur_acc = acc_reg;
                let mut cur_ty = acc_ty.clone();
                for (idx, elem_ty) in elem_tys.iter().enumerate() {
                    let elem_reg = self.emitter.emit_get_tuple_element(
                        &tree_reg, tree_ty, idx, elem_ty,
                    );
                    (cur_acc, cur_ty) = self.generate_tree_reduce(
                        lambda, elem_reg, elem_ty, cur_acc, &cur_ty,
                    )?;
                }
                Ok((cur_acc, cur_ty))
            }
            _ => {
                // Leaf: apply f(acc, leaf)
                match lambda {
                    CompiledExpr::Lambda { params, body } => {
                        if params.len() != 2 {
                            return Err(SheafError::Compile {
                                message: format!(
                                    "tree-reduce: lambda must have 2 params (acc, leaf), got {}",
                                    params.len()
                                ),
                                location: crate::core::error::SourceLocation::unknown(),
                            });
                        }
                        let saved = self.bindings.clone();
                        self.bindings.insert(params[0].clone(), (acc_reg, acc_ty.clone()));
                        self.bindings.insert(params[1].clone(), (tree_reg, tree_ty.clone()));
                        let result = self.generate(body);
                        self.bindings = saved;
                        result
                    }
                    // Builtin symbol like +, *, etc.: emit as (f acc leaf)
                    CompiledExpr::Symbol(name) => {
                        let call = CompiledExpr::FunctionCall {
                            name: name.clone(),
                            args: vec![
                                CompiledExpr::Symbol("__tree_reduce_acc".to_string()),
                                CompiledExpr::Symbol("__tree_reduce_leaf".to_string()),
                            ],
                            loc: None,
                        };
                        let saved = self.bindings.clone();
                        self.bindings.insert("__tree_reduce_acc".to_string(), (acc_reg, acc_ty.clone()));
                        self.bindings.insert("__tree_reduce_leaf".to_string(), (tree_reg, tree_ty.clone()));
                        let result = self.generate(&call);
                        self.bindings = saved;
                        result
                    }
                    _ => Err(SheafError::Compile {
                        message: "tree-reduce: first argument must be a lambda or builtin symbol".to_string(),
                        location: crate::core::error::SourceLocation::unknown(),
                    }),
                }
            }
        }
    }

    /// Static unrolling of `reduce` / `scan` over a collection with known type.
    ///
    /// Supports three collection shapes:
    /// - `Tuple([Tuple(...), ...])` — VecTuple (list of structs): each elem = get_tuple_element(i)
    /// - `Tuple([tensor[N,...], ...])` — stacked dict (scan): each elem = tuple of slices at i
    /// - `tensor[N, ...]` — plain tensor: each elem = index_axis0(i)
    ///
    /// Key layout (`tuple_key_layouts`) is inherited by the lambda's elem parameter so that
    /// `(get elem "field")` resolves correctly inside the body.
    pub(super) fn generate_reduce_scan(
        &mut self,
        lambda: &CompiledExpr,
        init: &CompiledExpr,
        coll: &CompiledExpr,
        is_scan: bool,
    ) -> SheafResult<(Register, StableHLOType)> {
        // Capture collection symbol before generating (for key layout lookup)
        let coll_sym = if let CompiledExpr::Symbol(s) = coll { Some(s.clone()) } else { None };
        let (coll_reg, coll_ty) = self.generate(coll)?;
        let (mut carry_reg, mut carry_ty) = self.generate(init)?;

        // Resolve the lambda (may be inline Lambda or a symbol in lambda_bindings)
        let lambda_resolved = match lambda {
            CompiledExpr::Lambda { .. } => lambda.clone(),
            CompiledExpr::Symbol(sym_name) => {
                self.lambda_bindings.get(sym_name).cloned().ok_or_else(|| SheafError::Compile {
                    message: format!("reduce/scan: lambda '{}' not found", sym_name),
                    location: crate::core::error::SourceLocation::unknown(),
                })?
            }
            _ => return Err(SheafError::Compile {
                message: "reduce/scan: first argument must be a lambda".to_string(),
                location: crate::core::error::SourceLocation::unknown(),
            }),
        };
        let (carry_param, elem_param, body) = match lambda_resolved {
            CompiledExpr::Lambda { params, body } if params.len() == 2 => {
                (params[0].clone(), params[1].clone(), *body)
            }
            _ => return Err(SheafError::Compile {
                message: "reduce/scan: lambda must have exactly 2 parameters (carry, elem)".to_string(),
                location: crate::core::error::SourceLocation::unknown(),
            }),
        };

        // Determine N and extraction strategy from collection type
        enum ElemKind {
            VecTuple(Vec<StableHLOType>),   // Tuple of sub-tuples
            StackedDict(Vec<StableHLOType>), // Tuple of stacked tensors
            PlainTensor,                     // Single tensor
        }
        let (n, kind) = match &coll_ty {
            StableHLOType::Tuple(types, _) if !types.is_empty() => {
                if types.iter().all(|t| matches!(t, StableHLOType::Tuple(..))) {
                    (types.len(), ElemKind::VecTuple(types.clone()))
                } else if types.iter().all(|t| !t.shape().is_empty()) {
                    let n = types[0].shape()[0] as usize;
                    if n == 0 {
                        return Err(SheafError::Compile {
                            message: "reduce/scan: stacked dict has zero elements".to_string(),
                            location: crate::core::error::SourceLocation::unknown(),
                        });
                    }
                    (n, ElemKind::StackedDict(types.clone()))
                } else {
                    return Err(SheafError::Compile {
                        message: "reduce/scan: mixed Tuple element types not supported".to_string(),
                        location: crate::core::error::SourceLocation::unknown(),
                    });
                }
            }
            t if !t.shape().is_empty() => (t.shape()[0] as usize, ElemKind::PlainTensor),
            _ => return Err(SheafError::Compile {
                message: "reduce/scan: cannot determine iteration count from collection type".to_string(),
                location: crate::core::error::SourceLocation::unknown(),
            }),
        };

        // Key layout for the elem parameter inherited from collection's layout.
        // Three resolution strategies:
        // 1. Symbol name lookup (e.g. "hidden" in standalone forward)
        // 2. Register-based lookup via layout_key_map (handles ANF-renamed vars)
        // 3. GetTupleElement chain resolution via idx_to_key
        let elem_layout = coll_sym
            .as_deref()
            .and_then(|s| self.tuple_key_layouts.get(s))
            .cloned()
            .or_else(|| {
                // Strategy 2: the register produced by generate(coll) may have
                // a layout key via layout_key_map (set by track_layout_key in GTE handler)
                let layout_key = self.layout_key_map.get(&coll_reg)?;
                self.tuple_key_layouts.get(layout_key).cloned()
            })
            .or_else(|| {
                // Strategy 3: GetTupleElement collections (e.g. (get params :h))
                if let CompiledExpr::GetTupleElement { param, indices } = coll {
                    let mut cur = param.clone();
                    for &idx in indices {
                        cur = self.idx_to_key.get(&(cur, idx))?.clone();
                    }
                    let list_layout = self.tuple_key_layouts.get(&cur)?;
                    let first_child_key = list_layout.values().next()
                        .and_then(|_| list_layout.keys().next())?;
                    self.tuple_key_layouts.get(first_child_key).cloned()
                } else {
                    None
                }
            });

        for i in 0..n {
            // Extract element i from the collection
            let (elem_reg, elem_ty) = match &kind {
                ElemKind::VecTuple(types) => {
                    let elem_ty = types[i].clone();
                    let reg = self.emitter.emit_get_tuple_element(&coll_reg, &coll_ty, i, &elem_ty);
                    (reg, elem_ty)
                }
                ElemKind::StackedDict(comp_types) => {
                    // Each component tensor: get + slice → then pack into a tuple
                    let mut parts = Vec::new();
                    let mut part_types = Vec::new();
                    for (k, comp_ty) in comp_types.iter().enumerate() {
                        let comp_reg = self.emitter.emit_get_tuple_element(
                            &coll_reg, &coll_ty, k, comp_ty,
                        );
                        let (slice_reg, slice_ty) =
                            self.emitter.emit_index_axis0(&comp_reg, comp_ty, i as i64);
                        parts.push(slice_reg);
                        part_types.push(slice_ty);
                    }
                    self.emitter.emit_tuple(&parts, &part_types)
                }
                ElemKind::PlainTensor => {
                    self.emitter.emit_index_axis0(&coll_reg, &coll_ty, i as i64)
                }
            };

            // Propagate key layout into elem parameter
            if let Some(ref layout) = elem_layout {
                self.tuple_key_layouts.insert(elem_param.clone(), layout.clone());
            }

            // Inline lambda body with carry and elem bound
            let saved_bindings = self.bindings.clone();
            self.bindings.insert(carry_param.clone(), (carry_reg, carry_ty.clone()));
            self.bindings.insert(elem_param.clone(), (elem_reg, elem_ty));
            let result = self.generate(&body);
            self.bindings = saved_bindings;

            // Clean up elem layout
            if elem_layout.is_some() {
                self.tuple_key_layouts.remove(&elem_param);
            }

            let (step_reg, step_ty) = result?;
            if is_scan {
                // Scan body returns [carry, output]; extract carry (element 0)
                match &step_ty {
                    StableHLOType::Tuple(elems, _) if elems.len() == 2 => {
                        let carry_elem_ty = elems[0].clone();
                        carry_reg = self.emitter.emit_get_tuple_element(
                            &step_reg, &step_ty, 0, &carry_elem_ty,
                        );
                        carry_ty = carry_elem_ty;
                    }
                    _ => {
                        return Err(SheafError::Compile {
                            message: "scan: lambda must return [carry output] (a 2-element tuple)".to_string(),
                            location: crate::core::error::SourceLocation::unknown(),
                        });
                    }
                }
            } else {
                carry_reg = step_reg;
                carry_ty = step_ty;
            }
        }

        Ok((carry_reg, carry_ty))
    }
}
