// Copyright (c) 2025 Damien Boureille
// Licensed under the MIT License.

//! Control flow codegen: lambda inlining, reduce/scan unrolling, tree-map, scan VJP.

use std::collections::{BTreeMap, HashMap};
use crate::autodiff::reverse::{to_anf, reverse_grad};
use crate::lowering::stablehlo::{Register, StableHLOType};
use crate::lowering::transforms::try_infer_shape;
use crate::core::expr::{BindingPattern, CompiledExpr};
use crate::core::error::{SheafError, SheafResult};
use super::CodeGenerator;

/// Build a deep index_map from `tuple_key_layouts` for a given root layout key.
/// Descends recursively into nested layouts to produce entries like
/// `["attn", "c_attn", "weight"] -> [0, 0, 0]`.
pub(super) fn build_deep_index_map(
    root_key: &str,
    tuple_key_layouts: &HashMap<String, BTreeMap<String, usize>>,
) -> BTreeMap<Vec<String>, Vec<usize>> {
    let mut map = BTreeMap::new();
    build_deep_index_map_rec(root_key, &[], &[], tuple_key_layouts, &mut map);
    map
}

pub(super) fn build_deep_index_map_rec(
    key: &str,
    path: &[String],
    indices: &[usize],
    layouts: &HashMap<String, BTreeMap<String, usize>>,
    map: &mut BTreeMap<Vec<String>, Vec<usize>>,
) {
    if !path.is_empty() {
        map.insert(path.to_vec(), indices.to_vec());
    }
    if let Some(child_layout) = layouts.get(key) {
        for (field_name, &field_idx) in child_layout {
            let mut child_path = path.to_vec();
            child_path.push(field_name.clone());
            let mut child_indices = indices.to_vec();
            child_indices.push(field_idx);
            build_deep_index_map_rec(field_name, &child_path, &child_indices, layouts, map);
        }
    }
}

/// Collection element extraction strategy for reduce/scan.
enum ElemKind {
    VecTuple(Vec<StableHLOType>),    // Tuple of sub-tuples
    StackedDict(Vec<StableHLOType>), // Tuple of stacked tensors
    PlainTensor,                      // Single tensor
}

impl CodeGenerator {
    /// Inline a lambda call: generate args, bind params->registers, generate body.
    pub(super) fn inline_lambda_call(
        &mut self,
        callee: &CompiledExpr,
        args: &[CompiledExpr],
    ) -> SheafResult<(Register, StableHLOType)> {
        if let CompiledExpr::FunctionCall { name, args: inner_args, .. } = callee {
            if name == "__value-and-grad-hof__" && inner_args.len() == 1 {
                return self.generate_vag_inline(&inner_args[0], args);
            }
        }

        // Resolve callee -- may be a Lambda directly, or a Symbol bound in lambda_bindings.
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

        // Bind param names -> (register, type) and generate body.
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
            StableHLOType::Tuple(first_elem_tys, keys) => {
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
                let (reg, _) = self.emitter.emit_tuple_with_keys(&result_regs, &result_tys, keys);
                Ok((reg, StableHLOType::Tuple(result_tys, keys.clone())))
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
    /// - `Tuple([Tuple(...), ...])` -- VecTuple (list of structs): each elem = get_tuple_element(i)
    /// - `Tuple([tensor[N,...], ...])` -- stacked dict (scan): each elem = tuple of slices at i
    /// - `tensor[N, ...]` -- plain tensor: each elem = index_axis0(i)
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
        // For a VecTuple (list of structs), the collection layout maps "0","1",... to indices.
        // We need the layout of one *element*, not the collection itself.
        // Helper: if a layout looks like a list (numeric keys), resolve to first child's layout.
        let resolve_elem_layout = |list_layout: &std::collections::BTreeMap<String, usize>| -> Option<std::collections::BTreeMap<String, usize>> {
            let first_key = list_layout.keys().next()?;
            // If the first key parses as a number, this is a list layout; descend one level
            if first_key.parse::<usize>().is_ok() {
                self.tuple_key_layouts.get(first_key).cloned()
            } else {
                // Already an element layout (keys are field names like "attn", "ln_1")
                Some(list_layout.clone())
            }
        };

        // Three resolution strategies:
        // 1. Symbol name lookup (e.g. "hidden" in standalone forward)
        // 2. Register-based lookup via layout_key_map (handles ANF-renamed vars)
        // 3. GetTupleElement chain resolution via idx_to_key
        let elem_layout = coll_sym
            .as_deref()
            .and_then(|s| self.tuple_key_layouts.get(s))
            .and_then(|layout| resolve_elem_layout(layout))
            .or_else(|| {
                // Strategy 2: the register produced by generate(coll) may have
                // a layout key via layout_key_map (set by track_layout_key in GTE handler)
                let layout_key = self.layout_key_map.get(&coll_reg)?;
                let layout = self.tuple_key_layouts.get(layout_key)?;
                resolve_elem_layout(layout)
            })
            .or_else(|| {
                // Strategy 3: GetTupleElement collections (e.g. (get params :h))
                if let CompiledExpr::GetTupleElement { param, indices } = coll {
                    let mut cur = param.clone();
                    for &idx in indices {
                        cur = self.idx_to_key.get(&(cur, idx))?.clone();
                    }
                    let list_layout = self.tuple_key_layouts.get(&cur)?;
                    resolve_elem_layout(list_layout)
                } else {
                    None
                }
            });

        let mut last_scan_result: Option<(Register, StableHLOType)> = None;

        for i in 0..n {
            // Extract element i from the collection
            let (elem_reg, elem_ty) = match &kind {
                ElemKind::VecTuple(types) => {
                    let elem_ty = types[i].clone();
                    let reg = self.emitter.emit_get_tuple_element(&coll_reg, &coll_ty, i, &elem_ty);
                    (reg, elem_ty)
                }
                ElemKind::StackedDict(comp_types) => {
                    // Each component tensor: get + slice -> then pack into a tuple
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
                match &step_ty {
                    // Body returns [carry, output]: extract carry for next iteration
                    StableHLOType::Tuple(elems, _) if elems.len() == 2 => {
                        let carry_elem_ty = elems[0].clone();
                        carry_reg = self.emitter.emit_get_tuple_element(
                            &step_reg, &step_ty, 0, &carry_elem_ty,
                        );
                        carry_ty = carry_elem_ty;
                        last_scan_result = Some((step_reg, step_ty));
                    }
                    // Body returns just carry (no output): wrap in [carry, ()] for first()
                    _ => {
                        carry_reg = step_reg;
                        carry_ty = step_ty.clone();
                        let unit_ty = StableHLOType::Tuple(vec![], None);
                        let (unit_reg, _) = self.emitter.emit_tuple(&[], &[]);
                        let (wrapped_reg, wrapped_ty) = self.emitter.emit_tuple(
                            &[carry_reg, unit_reg],
                            &[carry_ty.clone(), unit_ty],
                        );
                        last_scan_result = Some((wrapped_reg, wrapped_ty));
                    }
                }
            } else {
                carry_reg = step_reg;
                carry_ty = step_ty;
            }
        }

        // For scan: return the full [carry, output] tuple so that
        // first() correctly extracts carry with the right type.
        // In ANF, first(scan(...)) becomes separate bindings, so scan
        // must return the pair for first to extract from.
        if is_scan {
            if let Some(result) = last_scan_result {
                return Ok(result);
            }
        }
        Ok((carry_reg, carry_ty))
    }

    /// Determine iteration count N and element extraction strategy from collection type.
    fn determine_scan_params(coll_ty: &StableHLOType) -> SheafResult<(usize, ElemKind)> {
        match coll_ty {
            StableHLOType::Tuple(types, _) if !types.is_empty() => {
                if types.iter().all(|t| matches!(t, StableHLOType::Tuple(..))) {
                    Ok((types.len(), ElemKind::VecTuple(types.clone())))
                } else if types.iter().all(|t| !t.shape().is_empty()) {
                    let n = types[0].shape()[0] as usize;
                    if n == 0 {
                        return Err(SheafError::Compile {
                            message: "scan VJP: stacked dict has zero elements".to_string(),
                            location: crate::core::error::SourceLocation::unknown(),
                        });
                    }
                    Ok((n, ElemKind::StackedDict(types.clone())))
                } else {
                    Err(SheafError::Compile {
                        message: "scan VJP: mixed Tuple element types not supported".to_string(),
                        location: crate::core::error::SourceLocation::unknown(),
                    })
                }
            }
            t if !t.shape().is_empty() => Ok((t.shape()[0] as usize, ElemKind::PlainTensor)),
            _ => Err(SheafError::Compile {
                message: "scan VJP: cannot determine iteration count from collection type".to_string(),
                location: crate::core::error::SourceLocation::unknown(),
            }),
        }
    }

    /// Extract element `i` from a collection, given the extraction strategy.
    fn extract_scan_elem(
        &mut self,
        i: usize,
        coll_reg: &Register,
        coll_ty: &StableHLOType,
        kind: &ElemKind,
    ) -> (Register, StableHLOType) {
        match kind {
            ElemKind::VecTuple(types) => {
                let elem_ty = types[i].clone();
                let reg = self.emitter.emit_get_tuple_element(coll_reg, coll_ty, i, &elem_ty);
                (reg, elem_ty)
            }
            ElemKind::StackedDict(comp_types) => {
                let mut parts = Vec::new();
                let mut part_types = Vec::new();
                for (k, comp_ty) in comp_types.iter().enumerate() {
                    let comp_reg = self.emitter.emit_get_tuple_element(
                        coll_reg, coll_ty, k, comp_ty,
                    );
                    let (slice_reg, slice_ty) =
                        self.emitter.emit_index_axis0(&comp_reg, comp_ty, i as i64);
                    parts.push(slice_reg);
                    part_types.push(slice_ty);
                }
                self.emitter.emit_tuple(&parts, &part_types)
            }
            ElemKind::PlainTensor => {
                self.emitter.emit_index_axis0(coll_reg, coll_ty, i as i64)
            }
        }
    }

    /// Scan VJP: backward differentiation through a scan.
    ///
    /// Given `result = first(scan(body, init, coll))`, computes the adjoints
    /// `adj_init` and `adj_coll` by:
    /// 1. Forward scan: re-compute and store all intermediate carries
    /// 2. Differentiate the body lambda via `reverse_grad`
    /// 3. Backward scan (reverse): apply body VJP at each step
    /// 4. Stack element adjoints into `adj_coll`
    ///
    /// Returns a tuple `(adj_init, adj_coll)`.
    pub(super) fn generate_scan_vjp(
        &mut self,
        args: &[CompiledExpr],
    ) -> SheafResult<(Register, StableHLOType)> {
        // args: [lambda, init, coll, adj]
        let lambda_expr = &args[0];
        let init_expr = &args[1];
        let coll_expr = &args[2];
        let adj_expr = &args[3];

        // Resolve lambda
        let lambda = match lambda_expr {
            CompiledExpr::Lambda { .. } => lambda_expr.clone(),
            CompiledExpr::Symbol(s) => {
                self.lambda_bindings.get(s).cloned().ok_or_else(|| SheafError::Compile {
                    message: format!("scan VJP: lambda '{}' not found", s),
                    location: crate::core::error::SourceLocation::unknown(),
                })?
            }
            _ => return Err(SheafError::Compile {
                message: "scan VJP: first argument must be a lambda".to_string(),
                location: crate::core::error::SourceLocation::unknown(),
            }),
        };
        let (carry_param, elem_param, body) = match lambda {
            CompiledExpr::Lambda { params, body } if params.len() == 2 => {
                (params[0].clone(), params[1].clone(), *body)
            }
            _ => return Err(SheafError::Compile {
                message: "scan VJP: lambda must have exactly 2 parameters".to_string(),
                location: crate::core::error::SourceLocation::unknown(),
            }),
        };

        // Generate init, coll, adj
        let (init_reg, init_ty) = self.generate(init_expr)?;
        let (coll_reg, coll_ty) = self.generate(coll_expr)?;
        let (adj_reg, adj_ty) = self.generate(adj_expr)?;

        // Determine N and extraction strategy
        let (n, kind) = Self::determine_scan_params(&coll_ty)?;

        // Resolve elem layout from collection (same logic as generate_reduce_scan)
        let coll_sym = if let CompiledExpr::Symbol(s) = coll_expr { Some(s.clone()) } else { None };
        let elem_layout = coll_sym
            .as_deref()
            .and_then(|s| self.tuple_key_layouts.get(s))
            .cloned()
            .or_else(|| {
                let layout_key = self.layout_key_map.get(&coll_reg)?;
                self.tuple_key_layouts.get(layout_key).cloned()
            })
            .or_else(|| {
                if let CompiledExpr::GetTupleElement { param, indices } = coll_expr {
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

        // === Forward scan: store all intermediate carries ===
        let mut carries: Vec<(Register, StableHLOType)> = Vec::new();
        carries.push((init_reg, init_ty.clone()));
        let mut fwd_carry_reg = init_reg;
        let mut fwd_carry_ty = init_ty.clone();

        for i in 0..n {
            let (elem_reg, elem_ty) = self.extract_scan_elem(i, &coll_reg, &coll_ty, &kind);

            if let Some(ref layout) = elem_layout {
                self.tuple_key_layouts.insert(elem_param.clone(), layout.clone());
            }

            let saved = self.bindings.clone();
            self.bindings.insert(carry_param.clone(), (fwd_carry_reg, fwd_carry_ty.clone()));
            self.bindings.insert(elem_param.clone(), (elem_reg, elem_ty));
            let (result_reg, result_ty) = self.generate(&body)?;
            self.bindings = saved;

            if elem_layout.is_some() {
                self.tuple_key_layouts.remove(&elem_param);
            }

            // Extract carry from [carry, output]
            match &result_ty {
                StableHLOType::Tuple(elems, _) if elems.len() >= 2 => {
                    let carry_ty = elems[0].clone();
                    let carry_reg = self.emitter.emit_get_tuple_element(
                        &result_reg, &result_ty, 0, &carry_ty,
                    );
                    carries.push((carry_reg, carry_ty.clone()));
                    fwd_carry_reg = carry_reg;
                    fwd_carry_ty = carry_ty;
                }
                _ => {
                    carries.push((result_reg, result_ty.clone()));
                    fwd_carry_reg = result_reg;
                    fwd_carry_ty = result_ty;
                }
            }
        }

        // === Differentiate body ===

        // Lower body's get calls to GTE if we have an elem layout.
        // Inside scan body lambdas, (get elem "W") hasn't been lowered
        // because lower_get_calls only processes outer function params.
        let lowered_body = if let Some(ref layout) = elem_layout {
            let mut index_map = std::collections::BTreeMap::new();
            for (key, &idx) in layout.iter() {
                index_map.insert(vec![key.clone()], vec![idx]);
            }
            for (key, _) in layout {
                let deep = build_deep_index_map(key, &self.tuple_key_layouts);
                index_map.extend(deep);
            }
            crate::lowering::config::lower_get_calls(&body, &elem_param, &index_map)
        } else {
            body.clone()
        };

        // ANF-convert body
        let body_anf = to_anf(&lowered_body);
        let (body_bindings, body_result) = match &body_anf {
            CompiledExpr::Let { bindings, body } => (bindings.clone(), body.as_ref().clone()),
            other => (vec![], other.clone()),
        };

        // Extract the carry sub-expression from body result.
        // The body returns [carry', output] as a Vector. We differentiate carry' only.
        let carry_result = match &body_result {
            CompiledExpr::Vector(elems) if !elems.is_empty() => elems[0].clone(),
            _ => body_result.clone(),
        };

        // Build shapes for body differentiation.
        // carry_param -> carry shape, elem components -> their shapes.
        let mut body_shapes: HashMap<String, Vec<i64>> = HashMap::new();
        body_shapes.insert(carry_param.clone(), fwd_carry_ty.shape().to_vec());

        // Get a sample elem to determine its type/shape
        let sample_elem_ty = {
            let (_, ty) = self.extract_scan_elem(0, &coll_reg, &coll_ty, &kind);
            ty
        };

        match &sample_elem_ty {
            StableHLOType::Tuple(elem_tys, _) => {
                // For tuple-typed elem, register shapes for each GTE component
                for (idx, ety) in elem_tys.iter().enumerate() {
                    let key = format!("{}@[{}]", elem_param, idx);
                    body_shapes.insert(key, ety.shape().to_vec());
                }
            }
            _ => {
                body_shapes.insert(elem_param.clone(), sample_elem_ty.shape().to_vec());
            }
        }

        // Propagate shapes through body bindings
        for (name, val) in &body_bindings {
            if let BindingPattern::Simple(name_str) = name {
                if let Some(sh) = try_infer_shape(val, &body_shapes) {
                    body_shapes.insert(name_str.clone(), sh);
                }
            }
        }

        // Determine wrt list: carry + elem fields (GTE bindings) or elem directly
        let mut wrt = vec![carry_param.clone()];
        let elem_gte_names: Vec<(String, usize)> = if matches!(&sample_elem_ty, StableHLOType::Tuple(..)) {
            // Find GTE bindings referencing elem_param
            body_bindings.iter()
                .filter_map(|(name, val)| {
                    if let CompiledExpr::GetTupleElement { param, indices } = val {
                        if param == &elem_param && indices.len() == 1 {
                            if let BindingPattern::Simple(name_str) = name {
                                return Some((name_str.clone(), indices[0]));
                            }
                        }
                    }
                    None
                })
                .collect()
        } else {
            wrt.push(elem_param.clone());
            vec![]
        };
        for (name, _) in &elem_gte_names {
            wrt.push(name.clone());
        }

        // Convert body_bindings to string keys for reverse_grad
        let body_bindings_str: Vec<(String, CompiledExpr)> = body_bindings
            .iter()
            .filter_map(|(pat, expr)| {
                if let BindingPattern::Simple(s) = pat {
                    Some((s.clone(), expr.clone()))
                } else {
                    None
                }
            })
            .collect();

        // Run reverse_grad on body
        let (mut body_bwd, body_grad_map) = reverse_grad(
            &body_bindings_str, &carry_result, &wrt, &body_shapes,
        );

        // Replace seed (Float(1.0)) with Symbol("__scan_adj_carry")
        // so each backward iteration uses the current adjoint carry.
        if !body_bwd.is_empty() {
            body_bwd[0].1 = CompiledExpr::Symbol("__scan_adj_carry".to_string());
        }

        // === Backward scan: iterate in reverse ===
        let mut adj_carry_reg = adj_reg;
        let mut adj_carry_ty = adj_ty.clone();
        // Collect per-iteration elem adjoints (stored in reverse order: n-1, n-2, ..., 0)
        let mut adj_elem_parts: Vec<Vec<(Register, StableHLOType)>> = Vec::new();

        for i in (0..n).rev() {
            let (elem_reg, elem_ty) = self.extract_scan_elem(i, &coll_reg, &coll_ty, &kind);

            // Bind forward values for this iteration
            let saved_bindings = self.bindings.clone();
            let saved_layouts = self.tuple_key_layouts.clone();
            self.bindings.insert(carry_param.clone(), carries[i].clone());
            self.bindings.insert(elem_param.clone(), (elem_reg, elem_ty.clone()));
            if let Some(ref layout) = elem_layout {
                self.tuple_key_layouts.insert(elem_param.clone(), layout.clone());
            }

            // Re-generate forward body bindings (re-materialization)
            for (name, val) in &body_bindings {
                if let BindingPattern::Simple(name_str) = name {
                    self.generate_binding(name_str, val)?;
                }
            }

            // Bind adjoint seed
            self.bindings.insert("__scan_adj_carry".to_string(), (adj_carry_reg, adj_carry_ty.clone()));

            // Generate backward bindings
            for (name, val) in &body_bwd {
                self.generate_binding(name, val)?;
            }

            // Extract adj_carry (adjoint of carry_param)
            if let Some(grad_sym) = body_grad_map.get(&carry_param) {
                if let Some(&(reg, ref ty)) = self.bindings.get(grad_sym) {
                    adj_carry_reg = reg;
                    adj_carry_ty = ty.clone();
                }
            }

            // Extract adj_elem parts
            if matches!(&sample_elem_ty, StableHLOType::Tuple(..)) {
                let n_fields = match &sample_elem_ty {
                    StableHLOType::Tuple(tys, _) => tys.len(),
                    _ => 0,
                };
                let mut parts = vec![None; n_fields];
                for (gte_name, field_idx) in &elem_gte_names {
                    if let Some(grad_sym) = body_grad_map.get(gte_name) {
                        if let Some(&(reg, ref ty)) = self.bindings.get(grad_sym) {
                            parts[*field_idx] = Some((reg, ty.clone()));
                        }
                    }
                }
                // Fill missing parts with zeros
                let resolved_parts: Vec<(Register, StableHLOType)> = parts.into_iter()
                    .enumerate()
                    .map(|(idx, opt)| {
                        opt.unwrap_or_else(|| {
                            let field_ty = match &sample_elem_ty {
                                StableHLOType::Tuple(tys, _) => &tys[idx],
                                _ => unreachable!(),
                            };
                            self.emitter.emit_zeros(&field_ty.shape())
                        })
                    })
                    .collect();
                adj_elem_parts.push(resolved_parts);
            } else {
                if let Some(grad_sym) = body_grad_map.get(&elem_param) {
                    if let Some(&(reg, ref ty)) = self.bindings.get(grad_sym) {
                        adj_elem_parts.push(vec![(reg, ty.clone())]);
                    }
                }
            }

            self.bindings = saved_bindings;
            self.tuple_key_layouts = saved_layouts;
        }

        // Reverse to get order 0, 1, ..., n-1
        adj_elem_parts.reverse();

        // === Assemble adj_coll ===
        let (adj_coll_reg, adj_coll_ty) = match &kind {
            ElemKind::StackedDict(comp_types) => {
                // Stack each field's adjoints along axis 0
                let n_fields = comp_types.len();
                let mut stacked_fields = Vec::new();
                let mut stacked_types = Vec::new();
                for field_idx in 0..n_fields {
                    // Collect this field's adjoint from each iteration
                    let field_regs: Vec<Register> = adj_elem_parts.iter()
                        .map(|parts| parts[field_idx].0)
                        .collect();
                    let field_ty = &adj_elem_parts[0][field_idx].1;

                    // Stack: reshape each [d...] to [1, d...], then concatenate on axis 0
                    let old_shape = field_ty.shape();
                    let mut expanded_shape = vec![1i64];
                    expanded_shape.extend_from_slice(&old_shape);

                    let mut reshaped_regs = Vec::new();
                    let mut reshaped_types = Vec::new();
                    for &reg in &field_regs {
                        let (r, t) = self.emitter.emit_reshape(&reg, field_ty, &expanded_shape);
                        reshaped_regs.push(r);
                        reshaped_types.push(t);
                    }

                    let (concat_reg, concat_ty) = self.emitter.emit_concatenate(
                        &reshaped_regs, &reshaped_types, 0,
                    );
                    stacked_fields.push(concat_reg);
                    stacked_types.push(concat_ty);
                }
                self.emitter.emit_tuple(&stacked_fields, &stacked_types)
            }
            ElemKind::PlainTensor => {
                // Stack all element adjoints along axis 0
                let field_regs: Vec<Register> = adj_elem_parts.iter()
                    .map(|parts| parts[0].0)
                    .collect();
                let field_ty = &adj_elem_parts[0][0].1;

                let old_shape = field_ty.shape();
                let mut expanded_shape = vec![1i64];
                expanded_shape.extend_from_slice(&old_shape);

                let mut reshaped_regs = Vec::new();
                let mut reshaped_types = Vec::new();
                for &reg in &field_regs {
                    let (r, t) = self.emitter.emit_reshape(&reg, field_ty, &expanded_shape);
                    reshaped_regs.push(r);
                    reshaped_types.push(t);
                }

                self.emitter.emit_concatenate(&reshaped_regs, &reshaped_types, 0)
            }
            ElemKind::VecTuple(_types) => {
                // Reassemble into a VecTuple: tuple of per-iteration adj tuples
                let mut elem_regs = Vec::new();
                let mut elem_tys = Vec::new();
                for parts in &adj_elem_parts {
                    let regs: Vec<Register> = parts.iter().map(|(r, _)| *r).collect();
                    let tys: Vec<StableHLOType> = parts.iter().map(|(_, t)| t.clone()).collect();
                    let (r, t) = self.emitter.emit_tuple(&regs, &tys);
                    elem_regs.push(r);
                    elem_tys.push(t);
                }
                self.emitter.emit_tuple(&elem_regs, &elem_tys)
            }
        };

        // Return (adj_init, adj_coll)
        let adj_init_reg = adj_carry_reg;
        let adj_init_ty = adj_carry_ty;
        Ok(self.emitter.emit_tuple(
            &[adj_init_reg, adj_coll_reg],
            &[adj_init_ty, adj_coll_ty],
        ))
    }
}
