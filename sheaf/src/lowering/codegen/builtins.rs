// Copyright (c) 2025 Damien Boureille
// Licensed under the MIT License.

//! Built-in function call codegen: dispatches to specialized modules
//! (math, tensor, reduction, collection) and handles remaining builtins
//! (nn ops, autodiff helpers, control flow) inline.

use crate::lowering::stablehlo::{Register, StableHLOType};
use crate::core::expr::CompiledExpr;
use crate::core::error::{SheafError, SheafResult};
use super::CodeGenerator;

impl<'a> CodeGenerator<'a> {
    /// Generate code for a function call
    pub(super) fn generate_function_call(
        &mut self,
        name: &str,
        args: &[CompiledExpr],
    ) -> SheafResult<(Register, StableHLOType)> {
        // Check if this is a user-defined function in the registry
        // Clone the signature to avoid borrow checker issues
        let registry = self.function_registry;
        let signature = registry
            .and_then(|registry| registry.get(name))
            .and_then(|func_def| func_def.signature.clone());

        if let Some(_signature) = signature {
            // Generate code for each argument
            let mut arg_registers = Vec::new();
            let mut arg_types = Vec::new();

            for arg in args {
                let (reg, ty) = self.generate(arg)?;
                arg_registers.push(reg);
                arg_types.push(ty);
            }

            // Inline user-defined functions when their body is available and
            // the call is monomorphic (arg types are known). This avoids the
            // problem of emitting a func.call to a function compiled with the
            // wrong (scalar) type from inference.
            let func_def = registry.and_then(|registry| registry.get(name));
            if let Some(func_def) = func_def
                && let Some(body) = &func_def.body_compiled {
                    // Bind arg registers to param names in our bindings map
                    let saved_bindings = self.bindings.clone();
                    let saved_layouts = self.tuple_key_layouts.clone();
                    let saved_idx_to_key = self.idx_to_key.clone();
                    for (param, (arg_expr, (reg, ty))) in func_def
                        .params
                        .iter()
                        .zip(args.iter().zip(arg_registers.iter().zip(arg_types.iter())))
                    {
                        self.bindings
                            .insert(param.clone(), (*reg, ty.clone()));

                        // Propagate layout from caller arg to callee param name.
                        // When inlining forward(x, p), "params" in forward's body
                        // needs the same layout as "p" in the caller's namespace.
                        if let CompiledExpr::Symbol(arg_sym) = arg_expr {
                            if let Some(layout) = self.tuple_key_layouts.get(arg_sym).cloned() {
                                self.tuple_key_layouts.insert(param.clone(), layout);
                            }
                            // Copy idx_to_key entries: (caller_sym, idx) -> key
                            // becomes also (callee_param, idx) -> key
                            let entries: Vec<_> = self.idx_to_key.iter()
                                .filter(|((name, _), _)| name == arg_sym)
                                .map(|((_, idx), key)| (*idx, key.clone()))
                                .collect();
                            for (idx, key) in entries {
                                self.idx_to_key.insert((param.clone(), idx), key);
                            }
                        }
                    }
                    let body = body.clone();
                    let result = self.generate(&body);
                    self.bindings = saved_bindings;
                    self.tuple_key_layouts = saved_layouts;
                    self.idx_to_key = saved_idx_to_key;
                    return result;
                }

            // Fallback: emit func.call (may have type issues if not monomorphic)
            let sig = registry
                .and_then(|registry| registry.get(name))
                .and_then(|f| f.signature.clone())
                .unwrap();
            let result_reg =
                self.emitter
                    .emit_func_call(name, &arg_registers, &arg_types, &sig.return_type);
            return Ok((result_reg, sig.return_type.clone()));
        }

        // stop-gradient: forward is identity; only affects autodiff (gradient = 0)
        if name == "stop-gradient" && args.len() == 1 {
            return self.generate(&args[0]);
        }

        // Delegate to specialized modules
        if let Some(result) = self.generate_math_builtin(name, args) {
            return result;
        }
        if let Some(result) = self.generate_tensor_builtin(name, args) {
            return result;
        }
        if let Some(result) = self.generate_reduction_builtin(name, args) {
            return result;
        }
        if let Some(result) = self.generate_collection_builtin(name, args) {
            result
        }

        // einsum: (einsum "spec" lhs rhs)
        else if name == "einsum" && args.len() >= 3 {
            let spec = match &args[0] {
                CompiledExpr::String(s) => s.clone(),
                _ => {
                    return Err(SheafError::Compile {
                        message: "einsum: first argument must be a string spec".to_string(),
                        location: crate::core::error::SourceLocation::unknown(),
                    });
                }
            };
            let (lhs_reg, lhs_ty) = self.generate(&args[1])?;
            let (rhs_reg, rhs_ty) = self.generate(&args[2])?;
            self.emitter
                .emit_einsum(&lhs_reg, &rhs_reg, &lhs_ty, &rhs_ty, &spec)
                .map_err(|msg| SheafError::Compile {
                    message: msg,
                    location: crate::core::error::SourceLocation::unknown(),
                })
        }
        // sum_to_shape: (sum_to_shape x [M N]), used by autodiff
        else if name == "sum_to_shape" && args.len() == 2 {
            self.gen_sum_to_shape(args)
        }
        // slice_grad: (slice_grad adj [target_shape] start), used by autodiff
        else if name == "slice_grad" && args.len() == 3 {
            self.gen_slice_grad(args)
        }
        // reduce: (reduce fn init coll)
        else if name == "reduce" && args.len() == 3 {
            self.generate_reduce_scan(&args[0], &args[1], &args[2], false)
        }
        // scan: (scan fn init coll)
        else if name == "scan" && args.len() == 3 {
            self.generate_reduce_scan(&args[0], &args[1], &args[2], true)
        }
        // tree-map: (tree-map f tree1 tree2 ...)
        else if name == "tree-map" && args.len() >= 2 {
            let lambda = &args[0];
            let tree_args = &args[1..];

            let mut tree_regs = Vec::new();
            let mut tree_tys = Vec::new();
            for arg in tree_args {
                let (reg, ty) = self.generate(arg)?;
                tree_regs.push(reg);
                tree_tys.push(ty);
            }

            if !tree_tys.iter().all(|t| t.tuple_structure_matches(&tree_tys[0])) {
                                let ty_strs: Vec<String> = tree_tys.iter().map(|t| {
                                let mlir = t.to_mlir();
                                let len = if let StableHLOType::Tuple(e, _) = t { e.len() } else { 0 };
                                format!("(len={} {}...)", len, &mlir[..mlir.len().min(80)])
                            }).collect();
                            return Err(SheafError::Compile {
                                message: format!("tree-map: all tree arguments must have the same tuple structure (got: {:?})", ty_strs),
                                location: crate::core::error::SourceLocation::unknown(),
                            });
            }

            self.generate_tree_map(lambda, &tree_regs, &tree_tys)
        }
        // __scan_vjp__: backward differentiation through scan
        else if name == "__scan_vjp__" && args.len() == 4 {
            self.generate_scan_vjp(args)
        }
        // tree-reduce: (tree-reduce f tree init)
        else if name == "tree-reduce" && args.len() == 3 {
            let lambda = &args[0];
            let (tree_reg, tree_ty) = self.generate(&args[1])?;
            let (acc_reg, acc_ty) = self.generate(&args[2])?;
            self.generate_tree_reduce(lambda, tree_reg, &tree_ty, acc_reg, &acc_ty)
        }
        else {
            Err(SheafError::Compile {
                message: format!("Function call not yet supported: {} (arity {})", name, args.len()),
                location: crate::core::error::SourceLocation::unknown(),
            })
        }
    }

    fn gen_sum_to_shape(
        &mut self,
        args: &[CompiledExpr],
    ) -> SheafResult<(Register, StableHLOType)> {
        let (mut reg, mut ty) = self.generate(&args[0])?;
        if let CompiledExpr::Vector(shape_elems) = &args[1] {
            let target_shape = self.parse_shape_vec(shape_elems)?;
            let from_shape = ty.shape().to_vec();

            if from_shape == target_shape {
                return Ok((reg, ty));
            }

            // 1. Sum over leading extra dimensions
            let extra = from_shape.len().saturating_sub(target_shape.len());
            for _ in 0..extra {
                let (r, t) = self.emitter.emit_reduce_sum(&reg, &ty, 0, false);
                reg = r;
                ty = t;
            }

            // 2. Sum over dims where target is 1 but current is > 1
            let cur_shape = ty.shape().to_vec();
            for (i, (&cur_d, &tgt_d)) in
                cur_shape.iter().zip(target_shape.iter()).enumerate().rev()
            {
                if tgt_d == 1 && cur_d > 1 {
                    let (r, t) =
                        self.emitter.emit_reduce_sum(&reg, &ty, i as i64, true);
                    reg = r;
                    ty = t;
                }
            }

            Ok((reg, ty))
        } else {
            Err(SheafError::Compile {
                message: "sum_to_shape expects a vector shape argument".to_string(),
                location: crate::core::error::SourceLocation::unknown(),
            })
        }
    }

    fn gen_slice_grad(
        &mut self,
        args: &[CompiledExpr],
    ) -> SheafResult<(Register, StableHLOType)> {
        let (adj_reg, adj_ty) = self.generate(&args[0])?;
        if let (CompiledExpr::Vector(shape_elems), CompiledExpr::Integer(start)) =
            (&args[1], &args[2])
        {
            let target_shape = self.parse_shape_vec(shape_elems)?;
            let adj_shape = adj_ty.shape().to_vec();

            // Determine which axis was sliced: find the first axis where
            // target_shape[i] != adj_shape[i]
            let axis = adj_shape
                .iter()
                .zip(target_shape.iter())
                .position(|(a, t)| a != t)
                .unwrap_or(0);

            let start_val = *start;
            let end_val = start_val + adj_shape[axis];
            let total = target_shape[axis];
            let pad_low = start_val;
            let pad_high = total - end_val;

            let zero_reg = self.emitter.emit_constant_f32(0.0);
            let result_ty = StableHLOType::f32_tensor(target_shape);
            let result_reg = self.emitter.fresh_register();

            let ndim = adj_shape.len();
            let low: Vec<String> = (0..ndim)
                .map(|i| if i == axis { pad_low.to_string() } else { "0".to_string() })
                .collect();
            let high: Vec<String> = (0..ndim)
                .map(|i| if i == axis { pad_high.to_string() } else { "0".to_string() })
                .collect();
            let interior: Vec<String> = vec!["0".to_string(); ndim];

            self.emitter.body.push(format!(
                "    {} = stablehlo.pad {}, {}, low = [{}], high = [{}], interior = [{}] : ({}, {}) -> {}",
                result_reg.to_mlir(),
                adj_reg.to_mlir(),
                zero_reg.to_mlir(),
                low.join(", "),
                high.join(", "),
                interior.join(", "),
                adj_ty.to_mlir(),
                StableHLOType::scalar_f32().to_mlir(),
                result_ty.to_mlir(),
            ));

            Ok((result_reg, result_ty))
        } else {
            Err(SheafError::Compile {
                message: "slice_grad expects (adj, [shape], start)".to_string(),
                location: crate::core::error::SourceLocation::unknown(),
            })
        }
    }

}
