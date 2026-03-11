// Copyright (c) 2025 Damien Boureille
// Licensed under the MIT License.

//! Built-in function call codegen: arithmetic, tensor ops, nn ops, etc.

use crate::compiler::stablehlo::{Register, StableHLOType};
use crate::core::compiler::CompiledExpr;
use crate::core::error::{SheafError, SheafResult};
use crate::runtime::{math_ops, nn_ops, tensor_ops};
use super::CodeGenerator;

impl CodeGenerator {
    /// Generate code for a function call
    pub(super) fn generate_function_call(
        &mut self,
        name: &str,
        args: &[CompiledExpr],
    ) -> SheafResult<(Register, StableHLOType)> {
        // Check if this is a user-defined function in the registry
        // Clone the signature to avoid borrow checker issues
        let signature = self
            .function_registry
            .get(name)
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
            let func_def = self.function_registry.get(name).cloned();
            if let Some(func_def) = func_def {
                if let Some(body) = &func_def.body_compiled {
                    // Bind arg registers to param names in our bindings map
                    let saved_bindings = self.bindings.clone();
                    for (param, (reg, ty)) in func_def
                        .params
                        .iter()
                        .zip(arg_registers.iter().zip(arg_types.iter()))
                    {
                        self.bindings
                            .insert(param.clone(), (*reg, ty.clone()));
                    }
                    let body = body.clone();
                    let result = self.generate(&body);
                    self.bindings = saved_bindings;
                    return result;
                }
            }

            // Fallback: emit func.call (may have type issues if not monomorphic)
            let sig = self
                .function_registry
                .get(name)
                .and_then(|f| f.signature.clone())
                .unwrap();
            let result_reg =
                self.emitter
                    .emit_func_call(name, &arg_registers, &arg_types, &sig.return_type);
            return Ok((result_reg, sig.return_type.clone()));
        }

        // Arithmetic operations (binary or n-ary fold-left)
        if matches!(name, "+" | "-" | "*" | "/") && args.len() >= 2 {
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
        // Extended arithmetic: **, //, mod
        else if matches!(name, "**" | "//" | "%" | "mod") && args.len() == 2 {
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
        // Min/max operations
        else if matches!(name, "min" | "max") && args.len() == 2 {
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
        // Comparison operations
        else if matches!(name, "=" | "==" | "!=" | "<" | "<=" | ">" | ">=") && args.len() == 2 {
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
        // Matrix multiply
        else if name == "@" && args.len() == 2 {
            let (lhs_reg, lhs_ty) = self.generate(&args[0])?;
            let (rhs_reg, rhs_ty) = self.generate(&args[1])?;
            let (result_reg, result_ty) =
                math_ops::emit_matmul(&mut self.emitter, &lhs_reg, &rhs_reg, &lhs_ty, &rhs_ty);
            Ok((result_reg, result_ty))
        }
        // Boolean binary operations
        else if matches!(name, "and" | "or") && args.len() == 2 {
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
        // Math unary operations: sqrt, exp, log, abs
        else if matches!(name, "sqrt" | "exp" | "log" | "abs") && args.len() == 1 {
            let (operand_reg, operand_ty) = self.generate(&args[0])?;
            let result_reg =
                math_ops::emit_math_unary(&mut self.emitter, name, &operand_reg, &operand_ty);
            Ok((result_reg, operand_ty))
        }
        // Boolean not
        else if name == "not" && args.len() == 1 {
            let (operand_reg, operand_ty) = self.generate(&args[0])?;
            let result_reg = math_ops::emit_not(&mut self.emitter, &operand_reg, &operand_ty);
            Ok((result_reg, operand_ty))
        }
        // Neural network unary operations: relu, sigmoid, tanh
        else if name == "relu" && args.len() == 1 {
            let (operand_reg, operand_ty) = self.generate(&args[0])?;
            let result_reg = nn_ops::emit_relu(&mut self.emitter, &operand_reg, &operand_ty);
            Ok((result_reg, operand_ty))
        } else if name == "sigmoid" && args.len() == 1 {
            let (operand_reg, operand_ty) = self.generate(&args[0])?;
            let result_reg = nn_ops::emit_sigmoid(&mut self.emitter, &operand_reg, &operand_ty);
            Ok((result_reg, operand_ty))
        } else if name == "tanh" && args.len() == 1 {
            let (operand_reg, operand_ty) = self.generate(&args[0])?;
            let result_reg = nn_ops::emit_tanh(&mut self.emitter, &operand_reg, &operand_ty);
            Ok((result_reg, operand_ty))
        }
        // gelu: (gelu x)
        else if name == "gelu" && args.len() == 1 {
            let (operand_reg, operand_ty) = self.generate(&args[0])?;
            let (result_reg, result_ty) = nn_ops::emit_gelu(&mut self.emitter, &operand_reg, &operand_ty);
            Ok((result_reg, result_ty))
        }
        // silu: (silu x)
        else if name == "silu" && args.len() == 1 {
            let (operand_reg, operand_ty) = self.generate(&args[0])?;
            let (result_reg, result_ty) = nn_ops::emit_silu(&mut self.emitter, &operand_reg, &operand_ty);
            Ok((result_reg, result_ty))
        }
        // leaky-relu: (leaky-relu x :negative_slope 0.01)
        else if name == "leaky-relu" && !args.is_empty() {
            let (operand_reg, operand_ty) = self.generate(&args[0])?;
            let mut alpha = 0.01;
            if args.len() >= 3 {
                if let (CompiledExpr::Keyword(k), CompiledExpr::Float(v)) = (&args[1], &args[2]) {
                    if k == "negative_slope" { alpha = *v; }
                }
            }
            let (result_reg, result_ty) = nn_ops::emit_leaky_relu(&mut self.emitter, &operand_reg, &operand_ty, alpha);
            Ok((result_reg, result_ty))
        }
        // selu: (selu x)
        else if name == "selu" && args.len() == 1 {
            let (operand_reg, operand_ty) = self.generate(&args[0])?;
            let (result_reg, result_ty) = nn_ops::emit_selu(&mut self.emitter, &operand_reg, &operand_ty);
            Ok((result_reg, result_ty))
        }
        // celu: (celu x :alpha 1.0)
        else if name == "celu" && !args.is_empty() {
            let (operand_reg, operand_ty) = self.generate(&args[0])?;
            let mut alpha = 1.0;
            if args.len() >= 3 {
                if let (CompiledExpr::Keyword(k), CompiledExpr::Float(v)) = (&args[1], &args[2]) {
                    if k == "alpha" { alpha = *v; }
                }
            }
            let (result_reg, result_ty) = nn_ops::emit_celu(&mut self.emitter, &operand_reg, &operand_ty, alpha);
            Ok((result_reg, result_ty))
        }
        // softmax: (softmax x :axis N)
        else if name == "softmax" && !args.is_empty() {
            let (operand_reg, operand_ty) = self.generate(&args[0])?;
            let mut axis: i64 = -1; // default: last axis
            let mut i = 1;
            while i + 1 < args.len() {
                if let CompiledExpr::Keyword(k) = &args[i] {
                    if k == "axis" {
                        if let CompiledExpr::Integer(n) = &args[i + 1] {
                            axis = *n;
                        }
                        i += 2;
                        continue;
                    }
                }
                i += 1;
            }
            let (result_reg, result_ty) = nn_ops::emit_softmax(&mut self.emitter, &operand_reg, &operand_ty, axis);
            Ok((result_reg, result_ty))
        }
        // log-softmax: (log-softmax x :axis N)
        else if name == "log-softmax" && !args.is_empty() {
            let (operand_reg, operand_ty) = self.generate(&args[0])?;
            let mut axis: i64 = -1;
            let mut i = 1;
            while i + 1 < args.len() {
                if let CompiledExpr::Keyword(k) = &args[i] {
                    if k == "axis" {
                        if let CompiledExpr::Integer(n) = &args[i + 1] {
                            axis = *n;
                        }
                        i += 2;
                        continue;
                    }
                }
                i += 1;
            }
            let (result_reg, result_ty) = nn_ops::emit_log_softmax(&mut self.emitter, &operand_reg, &operand_ty, axis);
            Ok((result_reg, result_ty))
        }
        // mse-loss: (mse-loss pred target)
        else if name == "mse-loss" && args.len() == 2 {
            let (pred_reg, pred_ty) = self.generate(&args[0])?;
            let (target_reg, target_ty) = self.generate(&args[1])?;
            let (reg, ty) = nn_ops::emit_mse_loss(&mut self.emitter, &pred_reg, &target_reg, &pred_ty, &target_ty);
            Ok((reg, ty))
        }
        // mae-loss: (mae-loss pred target)
        else if name == "mae-loss" && args.len() == 2 {
            let (pred_reg, pred_ty) = self.generate(&args[0])?;
            let (target_reg, target_ty) = self.generate(&args[1])?;
            let (reg, ty) = nn_ops::emit_mae_loss(&mut self.emitter, &pred_reg, &target_reg, &pred_ty, &target_ty);
            Ok((reg, ty))
        }
        // sparse-cross-entropy: (sparse-cross-entropy logits labels)
        else if name == "sparse-cross-entropy" && args.len() == 2 {
            let (logits_reg, logits_ty) = self.generate(&args[0])?;
            let (labels_reg, labels_ty) = self.generate(&args[1])?;
            let (reg, ty) = nn_ops::emit_sparse_cross_entropy(&mut self.emitter, &logits_reg, &labels_reg, &logits_ty, &labels_ty);
            Ok((reg, ty))
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
        // shape: (shape tensor) or (shape tensor axis)
        else if name == "shape" && !args.is_empty() {
            let (_, operand_ty) = self.generate(&args[0])?;
            let dims = operand_ty.shape();
            if dims.is_empty() {
                return Err(SheafError::Compile {
                    message: "shape: cannot query shape of scalar or tuple".to_string(),
                    location: crate::core::error::SourceLocation::unknown(),
                });
            }
            if args.len() >= 2 {
                // (shape tensor axis) -> scalar integer
                if let CompiledExpr::Integer(ax) = &args[1] {
                    let idx = if *ax < 0 { (dims.len() as i64 + *ax) as usize } else { *ax as usize };
                    let reg = self.emitter.emit_constant_f32(dims[idx] as f64);
                    Ok((reg, StableHLOType::ScalarF32))
                } else {
                    Err(SheafError::Compile {
                        message: "shape: axis must be integer".to_string(),
                        location: crate::core::error::SourceLocation::unknown(),
                    })
                }
            } else {
                // (shape tensor) -> 1D tensor of dims
                let data: Vec<f64> = dims.iter().map(|&d| d as f64).collect();
                let shape = vec![data.len() as i64];
                let (reg, ty) = self.emitter.emit_nd_tensor_constant(&data, &shape);
                Ok((reg, ty))
            }
        }
        // first: (first x) — special case: (first (scan fn init coll)) = final carry
        else if name == "first" && args.len() == 1 {
            if let CompiledExpr::FunctionCall { name: inner, args: inner_args } = &args[0] {
                if inner == "scan" && inner_args.len() == 3 {
                    return self.generate_reduce_scan(&inner_args[0], &inner_args[1], &inner_args[2]);
                }
            }
            let (operand_reg, operand_ty) = self.generate(&args[0])?;
            match &operand_ty {
                StableHLOType::Tuple(elems) => {
                    let elem_ty = elems[0].clone();
                    let reg = self.emitter.emit_get_tuple_element(&operand_reg, &operand_ty, 0, &elem_ty);
                    Ok((reg, elem_ty))
                }
                _ if operand_ty.shape().is_empty() => {
                    Err(SheafError::Compile {
                        message: "first: cannot index a scalar".to_string(),
                        location: crate::core::error::SourceLocation::unknown(),
                    })
                }
                _ => {
                    let (reg, ty) = self.emitter.emit_index_axis0(&operand_reg, &operand_ty, 0);
                    Ok((reg, ty))
                }
            }
        }
        // second: (second x)
        else if name == "second" && args.len() == 1 {
            let (operand_reg, operand_ty) = self.generate(&args[0])?;
            match &operand_ty {
                StableHLOType::Tuple(elems) => {
                    let elem_ty = elems[1].clone();
                    let reg = self.emitter.emit_get_tuple_element(&operand_reg, &operand_ty, 1, &elem_ty);
                    Ok((reg, elem_ty))
                }
                _ if operand_ty.shape().is_empty() => {
                    Err(SheafError::Compile {
                        message: "second: cannot index a scalar".to_string(),
                        location: crate::core::error::SourceLocation::unknown(),
                    })
                }
                _ => {
                    let (reg, ty) = self.emitter.emit_index_axis0(&operand_reg, &operand_ty, 1);
                    Ok((reg, ty))
                }
            }
        }
        // zeros: (zeros [M N])
        else if name == "zeros" && args.len() == 1 {
            if let CompiledExpr::Vector(shape_elems) = &args[0] {
                let shape = Self::parse_shape_vec(shape_elems)?;
                let (reg, ty) = tensor_ops::emit_zeros(&mut self.emitter, &shape);
                Ok((reg, ty))
            } else {
                Err(SheafError::Compile {
                    message: "zeros expects a vector shape argument".to_string(),
                    location: crate::core::error::SourceLocation::unknown(),
                })
            }
        }
        // sum_to_shape: (sum_to_shape x [M N]) — reduce x by summing over
        // broadcast dimensions until its shape matches [M N].
        // Used by autodiff to undo implicit broadcasting in +, -, *, /.
        else if name == "sum_to_shape" && args.len() == 2 {
            let (mut reg, mut ty) = self.generate(&args[0])?;
            if let CompiledExpr::Vector(shape_elems) = &args[1] {
                let target_shape = Self::parse_shape_vec(shape_elems)?;
                let from_shape = ty.shape().to_vec();

                if from_shape == target_shape {
                    // Already matching — no-op
                    return Ok((reg, ty));
                }

                // 1. Sum over leading extra dimensions
                // e.g. from [4,256,384] target [256,384] → sum axis 0
                let extra = from_shape.len().saturating_sub(target_shape.len());
                for _ in 0..extra {
                    let (r, t) = self.emitter.emit_reduce_sum(&reg, &ty, 0, false);
                    reg = r;
                    ty = t;
                }

                // 2. Sum over dims where target is 1 but current is > 1
                // e.g. from [4,256,384] target [1,256,384] → sum axis 0 keepdims
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
        // slice_grad: (slice_grad adj [target_shape] start) — pad adj with zeros
        // to restore the original tensor shape before slicing.
        // The slice operates on axis 0 by default.
        else if name == "slice_grad" && args.len() == 3 {
            let (adj_reg, adj_ty) = self.generate(&args[0])?;
            if let (CompiledExpr::Vector(shape_elems), CompiledExpr::Integer(start)) =
                (&args[1], &args[2])
            {
                let target_shape = Self::parse_shape_vec(shape_elems)?;
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

                // Use stablehlo.pad
                let zero_reg = self.emitter.emit_constant_f32(0.0);
                let result_ty = StableHLOType::f32_tensor(target_shape);
                let result_reg = self.emitter.fresh_register();

                // Build pad config: edge_padding_low, edge_padding_high, interior_padding
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
        // broadcast: (broadcast x [M N]) — broadcast x to target shape
        else if name == "broadcast" && args.len() == 2 {
            let (operand_reg, operand_ty) = self.generate(&args[0])?;
            if let CompiledExpr::Vector(shape_elems) = &args[1] {
                let target_shape = Self::parse_shape_vec(shape_elems)?;
                let target_ty = StableHLOType::f32_tensor(target_shape);
                let reg = self.emitter.emit_broadcast(&operand_reg, &operand_ty, &target_ty);
                Ok((reg, target_ty))
            } else {
                Err(SheafError::Compile {
                    message: "broadcast expects a vector shape argument".to_string(),
                    location: crate::core::error::SourceLocation::unknown(),
                })
            }
        }
        // random-normal: (random-normal key [M N])
        else if name == "random-normal" && args.len() == 2 {
            if let CompiledExpr::Vector(shape_elems) = &args[1] {
                let shape = Self::parse_shape_vec(shape_elems)?;
                let (reg, ty) = tensor_ops::emit_random_normal(&mut self.emitter, &shape);
                Ok((reg, ty))
            } else {
                Err(SheafError::Compile {
                    message: "random-normal expects a vector shape argument".to_string(),
                    location: crate::core::error::SourceLocation::unknown(),
                })
            }
        }
        // ones: (ones [M N])
        else if name == "ones" && args.len() == 1 {
            if let CompiledExpr::Vector(shape_elems) = &args[0] {
                let shape = Self::parse_shape_vec(shape_elems)?;
                let (reg, ty) = tensor_ops::emit_ones(&mut self.emitter, &shape);
                Ok((reg, ty))
            } else {
                Err(SheafError::Compile {
                    message: "ones expects a vector shape argument".to_string(),
                    location: crate::core::error::SourceLocation::unknown(),
                })
            }
        }
        // eye: (eye N) or (eye N M)
        else if name == "eye" && !args.is_empty() && args.len() <= 2 {
            let n = match &args[0] {
                CompiledExpr::Integer(v) => *v,
                _ => return Err(SheafError::Compile {
                    message: "eye expects integer arguments".to_string(),
                    location: crate::core::error::SourceLocation::unknown(),
                }),
            };
            let m = if args.len() == 2 {
                match &args[1] {
                    CompiledExpr::Integer(v) => *v,
                    _ => n,
                }
            } else {
                n
            };
            let (reg, ty) = tensor_ops::emit_eye(&mut self.emitter, n, m);
            Ok((reg, ty))
        }
        // one-hot: (one-hot indices num_classes)
        else if name == "one-hot" && args.len() == 2 {
            let (indices_reg, indices_ty) = self.generate(&args[0])?;
            let num_classes = match &args[1] {
                CompiledExpr::Integer(v) => *v,
                _ => return Err(SheafError::Compile {
                    message: "one-hot expects integer num_classes".to_string(),
                    location: crate::core::error::SourceLocation::unknown(),
                }),
            };
            let (reg, ty) = tensor_ops::emit_one_hot(&mut self.emitter, &indices_reg, &indices_ty, num_classes);
            Ok((reg, ty))
        }
        // reshape: (reshape tensor [M N])
        else if name == "reshape" && args.len() == 2 {
            let (operand_reg, operand_ty) = self.generate(&args[0])?;
            if let CompiledExpr::Vector(shape_elems) = &args[1] {
                let mut new_shape = Self::parse_shape_vec(shape_elems)?;
                // Resolve -1 dimensions: infer from input total size
                if let Some(neg_idx) = new_shape.iter().position(|&d| d < 0) {
                    let input_size: i64 = operand_ty.shape().iter().product();
                    let known_size: i64 = new_shape.iter().filter(|&&d| d > 0).product();
                    if known_size > 0 {
                        new_shape[neg_idx] = input_size / known_size;
                    }
                }
                let (reg, ty) = tensor_ops::emit_reshape(
                    &mut self.emitter,
                    &operand_reg,
                    &operand_ty,
                    &new_shape,
                );
                Ok((reg, ty))
            } else {
                Err(SheafError::Compile {
                    message: "reshape expects a vector shape argument".to_string(),
                    location: crate::core::error::SourceLocation::unknown(),
                })
            }
        }
        // transpose: (transpose tensor [1 0]) or (transpose tensor) — default perm [1 0]
        else if name == "transpose" && (args.len() == 1 || args.len() == 2) {
            let (operand_reg, operand_ty) = self.generate(&args[0])?;
            let permutation: Vec<i64> = if args.len() == 2 {
                if let CompiledExpr::Vector(perm_elems) = &args[1] {
                    perm_elems
                        .iter()
                        .map(|e| match e {
                            CompiledExpr::Integer(n) => *n,
                            _ => panic!("Permutation element must be integer"),
                        })
                        .collect()
                } else {
                    return Err(SheafError::Compile {
                        message: "transpose expects a vector permutation argument".to_string(),
                        location: crate::core::error::SourceLocation::unknown(),
                    });
                }
            } else {
                // Default: swap last two dims (works for 2D matrices)
                let ndim = operand_ty.shape().len().max(2) as i64;
                let mut perm: Vec<i64> = (0..ndim).collect();
                perm.swap((ndim - 2) as usize, (ndim - 1) as usize);
                perm
            };
            let (reg, ty) = tensor_ops::emit_transpose(
                &mut self.emitter,
                &operand_reg,
                &operand_ty,
                &permutation,
            );
            Ok((reg, ty))
        }
        // arange/range: (arange N), (range N), (range start end)
        else if (name == "arange" || name == "range") && (args.len() == 1 || args.len() == 2) {
            if args.len() == 1 {
                // (range N) or (arange N) -> tensor<Nxf32> [0, 1, ..., N-1]
                if let CompiledExpr::Integer(n) = &args[0] {
                    let shape = vec![*n];
                    let (reg, ty) = tensor_ops::emit_arange(&mut self.emitter, &shape, 0);
                    Ok((reg, ty))
                } else {
                    Err(SheafError::Compile {
                        message: format!("{} expects an integer argument", name),
                        location: crate::core::error::SourceLocation::unknown(),
                    })
                }
            } else {
                // (range start end) -> tensor<(end-start)xf32> [start, start+1, ..., end-1]
                match (&args[0], &args[1]) {
                    (CompiledExpr::Integer(start), CompiledExpr::Integer(end)) => {
                        let len = end - start;
                        if len <= 0 {
                            return Err(SheafError::Compile {
                                message: format!("range: end ({}) must be greater than start ({})", end, start),
                                location: crate::core::error::SourceLocation::unknown(),
                            });
                        }
                        let shape = vec![len];
                        // iota gives [0, 1, ..., len-1], then add start
                        let (iota_reg, iota_ty) = tensor_ops::emit_arange(&mut self.emitter, &shape, 0);
                        if *start == 0 {
                            return Ok((iota_reg, iota_ty));
                        }
                        let start_reg = self.emitter.emit_constant_f32(*start as f64);
                        let start_ty = StableHLOType::scalar_f32();
                        let (reg, ty) = self.emitter.emit_binop("add", &iota_reg, &start_reg, &iota_ty, &start_ty);
                        Ok((reg, ty))
                    }
                    _ => Err(SheafError::Compile {
                        message: "range expects integer arguments".to_string(),
                        location: crate::core::error::SourceLocation::unknown(),
                    }),
                }
            }
        }
        // concat: (concat [tensor1 tensor2 ...] dim)
        else if name == "concat" && args.len() == 2 {
            if let CompiledExpr::Vector(tensor_exprs) = &args[0] {
                // Generate all tensor operands
                let mut operand_regs = Vec::new();
                let mut operand_types = Vec::new();
                for expr in tensor_exprs {
                    let (reg, ty) = self.generate(expr)?;
                    operand_regs.push(reg);
                    operand_types.push(ty);
                }

                // Get dimension
                if let CompiledExpr::Integer(dim) = &args[1] {
                    let (reg, ty) = tensor_ops::emit_concatenate(
                        &mut self.emitter,
                        &operand_regs,
                        &operand_types,
                        *dim,
                    );
                    Ok((reg, ty))
                } else {
                    Err(SheafError::Compile {
                        message: "concat expects an integer dimension argument".to_string(),
                        location: crate::core::error::SourceLocation::unknown(),
                    })
                }
            } else {
                Err(SheafError::Compile {
                    message: "concat expects a vector of tensors as first argument".to_string(),
                    location: crate::core::error::SourceLocation::unknown(),
                })
            }
        }
        // where: (where condition x y)
        else if name == "where" && args.len() == 3 {
            let (condition_reg, condition_ty) = self.generate(&args[0])?;
            let (x_reg, x_ty) = self.generate(&args[1])?;
            let (y_reg, y_ty) = self.generate(&args[2])?;
            let (reg, ty) = tensor_ops::emit_where(
                &mut self.emitter,
                &condition_reg,
                &x_reg,
                &y_reg,
                &condition_ty,
                &x_ty,
                &y_ty,
            );
            Ok((reg, ty))
        }
        // swapaxes: (swapaxes x axis1 axis2)
        else if name == "swapaxes" && args.len() == 3 {
            let (operand_reg, operand_ty) = self.generate(&args[0])?;
            let axis1 = match &args[1] {
                CompiledExpr::Integer(n) => *n,
                _ => {
                    return Err(SheafError::Compile {
                        message: "swapaxes axis1 must be an integer".to_string(),
                        location: crate::core::error::SourceLocation::unknown(),
                    });
                }
            };
            let axis2 = match &args[2] {
                CompiledExpr::Integer(n) => *n,
                _ => {
                    return Err(SheafError::Compile {
                        message: "swapaxes axis2 must be an integer".to_string(),
                        location: crate::core::error::SourceLocation::unknown(),
                    });
                }
            };
            let (reg, ty) = tensor_ops::emit_swapaxes(
                &mut self.emitter,
                &operand_reg,
                &operand_ty,
                axis1,
                axis2,
            );
            Ok((reg, ty))
        }
        // tril: (tril x)
        else if name == "tril" && args.len() == 1 {
            let (operand_reg, operand_ty) = self.generate(&args[0])?;
            let (reg, ty) = tensor_ops::emit_tril(&mut self.emitter, &operand_reg, &operand_ty);
            Ok((reg, ty))
        }
        // sum/mean: (sum x :axis N) or (sum x :axis N :keepdims true)
        // Without :axis → reduce all dimensions to scalar
        else if (name == "sum" || name == "mean") && !args.is_empty() {
            let (operand_reg, operand_ty) = self.generate(&args[0])?;

            // Parse keyword args: :axis N :keepdims [bool?]
            // :keepdims may be bare (no following value) — treat as true
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
                        keepdims = true; // bare :keepdims flag
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
                    // No :axis → reduce all dimensions sequentially (last to first)
                    let ndim = operand_ty.shape().len();
                    if ndim == 0 {
                        (operand_reg, operand_ty)
                    } else {
                        let mut cur_reg = operand_reg;
                        let mut cur_ty = operand_ty;
                        for _ in (0..ndim).rev() {
                            // Always reduce axis 0 of the current shape after prior reductions,
                            // but it's simpler to always reduce the last axis (-1)
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
        // product: (product x :axis N) or (product x)
        else if name == "product" && !args.is_empty() {
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
        // min/max reduction: (min x :axis N) or (max x :axis N)
        // 1-arg form with keyword args — reduction along axis
        else if matches!(name, "min" | "max") && !args.is_empty()
            && (args.len() == 1 || matches!(&args[1], CompiledExpr::Keyword(_)))
        {
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
        // argmax/argmin: (argmax x :axis N)
        else if matches!(name, "argmax" | "argmin") && !args.is_empty() {
            let (operand_reg, operand_ty) = self.generate(&args[0])?;

            let mut axis: i64 = -1; // default: last axis
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
        // reduce: (reduce fn init coll) — static unrolling when coll type is known
        else if name == "reduce" && args.len() == 3 {
            self.generate_reduce_scan(&args[0], &args[1], &args[2])
        }
        // scan: (scan fn init coll) — same as reduce for the final carry
        // Note: (first (scan ...)) is intercepted above in the "first" handler.
        else if name == "scan" && args.len() == 3 {
            self.generate_reduce_scan(&args[0], &args[1], &args[2])
        }
        // tree-map: (tree-map f tree1 tree2 ...)
        // Static unrolling when tree args have known tuple types.
        else if name == "tree-map" && args.len() >= 2 {
            let lambda = &args[0];
            let tree_args = &args[1..];

            // Generate tree arguments to get their registers and types
            let mut tree_regs = Vec::new();
            let mut tree_tys = Vec::new();
            for arg in tree_args {
                let (reg, ty) = self.generate(arg)?;
                tree_regs.push(reg);
                tree_tys.push(ty);
            }

            // All tree args must have the same tuple structure
            if !tree_tys.iter().all(|t| t.tuple_structure_matches(&tree_tys[0])) {
                return Err(SheafError::Compile {
                    message: "tree-map: all tree arguments must have the same tuple structure"
                        .to_string(),
                    location: crate::core::error::SourceLocation::unknown(),
                });
            }

            self.generate_tree_map(lambda, &tree_regs, &tree_tys)
        }
        // tree-reduce: (tree-reduce f tree init)
        // Fold a binary function over all leaves of a pytree.
        else if name == "tree-reduce" && args.len() == 3 {
            let lambda = &args[0];
            let (tree_reg, tree_ty) = self.generate(&args[1])?;
            let (acc_reg, acc_ty) = self.generate(&args[2])?;
            self.generate_tree_reduce(lambda, tree_reg, &tree_ty, acc_reg, &acc_ty)
        }
        // get: (get tensor idx) or (get tuple :key)
        else if name == "get" && args.len() >= 2 {
            // Peek at the symbol name before generating to look up key layout
            let sym_name = if let CompiledExpr::Symbol(s) = &args[0] { Some(s.clone()) } else { None };
            let (operand_reg, operand_ty) = self.generate(&args[0])?;
            match &operand_ty {
                // Tuple + keyword/string — try to resolve via tuple_key_layouts
                StableHLOType::Tuple(_) => {
                    // Resolve layout key from symbol name or layout_key_map
                    let layout_key = sym_name.clone()
                        .or_else(|| self.layout_key_map.get(&operand_reg).cloned());
                    if let Some(ref start_key) = layout_key {
                        // Collect all keyword args for multi-key get: (get x :k1 :k2 ...)
                        let keywords: Vec<String> = args[1..].iter().filter_map(|a| match a {
                            CompiledExpr::Keyword(k) | CompiledExpr::String(k) => Some(k.clone()),
                            _ => None,
                        }).collect();
                        if keywords.len() == args.len() - 1 && !keywords.is_empty() {
                            let mut cur_reg = operand_reg.clone();
                            let mut cur_ty = operand_ty.clone();
                            let mut cur_key = start_key.clone();
                            let mut ok = true;
                            for key in &keywords {
                                if let StableHLOType::Tuple(sub_types) = &cur_ty {
                                    if let Some(layout) = self.tuple_key_layouts.get(&cur_key).cloned() {
                                        if let Some(&idx) = layout.get(key) {
                                            let sub_ty = sub_types[idx].clone();
                                            cur_reg = self.emitter.emit_get_tuple_element(
                                                &cur_reg, &cur_ty, idx, &sub_ty,
                                            );
                                            cur_ty = sub_ty;
                                            cur_key = key.clone();
                                            continue;
                                        }
                                    }
                                }
                                ok = false;
                                break;
                            }
                            if ok {
                                // Track the result's layout key for nested gets
                                self.layout_key_map.insert(cur_reg, cur_key);
                                return Ok((cur_reg, cur_ty));
                            }
                        }
                    }
                    Err(SheafError::Compile {
                        message: format!(
                            "get on dict/tuple requires type info (sym={:?} keys={:?})",
                            sym_name,
                            args[1..].iter().map(|a| format!("{:?}", a)).collect::<Vec<_>>()
                        ),
                        location: crate::core::error::SourceLocation::unknown(),
                    })
                }
                // Tensor + integer → index axis 0
                _ if !operand_ty.shape().is_empty() => {
                    match &args[1] {
                        CompiledExpr::Integer(idx) => {
                            let actual_idx = if *idx < 0 {
                                operand_ty.shape()[0] + *idx
                            } else {
                                *idx
                            };
                            let (reg, ty) = self.emitter.emit_index_axis0(&operand_reg, &operand_ty, actual_idx);
                            Ok((reg, ty))
                        }
                        // (get tensor ... idx) — ellipsis: index last axis
                        CompiledExpr::Symbol(s) if s == "..." => {
                            if args.len() >= 3 {
                                if let CompiledExpr::Integer(idx) = &args[2] {
                                    let shape = operand_ty.shape();
                                    let ndim = shape.len();
                                    let last_axis_size = shape[ndim - 1];
                                    let actual_idx = if *idx < 0 { last_axis_size + *idx } else { *idx };
                                    let (reg, ty) = self.emitter.emit_slice_last_axis(
                                        &operand_reg, &operand_ty, actual_idx, actual_idx + 1,
                                    );
                                    Ok((reg, ty))
                                } else {
                                    Err(SheafError::Compile {
                                        message: "get with ellipsis: index must be integer".to_string(),
                                        location: crate::core::error::SourceLocation::unknown(),
                                    })
                                }
                            } else {
                                Err(SheafError::Compile {
                                    message: "get with ellipsis: missing index after ...".to_string(),
                                    location: crate::core::error::SourceLocation::unknown(),
                                })
                            }
                        }
                        _ => {
                            // Tensor-indexed gather: (get operand indices)
                            let (idx_reg, idx_ty) = self.generate(&args[1])?;
                            if !idx_ty.shape().is_empty() {
                                let (reg, ty) = tensor_ops::emit_gather_axis0(
                                    &mut self.emitter,
                                    &operand_reg, &operand_ty,
                                    &idx_reg, &idx_ty,
                                );
                                Ok((reg, ty))
                            } else {
                                Err(SheafError::Compile {
                                    message: "get on tensor: index must be integer, ellipsis, or tensor".to_string(),
                                    location: crate::core::error::SourceLocation::unknown(),
                                })
                            }
                        }
                    }
                }
                _ => {
                    Err(SheafError::Compile {
                        message: format!("get: unsupported operand type {}", operand_ty.to_mlir()),
                        location: crate::core::error::SourceLocation::unknown(),
                    })
                }
            }
        }
        // get-in: (get-in tuple [:k1 :k2]) — should be lowered by lower_get_calls
        else if name == "get-in" && args.len() >= 2 {
            Err(SheafError::Compile {
                message: "get-in requires type info (use --trace-with or auto-trace)".to_string(),
                location: crate::core::error::SourceLocation::unknown(),
            })
        }
        // minimum/maximum: element-wise min/max
        else if matches!(name, "minimum" | "maximum") && args.len() == 2 {
            let (lhs_reg, lhs_ty) = self.generate(&args[0])?;
            let (rhs_reg, rhs_ty) = self.generate(&args[1])?;
            let op = if name == "minimum" { "min" } else { "max" };
            let (result_reg, result_ty) = math_ops::emit_minmax(
                &mut self.emitter, op, &lhs_reg, &rhs_reg, &lhs_ty, &rhs_ty,
            );
            Ok((result_reg, result_ty))
        }
        // ndim: (ndim tensor) → compile-time rank
        else if name == "ndim" && args.len() == 1 {
            let (_, operand_ty) = self.generate(&args[0])?;
            let ndim = operand_ty.shape().len() as i64;
            let reg = self.emitter.emit_constant_f32(ndim as f64);
            Ok((reg, StableHLOType::ScalarF32))
        }
        // var: (var x :axis N) or (var x)
        else if name == "var" && !args.is_empty() {
            let (operand_reg, operand_ty) = self.generate(&args[0])?;

            // Parse keyword args: :axis N :keepdims [bool?]
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
                    // No axis → reduce all dimensions sequentially
                    let ndim = operand_ty.shape().len();
                    if ndim == 0 {
                        // var of scalar = 0
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
        // normalize: (normalize x :axis N) or (normalize x)
        else if name == "normalize" && !args.is_empty() {
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
        // dynamic-slice: (dynamic-slice tensor start end)
        else if name == "dynamic-slice" && args.len() == 3 {
            let (operand_reg, operand_ty) = self.generate(&args[0])?;
            let start = match &args[1] {
                CompiledExpr::Integer(n) => *n,
                _ => {
                    return Err(SheafError::Compile {
                        message: "dynamic-slice: start must be integer".to_string(),
                        location: crate::core::error::SourceLocation::unknown(),
                    });
                }
            };
            let end = match &args[2] {
                CompiledExpr::Integer(n) => *n,
                _ => {
                    return Err(SheafError::Compile {
                        message: "dynamic-slice: end must be integer".to_string(),
                        location: crate::core::error::SourceLocation::unknown(),
                    });
                }
            };
            let (reg, ty) = tensor_ops::emit_dynamic_slice(&mut self.emitter, &operand_reg, &operand_ty, start, end);
            Ok((reg, ty))
        }
        // roll: (roll tensor shift)
        else if name == "roll" && args.len() == 2 {
            let (operand_reg, operand_ty) = self.generate(&args[0])?;
            let shift = match &args[1] {
                CompiledExpr::Integer(n) => *n,
                _ => {
                    return Err(SheafError::Compile {
                        message: "roll: shift must be integer".to_string(),
                        location: crate::core::error::SourceLocation::unknown(),
                    });
                }
            };
            let (reg, ty) = tensor_ops::emit_roll(&mut self.emitter, &operand_reg, &operand_ty, shift);
            Ok((reg, ty))
        }
        // index-update: (index-update tensor idx new-value)
        else if name == "index-update" && args.len() == 3 {
            let (operand_reg, operand_ty) = self.generate(&args[0])?;
            let idx = match &args[1] {
                CompiledExpr::Integer(n) => *n,
                _ => {
                    return Err(SheafError::Compile {
                        message: "index-update: index must be integer".to_string(),
                        location: crate::core::error::SourceLocation::unknown(),
                    });
                }
            };
            let (value_reg, value_ty) = self.generate(&args[2])?;
            let (reg, ty) = tensor_ops::emit_index_update(&mut self.emitter, &operand_reg, &operand_ty, idx, &value_reg, &value_ty);
            Ok((reg, ty))
        }
        // append-and-roll: (append-and-roll tensor value)
        else if name == "append-and-roll" && args.len() == 2 {
            let (operand_reg, operand_ty) = self.generate(&args[0])?;
            let (value_reg, value_ty) = self.generate(&args[1])?;
            let (reg, ty) = tensor_ops::emit_append_and_roll(&mut self.emitter, &operand_reg, &operand_ty, &value_reg, &value_ty);
            Ok((reg, ty))
        }
        // last: (last x) — last element along axis 0
        else if name == "last" && args.len() == 1 {
            let (operand_reg, operand_ty) = self.generate(&args[0])?;
            let shape = operand_ty.shape();
            if shape.is_empty() {
                return Err(SheafError::Compile {
                    message: "last: cannot index a scalar".to_string(),
                    location: crate::core::error::SourceLocation::unknown(),
                });
            }
            let last_idx = shape[0] - 1;
            let (reg, ty) = self.emitter.emit_index_axis0(&operand_reg, &operand_ty, last_idx);
            Ok((reg, ty))
        }
        // int: (int x) — cast to integer (identity in codegen, shape info is already i64)
        else if name == "int" && args.len() == 1 {
            self.generate(&args[0])
        }
        // float: (float x) — cast to float (identity in codegen)
        else if name == "float" && args.len() == 1 {
            self.generate(&args[0])
        }
        // slice: (slice tensor start end :axis N) — start inclusive, end exclusive
        else if name == "slice" && args.len() >= 2 {
            let (operand_reg, operand_ty) = self.generate(&args[0])?;
            // Parse positional args (start, end) and keyword :axis
            let mut positionals = Vec::new();
            let mut axis: Option<i64> = None;
            let mut i = 1;
            while i < args.len() {
                if let CompiledExpr::Keyword(k) = &args[i] {
                    if k == "axis" && i + 1 < args.len() {
                        if let CompiledExpr::Integer(n) = &args[i + 1] {
                            axis = Some(*n);
                        }
                        i += 2;
                        continue;
                    }
                }
                positionals.push(&args[i]);
                i += 1;
            }
            let start = match positionals.first() {
                Some(CompiledExpr::Integer(n)) => *n,
                _ => {
                    return Err(SheafError::Compile {
                        message: "slice: start must be integer".to_string(),
                        location: crate::core::error::SourceLocation::unknown(),
                    });
                }
            };
            let shape = operand_ty.shape();
            let axis_val = axis.unwrap_or(0);
            let axis_usize = if axis_val < 0 { (shape.len() as i64 + axis_val) as usize } else { axis_val as usize };
            let end = if positionals.len() > 1 {
                match positionals[1] {
                    CompiledExpr::Integer(n) => *n,
                    _ => {
                        return Err(SheafError::Compile {
                            message: "slice: end must be integer".to_string(),
                            location: crate::core::error::SourceLocation::unknown(),
                        });
                    }
                }
            } else {
                shape[axis_usize]
            };
            let (reg, ty) = tensor_ops::emit_slice(&mut self.emitter, &operand_reg, &operand_ty, start, end, axis_usize);
            Ok((reg, ty))
        }
        // tensor-split: (tensor-split tensor num-sections)
        else if name == "tensor-split" && args.len() == 2 {
            let (operand_reg, operand_ty) = self.generate(&args[0])?;
            let num_sections = match &args[1] {
                CompiledExpr::Integer(n) => *n,
                _ => {
                    return Err(SheafError::Compile {
                        message: "tensor-split: num-sections must be integer".to_string(),
                        location: crate::core::error::SourceLocation::unknown(),
                    });
                }
            };
            let (reg, ty) = tensor_ops::emit_tensor_split(&mut self.emitter, &operand_reg, &operand_ty, num_sections);
            Ok((reg, ty))
        }
        // count/len: (count x) or (len x) — compile-time length
        else if (name == "count" || name == "len") && args.len() == 1 {
            let (_, operand_ty) = self.generate(&args[0])?;
            let shape = operand_ty.shape();
            if shape.is_empty() {
                return Err(SheafError::Compile {
                    message: format!("{}: cannot get length of scalar", name),
                    location: crate::core::error::SourceLocation::unknown(),
                });
            }
            let len = shape[0];
            let reg = self.emitter.emit_constant_f32(len as f64);
            Ok((reg, StableHLOType::ScalarF32))
        }
        // static: deprecated identity — evaluate inner expression directly
        else if name == "static" && args.len() == 1 {
            self.generate(&args[0])
        }
        // assoc: (assoc tuple :k1 v1 :k2 v2 ...) — create new tuple with replaced elements
        else if name == "assoc" && args.len() >= 3 && args.len() % 2 == 1 {
            let sym_name = if let CompiledExpr::Symbol(s) = &args[0] { Some(s.clone()) } else { None };
            let (base_reg, base_ty) = self.generate(&args[0])?;
            let elems = match &base_ty {
                StableHLOType::Tuple(e) => e.clone(),
                _ => {
                    return Err(SheafError::Compile {
                        message: "assoc: base must be a tuple".to_string(),
                        location: crate::core::error::SourceLocation::unknown(),
                    })
                }
            };
            let layout_key = sym_name.clone().or_else(|| self.layout_key_map.get(&base_reg).cloned());
            let layout = layout_key
                .as_ref()
                .and_then(|k| self.tuple_key_layouts.get(k).cloned());
            let layout = match layout {
                Some(l) => l,
                None => {
                    return Err(SheafError::Compile {
                        message: "assoc: no layout for tuple".to_string(),
                        location: crate::core::error::SourceLocation::unknown(),
                    })
                }
            };
            // Collect replacement values by tuple index
            let mut replacements: std::collections::HashMap<usize, (Register, StableHLOType)> =
                std::collections::HashMap::new();
            let mut i = 1;
            while i + 1 < args.len() {
                let key_name = match &args[i] {
                    CompiledExpr::Keyword(k) | CompiledExpr::String(k) => k.clone(),
                    _ => {
                        return Err(SheafError::Compile {
                            message: format!("assoc: expected keyword key, got {:?}", args[i]),
                            location: crate::core::error::SourceLocation::unknown(),
                        })
                    }
                };
                let idx = match layout.get(&key_name) {
                    Some(&idx) => idx,
                    None => {
                        return Err(SheafError::Compile {
                            message: format!("assoc: unknown key {:?}", key_name),
                            location: crate::core::error::SourceLocation::unknown(),
                        })
                    }
                };
                let (val_reg, val_ty) = self.generate(&args[i + 1])?;
                replacements.insert(idx, (val_reg, val_ty));
                i += 2;
            }
            // Build new tuple: use replacement or extract from original
            let mut new_regs = Vec::new();
            let mut new_tys = Vec::new();
            for (j, elem_ty) in elems.iter().enumerate() {
                if let Some((r, t)) = replacements.get(&j) {
                    new_regs.push(*r);
                    new_tys.push(t.clone());
                } else {
                    let r = self.emitter.emit_get_tuple_element(
                        &base_reg, &base_ty, j, elem_ty,
                    );
                    new_regs.push(r);
                    new_tys.push(elem_ty.clone());
                }
            }
            Ok(self.emitter.emit_tuple(&new_regs, &new_tys))
        }
        else {
            Err(SheafError::Compile {
                message: format!("Function call not yet supported: {} (arity {})", name, args.len()),
                location: crate::core::error::SourceLocation::unknown(),
            })
        }
    }
}
