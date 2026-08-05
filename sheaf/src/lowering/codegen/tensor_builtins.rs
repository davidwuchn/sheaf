// Copyright (c) 2025 Damien Boureille
// Licensed under the MIT License.

//! Tensor creation, manipulation, and slicing builtin codegen.

use crate::lowering::stablehlo::{Register, StableHLOType};
use crate::core::expr::CompiledExpr;
use crate::core::error::{SheafError, SheafResult};
use crate::core::ast::SheafValue;
use super::CodeGenerator;

impl CodeGenerator {
    pub(super) fn generate_tensor_builtin(
        &mut self,
        name: &str,
        args: &[CompiledExpr],
    ) -> Option<SheafResult<(Register, StableHLOType)>> {
        match name {
            "zeros" if args.len() == 1 => Some(self.gen_zeros(args)),
            "ones" if args.len() == 1 => Some(self.gen_ones(args)),
            "eye" if !args.is_empty() && args.len() <= 2 => Some(self.gen_eye(args)),
            "one-hot" if args.len() == 2 => Some(self.gen_one_hot(args)),
            "reshape" if args.len() == 2 => Some(self.gen_reshape(args)),
            "transpose" | "tr" if args.len() == 1 || args.len() == 2 =>
                Some(self.gen_transpose(args)),
            "broadcast" if args.len() == 2 => Some(self.gen_broadcast(args)),
            "cast" if args.len() == 2 => Some(self.gen_cast(args)),
            "arange" | "range" if args.len() == 1 || args.len() == 2 =>
                Some(self.gen_arange(name, args)),
            "concat" if args.len() == 2 => Some(self.gen_concat(args)),
            "swapaxes" if args.len() == 3 => Some(self.gen_swapaxes(args)),
            "tril" if args.len() == 1 => Some(self.gen_tril(args)),
            "where" if args.len() == 3 => Some(self.gen_where(args)),
            "slice" if args.len() >= 2 => Some(self.gen_slice(args)),
            "dynamic-slice" if args.len() == 3 => Some(self.gen_dynamic_slice(args)),
            "tensor-split" if args.len() == 2 => Some(self.gen_tensor_split(args)),
            "roll" if args.len() == 2 => Some(self.gen_roll(args)),
        "flip" if args.len() == 1 => Some(self.gen_flip(args)),
            "index-update" if args.len() == 3 => Some(self.gen_index_update(args)),
            "append-and-roll" if args.len() == 2 => Some(self.gen_append_and_roll(args)),
            "random-key" if args.len() == 1 => Some(self.gen_random_key(args)),
            "random-normal" if args.len() == 2 => Some(self.gen_random_normal(args)),
            "random-uniform" if args.len() == 2 => Some(self.gen_random_uniform(args)),
            "random-randint" if args.len() == 4 => Some(self.gen_random_randint(args)),
            "random-split" if args.len() == 1 || args.len() == 2 =>
                Some(self.gen_random_split(args)),
            "choice" => Some(self.gen_choice(args)),
            _ => None,
        }
    }

    fn gen_zeros(&mut self, args: &[CompiledExpr]) -> SheafResult<(Register, StableHLOType)> {
        if let CompiledExpr::Vector(shape_elems) = &args[0] {
            let shape = self.parse_shape_vec(shape_elems)?;
            let (reg, ty) = self.emitter.emit_zeros(&shape);
            Ok((reg, ty))
        } else {
            Err(SheafError::Compile {
                message: "zeros expects a vector shape argument".to_string(),
                location: crate::core::error::SourceLocation::unknown(),
            })
        }
    }

    fn gen_ones(&mut self, args: &[CompiledExpr]) -> SheafResult<(Register, StableHLOType)> {
        if let CompiledExpr::Vector(shape_elems) = &args[0] {
            let shape = self.parse_shape_vec(shape_elems)?;
            let (reg, ty) = self.emitter.emit_ones(&shape);
            Ok((reg, ty))
        } else {
            Err(SheafError::Compile {
                message: "ones expects a vector shape argument".to_string(),
                location: crate::core::error::SourceLocation::unknown(),
            })
        }
    }

    fn gen_eye(&mut self, args: &[CompiledExpr]) -> SheafResult<(Register, StableHLOType)> {
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
        let (reg, ty) = self.emitter.emit_eye(n, m);
        Ok((reg, ty))
    }

    fn gen_one_hot(&mut self, args: &[CompiledExpr]) -> SheafResult<(Register, StableHLOType)> {
        let (indices_reg, indices_ty) = self.generate(&args[0])?;
        let num_classes = match &args[1] {
            CompiledExpr::Integer(v) => *v,
            CompiledExpr::Float(f) => *f as i64,
            _ => {
                // Try resolving via codegen (e.g. shape(x, -1) returns a known scalar)
                if let Ok((reg, _)) = self.generate(&args[1]) {
                    if let Some(v) = self.emitter.known_scalar_value(&reg) {
                        v as i64
                    } else {
                        return Err(SheafError::Compile {
                            message: "one-hot expects a constant integer for num_classes".to_string(),
                            location: crate::core::error::SourceLocation::unknown(),
                        });
                    }
                } else {
                    return Err(SheafError::Compile {
                        message: "one-hot expects a constant integer for num_classes".to_string(),
                        location: crate::core::error::SourceLocation::unknown(),
                    });
                }
            }
        };
        let (reg, ty) = self.emitter.emit_one_hot(&indices_reg, &indices_ty, num_classes);
        Ok((reg, ty))
    }

    fn gen_reshape(&mut self, args: &[CompiledExpr]) -> SheafResult<(Register, StableHLOType)> {
        let (operand_reg, operand_ty) = self.generate(&args[0])?;
        let shape_elems_owned;
        let shape_elems: &[CompiledExpr] = if let CompiledExpr::Vector(elems) = &args[1] {
            elems
        } else if let CompiledExpr::Quoted(val) = &args[1] {
            if let SheafValue::Vector(elems, _) = val.as_ref() {
                shape_elems_owned = elems.iter().map(|e| match e {
                    SheafValue::Integer(n, _) => CompiledExpr::Integer(*n),
                    SheafValue::Float(f, _) => CompiledExpr::Float(*f),
                    _ => CompiledExpr::Integer(0),
                }).collect::<Vec<_>>();
                &shape_elems_owned
            } else {
                return Err(SheafError::Compile {
                    message: "reshape expects a vector shape argument".to_string(),
                    location: crate::core::error::SourceLocation::unknown(),
                });
            }
        } else {
            return Err(SheafError::Compile {
                message: "reshape expects a vector shape argument".to_string(),
                location: crate::core::error::SourceLocation::unknown(),
            });
        };
        let mut new_shape = self.parse_shape_vec(shape_elems)?;
        if let Some(neg_idx) = new_shape.iter().position(|&d| d < 0) {
            let input_size: i64 = operand_ty.shape().iter().product();
            let known_size: i64 = new_shape.iter().filter(|&&d| d > 0).product();
            if known_size > 0 {
                new_shape[neg_idx] = input_size / known_size;
            }
        }
        let (reg, ty) = self.emitter.emit_reshape(
            &operand_reg,
            &operand_ty,
            &new_shape,
        );
        Ok((reg, ty))
    }

    fn gen_transpose(&mut self, args: &[CompiledExpr]) -> SheafResult<(Register, StableHLOType)> {
        let (operand_reg, operand_ty) = self.generate(&args[0])?;
        let permutation: Vec<i64> = if args.len() == 2 {
            match &args[1] {
                CompiledExpr::Vector(perm_elems) => perm_elems
                    .iter()
                    .map(|e| match e {
                        CompiledExpr::Integer(n) => Ok(*n),
                        _ => Err(SheafError::Compile {
                            message: "transpose: permutation elements must be integers".to_string(),
                            location: crate::core::error::SourceLocation::unknown(),
                        }),
                    })
                    .collect::<SheafResult<_>>()?,
                CompiledExpr::Quoted(val) => match val.as_ref() {
                    SheafValue::Vector(elems, _) => elems
                        .iter()
                        .map(|e| match e {
                            SheafValue::Integer(n, _) => Ok(*n),
                            _ => Err(SheafError::Compile {
                                message: "transpose: permutation elements must be integers".to_string(),
                                location: crate::core::error::SourceLocation::unknown(),
                            }),
                        })
                        .collect::<SheafResult<_>>()?,
                    _ => return Err(SheafError::Compile {
                        message: "transpose expects a vector permutation argument".to_string(),
                        location: crate::core::error::SourceLocation::unknown(),
                    }),
                },
                _ => return Err(SheafError::Compile {
                    message: "transpose expects a vector permutation argument".to_string(),
                    location: crate::core::error::SourceLocation::unknown(),
                }),
            }
        } else {
            let ndim = operand_ty.shape().len().max(2) as i64;
            let mut perm: Vec<i64> = (0..ndim).collect();
            perm.swap((ndim - 2) as usize, (ndim - 1) as usize);
            perm
        };
        let (reg, ty) = self.emitter.emit_transpose(
            &operand_reg,
            &operand_ty,
            &permutation,
        );
        Ok((reg, ty))
    }

    fn gen_broadcast(&mut self, args: &[CompiledExpr]) -> SheafResult<(Register, StableHLOType)> {
        let (operand_reg, operand_ty) = self.generate(&args[0])?;
        if let CompiledExpr::Vector(shape_elems) = &args[1] {
            let target_shape = self.parse_shape_vec(shape_elems)?;
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

    fn gen_cast(&mut self, args: &[CompiledExpr]) -> SheafResult<(Register, StableHLOType)> {
        let (src_reg, src_ty) = self.generate(&args[0])?;
        if let CompiledExpr::Keyword(dtype_str) = &args[1] {
            let target_dtype = match dtype_str.as_str() {
                "bf16" => "bf16",
                "f32" => "f32",
                "i32" => "i32",
                other => return Err(SheafError::Compile {
                    message: format!("cast: unsupported dtype :{}", other),
                    location: crate::core::error::SourceLocation::unknown(),
                }),
            };
            let target_ty = StableHLOType::typed_tensor(
                src_ty.shape().to_vec(),
                target_dtype,
            );
            let reg = self.emitter.emit_convert(&src_reg, &src_ty, &target_ty);
            Ok((reg, target_ty))
        } else {
            Err(SheafError::Compile {
                message: "cast expects a keyword dtype argument (:bf16, :f32)".to_string(),
                location: crate::core::error::SourceLocation::unknown(),
            })
        }
    }

    fn gen_arange(&mut self, name: &str, args: &[CompiledExpr]) -> SheafResult<(Register, StableHLOType)> {
        if args.len() == 1 {
            if let CompiledExpr::Integer(n) = &args[0] {
                let shape = vec![*n];
                let (reg, ty) = self.emitter.emit_iota(&shape, 0);
                Ok((reg, ty))
            } else {
                Err(SheafError::Compile {
                    message: format!("{} expects an integer argument", name),
                    location: crate::core::error::SourceLocation::unknown(),
                })
            }
        } else {
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
                    let (iota_reg, iota_ty) = self.emitter.emit_iota(&shape, 0);
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

    fn gen_concat(&mut self, args: &[CompiledExpr]) -> SheafResult<(Register, StableHLOType)> {
        if let CompiledExpr::Vector(tensor_exprs) = &args[0] {
            let mut operand_regs = Vec::new();
            let mut operand_types = Vec::new();
            for expr in tensor_exprs {
                let (reg, ty) = self.generate(expr)?;
                operand_regs.push(reg);
                operand_types.push(ty);
            }
            if let CompiledExpr::Integer(dim) = &args[1] {
                let (reg, ty) = self.emitter.emit_concatenate(
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

    fn gen_swapaxes(&mut self, args: &[CompiledExpr]) -> SheafResult<(Register, StableHLOType)> {
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
        let (reg, ty) = self.emitter.emit_swapaxes(
            &operand_reg,
            &operand_ty,
            axis1,
            axis2,
        );
        Ok((reg, ty))
    }

    fn gen_tril(&mut self, args: &[CompiledExpr]) -> SheafResult<(Register, StableHLOType)> {
        let (operand_reg, operand_ty) = self.generate(&args[0])?;
        let (reg, ty) = self.emitter.emit_tril(&operand_reg, &operand_ty);
        Ok((reg, ty))
    }

    fn gen_where(&mut self, args: &[CompiledExpr]) -> SheafResult<(Register, StableHLOType)> {
        let (condition_reg, condition_ty) = self.generate(&args[0])?;
        let (x_reg, x_ty) = self.generate(&args[1])?;
        let (y_reg, y_ty) = self.generate(&args[2])?;
        let (reg, ty) = self.emitter.emit_where(
            &condition_reg,
            &x_reg,
            &y_reg,
            &condition_ty,
            &x_ty,
            &y_ty,
        );
        Ok((reg, ty))
    }

    fn gen_slice(&mut self, args: &[CompiledExpr]) -> SheafResult<(Register, StableHLOType)> {
        let (operand_reg, operand_ty) = self.generate(&args[0])?;
        let mut positionals = Vec::new();
        let mut axis: Option<i64> = None;
        let mut i = 1;
        while i < args.len() {
            if let CompiledExpr::Keyword(k) = &args[i]
               && k == "axis" && i + 1 < args.len() 
            {
                if let CompiledExpr::Integer(n) = &args[i + 1] {
                    axis = Some(*n);
                }
                i += 2;
                continue;
            }
            positionals.push(&args[i]);
            i += 1;
        }
        let start = match positionals.first() {
            Some(CompiledExpr::Integer(n)) => *n,
            Some(CompiledExpr::Float(f)) if f.fract() == 0.0 => *f as i64,
            _ => {
                return Err(SheafError::Compile {
                    message: format!("slice: start must be integer, got {:?}", positionals.first()),
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
                CompiledExpr::Float(f) if f.fract() == 0.0 => *f as i64,
                _ => {
                    return Err(SheafError::Compile {
                        message: format!("slice: end must be integer, got {:?}", positionals[1]),
                        location: crate::core::error::SourceLocation::unknown(),
                    });
                }
            }
        } else {
            shape[axis_usize]
        };
        let (reg, ty) = self.emitter.emit_slice_axis(&operand_reg, &operand_ty, start, end, axis_usize);
        Ok((reg, ty))
    }

    fn gen_dynamic_slice(&mut self, args: &[CompiledExpr]) -> SheafResult<(Register, StableHLOType)> {
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
        let (reg, ty) = self.emitter.emit_slice_range(&operand_reg, &operand_ty, start, end);
        Ok((reg, ty))
    }

    fn gen_tensor_split(&mut self, args: &[CompiledExpr]) -> SheafResult<(Register, StableHLOType)> {
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
        let (reg, ty) = self.emitter.emit_tensor_split(&operand_reg, &operand_ty, num_sections);
        Ok((reg, ty))
    }

    fn gen_roll(&mut self, args: &[CompiledExpr]) -> SheafResult<(Register, StableHLOType)> {
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
        let (reg, ty) = self.emitter.emit_roll(&operand_reg, &operand_ty, shift);
        Ok((reg, ty))
    }

    fn gen_flip(&mut self, args: &[CompiledExpr]) -> SheafResult<(Register, StableHLOType)> {
        let (operand_reg, operand_ty) = self.generate(&args[0])?;
        let axis = 0;
        let (reg, ty) = self.emitter.emit_reverse(&operand_reg, &operand_ty, axis);
        Ok((reg, ty))
    }

    fn gen_index_update(&mut self, args: &[CompiledExpr]) -> SheafResult<(Register, StableHLOType)> {
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
        let (reg, ty) = self.emitter.emit_index_update(&operand_reg, &operand_ty, idx, &value_reg, &value_ty);
        Ok((reg, ty))
    }

    fn gen_append_and_roll(&mut self, args: &[CompiledExpr]) -> SheafResult<(Register, StableHLOType)> {
        let (operand_reg, operand_ty) = self.generate(&args[0])?;
        let (value_reg, value_ty) = self.generate(&args[1])?;
        let n = operand_ty.shape()[0];
        let (tail, tail_ty) = self.emitter.emit_slice_range(&operand_reg, &operand_ty, 1, n - 1);
        let val_1d_ty = StableHLOType::f32_tensor(vec![1]);
        let val_1d = self.emitter.fresh_register();
        self.emitter.body.push(format!(
            "    {} = stablehlo.reshape {} : ({}) -> {}",
            val_1d.to_mlir(),
            value_reg.to_mlir(),
            value_ty.to_mlir(),
            val_1d_ty.to_mlir(),
        ));
        let (reg, ty) = self.emitter.emit_concatenate(&[tail, val_1d], &[tail_ty, val_1d_ty], 0);
        Ok((reg, ty))
    }

    fn gen_random_key(&mut self, args: &[CompiledExpr]) -> SheafResult<(Register, StableHLOType)> {
        if let CompiledExpr::Integer(seed) = &args[0] {
            let (reg, ty) = self.emitter.emit_random_key( *seed);
            Ok((reg, ty))
        } else if let CompiledExpr::Float(f) = &args[0] {
            let (reg, ty) = self.emitter.emit_random_key( *f as i64);
            Ok((reg, ty))
        } else {
            Err(SheafError::Compile {
                message: "random-key expects an integer seed".to_string(),
                location: crate::core::error::SourceLocation::unknown(),
            })
        }
    }

    fn gen_random_normal(&mut self, args: &[CompiledExpr]) -> SheafResult<(Register, StableHLOType)> {
        if let CompiledExpr::Vector(shape_elems) = &args[1] {
            let shape = self.parse_shape_vec(shape_elems)?;
            let (key_reg, key_ty) = self.generate(&args[0])?;
            let (reg, ty) = self.emitter.emit_random_normal(&key_reg, &key_ty, &shape);
            Ok((reg, ty))
        } else {
            Err(SheafError::Compile {
                message: "random-normal expects a vector shape argument".to_string(),
                location: crate::core::error::SourceLocation::unknown(),
            })
        }
    }

    fn gen_random_uniform(&mut self, args: &[CompiledExpr]) -> SheafResult<(Register, StableHLOType)> {
        if let CompiledExpr::Vector(shape_elems) = &args[1] {
            let shape = self.parse_shape_vec(shape_elems)?;
            let (key_reg, key_ty) = self.generate(&args[0])?;
            let (reg, ty) = self.emitter.emit_random_uniform(&key_reg, &key_ty, &shape);
            Ok((reg, ty))
        } else {
            Err(SheafError::Compile {
                message: "random-uniform expects a vector shape argument".to_string(),
                location: crate::core::error::SourceLocation::unknown(),
            })
        }
    }

    fn gen_random_randint(&mut self, args: &[CompiledExpr]) -> SheafResult<(Register, StableHLOType)> {
        if let CompiledExpr::Vector(shape_elems) = &args[1] {
            let shape = self.parse_shape_vec(shape_elems)?;
            let (key_reg, key_ty) = self.generate(&args[0])?;
            let low = match &args[2] {
                CompiledExpr::Integer(n) => *n,
                CompiledExpr::Float(f) => *f as i64,
                _ => return Err(SheafError::Compile {
                    message: "random-randint: low must be an integer literal".to_string(),
                    location: crate::core::error::SourceLocation::unknown(),
                }),
            };
            let high = match &args[3] {
                CompiledExpr::Integer(n) => *n,
                CompiledExpr::Float(f) => *f as i64,
                _ => return Err(SheafError::Compile {
                    message: "random-randint: high must be an integer literal".to_string(),
                    location: crate::core::error::SourceLocation::unknown(),
                }),
            };
            let (reg, ty) = self.emitter.emit_random_randint(&key_reg, &key_ty, &shape, low, high);
            Ok((reg, ty))
        } else {
            Err(SheafError::Compile {
                message: "random-randint expects a vector shape argument".to_string(),
                location: crate::core::error::SourceLocation::unknown(),
            })
        }
    }

    fn gen_random_split(&mut self, args: &[CompiledExpr]) -> SheafResult<(Register, StableHLOType)> {
        let n = if args.len() == 2 {
            match &args[1] {
                CompiledExpr::Integer(n) => *n as usize,
                _ => return Err(SheafError::Compile {
                    message: "random-split: N must be an integer literal".to_string(),
                    location: crate::core::error::SourceLocation::unknown(),
                }),
            }
        } else {
            2
        };
        let (key_reg, key_ty) = self.generate(&args[0])?;
        Ok(self.emitter.emit_random_split_n(&key_reg, &key_ty, n))
    }

    fn gen_choice(&mut self, args: &[CompiledExpr]) -> SheafResult<(Register, StableHLOType)> {
        let (key_reg, key_ty) = self.generate(&args[0])?;
        let mut probs_expr = None;
        let mut i = 1;
        while i < args.len() {
            if let CompiledExpr::Keyword(k) = &args[i]
               && k == "p" && i + 1 < args.len() {
                    probs_expr = Some(&args[i + 1]);
                    i += 2;
                    continue;
                }
            i += 1;
        }
        match probs_expr {
            Some(expr) => {
                let (probs_reg, probs_ty) = self.generate(expr)?;
                let (reg, ty) = self.emitter.emit_choice(
                    &key_reg, &key_ty, &probs_reg, &probs_ty,
                );
                Ok((reg, ty))
            }
            None => Err(SheafError::Compile {
                message: "choice: requires :p probs argument for codegen".to_string(),
                location: crate::core::error::SourceLocation::unknown(),
            }),
        }
    }
}
