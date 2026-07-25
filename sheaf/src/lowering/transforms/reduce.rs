// Copyright (c) 2025 Damien Boureille
// Licensed under the MIT License.

use crate::autodiff::replace_symbol;
use crate::core::expr::{BindingPattern, CompiledExpr};
use crate::lowering::stablehlo::StableHLOType;
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};

static UNROLL_COUNTER: AtomicUsize = AtomicUsize::new(0);

/// Look up a nested tuple type.
fn resolve_type_at_indices(
    param: &str,
    indices: &[usize],
    param_types: &[(String, StableHLOType)],
) -> Option<StableHLOType> {
    let base = param_types
        .iter()
        .find(|(n, _)| n == param)
        .map(|(_, t)| t)?;
    let mut current = base;
    for &idx in indices {
        match current {
            StableHLOType::Tuple(elems, _) if idx < elems.len() => {
                current = &elems[idx];
            }
            _ => return None,
        }
    }
    Some(current.clone())
}

/// Unroll statically sized `reduce` calls into `Let` chains for symbolic AD.
pub fn unroll_reduces(
    expr: &CompiledExpr,
    param_types: &[(String, StableHLOType)],
) -> CompiledExpr {
    unroll_reduces_rec(expr, param_types, &HashMap::new())
}

fn unroll_reduces_rec(
    expr: &CompiledExpr,
    param_types: &[(String, StableHLOType)],
    let_env: &HashMap<String, CompiledExpr>,
) -> CompiledExpr {
    match expr {
        CompiledExpr::FunctionCall { name, args, .. } if name == "reduce" && args.len() == 3 => {
            let (carry_p, elem_p, body) = match &args[0] {
                CompiledExpr::Lambda { params, body } if params.len() == 2 => {
                    (&params[0], &params[1], body.as_ref())
                }
                _ => {
                    return CompiledExpr::FunctionCall {
                        name: name.clone(),
                        args: args
                            .iter()
                            .map(|a| unroll_reduces_rec(a, param_types, let_env))
                            .collect(),
                        loc: None,
                    };
                }
            };

            let init = unroll_reduces_rec(&args[1], param_types, let_env);
            let coll = unroll_reduces_rec(&args[2], param_types, let_env);

            let resolved_coll = match &coll {
                CompiledExpr::Symbol(s) => let_env.get(s).unwrap_or(&coll),
                other => other,
            };

            let unroll_info = match resolved_coll {
                CompiledExpr::GetTupleElement { param, indices } => {
                    resolve_type_at_indices(param, indices, param_types).and_then(|ty| match ty {
                        StableHLOType::Tuple(elems, _) => Some((
                            elems.len(),
                            UnrollColl::TupleElement {
                                param: param.clone(),
                                base_indices: indices.clone(),
                            },
                        )),
                        _ => None,
                    })
                }
                CompiledExpr::Vector(elems) => {
                    Some((elems.len(), UnrollColl::Vector(elems.clone())))
                }
                _ => None,
            };

            let (n, coll_info) = match unroll_info {
                Some(info) => info,
                None => {
                    return CompiledExpr::FunctionCall {
                        name: name.clone(),
                        args: vec![args[0].clone(), init, coll],
                        loc: None,
                    };
                }
            };

            if n == 0 {
                return init;
            }

            let id = UNROLL_COUNTER.fetch_add(1, Ordering::Relaxed);
            let mut bindings = Vec::with_capacity(n);

            for i in 0..n {
                let var_name = format!("__reduce_{}_{}", id, i);

                let elem_expr = match &coll_info {
                    UnrollColl::TupleElement {
                        param,
                        base_indices,
                    } => {
                        let mut indices = base_indices.clone();
                        indices.push(i);
                        CompiledExpr::GetTupleElement {
                            param: param.clone(),
                            indices,
                        }
                    }
                    UnrollColl::Vector(elems) => elems[i].clone(),
                };

                let carry_expr = if i == 0 {
                    init.clone()
                } else {
                    CompiledExpr::Symbol(format!("__reduce_{}_{}", id, i - 1))
                };

                let mut iteration_body = body.clone();
                iteration_body = replace_symbol(&iteration_body, carry_p, &carry_expr);
                iteration_body = replace_symbol(&iteration_body, elem_p, &elem_expr);

                iteration_body = unroll_reduces_rec(&iteration_body, param_types, let_env);

                bindings.push((BindingPattern::Simple(var_name), iteration_body));
            }

            let last_var = format!("__reduce_{}_{}", id, n - 1);
            CompiledExpr::Let {
                bindings,
                body: Box::new(CompiledExpr::Symbol(last_var)),
            }
        }

        CompiledExpr::FunctionCall { name, args, .. } => CompiledExpr::FunctionCall {
            name: name.clone(),
            args: args
                .iter()
                .map(|a| unroll_reduces_rec(a, param_types, let_env))
                .collect(),
            loc: None,
        },
        CompiledExpr::Let { bindings, body } => {
            let mut new_env = let_env.clone();
            let new_bindings: Vec<_> = bindings
                .iter()
                .map(|(k, v)| {
                    let resolved = unroll_reduces_rec(v, param_types, &new_env);
                    if let BindingPattern::Simple(k_str) = k {
                        new_env.insert(k_str.clone(), resolved.clone());
                    }
                    (k.clone(), resolved)
                })
                .collect();
            CompiledExpr::Let {
                bindings: new_bindings,
                body: Box::new(unroll_reduces_rec(body, param_types, &new_env)),
            }
        }
        CompiledExpr::Do(exprs) => CompiledExpr::Do(
            exprs
                .iter()
                .map(|e| unroll_reduces_rec(e, param_types, let_env))
                .collect(),
        ),
        CompiledExpr::If {
            condition,
            then_branch,
            else_branch,
        } => CompiledExpr::If {
            condition: Box::new(unroll_reduces_rec(condition, param_types, let_env)),
            then_branch: Box::new(unroll_reduces_rec(then_branch, param_types, let_env)),
            else_branch: else_branch
                .as_ref()
                .map(|e| Box::new(unroll_reduces_rec(e, param_types, let_env))),
        },
        CompiledExpr::Lambda { params, body } => CompiledExpr::Lambda {
            params: params.clone(),
            body: Box::new(unroll_reduces_rec(body, param_types, let_env)),
        },
        CompiledExpr::LambdaCall { callee, args } => CompiledExpr::LambdaCall {
            callee: Box::new(unroll_reduces_rec(callee, param_types, let_env)),
            args: args
                .iter()
                .map(|a| unroll_reduces_rec(a, param_types, let_env))
                .collect(),
        },
        CompiledExpr::Repeat {
            index_var,
            count,
            acc_var,
            acc_init,
            body,
        } => CompiledExpr::Repeat {
            index_var: index_var.clone(),
            count: Box::new(unroll_reduces_rec(count, param_types, let_env)),
            acc_var: acc_var.clone(),
            acc_init: Box::new(unroll_reduces_rec(acc_init, param_types, let_env)),
            body: Box::new(unroll_reduces_rec(body, param_types, let_env)),
        },
        CompiledExpr::Vector(elems) => CompiledExpr::Vector(
            elems
                .iter()
                .map(|e| unroll_reduces_rec(e, param_types, let_env))
                .collect(),
        ),
        other => other.clone(),
    }
}

enum UnrollColl {
    TupleElement {
        param: String,
        base_indices: Vec<usize>,
    },
    Vector(Vec<CompiledExpr>),
}
