// Copyright (c) 2025 Damien Boureille
// Licensed under the MIT License.

//! Compiler transforms: constant resolution, inlined-get lowering, layout propagation,
//! and scalar constant extraction. Shared between `sheaf build` (AOT) and JIT compilation.

use crate::core::expr::BindingPattern;
use crate::core::expr::CompiledExpr;
use crate::interpreter::value::Value;
use std::collections::{BTreeMap, HashMap};

use crate::core::error::SheafResult;
use crate::core::inference::{expr_is_tensor, infer_type_with_context};

/// Resolve scalar constants.
pub fn resolve_static_constants(
    expr: &CompiledExpr,
    constants: &HashMap<(String, Vec<usize>), f64>,
    param_shapes: &HashMap<String, Vec<i64>>,
    skip_lambda: bool,
) -> CompiledExpr {
    let mut locals: HashMap<String, CompiledExpr> = HashMap::new();
    let mut shapes: HashMap<String, Vec<i64>> = param_shapes.clone();
    resolve_constants_rec(expr, constants, &mut locals, &mut shapes, skip_lambda)
}

fn resolve_constants_rec(
    expr: &CompiledExpr,
    constants: &HashMap<(String, Vec<usize>), f64>,
    locals: &mut HashMap<String, CompiledExpr>,
    shapes: &mut HashMap<String, Vec<i64>>,
    skip_lambda: bool,
) -> CompiledExpr {
    match expr {
        CompiledExpr::GetTupleElement { param, indices } => {
            let key = (param.clone(), indices.clone());
            match constants.get(&key) {
                Some(&val) => f64_to_const(val),
                None => expr.clone(),
            }
        }
        CompiledExpr::Symbol(name) => locals.get(name).cloned().unwrap_or_else(|| expr.clone()),
        CompiledExpr::FunctionCall { name, args, .. } if name == "shape" && args.len() == 1 => {
            let shape_from_sym = match &args[0] {
                CompiledExpr::Symbol(s) => shapes.get(s.as_str()).cloned(),
                _ => None,
            };
            let shape = shape_from_sym.or_else(|| {
                let resolved =
                    resolve_constants_rec(&args[0], constants, locals, shapes, skip_lambda);
                try_infer_shape(&resolved, shapes)
            });
            if let Some(sh) = shape {
                return CompiledExpr::Vector(
                    sh.iter().map(|&d| CompiledExpr::Integer(d)).collect(),
                );
            }
            CompiledExpr::FunctionCall {
                name: name.clone(),
                args: args
                    .iter()
                    .map(|a| resolve_constants_rec(a, constants, locals, shapes, skip_lambda))
                    .collect(),
                loc: None,
            }
        }
        CompiledExpr::FunctionCall { name, args, .. } if name == "get" && args.len() == 2 => {
            let recv = resolve_constants_rec(&args[0], constants, locals, shapes, skip_lambda);
            let idx = resolve_constants_rec(&args[1], constants, locals, shapes, skip_lambda);
            if let (CompiledExpr::Vector(elems), CompiledExpr::Integer(i)) = (&recv, &idx) {
                let len = elems.len() as i64;
                let norm = if *i < 0 { len + i } else { *i };
                if norm >= 0 && norm < len {
                    return elems[norm as usize].clone();
                }
            }
            CompiledExpr::FunctionCall {
                name: name.clone(),
                args: vec![recv, idx],
                loc: None,
            }
        }
        CompiledExpr::FunctionCall { name, args, .. } if name == "cons" && args.len() == 2 => {
            let head = resolve_constants_rec(&args[0], constants, locals, shapes, skip_lambda);
            let tail = resolve_constants_rec(&args[1], constants, locals, shapes, skip_lambda);
            let tail_elems = match &tail {
                CompiledExpr::Vector(v) => Some(v.clone()),
                CompiledExpr::Quoted(inner) => {
                    use crate::core::ast::SheafValue;
                    match inner.as_ref() {
                        SheafValue::Vector(v, _loc) => {
                            // Convert SheafValue elements to CompiledExpr
                            let elems: Vec<CompiledExpr> = v
                                .iter()
                                .filter_map(|sv| match sv {
                                    SheafValue::Integer(n, _) => Some(CompiledExpr::Integer(*n)),
                                    SheafValue::Float(f, _) => Some(CompiledExpr::Float(*f)),
                                    _ => None,
                                })
                                .collect();
                            if elems.len() == v.len() {
                                Some(elems)
                            } else {
                                None
                            }
                        }
                        _ => None,
                    }
                }
                _ => None,
            };
            if let Some(mut elems) = tail_elems {
                elems.insert(0, head);
                CompiledExpr::Vector(elems)
            } else {
                CompiledExpr::FunctionCall {
                    name: name.clone(),
                    args: vec![head, tail],
                    loc: None,
                }
            }
        }
        CompiledExpr::FunctionCall { name, args, .. } if name == "int" && args.len() == 1 => {
            let inner = resolve_constants_rec(&args[0], constants, locals, shapes, skip_lambda);
            match &inner {
                CompiledExpr::Integer(_) => inner,
                CompiledExpr::Float(f) => CompiledExpr::Integer(*f as i64),
                _ => CompiledExpr::FunctionCall {
                    name: name.clone(),
                    args: vec![inner],
                    loc: None,
                },
            }
        }
        CompiledExpr::FunctionCall { name, args, .. }
            if (name == "first" || name == "last") && args.len() == 1 =>
        {
            let recv = resolve_constants_rec(&args[0], constants, locals, shapes, skip_lambda);
            push_first_last(name, recv)
        }
        CompiledExpr::FunctionCall { name, args, .. } => {
            let resolved: Vec<_> = args
                .iter()
                .map(|a| resolve_constants_rec(a, constants, locals, shapes, skip_lambda))
                .collect();
            try_fold_arithmetic(name, &resolved).unwrap_or_else(|| CompiledExpr::FunctionCall {
                name: name.clone(),
                args: resolved,
                loc: None,
            })
        }
        CompiledExpr::Let { bindings, body } => {
            let new_bindings: Vec<_> = bindings
                .iter()
                .map(|(k, v)| {
                    let resolved = resolve_constants_rec(v, constants, locals, shapes, skip_lambda);
                    match &resolved {
                        CompiledExpr::Integer(_) | CompiledExpr::Float(_) => {
                            if let BindingPattern::Simple(k_str) = k {
                                locals.insert(k_str.clone(), resolved.clone());
                            }
                        }
                        CompiledExpr::Symbol(aliased) => {
                            if let Some(sh) = shapes.get(aliased).cloned() {
                                if let BindingPattern::Simple(k_str) = k {
                                    shapes.insert(k_str.clone(), sh);
                                }
                            }
                        }
                        CompiledExpr::Vector(elems)
                            if elems.iter().all(|e| matches!(e, CompiledExpr::Integer(_))) =>
                        {
                            let sh: Vec<i64> = elems
                                .iter()
                                .filter_map(|e| {
                                    if let CompiledExpr::Integer(n) = e {
                                        Some(*n)
                                    } else {
                                        None
                                    }
                                })
                                .collect();
                            if let BindingPattern::Simple(k_str) = k {
                                shapes.insert(k_str.clone(), sh);
                            }
                        }
                        _ => {
                            if let BindingPattern::Simple(k_str) = k {
                                if let Some(sh) = try_infer_shape(&resolved, shapes) {
                                    shapes.insert(k_str.clone(), sh);
                                }
                            }
                        }
                    }
                    (k.clone(), resolved)
                })
                .collect();
            CompiledExpr::Let {
                bindings: new_bindings,
                body: Box::new(resolve_constants_rec(
                    body,
                    constants,
                    locals,
                    shapes,
                    skip_lambda,
                )),
            }
        }
        CompiledExpr::Do(exprs) => CompiledExpr::Do(
            exprs
                .iter()
                .map(|e| resolve_constants_rec(e, constants, locals, shapes, skip_lambda))
                .collect(),
        ),
        CompiledExpr::If {
            condition,
            then_branch,
            else_branch,
        } => CompiledExpr::If {
            condition: Box::new(resolve_constants_rec(
                condition,
                constants,
                locals,
                shapes,
                skip_lambda,
            )),
            then_branch: Box::new(resolve_constants_rec(
                then_branch,
                constants,
                locals,
                shapes,
                skip_lambda,
            )),
            else_branch: else_branch.as_ref().map(|e| {
                Box::new(resolve_constants_rec(
                    e,
                    constants,
                    locals,
                    shapes,
                    skip_lambda,
                ))
            }),
        },
        CompiledExpr::Lambda { params, body } => {
            if skip_lambda {
                let _ = (constants, locals, shapes);
                CompiledExpr::Lambda {
                    params: params.clone(),
                    body: body.clone(),
                }
            } else {
                let saved_locals = locals.clone();
                for p in params {
                    locals.remove(p);
                }
                let resolved_body =
                    resolve_constants_rec(body, constants, locals, shapes, skip_lambda);
                *locals = saved_locals;
                CompiledExpr::Lambda {
                    params: params.clone(),
                    body: Box::new(resolved_body),
                }
            }
        }
        CompiledExpr::LambdaCall { callee, args } => CompiledExpr::LambdaCall {
            callee: Box::new(resolve_constants_rec(
                callee,
                constants,
                locals,
                shapes,
                skip_lambda,
            )),
            args: args
                .iter()
                .map(|a| resolve_constants_rec(a, constants, locals, shapes, skip_lambda))
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
            count: Box::new(resolve_constants_rec(
                count,
                constants,
                locals,
                shapes,
                skip_lambda,
            )),
            acc_var: acc_var.clone(),
            acc_init: Box::new(resolve_constants_rec(
                acc_init,
                constants,
                locals,
                shapes,
                skip_lambda,
            )),
            body: Box::new(resolve_constants_rec(
                body,
                constants,
                locals,
                shapes,
                skip_lambda,
            )),
        },
        CompiledExpr::Vector(elems) => CompiledExpr::Vector(
            elems
                .iter()
                .map(|e| resolve_constants_rec(e, constants, locals, shapes, skip_lambda))
                .collect(),
        ),
        other => other.clone(),
    }
}

/// Lower dictionary accesses through `Let` aliases.
pub fn lower_inlined_gets(
    body: &CompiledExpr,
    index_maps: &[(String, BTreeMap<Vec<String>, Vec<usize>>)],
) -> CompiledExpr {
    let mut reverse: HashMap<(String, Vec<usize>), Vec<String>> = HashMap::new();
    for (param, imap) in index_maps {
        for (path, indices) in imap {
            reverse.insert((param.clone(), indices.clone()), path.clone());
        }
    }
    lower_inlined_gets_rec(body, index_maps, &mut HashMap::new(), &reverse)
}

fn lower_inlined_gets_rec(
    expr: &CompiledExpr,
    index_maps: &[(String, BTreeMap<Vec<String>, Vec<usize>>)],
    aliases: &mut HashMap<String, (String, Vec<String>)>,
    reverse: &HashMap<(String, Vec<usize>), Vec<String>>,
) -> CompiledExpr {
    match expr {
        CompiledExpr::FunctionCall { name, args, .. }
            if (name == "get" || name == "get-in") && args.len() >= 2 =>
        {
            if let CompiledExpr::Symbol(alias) = &args[0] {
                if let Some((root_param, prefix_path)) = aliases.get(alias) {
                    let mut path = prefix_path.clone();
                    let resolved = if name == "get-in" {
                        // (get-in alias [:k1 :k2 ...])
                        if let CompiledExpr::Vector(keys) = &args[1] {
                            let mut ok = true;
                            for k in keys {
                                match k {
                                    CompiledExpr::Keyword(s) => path.push(s.clone()),
                                    _ => {
                                        ok = false;
                                        break;
                                    }
                                }
                            }
                            ok
                        } else {
                            false
                        }
                    } else {
                        // (get alias :k1 :k2 ...)
                        let mut ok = true;
                        for arg in &args[1..] {
                            match arg {
                                CompiledExpr::Keyword(k) => path.push(k.clone()),
                                _ => {
                                    ok = false;
                                    break;
                                }
                            }
                        }
                        ok
                    };
                    if resolved {
                        if let Some(imap) = index_maps
                            .iter()
                            .find(|(p, _)| p == root_param)
                            .map(|(_, m)| m)
                        {
                            if let Some(indices) = imap.get(&path) {
                                return CompiledExpr::GetTupleElement {
                                    param: root_param.clone(),
                                    indices: indices.clone(),
                                };
                            }
                        }
                    }
                }
            }
            let resolved_args: Vec<_> = args
                .iter()
                .map(|a| lower_inlined_gets_rec(a, index_maps, aliases, reverse))
                .collect();
            if let CompiledExpr::GetTupleElement { param, indices } = &resolved_args[0] {
                let key = (param.clone(), indices.clone());
                if let Some(prefix_path) = reverse.get(&key) {
                    let mut path = prefix_path.clone();
                    let keys_ok = if name == "get-in" {
                        if let CompiledExpr::Vector(keys) = &resolved_args[1] {
                            let mut ok = true;
                            for k in keys {
                                match k {
                                    CompiledExpr::Keyword(s) => path.push(s.clone()),
                                    _ => {
                                        ok = false;
                                        break;
                                    }
                                }
                            }
                            ok
                        } else {
                            false
                        }
                    } else {
                        let mut ok = true;
                        for arg in &resolved_args[1..] {
                            match arg {
                                CompiledExpr::Keyword(k) => path.push(k.clone()),
                                _ => {
                                    ok = false;
                                    break;
                                }
                            }
                        }
                        ok
                    };
                    if keys_ok {
                        if let Some(imap) =
                            index_maps.iter().find(|(p, _)| p == param).map(|(_, m)| m)
                        {
                            if let Some(full_indices) = imap.get(&path) {
                                return CompiledExpr::GetTupleElement {
                                    param: param.clone(),
                                    indices: full_indices.clone(),
                                };
                            }
                        }
                    }
                }
            }
            CompiledExpr::FunctionCall {
                name: name.clone(),
                args: resolved_args,
                loc: None,
            }
        }
        CompiledExpr::Let { bindings, body } => {
            let new_bindings: Vec<_> = bindings
                .iter()
                .map(|(k, v)| {
                    let resolved = lower_inlined_gets_rec(v, index_maps, aliases, reverse);
                    match &resolved {
                        CompiledExpr::GetTupleElement { param, indices } => {
                            let key = (param.clone(), indices.clone());
                            if let Some(path) = reverse.get(&key) {
                                if let BindingPattern::Simple(k_str) = k {
                                    aliases.insert(k_str.clone(), (param.clone(), path.clone()));
                                }
                            }
                        }
                        CompiledExpr::Symbol(s) => {
                            if let Some(existing) = aliases.get(s).cloned() {
                                if let BindingPattern::Simple(k_str) = k {
                                    aliases.insert(k_str.clone(), existing);
                                }
                            }
                            else if index_maps.iter().any(|(p, _)| p == s) {
                                if let BindingPattern::Simple(k_str) = k {
                                    aliases.insert(k_str.clone(), (s.clone(), vec![]));
                                }
                            }
                        }
                        _ => {}
                    }
                    (k.clone(), resolved)
                })
                .collect();
            CompiledExpr::Let {
                bindings: new_bindings,
                body: Box::new(lower_inlined_gets_rec(body, index_maps, aliases, reverse)),
            }
        }
        CompiledExpr::FunctionCall { name, args, .. } => CompiledExpr::FunctionCall {
            name: name.clone(),
            args: args
                .iter()
                .map(|a| lower_inlined_gets_rec(a, index_maps, aliases, reverse))
                .collect(),
            loc: None,
        },
        CompiledExpr::Do(exprs) => CompiledExpr::Do(
            exprs
                .iter()
                .map(|e| lower_inlined_gets_rec(e, index_maps, aliases, reverse))
                .collect(),
        ),
        CompiledExpr::If {
            condition,
            then_branch,
            else_branch,
        } => CompiledExpr::If {
            condition: Box::new(lower_inlined_gets_rec(
                condition, index_maps, aliases, reverse,
            )),
            then_branch: Box::new(lower_inlined_gets_rec(
                then_branch,
                index_maps,
                aliases,
                reverse,
            )),
            else_branch: else_branch
                .as_ref()
                .map(|e| Box::new(lower_inlined_gets_rec(e, index_maps, aliases, reverse))),
        },
        CompiledExpr::Lambda { params, body } => CompiledExpr::Lambda {
            params: params.clone(),
            body: Box::new(lower_inlined_gets_rec(body, index_maps, aliases, reverse)),
        },
        CompiledExpr::LambdaCall { callee, args } => CompiledExpr::LambdaCall {
            callee: Box::new(lower_inlined_gets_rec(callee, index_maps, aliases, reverse)),
            args: args
                .iter()
                .map(|a| lower_inlined_gets_rec(a, index_maps, aliases, reverse))
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
            count: Box::new(lower_inlined_gets_rec(count, index_maps, aliases, reverse)),
            acc_var: acc_var.clone(),
            acc_init: Box::new(lower_inlined_gets_rec(
                acc_init, index_maps, aliases, reverse,
            )),
            body: Box::new(lower_inlined_gets_rec(body, index_maps, aliases, reverse)),
        },
        CompiledExpr::Vector(elems) => CompiledExpr::Vector(
            elems
                .iter()
                .map(|e| lower_inlined_gets_rec(e, index_maps, aliases, reverse))
                .collect(),
        ),
        other => other.clone(),
    }
}

/// Propagate dictionary layouts through `Let` bindings.
pub fn propagate_let_layouts(
    expr: &CompiledExpr,
    idx_to_key: &HashMap<(String, usize), String>,
    layouts: &mut HashMap<String, BTreeMap<String, usize>>,
) {
    match expr {
        CompiledExpr::Let { bindings, body } => {
            for (var_name, value_expr) in bindings {
                if let CompiledExpr::GetTupleElement { param, indices } = value_expr {
                    let mut cur = param.clone();
                    let mut resolved = true;
                    for &idx in indices {
                        if let Some(key) = idx_to_key.get(&(cur.clone(), idx)) {
                            cur = key.clone();
                        } else {
                            resolved = false;
                            break;
                        }
                    }
                    if resolved {
                        if let Some(sub_layout) = layouts.get(&cur).cloned() {
                            if let BindingPattern::Simple(var_name_str) = var_name {
                                layouts.insert(var_name_str.clone(), sub_layout);
                            }
                        }
                    }
                }
                propagate_let_layouts(value_expr, idx_to_key, layouts);
            }
            propagate_let_layouts(body, idx_to_key, layouts);
        }
        CompiledExpr::FunctionCall { args, .. } => {
            for a in args {
                propagate_let_layouts(a, idx_to_key, layouts);
            }
        }
        CompiledExpr::Do(exprs) => {
            for e in exprs {
                propagate_let_layouts(e, idx_to_key, layouts);
            }
        }
        CompiledExpr::If {
            condition,
            then_branch,
            else_branch,
        } => {
            propagate_let_layouts(condition, idx_to_key, layouts);
            propagate_let_layouts(then_branch, idx_to_key, layouts);
            if let Some(e) = else_branch {
                propagate_let_layouts(e, idx_to_key, layouts);
            }
        }
        CompiledExpr::Lambda { body, .. } => {
            propagate_let_layouts(body, idx_to_key, layouts);
        }
        _ => {}
    }
}

/// Replace free occurrences of a symbol with a scalar constant.
pub fn substitute_scalar_param(expr: &CompiledExpr, name: &str, value: f64) -> CompiledExpr {
    match expr {
        CompiledExpr::Symbol(s) if s == name => f64_to_const(value),
        CompiledExpr::Lambda { params, body } => {
            if params.iter().any(|p| p == name) {
                expr.clone()
            } else {
                CompiledExpr::Lambda {
                    params: params.clone(),
                    body: Box::new(substitute_scalar_param(body, name, value)),
                }
            }
        }
        CompiledExpr::Let { bindings, body } => {
            let mut new_bindings = Vec::new();
            let mut shadowed = false;
            for (bname, bexpr) in bindings {
                let new_expr = if shadowed {
                    bexpr.clone()
                } else {
                    substitute_scalar_param(bexpr, name, value)
                };
                if let BindingPattern::Simple(bname_str) = bname {
                    if bname_str == name {
                        shadowed = true;
                    }
                }
                new_bindings.push((bname.clone(), new_expr));
            }
            CompiledExpr::Let {
                bindings: new_bindings,
                body: if shadowed {
                    body.clone()
                } else {
                    Box::new(substitute_scalar_param(body, name, value))
                },
            }
        }
        CompiledExpr::FunctionCall {
            name: fn_name,
            args,
            ..
        } => CompiledExpr::FunctionCall {
            name: fn_name.clone(),
            args: args
                .iter()
                .map(|a| substitute_scalar_param(a, name, value))
                .collect(),
            loc: None,
        },
        CompiledExpr::LambdaCall { callee, args } => CompiledExpr::LambdaCall {
            callee: Box::new(substitute_scalar_param(callee, name, value)),
            args: args
                .iter()
                .map(|a| substitute_scalar_param(a, name, value))
                .collect(),
        },
        CompiledExpr::If {
            condition,
            then_branch,
            else_branch,
        } => CompiledExpr::If {
            condition: Box::new(substitute_scalar_param(condition, name, value)),
            then_branch: Box::new(substitute_scalar_param(then_branch, name, value)),
            else_branch: else_branch
                .as_ref()
                .map(|e| Box::new(substitute_scalar_param(e, name, value))),
        },
        CompiledExpr::Vector(elems) => CompiledExpr::Vector(
            elems
                .iter()
                .map(|e| substitute_scalar_param(e, name, value))
                .collect(),
        ),
        CompiledExpr::Repeat {
            index_var,
            count,
            acc_var,
            acc_init,
            body,
        } => {
            let new_init = Box::new(substitute_scalar_param(acc_init, name, value));
            let shadowed = index_var == name || acc_var == name;
            CompiledExpr::Repeat {
                index_var: index_var.clone(),
                count: count.clone(),
                acc_var: acc_var.clone(),
                acc_init: new_init,
                body: if shadowed {
                    body.clone()
                } else {
                    Box::new(substitute_scalar_param(body, name, value))
                },
            }
        }
        _ => expr.clone(),
    }
}

/// Extract scalar values from a dict Value for compile-time constant propagation.
pub fn extract_scalar_constants(
    val: &Value,
    param_name: &str,
    index_map: &BTreeMap<Vec<String>, Vec<usize>>,
    out: &mut HashMap<(String, Vec<usize>), f64>,
) {
    extract_scalars_rec(val, param_name, &mut vec![], index_map, out);
}

fn extract_scalars_rec(
    val: &Value,
    param_name: &str,
    path: &mut Vec<String>,
    index_map: &BTreeMap<Vec<String>, Vec<usize>>,
    out: &mut HashMap<(String, Vec<usize>), f64>,
) {
    let scalar = match val {
        Value::Int(n) => Some(*n as f64),
        Value::Float(f) => Some(*f as f64),
        Value::Tensor { data, .. } if data.is_empty() => None,
        Value::Tensor { data, .. } if data.len() == 1 => {
            Some(data.iter().next().copied().unwrap() as f64)
        }
        Value::Dict(map) => {
            for (key, child) in map {
                path.push(key.clone());
                extract_scalars_rec(child, param_name, path, index_map, out);
                path.pop();
            }
            return;
        }
        _ => return,
    };
    if let (Some(v), Some(indices)) = (scalar, index_map.get(path)) {
        out.insert((param_name.to_string(), indices.clone()), v);
    }
}

fn push_first_last(name: &str, expr: CompiledExpr) -> CompiledExpr {
    match expr {
        CompiledExpr::Vector(elems) => if name == "first" {
            elems.into_iter().next()
        } else {
            elems.into_iter().last()
        }
        .unwrap_or_else(|| CompiledExpr::Vector(vec![])),
        CompiledExpr::Let { bindings, body } => CompiledExpr::Let {
            bindings,
            body: Box::new(push_first_last(name, *body)),
        },
        other => CompiledExpr::FunctionCall {
            name: name.to_string(),
            args: vec![other],
            loc: None,
        },
    }
}

pub fn try_infer_shape(
    expr: &CompiledExpr,
    shapes: &HashMap<String, Vec<i64>>,
) -> Option<Vec<i64>> {
    match expr {
        CompiledExpr::Symbol(s) => shapes.get(s).cloned(),
        CompiledExpr::Vector(elems) => {
            if elems.is_empty() {
                return Some(vec![0]);
            }
            if let CompiledExpr::Vector(inner) = &elems[0] {
                Some(vec![elems.len() as i64, inner.len() as i64])
            } else {
                Some(vec![elems.len() as i64])
            }
        }
        CompiledExpr::Let { bindings, body } => {
            let mut inner = shapes.clone();
            for (name, rhs) in bindings {
                if let Some(sh) = try_infer_shape(rhs, &inner) {
                    if let BindingPattern::Simple(name_str) = name {
                        inner.insert(name_str.clone(), sh);
                    }
                }
            }
            try_infer_shape(body, &inner)
        }
        CompiledExpr::FunctionCall { name, args, .. } => match name.as_str() {
            "+" | "-" | "*" | "/" | "**" | "sqrt" | "exp" | "log" | "abs" | "relu" | "gelu"
            | "tanh" | "sigmoid" | "neg" | "maximum" | "minimum" | "clamp" | "==" | "!=" | "<"
            | "<=" | ">" | ">=" | "and" | "or" | "not" | "where" => {
                args.iter().find_map(|a| try_infer_shape(a, shapes))
            }
            "layer-norm" | "softmax" | "normalize" | "log-softmax" => {
                args.first().and_then(|a| try_infer_shape(a, shapes))
            }
            "transpose" | "tr" if args.len() == 1 || args.len() == 2 => {
                let sh = try_infer_shape(&args[0], shapes)?;
                if sh.len() >= 2 {
                    let mut out = sh;
                    let n = out.len();
                    out.swap(n - 2, n - 1);
                    Some(out)
                } else {
                    None
                }
            }
            "first" => args.first().and_then(|a| try_infer_shape(a, shapes)),
            "reshape" if args.len() == 2 => {
                if let CompiledExpr::Vector(elems) = &args[1] {
                    elems
                        .iter()
                        .map(|e| {
                            if let CompiledExpr::Integer(n) = e {
                                Some(*n)
                            } else {
                                None
                            }
                        })
                        .collect()
                } else {
                    None
                }
            }
            "swapaxes" if args.len() == 3 => {
                if let (Some(sh), Some(CompiledExpr::Integer(a)), Some(CompiledExpr::Integer(b))) =
                    (try_infer_shape(&args[0], shapes), args.get(1), args.get(2))
                {
                    let ndim = sh.len() as i64;
                    let ai = if *a < 0 {
                        (ndim + a) as usize
                    } else {
                        *a as usize
                    };
                    let bi = if *b < 0 {
                        (ndim + b) as usize
                    } else {
                        *b as usize
                    };
                    if ai < sh.len() && bi < sh.len() {
                        let mut out = sh;
                        out.swap(ai, bi);
                        Some(out)
                    } else {
                        None
                    }
                } else {
                    None
                }
            }
            "@" if args.len() == 2 => {
                let lhs = try_infer_shape(&args[0], shapes)?;
                let rhs = try_infer_shape(&args[1], shapes)?;
                match (lhs.len(), rhs.len()) {
                    (1, 1) => Some(vec![]),
                    (1, r) if r >= 2 => {
                        let mut out = rhs[..r - 2].to_vec();
                        out.extend_from_slice(&rhs[r - 1..]);
                        Some(out)
                    }
                    (l, 1) if l >= 1 => Some(lhs[..l - 1].to_vec()),
                    (l, r) if l >= 2 && r >= 2 => {
                        let mut out = lhs[..lhs.len() - 1].to_vec();
                        out.push(*rhs.last().unwrap());
                        Some(out)
                    }
                    _ => None,
                }
            }
            "get" if args.len() == 2 => {
                let tensor_shape = try_infer_shape(&args[0], shapes)?;
                if tensor_shape.is_empty() {
                    return None;
                }
                let trailing = &tensor_shape[1..];
                match try_infer_shape(&args[1], shapes) {
                    Some(idx_shape) => {
                        let mut out = idx_shape;
                        out.extend_from_slice(trailing);
                        Some(out)
                    }
                    None => Some(trailing.to_vec()), // scalar index
                }
            }
            "slice" if args.len() >= 3 => {
                let tensor_shape = try_infer_shape(&args[0], shapes)?;
                if tensor_shape.is_empty() {
                    return None;
                }
                let mut axis_raw: i64 = 0;
                for (i, a) in args.iter().enumerate() {
                    if let CompiledExpr::Keyword(k) = a {
                        if k == "axis" {
                            if let Some(CompiledExpr::Integer(n)) = args.get(i + 1) {
                                axis_raw = *n;
                            }
                        }
                    }
                }
                let ndim = tensor_shape.len() as i64;
                let axis = if axis_raw < 0 {
                    (ndim + axis_raw) as usize
                } else {
                    axis_raw as usize
                };
                if axis >= tensor_shape.len() {
                    return None;
                }
                if let (CompiledExpr::Integer(start), CompiledExpr::Integer(end)) =
                    (&args[1], &args[2])
                {
                    let sliced_dim = end - start;
                    let mut out = tensor_shape.clone();
                    out[axis] = sliced_dim;
                    Some(out)
                } else {
                    None
                }
            }
            _ => None,
        },
        CompiledExpr::GetTupleElement { param, indices } => {
            let key = format!("{}@{:?}", param, indices);
            shapes.get(&key).cloned()
        }
        _ => None,
    }
}

fn f64_to_const(val: f64) -> CompiledExpr {
    if val.fract() == 0.0 && val.abs() < i64::MAX as f64 {
        CompiledExpr::Integer(val as i64)
    } else {
        CompiledExpr::Float(val)
    }
}

fn try_fold_arithmetic(name: &str, args: &[CompiledExpr]) -> Option<CompiledExpr> {
    if args.len() != 2 {
        return None;
    }
    let a = extract_numeric(&args[0])?;
    let b = extract_numeric(&args[1])?;
    let result = match name {
        "+" => a + b,
        "-" => a - b,
        "*" => a * b,
        "/" => a / b,
        "//" => (a / b).floor(),
        _ => return None,
    };
    Some(f64_to_const(result))
}

fn extract_numeric(expr: &CompiledExpr) -> Option<f64> {
    match expr {
        CompiledExpr::Integer(n) => Some(*n as f64),
        CompiledExpr::Float(f) => Some(*f),
        _ => None,
    }
}

// ── Reduce unrolling ──

use crate::autodiff::replace_symbol;
use crate::core::error::SheafError;
use crate::lowering::stablehlo::StableHLOType;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

static UNROLL_COUNTER: AtomicUsize = AtomicUsize::new(0);
static DESTRUCTURE_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Generate a fresh temporary variable name for destructuring desugaring.
fn fresh_temp() -> String {
    let n = DESTRUCTURE_COUNTER.fetch_add(1, Ordering::Relaxed);
    format!("__dst{}", n)
}

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

pub fn classify_vectors(
    expr: CompiledExpr,
    param_shapes: &HashMap<String, Vec<i64>>,
) -> CompiledExpr {
    let mut symbol_types = HashMap::new();
    for (name, shape) in param_shapes {
        symbol_types.insert(name.clone(), StableHLOType::f32_tensor(shape.clone()));
    }
    classify_vectors_rec(expr, &mut symbol_types)
}

fn classify_vectors_rec(
    expr: CompiledExpr,
    symbol_types: &mut HashMap<String, StableHLOType>,
) -> CompiledExpr {
    match expr {
        CompiledExpr::Let { bindings, body } => {
            let mut new_bindings = Vec::with_capacity(bindings.len());
            for (pattern, value) in bindings {
                let value_classified = classify_vectors_rec(value, symbol_types);
                if let BindingPattern::Simple(name) = &pattern {
                    let ty = infer_type_with_context(&value_classified, &*symbol_types)
                        .unwrap_or_else(|_| StableHLOType::scalar_f32());
                    symbol_types.insert(name.clone(), ty);
                }
                new_bindings.push((pattern.clone(), value_classified));
            }
            let body_classified = classify_vectors_rec(*body, symbol_types);
            CompiledExpr::Let {
                bindings: new_bindings,
                body: Box::new(body_classified),
            }
        }
        CompiledExpr::Lambda { params, body } => {
            let mut saved = Vec::new();
            for param in &params {
                let old_ty = symbol_types.remove(param);
                let active = old_ty
                    .clone()
                    .unwrap_or_else(|| StableHLOType::scalar_f32());
                symbol_types.insert(param.clone(), active);
                saved.push((param.clone(), old_ty));
            }
            let body_classified = classify_vectors_rec(*body, symbol_types);
            for (param, old_ty) in saved {
                match old_ty {
                    Some(old) => {
                        symbol_types.insert(param, old);
                    }
                    None => {
                        symbol_types.remove(&param);
                    }
                }
            }
            CompiledExpr::Lambda {
                params,
                body: Box::new(body_classified),
            }
        }
        CompiledExpr::Vector(elems) => {
            let mut elems_classified = Vec::with_capacity(elems.len());
            for e in elems {
                elems_classified.push(classify_vectors_rec(e, symbol_types));
            }
            if crate::lowering::codegen::try_flatten_to_constant(&elems_classified).is_some() {
                return CompiledExpr::Vector(elems_classified);
            }
            let mut all_scalar = true;
            for e in &elems_classified {
                if expr_is_tensor(e, symbol_types) {
                    all_scalar = false;
                    break;
                }
            }
            if all_scalar {
                return CompiledExpr::Vector(elems_classified);
            }
            CompiledExpr::Tuple(elems_classified)
        }
        CompiledExpr::Tuple(elems) => {
            let mut elems_classified = Vec::with_capacity(elems.len());
            for e in elems {
                elems_classified.push(classify_vectors_rec(e, symbol_types));
            }
            CompiledExpr::Tuple(elems_classified)
        }
        CompiledExpr::FunctionCall { name, args, loc } => {
            let mut args_classified = Vec::with_capacity(args.len());
            for a in args {
                args_classified.push(classify_vectors_rec(a, symbol_types));
            }
            CompiledExpr::FunctionCall {
                name,
                args: args_classified,
                loc,
            }
        }
        CompiledExpr::If {
            condition,
            then_branch,
            else_branch,
        } => {
            let condition_classified = classify_vectors_rec(*condition, symbol_types);
            let then_classified = classify_vectors_rec(*then_branch, symbol_types);
            let else_classified =
                else_branch.map(|e| Box::new(classify_vectors_rec(*e, symbol_types)));
            CompiledExpr::If {
                condition: Box::new(condition_classified),
                then_branch: Box::new(then_classified),
                else_branch: else_classified,
            }
        }
        CompiledExpr::Do(exprs) => {
            let mut exprs_classified = Vec::with_capacity(exprs.len());
            for e in exprs {
                exprs_classified.push(classify_vectors_rec(e, symbol_types));
            }
            CompiledExpr::Do(exprs_classified)
        }
        CompiledExpr::LambdaCall { callee, args } => {
            let callee_classified = classify_vectors_rec(*callee, symbol_types);
            let mut args_classified = Vec::with_capacity(args.len());
            for a in args {
                args_classified.push(classify_vectors_rec(a, symbol_types));
            }
            CompiledExpr::LambdaCall {
                callee: Box::new(callee_classified),
                args: args_classified,
            }
        }
        CompiledExpr::Repeat {
            index_var,
            count,
            acc_var,
            acc_init,
            body,
        } => {
            let count_classified = classify_vectors_rec(*count, symbol_types);
            let acc_init_classified = classify_vectors_rec(*acc_init, symbol_types);
            let body_classified = classify_vectors_rec(*body, symbol_types);
            CompiledExpr::Repeat {
                index_var,
                count: Box::new(count_classified),
                acc_var,
                acc_init: Box::new(acc_init_classified),
                body: Box::new(body_classified),
            }
        }
        CompiledExpr::Integer(_)
        | CompiledExpr::Float(_)
        | CompiledExpr::Boolean(_)
        | CompiledExpr::Nil
        | CompiledExpr::String(_)
        | CompiledExpr::Keyword(_)
        | CompiledExpr::Dict(_)
        | CompiledExpr::Quoted(_)
        | CompiledExpr::FunctionRef(_)
        | CompiledExpr::Symbol(_)
        | CompiledExpr::GetTupleElement { .. }
        | CompiledExpr::ValueAndGrad { .. }
        | CompiledExpr::Def { .. }
        | CompiledExpr::Guard { .. }
        | CompiledExpr::While { .. } => expr,
    }
}

/// Classify a destructuring source.
enum DestructKind {
    Tuple,
    Tensor1D,
    Vector1D, // literal Vector of scalars
    Unknown,
}

fn kind_of(expr: &CompiledExpr, symbol_types: &HashMap<String, StableHLOType>) -> DestructKind {
    match expr {
        CompiledExpr::Tuple(_) => DestructKind::Tuple,
        CompiledExpr::Vector(_) => DestructKind::Vector1D,
        CompiledExpr::Symbol(s) => {
            match symbol_types.get(s) {
                Some(StableHLOType::Tuple(_, _)) => DestructKind::Tuple,
                Some(ty) => {
                    let sh = ty.shape();
                    if sh.len() == 1 && !sh.is_empty() {
                        DestructKind::Tensor1D
                    } else {
                        DestructKind::Unknown
                    }
                }
                None => DestructKind::Unknown,
            }
        }
        _ => DestructKind::Unknown,
    }
}

/// Return the statically known length of a destructuring source.
fn static_length_with_types(
    expr: &CompiledExpr,
    symbol_types: &HashMap<String, StableHLOType>,
) -> Option<usize> {
    match expr {
        CompiledExpr::Vector(elems) => Some(elems.len()),
        CompiledExpr::Tuple(elems) => Some(elems.len()),
        CompiledExpr::Symbol(s) => symbol_types.get(s).and_then(|ty| match ty {
            StableHLOType::Tuple(tys, _) => Some(tys.len()),
            other => {
                let sh = other.shape();
                if sh.len() == 1 && !sh.is_empty() {
                    Some(sh[0] as usize)
                } else {
                    None
                }
            }
        }),
        _ => None,
    }
}

/// Desugar destructuring `Let` bindings.

/// Shape-bearing operations and the argument positions that require compile-time integers.
const SHAPE_BEARING_OPS: &[(&str, &[usize])] = &[
    ("reshape", &[1]),           // dimensions
    ("slice", &[1, 2]),          // start_indices, limit_indices
    ("dynamic-slice", &[1, 2]),  // start_indices, limit_indices
    ("one-hot", &[1]),           // num_classes
    ("random-randint", &[2, 3]), // low, high
    ("random-split", &[1]),      // N
    ("repeat", &[1]),            // count
];

/// Resolve a `Let` alias to a tuple element.
fn resolve_to_gte<'a>(
    e: &'a crate::core::expr::CompiledExpr,
    locals: &'a HashMap<String, crate::core::expr::CompiledExpr>,
) -> Option<(&'a str, &'a Vec<usize>)> {
    use crate::core::expr::CompiledExpr;
    match e {
        CompiledExpr::GetTupleElement { param, indices } => Some((param, indices)),
        CompiledExpr::Symbol(name) => {
            let resolved = locals.get(name)?;
            resolve_to_gte(resolved, locals)
        }
        _ => None,
    }
}

/// Collect tuple elements used as static shape arguments.
pub(crate) fn collect_shape_gtes(
    body: &crate::core::expr::CompiledExpr,
) -> std::collections::HashSet<(String, Vec<usize>)> {
    use crate::core::expr::{BindingPattern, CompiledExpr};
    let mut result = std::collections::HashSet::new();
    let root_locals: HashMap<String, CompiledExpr> = HashMap::new();

    fn visit(
        e: &CompiledExpr,
        shape_gtes: &mut std::collections::HashSet<(String, Vec<usize>)>,
        locals: &HashMap<String, CompiledExpr>,
    ) {
        if let CompiledExpr::FunctionCall { name, args, .. } = e {
            if let Some(pos_list) = SHAPE_BEARING_OPS
                .iter()
                .find(|(n, _)| *n == name)
                .map(|(_, p)| *p)
            {
                for &pos in pos_list {
                    if pos < args.len() {
                        if let Some((param, indices)) = resolve_to_gte(&args[pos], locals) {
                            shape_gtes.insert((param.to_string(), indices.clone()));
                        } else if let CompiledExpr::Symbol(param) = &args[pos]
                            && !locals.contains_key(param)
                        {
                            shape_gtes.insert((param.clone(), Vec::new()));
                        }
                        if let CompiledExpr::Vector(elems) | CompiledExpr::Tuple(elems) = &args[pos]
                        {
                            for elem in elems {
                                if let Some((param, indices)) = resolve_to_gte(elem, locals) {
                                    shape_gtes.insert((param.to_string(), indices.clone()));
                                } else if let CompiledExpr::Symbol(param) = elem
                                    && !locals.contains_key(param)
                                {
                                    shape_gtes.insert((param.clone(), Vec::new()));
                                }
                            }
                        }
                    }
                }
            }
        }
        match e {
            CompiledExpr::FunctionCall { args, .. } => {
                for a in args {
                    visit(a, shape_gtes, locals);
                }
            }
            CompiledExpr::Let { bindings, body: b } => {
                let mut scope = locals.clone();
                for (pat, rhs) in bindings {
                    visit(rhs, shape_gtes, &scope);
                    if let BindingPattern::Simple(name) = pat {
                        scope.insert(name.clone(), rhs.clone());
                    }
                }
                visit(b, shape_gtes, &scope);
            }
            CompiledExpr::Vector(elems) | CompiledExpr::Tuple(elems) => {
                for e in elems {
                    visit(e, shape_gtes, locals);
                }
            }
            CompiledExpr::Dict(pairs) => {
                for (_, v) in pairs {
                    visit(v, shape_gtes, locals);
                }
            }
            CompiledExpr::Lambda { body: b, .. } => {
                visit(b, shape_gtes, locals);
            }
            CompiledExpr::LambdaCall { callee, args } => {
                visit(callee, shape_gtes, locals);
                for a in args {
                    visit(a, shape_gtes, locals);
                }
            }
            CompiledExpr::If {
                condition: c,
                then_branch: t,
                else_branch: e,
            } => {
                visit(c, shape_gtes, locals);
                visit(t, shape_gtes, locals);
                if let Some(e) = e {
                    visit(e, shape_gtes, locals);
                }
            }
            CompiledExpr::Do(stmts) => {
                for s in stmts {
                    visit(s, shape_gtes, locals);
                }
            }
            CompiledExpr::Repeat {
                count: c,
                acc_init: a,
                body: b,
                ..
            } => {
                visit(c, shape_gtes, locals);
                visit(a, shape_gtes, locals);
                visit(b, shape_gtes, locals);
            }
            CompiledExpr::While {
                condition: c,
                acc_init: a,
                body: b,
                ..
            } => {
                visit(c, shape_gtes, locals);
                visit(a, shape_gtes, locals);
                visit(b, shape_gtes, locals);
            }
            CompiledExpr::Guard { check: _, expr: e } => {
                visit(e, shape_gtes, locals);
            }
            CompiledExpr::Def { value: v, .. } => {
                visit(v, shape_gtes, locals);
            }
            CompiledExpr::ValueAndGrad { shape_config, .. } => {
                let _ = shape_config;
            }
            CompiledExpr::Integer(_)
            | CompiledExpr::Float(_)
            | CompiledExpr::Boolean(_)
            | CompiledExpr::Nil
            | CompiledExpr::String(_)
            | CompiledExpr::Keyword(_)
            | CompiledExpr::FunctionRef(_)
            | CompiledExpr::Symbol(_)
            | CompiledExpr::GetTupleElement { .. }
            | CompiledExpr::Quoted(_) => {}
        }
    }

    visit(body, &mut result, &root_locals);
    result
}

/// Keep captured scalars used as static shape arguments.
pub(crate) fn filter_constants_for_shape_positions(
    constants: &HashMap<(String, Vec<usize>), f64>,
    body: &crate::core::expr::CompiledExpr,
) -> HashMap<(String, Vec<usize>), f64> {
    let shape_gtes = collect_shape_gtes(body);
    constants
        .iter()
        .filter(|((param, indices), _)| shape_gtes.contains(&(param.clone(), indices.clone())))
        .map(|((p, i), v)| ((p.clone(), i.clone()), *v))
        .collect()
}

pub fn lower_tuples_and_destructuring(
    body: CompiledExpr,
    param_shapes: &HashMap<String, Vec<i64>>,
) -> SheafResult<CompiledExpr> {
    let mut symbol_types: HashMap<String, StableHLOType> = HashMap::new();
    for (name, shape) in param_shapes {
        symbol_types.insert(name.clone(), StableHLOType::f32_tensor(shape.clone()));
    }
    let classified = classify_vectors_rec(body, &mut symbol_types);
    desugar_destructuring_lets(classified, &symbol_types)
}

pub fn desugar_destructuring_lets(
    expr: CompiledExpr,
    symbol_types: &HashMap<String, StableHLOType>,
) -> SheafResult<CompiledExpr> {
    match expr {
        CompiledExpr::Let { bindings, body } => {
            let mut new_bindings = Vec::new();
            let mut body = desugar_destructuring_lets(*body, symbol_types)?;

            for (pattern, value) in bindings {
                match pattern {
                    BindingPattern::Simple(_) => {
                        new_bindings
                            .push((pattern, desugar_destructuring_lets(value, symbol_types)?));
                    }
                    BindingPattern::Destructure(names) => {
                        for name in &names {
                            if let BindingPattern::Destructure(_) = name {
                                return Err(SheafError::Compile {
                                    message: "Nested destructuring patterns are not yet supported"
                                        .to_string(),
                                    location: crate::core::error::SourceLocation::unknown(),
                                });
                            }
                        }

                        let tmp = fresh_temp();
                        let value = desugar_destructuring_lets(value, symbol_types)?;

                        let kind = kind_of(&value, symbol_types);
                        let len =
                            static_length_with_types(&value, symbol_types).ok_or_else(|| {
                                SheafError::Compile {
                                    message:
                                        "Destructuring source has a dynamic length that cannot be \
                                          resolved statically: it is neither a literal nor a typed \
                                          parameter of known shape"
                                            .to_string(),
                                    location: crate::core::error::SourceLocation::unknown(),
                                }
                            })?;

                        if len != names.len() {
                            return Err(SheafError::Compile {
                                message: format!(
                                    "Destructuring arity mismatch: pattern has {} names, source has {} elements",
                                    names.len(),
                                    len
                                ),
                                location: crate::core::error::SourceLocation::unknown(),
                            });
                        }

                        for (i, name) in names.into_iter().enumerate().rev() {
                            let extraction = match kind {
                                DestructKind::Tuple => CompiledExpr::GetTupleElement {
                                    param: tmp.clone(),
                                    indices: vec![i],
                                },
                                DestructKind::Vector1D | DestructKind::Tensor1D => {
                                    CompiledExpr::FunctionCall {
                                        name: "get".to_string(),
                                        args: vec![
                                            CompiledExpr::Symbol(tmp.clone()),
                                            CompiledExpr::Integer(i as i64),
                                        ],
                                        loc: None,
                                    }
                                }
                                DestructKind::Unknown => {
                                    return Err(SheafError::Compile {
                                        message: format!(
                                            "Destructuring the result of a runtime tensor of unknown shape is not supported"
                                        ),
                                        location: crate::core::error::SourceLocation::unknown(),
                                    });
                                }
                            };

                            body = CompiledExpr::Let {
                                bindings: vec![(name, extraction)],
                                body: Box::new(body),
                            };
                        }

                        new_bindings.push((BindingPattern::Simple(tmp), value));
                    }
                }
            }

            Ok(CompiledExpr::Let {
                bindings: new_bindings,
                body: Box::new(body),
            })
        }
        CompiledExpr::FunctionCall { name, args, loc } => {
            let mut new_args = Vec::new();
            for arg in args {
                new_args.push(desugar_destructuring_lets(arg, symbol_types)?);
            }
            Ok(CompiledExpr::FunctionCall {
                name,
                args: new_args,
                loc,
            })
        }
        CompiledExpr::Lambda { params, body } => {
            Ok(CompiledExpr::Lambda {
                params,
                body: Box::new(desugar_destructuring_lets(*body, symbol_types)?),
            })
        }
        CompiledExpr::LambdaCall { callee, args } => {
            let new_args: Vec<CompiledExpr> = args
                .into_iter()
                .map(|a| desugar_destructuring_lets(a, symbol_types))
                .collect::<SheafResult<_>>()?;
            Ok(CompiledExpr::LambdaCall {
                callee: Box::new(desugar_destructuring_lets(*callee, symbol_types)?),
                args: new_args,
            })
        }
        CompiledExpr::If {
            condition,
            then_branch,
            else_branch,
        } => Ok(CompiledExpr::If {
            condition: Box::new(desugar_destructuring_lets(*condition, symbol_types)?),
            then_branch: Box::new(desugar_destructuring_lets(*then_branch, symbol_types)?),
            else_branch: match else_branch {
                Some(e) => Some(Box::new(desugar_destructuring_lets(*e, symbol_types)?)),
                None => None,
            },
        }),
        CompiledExpr::Do(exprs) => Ok(CompiledExpr::Do(
            exprs
                .into_iter()
                .map(|e| desugar_destructuring_lets(e, symbol_types))
                .collect::<SheafResult<_>>()?,
        )),
        CompiledExpr::Repeat {
            index_var,
            count,
            acc_var,
            acc_init,
            body,
        } => Ok(CompiledExpr::Repeat {
            index_var,
            count: Box::new(desugar_destructuring_lets(*count, symbol_types)?),
            acc_var,
            acc_init: Box::new(desugar_destructuring_lets(*acc_init, symbol_types)?),
            body: Box::new(desugar_destructuring_lets(*body, symbol_types)?),
        }),
        CompiledExpr::Vector(elems) => Ok(CompiledExpr::Vector(
            elems
                .into_iter()
                .map(|e| desugar_destructuring_lets(e, symbol_types))
                .collect::<SheafResult<_>>()?,
        )),
        CompiledExpr::Tuple(elems) => Ok(CompiledExpr::Tuple(
            elems
                .into_iter()
                .map(|e| desugar_destructuring_lets(e, symbol_types))
                .collect::<SheafResult<_>>()?,
        )),
        other => Ok(other),
    }
}

#[cfg(test)]
mod shape_classifier_tests {
    use super::{collect_shape_gtes, filter_constants_for_shape_positions};
    use crate::core::expr::{BindingPattern, CompiledExpr};
    use std::collections::HashMap;

    fn gte(param: &str, idx: usize) -> CompiledExpr {
        CompiledExpr::GetTupleElement {
            param: param.into(),
            indices: vec![idx],
        }
    }
    fn sym(name: &str) -> CompiledExpr {
        CompiledExpr::Symbol(name.into())
    }
    fn call(name: &str, args: Vec<CompiledExpr>) -> CompiledExpr {
        CompiledExpr::FunctionCall {
            name: name.into(),
            args,
            loc: None,
        }
    }
    fn let_(bindings: Vec<(&str, CompiledExpr)>, body: CompiledExpr) -> CompiledExpr {
        CompiledExpr::Let {
            bindings: bindings
                .into_iter()
                .map(|(n, rhs)| (BindingPattern::Simple(n.into()), rhs))
                .collect(),
            body: Box::new(body),
        }
    }

    #[test]
    fn shape_scalar_via_let_alias_is_collected() {
        let body = let_(
            vec![("n", gte("cfg", 0))],
            call(
                "reshape",
                vec![sym("x"), CompiledExpr::Vector(vec![sym("n"), sym("n")])],
            ),
        );
        let gtes = collect_shape_gtes(&body);
        assert!(
            gtes.contains(&("cfg".to_string(), vec![0])),
            "let-aliased shape scalar must be detected, got {:?}",
            gtes
        );
    }

    #[test]
    fn shape_scalar_via_nested_let_alias_is_collected() {
        let body = let_(
            vec![("n", gte("cfg", 0))],
            let_(
                vec![("m", sym("n"))],
                call(
                    "reshape",
                    vec![sym("x"), CompiledExpr::Vector(vec![sym("m"), sym("m")])],
                ),
            ),
        );
        let gtes = collect_shape_gtes(&body);
        assert!(gtes.contains(&("cfg".to_string(), vec![0])));
    }

    #[test]
    fn arithmetic_only_scalar_is_not_collected() {
        let body = let_(
            vec![("lr", gte("cfg", 0))],
            call("+", vec![sym("w"), call("*", vec![sym("lr"), sym("x")])]),
        );
        let gtes = collect_shape_gtes(&body);
        assert!(
            gtes.is_empty(),
            "arithmetic-only scalar must not be baked, got {:?}",
            gtes
        );
    }

    #[test]
    fn filter_keeps_shape_drops_arithmetic() {
        let body = let_(
            vec![
                ("n", gte("cfg", 0)),
                ("scale", gte("cfg", 1)),
                (
                    "r",
                    call(
                        "reshape",
                        vec![sym("x"), CompiledExpr::Vector(vec![sym("n"), sym("n")])],
                    ),
                ),
            ],
            call("*", vec![sym("scale"), sym("r")]),
        );
        let mut constants = HashMap::new();
        constants.insert(("cfg".to_string(), vec![0]), 8.0); // n
        constants.insert(("cfg".to_string(), vec![1]), 2.0); // scale
        let filtered = filter_constants_for_shape_positions(&constants, &body);
        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered.get(&("cfg".to_string(), vec![0])), Some(&8.0));
    }

    #[test]
    fn sequential_let_accumulates_scope() {
        let body = let_(
            vec![
                ("n", gte("cfg", 0)),
                (
                    "r",
                    call(
                        "reshape",
                        vec![sym("x"), CompiledExpr::Vector(vec![sym("n"), sym("n")])],
                    ),
                ),
            ],
            sym("r"),
        );
        let gtes = collect_shape_gtes(&body);
        assert!(gtes.contains(&("cfg".to_string(), vec![0])));
    }
}
