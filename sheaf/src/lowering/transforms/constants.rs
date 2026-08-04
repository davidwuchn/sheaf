// Copyright (c) 2025 Damien Boureille
// Licensed under the MIT License.

use crate::core::expr::{BindingPattern, CompiledExpr};
use crate::interpreter::value::Value;
use std::collections::{BTreeMap, HashMap};

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
                            if let Some(sh) = shapes.get(aliased).cloned()
                                && let BindingPattern::Simple(k_str) = k
                            {
                                shapes.insert(k_str.clone(), sh);
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
                            if let BindingPattern::Simple(k_str) = k
                                && let Some(sh) = try_infer_shape(&resolved, shapes)
                            {
                                shapes.insert(k_str.clone(), sh);
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
                if let BindingPattern::Simple(bname_str) = bname
                    && bname_str == name
                {
                    shadowed = true;
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
                if let Some(sh) = try_infer_shape(rhs, &inner)
                    && let BindingPattern::Simple(name_str) = name
                {
                    inner.insert(name_str.clone(), sh);
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
                    if let CompiledExpr::Keyword(k) = a
                        && k == "axis"
                        && let Some(CompiledExpr::Integer(n)) = args.get(i + 1)
                    {
                        axis_raw = *n;
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
        if let CompiledExpr::FunctionCall { name, args, .. } = e
            && let Some(pos_list) = SHAPE_BEARING_OPS
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
