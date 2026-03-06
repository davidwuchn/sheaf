// Copyright (c) 2025 Damien Boureille
// Licensed under the MIT License.

//! Compiler transforms for the build pipeline: inlining, constant resolution,
//! value-and-grad desugaring, and layout propagation.

use std::process::exit;
use sheaf_compiler::core::compiler::CompiledExpr;

/// Substitute known scalar constants and propagate Let-bound constants.
/// Handles: GetTupleElement → Integer, (static expr) → evaluate, Symbol → local constant,
/// and constant folding of arithmetic on known values.
/// Lower (get alias :key) patterns introduced after inlining stdlib functions.
///
/// After inlining, a call like (layer-norm h (GetTupleElement layer-p [1]) 2)
/// becomes (let [p (GetTupleElement layer-p [1])] (let [gamma (get p :gamma)] ...)).
/// This pass tracks such aliases and resolves (get alias :key) to
/// GetTupleElement { param: root_param, indices: full_indices }.
pub(super) fn lower_inlined_gets(
    body: &CompiledExpr,
    index_maps: &[(String, std::collections::BTreeMap<Vec<String>, Vec<usize>>)],
) -> CompiledExpr {
    // Build reverse map: (param, indices) → key_path for alias tracking
    let mut reverse: std::collections::HashMap<(String, Vec<usize>), Vec<String>> = std::collections::HashMap::new();
    for (param, imap) in index_maps {
        for (path, indices) in imap {
            reverse.insert((param.clone(), indices.clone()), path.clone());
        }
    }
    lower_inlined_gets_rec(body, index_maps, &mut std::collections::HashMap::new(), &reverse)
}

// aliases: symbol → (root_param, prefix_path)
fn lower_inlined_gets_rec(
    expr: &CompiledExpr,
    index_maps: &[(String, std::collections::BTreeMap<Vec<String>, Vec<usize>>)],
    aliases: &mut std::collections::HashMap<String, (String, Vec<String>)>,
    reverse: &std::collections::HashMap<(String, Vec<usize>), Vec<String>>,
) -> CompiledExpr {
    match expr {
        // (get alias :key ...) where alias is a tracked tuple alias
        CompiledExpr::FunctionCall { name, args } if name == "get" && args.len() >= 2 => {
            if let CompiledExpr::Symbol(alias) = &args[0] {
                if let Some((root_param, prefix_path)) = aliases.get(alias) {
                    let mut path = prefix_path.clone();
                    let mut all_kw = true;
                    for arg in &args[1..] {
                        match arg {
                            CompiledExpr::Keyword(k) => path.push(k.clone()),
                            _ => { all_kw = false; break; }
                        }
                    }
                    if all_kw {
                        if let Some(imap) = index_maps.iter().find(|(p, _)| p == root_param).map(|(_, m)| m) {
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
            CompiledExpr::FunctionCall {
                name: name.clone(),
                args: args.iter().map(|a| lower_inlined_gets_rec(a, index_maps, aliases, reverse)).collect(),
            }
        }
        CompiledExpr::Let { bindings, body } => {
            let new_bindings: Vec<_> = bindings.iter().map(|(k, v)| {
                let resolved = lower_inlined_gets_rec(v, index_maps, aliases, reverse);
                // Track alias: if the value is GetTupleElement, record what key path it corresponds to
                if let CompiledExpr::GetTupleElement { param, indices } = &resolved {
                    let key = (param.clone(), indices.clone());
                    if let Some(path) = reverse.get(&key) {
                        aliases.insert(k.clone(), (param.clone(), path.clone()));
                    }
                }
                (k.clone(), resolved)
            }).collect();
            CompiledExpr::Let {
                bindings: new_bindings,
                body: Box::new(lower_inlined_gets_rec(body, index_maps, aliases, reverse)),
            }
        }
        CompiledExpr::FunctionCall { name, args } => CompiledExpr::FunctionCall {
            name: name.clone(),
            args: args.iter().map(|a| lower_inlined_gets_rec(a, index_maps, aliases, reverse)).collect(),
        },
        CompiledExpr::Do(exprs) => CompiledExpr::Do(
            exprs.iter().map(|e| lower_inlined_gets_rec(e, index_maps, aliases, reverse)).collect(),
        ),
        CompiledExpr::If { condition, then_branch, else_branch } => CompiledExpr::If {
            condition: Box::new(lower_inlined_gets_rec(condition, index_maps, aliases, reverse)),
            then_branch: Box::new(lower_inlined_gets_rec(then_branch, index_maps, aliases, reverse)),
            else_branch: else_branch.as_ref().map(|e| Box::new(lower_inlined_gets_rec(e, index_maps, aliases, reverse))),
        },
        CompiledExpr::Lambda { params, body } => CompiledExpr::Lambda {
            params: params.clone(),
            body: Box::new(lower_inlined_gets_rec(body, index_maps, aliases, reverse)),
        },
        CompiledExpr::LambdaCall { callee, args } => CompiledExpr::LambdaCall {
            callee: Box::new(lower_inlined_gets_rec(callee, index_maps, aliases, reverse)),
            args: args.iter().map(|a| lower_inlined_gets_rec(a, index_maps, aliases, reverse)).collect(),
        },
        CompiledExpr::Repeat { index_var, count, acc_var, acc_init, body } => CompiledExpr::Repeat {
            index_var: index_var.clone(),
            count: Box::new(lower_inlined_gets_rec(count, index_maps, aliases, reverse)),
            acc_var: acc_var.clone(),
            acc_init: Box::new(lower_inlined_gets_rec(acc_init, index_maps, aliases, reverse)),
            body: Box::new(lower_inlined_gets_rec(body, index_maps, aliases, reverse)),
        },
        CompiledExpr::Vector(elems) => CompiledExpr::Vector(
            elems.iter().map(|e| lower_inlined_gets_rec(e, index_maps, aliases, reverse)).collect(),
        ),
        other => other.clone(),
    }
}

pub(super) fn resolve_static_constants(
    expr: &CompiledExpr,
    constants: &std::collections::HashMap<(String, Vec<usize>), f64>,
    param_shapes: &std::collections::HashMap<String, Vec<i64>>,
) -> CompiledExpr {
    let mut locals: std::collections::HashMap<String, CompiledExpr> = std::collections::HashMap::new();
    let mut shapes: std::collections::HashMap<String, Vec<i64>> = param_shapes.clone();
    resolve_constants_rec(expr, constants, &mut locals, &mut shapes)
}

fn resolve_constants_rec(
    expr: &CompiledExpr,
    constants: &std::collections::HashMap<(String, Vec<usize>), f64>,
    locals: &mut std::collections::HashMap<String, CompiledExpr>,
    shapes: &mut std::collections::HashMap<String, Vec<i64>>,
) -> CompiledExpr {
    match expr {
        CompiledExpr::GetTupleElement { param, indices } => {
            let key = (param.clone(), indices.clone());
            match constants.get(&key) {
                Some(&val) => f64_to_const(val),
                None => expr.clone(),
            }
        }
        CompiledExpr::Symbol(name) => {
            locals.get(name).cloned().unwrap_or_else(|| expr.clone())
        }
        CompiledExpr::FunctionCall { name, args } if name == "static" && args.len() == 1 => {
            resolve_constants_rec(&args[0], constants, locals, shapes)
        }
        // (shape expr) → Vector([Integer(d0), Integer(d1), ...]) when shape is known
        CompiledExpr::FunctionCall { name, args } if name == "shape" && args.len() == 1 => {
            // Try direct symbol lookup first
            let shape_from_sym = match &args[0] {
                CompiledExpr::Symbol(s) => shapes.get(s.as_str()).cloned(),
                _ => None,
            };
            // Fallback: infer shape from expression (handles inlined code)
            let shape = shape_from_sym.or_else(|| {
                let resolved = resolve_constants_rec(&args[0], constants, locals, shapes);
                try_infer_shape(&resolved, shapes)
            });
            if let Some(sh) = shape {
                return CompiledExpr::Vector(
                    sh.iter().map(|&d| CompiledExpr::Integer(d)).collect(),
                );
            }
            CompiledExpr::FunctionCall {
                name: name.clone(),
                args: args.iter().map(|a| resolve_constants_rec(a, constants, locals, shapes)).collect(),
            }
        }
        // (get Vector idx) → fold with negative index support
        CompiledExpr::FunctionCall { name, args } if name == "get" && args.len() == 2 => {
            let recv = resolve_constants_rec(&args[0], constants, locals, shapes);
            let idx = resolve_constants_rec(&args[1], constants, locals, shapes);
            if let (CompiledExpr::Vector(elems), CompiledExpr::Integer(i)) = (&recv, &idx) {
                let len = elems.len() as i64;
                let norm = if *i < 0 { len + i } else { *i };
                if norm >= 0 && norm < len {
                    return elems[norm as usize].clone();
                }
            }
            CompiledExpr::FunctionCall { name: name.clone(), args: vec![recv, idx] }
        }
        // (first [a b ...]) → a, (last [a b ...]) → last element — fold at compile time
        // Recursively pushes first/last through nested Lets (e.g. inlined functions).
        CompiledExpr::FunctionCall { name, args }
            if (name == "first" || name == "last") && args.len() == 1 =>
        {
            let recv = resolve_constants_rec(&args[0], constants, locals, shapes);
            return push_first_last(name, recv);
        }
        CompiledExpr::FunctionCall { name, args } => {
            let resolved: Vec<_> = args.iter()
                .map(|a| resolve_constants_rec(a, constants, locals, shapes))
                .collect();
            try_fold_arithmetic(name, &resolved)
                .unwrap_or_else(|| CompiledExpr::FunctionCall { name: name.clone(), args: resolved })
        }
        CompiledExpr::Let { bindings, body } => {
            let new_bindings: Vec<_> = bindings.iter().map(|(k, v)| {
                let resolved = resolve_constants_rec(v, constants, locals, shapes);
                match &resolved {
                    CompiledExpr::Integer(_) | CompiledExpr::Float(_) => {
                        locals.insert(k.clone(), resolved.clone());
                    }
                    // Alias: (let [X x]) — propagate shape of x to X
                    CompiledExpr::Symbol(aliased) => {
                        if let Some(sh) = shapes.get(aliased).cloned() {
                            shapes.insert(k.clone(), sh);
                        }
                    }
                    // Known vector shape (e.g. from (shape X)) — track for nested (get (shape ...) -1)
                    CompiledExpr::Vector(elems) if elems.iter().all(|e| matches!(e, CompiledExpr::Integer(_))) => {
                        let sh: Vec<i64> = elems.iter().filter_map(|e| {
                            if let CompiledExpr::Integer(n) = e { Some(*n) } else { None }
                        }).collect();
                        shapes.insert(k.clone(), sh);
                    }
                    _ => {
                        // For FunctionCall/Let RHS (e.g. inlined layer-norm), infer shape transitively
                        if let Some(sh) = try_infer_shape(&resolved, shapes) {
                            shapes.insert(k.clone(), sh);
                        }
                    }
                }
                (k.clone(), resolved)
            }).collect();
            CompiledExpr::Let {
                bindings: new_bindings,
                body: Box::new(resolve_constants_rec(body, constants, locals, shapes)),
            }
        }
        CompiledExpr::Do(exprs) => CompiledExpr::Do(
            exprs.iter().map(|e| resolve_constants_rec(e, constants, locals, shapes)).collect(),
        ),
        CompiledExpr::If { condition, then_branch, else_branch } => CompiledExpr::If {
            condition: Box::new(resolve_constants_rec(condition, constants, locals, shapes)),
            then_branch: Box::new(resolve_constants_rec(then_branch, constants, locals, shapes)),
            else_branch: else_branch.as_ref().map(|e| Box::new(resolve_constants_rec(e, constants, locals, shapes))),
        },
        CompiledExpr::Lambda { params, body } => CompiledExpr::Lambda {
            params: params.clone(),
            body: Box::new(resolve_constants_rec(body, constants, locals, shapes)),
        },
        CompiledExpr::LambdaCall { callee, args } => CompiledExpr::LambdaCall {
            callee: Box::new(resolve_constants_rec(callee, constants, locals, shapes)),
            args: args.iter().map(|a| resolve_constants_rec(a, constants, locals, shapes)).collect(),
        },
        CompiledExpr::Repeat { index_var, count, acc_var, acc_init, body } => CompiledExpr::Repeat {
            index_var: index_var.clone(),
            count: Box::new(resolve_constants_rec(count, constants, locals, shapes)),
            acc_var: acc_var.clone(),
            acc_init: Box::new(resolve_constants_rec(acc_init, constants, locals, shapes)),
            body: Box::new(resolve_constants_rec(body, constants, locals, shapes)),
        },
        CompiledExpr::Vector(elems) => CompiledExpr::Vector(
            elems.iter().map(|e| resolve_constants_rec(e, constants, locals, shapes)).collect(),
        ),
        other => other.clone(),
    }
}

/// Push `first`/`last` through nested Let expressions until a Vector is found.
/// (first (let [binds] (let [binds2] [a b ...]))) → (let [binds] (let [binds2] a))
fn push_first_last(name: &str, expr: CompiledExpr) -> CompiledExpr {
    match expr {
        CompiledExpr::Vector(elems) => {
            if name == "first" {
                elems.into_iter().next()
            } else {
                elems.into_iter().last()
            }
            .unwrap_or_else(|| CompiledExpr::Vector(vec![]))
        }
        CompiledExpr::Let { bindings, body } => CompiledExpr::Let {
            bindings,
            body: Box::new(push_first_last(name, *body)),
        },
        other => CompiledExpr::FunctionCall {
            name: name.to_string(),
            args: vec![other],
        },
    }
}

/// Heuristically infer the output shape of an expression given a known shapes map.
/// Used to propagate shapes through Let-bound intermediate variables after inlining.
fn try_infer_shape(
    expr: &CompiledExpr,
    shapes: &std::collections::HashMap<String, Vec<i64>>,
) -> Option<Vec<i64>> {
    match expr {
        CompiledExpr::Symbol(s) => shapes.get(s).cloned(),
        CompiledExpr::Let { bindings, body } => {
            let mut inner = shapes.clone();
            for (name, rhs) in bindings {
                if let Some(sh) = try_infer_shape(rhs, &inner) {
                    inner.insert(name.clone(), sh);
                }
            }
            try_infer_shape(body, &inner)
        }
        CompiledExpr::FunctionCall { name, args } => match name.as_str() {
            // Element-wise ops: output shape = any known tensor arg's shape
            "+" | "-" | "*" | "/" | "**" | "sqrt" | "exp" | "log" | "abs"
            | "relu" | "gelu" | "tanh" | "sigmoid" | "neg"
            | "maximum" | "minimum" | "clamp"
            | "==" | "!=" | "<" | "<=" | ">" | ">=" | "and" | "or" | "not"
            | "where" => args.iter().find_map(|a| try_infer_shape(a, shapes)),
            // Shape-preserving ops: output has same shape as first tensor arg
            "layer-norm" | "softmax" | "normalize"
                => args.first().and_then(|a| try_infer_shape(a, shapes)),
            "first" => args.first().and_then(|a| try_infer_shape(a, shapes)),
            // reshape: output shape is the second arg (a vector of ints)
            "reshape" if args.len() == 2 => {
                if let CompiledExpr::Vector(elems) = &args[1] {
                    let sh: Option<Vec<i64>> = elems.iter().map(|e| {
                        if let CompiledExpr::Integer(n) = e { Some(*n) } else { None }
                    }).collect();
                    sh
                } else {
                    None
                }
            }
            // swapaxes: permute shape dims
            "swapaxes" if args.len() == 3 => {
                if let (Some(sh), Some(CompiledExpr::Integer(a)), Some(CompiledExpr::Integer(b)))
                    = (try_infer_shape(&args[0], shapes), args.get(1), args.get(2))
                {
                    let ndim = sh.len() as i64;
                    let ai = if *a < 0 { (ndim + a) as usize } else { *a as usize };
                    let bi = if *b < 0 { (ndim + b) as usize } else { *b as usize };
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
            // matmul (@): [..., M, K] @ [..., K, N] → [..., M, N]
            "@" if args.len() == 2 => {
                let lhs = try_infer_shape(&args[0], shapes)?;
                let rhs = try_infer_shape(&args[1], shapes)?;
                if lhs.len() >= 2 && rhs.len() >= 2 {
                    let mut out = lhs[..lhs.len()-1].to_vec();
                    out.push(*rhs.last().unwrap());
                    Some(out)
                } else {
                    None
                }
            }
            _ => None,
        },
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
    if args.len() != 2 { return None; }
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

pub(super) fn resolve_vag_decls(
    compiler: &sheaf_compiler::core::compiler::CompilerContext,
    compiled_exprs: &[sheaf_compiler::core::compiler::CompiledExpr],
    verbosity: u8,
) -> Vec<String> {
    use sheaf_compiler::autodiff::value_and_grad::{GradParam, emit_value_and_grad_func};
    use sheaf_compiler::core::inference::infer_function_signature_with_known;
    use sheaf_compiler::StableHLOType;

    let mut vag_nodes = Vec::new();
    for expr in compiled_exprs {
        collect_vag_nodes(expr, &mut vag_nodes);
    }

    let mut decls = Vec::new();
    for (fn_name, src_fn_name, wrt_params, shape_config) in vag_nodes {
        let func_def = match compiler.registry.get(src_fn_name) {
            Some(fd) => fd,
            None => continue,
        };
        let body_compiled = match &func_def.body_compiled {
            Some(b) => b,
            None => continue,
        };

        let known_types: Vec<(String, StableHLOType)> = shape_config
            .iter()
            .map(|(name, dims)| {
                let ty = if dims.is_empty() {
                    StableHLOType::scalar_f32()
                } else {
                    StableHLOType::f32_tensor(dims.clone())
                };
                (name.clone(), ty)
            })
            .collect();

        let signature = if !known_types.is_empty() {
            match infer_function_signature_with_known(
                compiler,
                &func_def.params,
                body_compiled,
                &known_types,
            ) {
                Ok(sig) => sig,
                Err(e) => {
                    eprintln!(
                        "value-and-grad '{}': signature inference failed: {}",
                        fn_name, e
                    );
                    exit(1);
                }
            }
        } else {
            match &func_def.signature {
                Some(sig) => sig.clone(),
                None => {
                    eprintln!(
                        "value-and-grad '{}': function '{}' has no inferred signature",
                        fn_name, src_fn_name
                    );
                    exit(1);
                }
            }
        };

        let grad_params: Vec<GradParam> = wrt_params
            .iter()
            .map(|wrt_name| {
                let idx = func_def
                    .params
                    .iter()
                    .position(|p| p == wrt_name)
                    .unwrap_or_else(|| {
                        eprintln!(
                            "value-and-grad '{}': '{}' is not a parameter of '{}'",
                            fn_name, wrt_name, src_fn_name
                        );
                        exit(1);
                    });
                GradParam {
                    name: wrt_name.clone(),
                    ty: signature.param_types[idx].clone(),
                }
            })
            .collect();

        if verbosity >= 1 {
            println!("Emitting value-and-grad '{}'...", fn_name);
        }

        let func_decl = emit_value_and_grad_func(
            fn_name,
            &func_def.params,
            &signature.param_types,
            body_compiled,
            &grad_params,
            compiler.registry.clone(),
        )
        .unwrap_or_else(|e| {
            eprintln!("value-and-grad '{}': codegen failed: {}", fn_name, e);
            exit(1);
        });

        decls.push(func_decl);
    }
    decls
}

fn collect_vag_nodes<'a>(
    expr: &'a sheaf_compiler::core::compiler::CompiledExpr,
    out: &mut Vec<(&'a str, &'a str, &'a Vec<String>, &'a Vec<(String, Vec<i64>)>)>,
) {
    use sheaf_compiler::core::compiler::CompiledExpr;
    match expr {
        CompiledExpr::ValueAndGrad {
            fn_name,
            src_fn_name,
            wrt_params,
            shape_config,
        } => {
            out.push((fn_name, src_fn_name, wrt_params, shape_config));
        }
        CompiledExpr::Do(exprs) => {
            for e in exprs {
                collect_vag_nodes(e, out);
            }
        }
        CompiledExpr::Let { bindings, body } => {
            for (_, v) in bindings {
                collect_vag_nodes(v, out);
            }
            collect_vag_nodes(body, out);
        }
        _ => {}
    }
}

/// Scan a lowered body for Let bindings of the form `var = GetTupleElement(param, [i])`.
/// For each such binding, propagate the tuple_key_layout from the dict key name to the
/// variable name. This allows `(get input-layer "W")` to resolve when `input-layer` is
/// a let binding for `(get params "input")`.
pub(super) fn propagate_let_layouts(
    expr: &sheaf_compiler::core::compiler::CompiledExpr,
    idx_to_key: &std::collections::HashMap<(String, usize), String>,
    layouts: &mut std::collections::HashMap<String, std::collections::BTreeMap<String, usize>>,
) {
    use sheaf_compiler::core::compiler::CompiledExpr;
    match expr {
        CompiledExpr::Let { bindings, body } => {
            for (var_name, value_expr) in bindings {
                if let CompiledExpr::GetTupleElement { param, indices } = value_expr {
                    if indices.len() == 1 {
                        let lookup = (param.clone(), indices[0]);
                        if let Some(key_name) = idx_to_key.get(&lookup) {
                            if let Some(sub_layout) = layouts.get(key_name).cloned() {
                                layouts.insert(var_name.clone(), sub_layout);
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
        CompiledExpr::If { condition, then_branch, else_branch } => {
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
