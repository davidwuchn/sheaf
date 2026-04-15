// Copyright (c) 2026 Damien Boureille
// Licensed under the MIT License.

//! ML-specific special forms: with-params, grad, value-and-grad

use crate::core::ast::SheafValue;
use crate::lowering::stablehlo::StableHLOType;
use crate::core::expr::{CompiledExpr, CompilerContext, ParamField, ParamLayout};
use crate::core::error::{SheafError, SheafResult, SourceLocation};
use crate::forms::base::{SpecialForm, check_min_arity};

// ---------------------------------------------------------------------------
// with-params
// ---------------------------------------------------------------------------

/// with-params - Destructure a typed parameter tuple into local scope.
///
/// Syntax:
///   (with-params [param-name] body...)
///   (with-params [param-name :key] body...)   ; access sub-dict
///
/// The parameter must be typed with :as in the enclosing defn signature,
/// or the type must be inferable from context.
///
/// Example:
///   (defn linear [x (p :as Linear)]
///     (with-params [p]
///       (+ (@ x W) b)))   ; W and b resolved from p via tuple indices
pub struct WithParamsForm;

impl SpecialForm for WithParamsForm {
    fn name(&self) -> &'static str {
        "with-params"
    }

    fn compile(
        &self,
        compiler: &mut CompilerContext,
        args: &[SheafValue],
        loc: &SourceLocation,
    ) -> SheafResult<CompiledExpr> {
        check_min_arity("with-params", args, 2, loc)?;

        // First arg: binding vector [param-name] or [param-name :key]
        let binding = &args[0];
        let body = &args[1..];

        let (param_name, opt_key) = parse_with_params_binding(binding, loc)?;

        // Find the layout for this param.
        // Look up the param's type annotation from local_vars metadata.
        // If no annotation is present, fall back to interpreter-mode dict access:
        // emit a WithParamsDynamic node so the interpreter can do runtime key lookup.
        let layout_name = match compiler
            .local_vars
            .get(&format!("__type__{}", param_name))
            .and_then(|v| v.as_symbol())
            .map(|s| s.to_string())
        {
            Some(name) => name,
            None => {
                // Interpreter fallback: no layout annotation.
                // Strategy: collect all free symbols in the body AST,
                // register them as (get sub_dict :name) in local scope,
                // compile the body, then restore scope.
                //
                // sub_dict = if opt_key: (get param :key), else: param itself
                let sub_dict_expr = if let Some(ref key) = opt_key {
                    SheafValue::List(
                        vec![
                            SheafValue::Symbol("get".to_string(), loc.clone()),
                            SheafValue::Symbol(param_name.to_string(), loc.clone()),
                            SheafValue::Keyword(key.clone(), loc.clone()),
                        ],
                        loc.clone(),
                    )
                } else {
                    SheafValue::Symbol(param_name.to_string(), loc.clone())
                };

                // Collect candidate symbols from the body (simple heuristic: all symbols
                // that are not already in scope and not function names).
                let free_syms = collect_free_symbols(body, compiler);

                // Register each free symbol as (get sub_dict :sym) in env so that
                // compiler.compile(Symbol("W")) will call compile(&getter_ast).
                // We use env (not local_vars) because env entries are compiled on lookup.
                let saved_env: Vec<(String, SheafValue)> = free_syms
                    .iter()
                    .filter_map(|sym| compiler.env.get(sym).map(|v| (sym.clone(), v.clone())))
                    .collect();

                for sym in &free_syms {
                    let getter = SheafValue::List(
                        vec![
                            SheafValue::Symbol("get".to_string(), loc.clone()),
                            sub_dict_expr.clone(),
                            SheafValue::Keyword(sym.clone(), loc.clone()),
                        ],
                        loc.clone(),
                    );
                    compiler.env.insert(sym.clone(), getter);
                }

                let compiled_body: SheafResult<Vec<CompiledExpr>> =
                    body.iter().map(|e| compiler.compile(e)).collect();
                let compiled_body = compiled_body?;

                // Restore env entries we overwrote
                for sym in &free_syms {
                    compiler.env.remove(sym);
                }
                for (sym, val) in saved_env {
                    compiler.env.insert(sym, val);
                }

                return Ok(if compiled_body.len() == 1 {
                    compiled_body.into_iter().next().unwrap()
                } else {
                    CompiledExpr::Do(compiled_body)
                });
            }
        };

        let layout = compiler
            .param_types
            .get(&layout_name)
            .cloned()
            .ok_or_else(|| SheafError::Compile {
                message: format!(
                    "with-params: unknown param type '{}'",
                    layout_name
                ),
                location: loc.clone(),
            })?;

        // Determine which fields to expose based on optional key filter
        let fields_to_expose: Vec<&ParamField> = if let Some(ref key) = opt_key {
            layout.fields_under(key)
        } else {
            layout.fields.iter().collect()
        };

        // Save current param_scope to restore after body
        let saved_scope = compiler.param_scope.clone();

        // Populate param_scope: last path component -> (param_name, tuple_indices)
        for field in &fields_to_expose {
            let local_name = field.path.last().unwrap().clone();
            compiler
                .param_scope
                .insert(local_name, (param_name.clone(), field.tuple_index.clone()));
        }

        // Compile body with extended scope
        let compiled_body: SheafResult<Vec<CompiledExpr>> =
            body.iter().map(|e| compiler.compile(e)).collect();
        let compiled_body = compiled_body?;

        // Restore scope
        compiler.param_scope = saved_scope;

        // Return last expression (like do)
        if compiled_body.len() == 1 {
            Ok(compiled_body.into_iter().next().unwrap())
        } else {
            Ok(CompiledExpr::Do(compiled_body))
        }
    }
}

/// Collect free symbols in an AST body: symbols not already defined in the compiler scope.
/// Used by with-params fallback to find field names like W, b.
fn collect_free_symbols(body: &[SheafValue], compiler: &CompilerContext) -> Vec<String> {
    let mut syms = std::collections::HashSet::new();
    for expr in body {
        collect_symbols_in(expr, compiler, &mut syms);
    }
    syms.into_iter().collect()
}

fn collect_symbols_in(
    expr: &SheafValue,
    compiler: &CompilerContext,
    out: &mut std::collections::HashSet<String>,
) {
    match expr {
        SheafValue::Symbol(name, _) => {
            // Only collect if not already known (not in registry, not in local_vars, not a special form)
            if !compiler.registry.contains_key(name)
                && !compiler.local_vars.contains_key(name)
                && !compiler.env.contains_key(name)
                && !is_special_form(name)
            {
                out.insert(name.clone());
            }
        }
        SheafValue::List(elems, _) | SheafValue::Vector(elems, _) => {
            // Skip first element if it looks like a function call (symbol = function name)
            let start = if matches!(elems.first(), Some(SheafValue::Symbol(..))) { 1 } else { 0 };
            for e in &elems[start..] {
                collect_symbols_in(e, compiler, out);
            }
        }
        _ => {}
    }
}

fn is_special_form(name: &str) -> bool {
    matches!(name, "let" | "fn" | "defn" | "if" | "do" | "and" | "or" | "not"
        | "quote" | "use" | "get" | "get-in" | "dict" | "assoc" | "last"
        | "with-params" | "grad" | "value-and-grad"
        | "repeat" | "while" | "case" | "as->" | "->")
}

/// Parse the binding vector of with-params.
/// Returns (param_name, optional_sub_key)
fn parse_with_params_binding(
    binding: &SheafValue,
    loc: &SourceLocation,
) -> SheafResult<(String, Option<String>)> {
    match binding {
        SheafValue::Vector(elems, _) => {
            if elems.is_empty() {
                return Err(SheafError::Compile {
                    message: "with-params: binding vector must not be empty".to_string(),
                    location: loc.clone(),
                });
            }

            let param_name = elems[0].as_symbol().ok_or_else(|| SheafError::Compile {
                message: "with-params: first element of binding must be a symbol".to_string(),
                location: loc.clone(),
            })?;

            let opt_key = if elems.len() >= 2 {
                match &elems[1] {
                    SheafValue::Keyword(k, _) => Some(k.clone()),
                    other => {
                        return Err(SheafError::Compile {
                            message: format!(
                                "with-params: second element of binding must be a keyword, got: {}",
                                other
                            ),
                            location: loc.clone(),
                        });
                    }
                }
            } else {
                None
            };

            Ok((param_name.to_string(), opt_key))
        }
        SheafValue::Symbol(s, _) => Err(SheafError::Compile {
            message: format!(
                "with-params: expected [{}] (vector), got bare symbol. Use (with-params [{}] ...)",
                s, s
            ),
            location: loc.clone(),
        }),
        other => Err(SheafError::Compile {
            message: format!(
                "with-params: expected binding vector [param] or [param :key], got: {}",
                other
            ),
            location: loc.clone(),
        }),
    }
}


// ---------------------------------------------------------------------------
// value-and-grad
// ---------------------------------------------------------------------------

/// Compile a `value-and-grad` form into a deferred IR node.
///
/// Syntax:
///   (value-and-grad name f config :wrt [p1 p2 ...])
///   (value-and-grad name f :wrt [p1 p2 ...])
///
/// - `name`: symbol -- name for the generated function
/// - `f`: symbol -- must refer to an existing `defn` in the registry
/// - `config`: optional dict `{:param [dim ...] ...}` -- shapes for specialization
/// - `:wrt [p1 p2 ...]`: parameter names to differentiate
///
/// Returns `CompiledExpr::ValueAndGrad` recording the intent. Actual codegen
/// or interpretation is deferred to the backend.

/// Parse a shape config dict into raw (name, dims) pairs, without
/// constructing StableHLOType -- keeps the frontend free from codegen types.
fn parse_shape_dims(
    dict: &SheafValue,
    loc: &SourceLocation,
) -> SheafResult<Vec<(String, Vec<i64>)>> {
    let pairs = match dict {
        SheafValue::Dict(pairs, _) => pairs,
        other => {
            return Err(SheafError::Compile {
                message: format!("Expected a dict for shape config, got: {}", other),
                location: loc.clone(),
            });
        }
    };
    let mut result = Vec::new();
    for (key, val) in pairs {
        let name = match key {
            SheafValue::Keyword(k, _) => k.clone(),
            SheafValue::Symbol(s, _) => s.clone(),
            other => {
                return Err(SheafError::Compile {
                    message: format!("Shape dict key must be a keyword or symbol, got: {}", other),
                    location: loc.clone(),
                });
            }
        };
        let dims = match val {
            SheafValue::Vector(elems, _) => {
                let d: SheafResult<Vec<i64>> = elems
                    .iter()
                    .map(|e| match e {
                        SheafValue::Integer(n, _) => Ok(*n),
                        other => Err(SheafError::Compile {
                            message: format!("Shape dimensions must be integers, got: {}", other),
                            location: loc.clone(),
                        }),
                    })
                    .collect();
                d?
            }
            other => {
                return Err(SheafError::Compile {
                    message: format!(
                        "Shape value for '{}' must be a vector of dims, got: {}",
                        name, other
                    ),
                    location: loc.clone(),
                });
            }
        };
        result.push((name, dims));
    }
    Ok(result)
}

pub struct ValueAndGradForm;

impl SpecialForm for ValueAndGradForm {
    fn name(&self) -> &'static str {
        "value-and-grad"
    }

    fn compile(
        &self,
        compiler: &mut CompilerContext,
        args: &[SheafValue],
        loc: &SourceLocation,
    ) -> SheafResult<CompiledExpr> {
        // Interpreter HOF form: (value-and-grad f) where f is a lambda or function ref.
        // Returns a function that, when called with params, returns [loss, grad].
        if args.len() == 1 {
            let f = compiler.compile(&args[0])?;
            return Ok(CompiledExpr::FunctionCall {
                name: "__value-and-grad-hof__".to_string(),
                args: vec![f],
                loc: None,
            });
        }

        if args.len() < 4 {
            return Err(SheafError::Compile {
                message: "value-and-grad: expected (value-and-grad name f [:wrt | config])"
                    .to_string(),
                location: loc.clone(),
            });
        }

        let new_fn_name = args[0].as_symbol().ok_or_else(|| SheafError::Compile {
            message: "value-and-grad: first argument must be a symbol (new function name)"
                .to_string(),
            location: loc.clone(),
        })?;

        let src_fn_name = args[1].as_symbol().ok_or_else(|| SheafError::Compile {
            message: "value-and-grad: second argument must be a symbol (source function)"
                .to_string(),
            location: loc.clone(),
        })?;

        // args[2] may be a config dict, a symbol bound to a dict, or the start of :wrt
        let config_arg = match &args[2] {
            SheafValue::Dict(..) => Some(&args[2]),
            SheafValue::Symbol(name, _) => compiler.local_vars.get(name.as_str()),
            _ => None,
        };
        let (shape_config, rest_start) = match config_arg {
            Some(val @ SheafValue::Dict(..)) => (parse_shape_dims(val, loc)?, 3),
            _ => (Vec::new(), 2),
        };

        // Parse :wrt [p1 p2 ...]
        let mut wrt_names: Vec<String> = Vec::new();
        let mut i = rest_start;
        while i < args.len() {
            if let SheafValue::Keyword(k, _) = &args[i] {
                if k == "wrt" {
                    i += 1;
                    match args.get(i) {
                        Some(SheafValue::Vector(elems, _)) => {
                            for elem in elems {
                                let name = elem.as_symbol().ok_or_else(|| SheafError::Compile {
                                    message: "value-and-grad: :wrt elements must be symbols"
                                        .to_string(),
                                    location: loc.clone(),
                                })?;
                                wrt_names.push(name.to_string());
                            }
                        }
                        other => {
                            return Err(SheafError::Compile {
                                message: format!(
                                    "value-and-grad: :wrt must be followed by a vector, got {}",
                                    other.map(|v| v.to_string()).unwrap_or("nothing".into())
                                ),
                                location: loc.clone(),
                            });
                        }
                    }
                }
            }
            i += 1;
        }

        if wrt_names.is_empty() {
            return Err(SheafError::Compile {
                message: "value-and-grad: :wrt list is empty or missing".to_string(),
                location: loc.clone(),
            });
        }

        // Validate source function exists
        if !compiler.registry.contains_key(src_fn_name) {
            return Err(SheafError::Compile {
                message: format!(
                    "value-and-grad: function '{}' not found. Did you forget (defn {})?",
                    src_fn_name, src_fn_name
                ),
                location: loc.clone(),
            });
        }

        Ok(CompiledExpr::ValueAndGrad {
            fn_name: new_fn_name.to_string(),
            src_fn_name: src_fn_name.to_string(),
            wrt_params: wrt_names,
            shape_config,
        })
    }
}

// ---------------------------------------------------------------------------
// ParamLayout -> StableHLOType conversion
// ---------------------------------------------------------------------------

/// Convert a ParamLayout into a StableHLO tuple type.
///
/// Flat layout:  {:W [4 8] :b [8]}
///   -> tuple<tensor<4x8xf32>, tensor<8xf32>>
///
/// Nested layout: {:attn {:Wq [512 512] :Wk [512 512]} :mlp {:W1 [512 2048]}}
///   -> tuple<tuple<tensor<512x512xf32>, tensor<512x512xf32>>, tuple<tensor<512x2048xf32>>>
pub fn param_layout_to_stablehlo_type(layout: &ParamLayout) -> StableHLOType {
    let field_refs: Vec<&ParamField> = layout.fields.iter().collect();
    fields_to_tuple_type(&field_refs, 0)
}

fn fields_to_tuple_type(fields: &[&ParamField], depth: usize) -> StableHLOType {
    // Group fields by path[depth], preserving insertion order
    let mut group_keys: Vec<String> = Vec::new();
    let mut groups: std::collections::HashMap<String, Vec<&ParamField>> =
        std::collections::HashMap::new();
    for field in fields {
        let key = &field.path[depth];
        if !group_keys.contains(key) {
            group_keys.push(key.clone());
        }
        groups.entry(key.clone()).or_default().push(field);
    }

    let elements: Vec<StableHLOType> = group_keys
        .iter()
        .map(|key| {
            let group = &groups[key];
            if group.len() == 1 && group[0].path.len() == depth + 1 {
                field_to_tensor_type(group[0])
            } else {
                fields_to_tuple_type(group, depth + 1)
            }
        })
        .collect();

    StableHLOType::Tuple(elements, None)
}

fn field_to_tensor_type(field: &ParamField) -> StableHLOType {
    if field.shape.is_empty() {
        StableHLOType::scalar_f32()
    } else {
        StableHLOType::f32_tensor(field.shape.clone())
    }
}
