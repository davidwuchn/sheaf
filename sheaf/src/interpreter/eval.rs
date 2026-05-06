// Copyright (c) 2025 Damien Boureille
// Licensed under the MIT License.

//! Top-level evaluation entry points for the Sheaf interpreter.
//!
//! Provides stateless (`eval_source`) and stateful (`Interpreter`) interfaces,
//! the latter being used by the REPL to persist bindings across inputs.

use crate::core::expr::{CompiledExpr, CompilerContext};
use crate::core::error::SheafError;
use crate::interpreter::builtins::register_builtins;
use crate::interpreter::env::Env;
use crate::interpreter::tracer::{Tracer, TracerConfig};
use crate::interpreter::value::Value;
use crate::interpreter;

/// Evaluate a complete Sheaf source string and return the last value.
/// Each call is fully independent (no shared state).
/// `file_path` is used to resolve relative `(use ...)` paths; pass `None` for inline expressions.
pub fn eval_source(source: &str) -> Result<Value, SheafError> {
    eval_source_with_path(source, None)
}

pub fn eval_source_with_path(
    source: &str,
    file_path: Option<&std::path::Path>,
) -> Result<Value, SheafError> {
    let filename = file_path
        .and_then(|p| p.to_str())
        .unwrap_or("<eval>");
    let exprs = crate::core::parse(source, filename)?;
    let mut compiler = CompilerContext::new();
    // Set current_dir so (use module) resolves relative to the file being evaluated
    if let Some(path) = file_path {
        if let Some(dir) = path.parent() {
            compiler.set_current_dir(
                dir.canonicalize().unwrap_or_else(|_| dir.to_path_buf()),
            );
        }
    }
    let mut compiled = Vec::new();
    for expr in &exprs {
        compiled.push(compiler.compile(expr)?);
    }
    // Auto-detect companion VMFB for the file being run
    #[cfg(iree_runtime)]
    if let Some(path) = file_path {
        let all_fns: Vec<String> = compiler.registry.keys().cloned().collect();
        crate::runtime::vmfb_loader::try_load_vmfb(&mut compiler, path, &all_fns);
    }

    let mut env = Env::with_registry(compiler.registry.clone());
    env.vmfb_sessions = compiler.vmfb_sessions.clone();
    register_builtins(&mut env);
    let mut last = Value::Nil;
    for c in &compiled {
        if !matches!(c, CompiledExpr::Nil) {
            last = interpreter::eval(c, &mut env)?;
        }
    }
    Ok(last)
}

/// Evaluate source with tracing and/or CLI guards enabled.
pub fn eval_source_with_tracing(
    source: &str,
    file_path: Option<&std::path::Path>,
    config: TracerConfig,
) -> Result<Value, SheafError> {
    let filename = file_path
        .and_then(|p| p.to_str())
        .unwrap_or("<eval>");
    let exprs = crate::core::parse(source, filename)?;
    let mut compiler = CompilerContext::new();
    if let Some(path) = file_path {
        if let Some(dir) = path.parent() {
            compiler.set_current_dir(
                dir.canonicalize().unwrap_or_else(|_| dir.to_path_buf()),
            );
        }
    }
    let mut compiled = Vec::new();
    for expr in &exprs {
        compiled.push(compiler.compile(expr)?);
    }

    #[cfg(iree_runtime)]
    if let Some(path) = file_path {
        let all_fns: Vec<String> = compiler.registry.keys().cloned().collect();
        crate::runtime::vmfb_loader::try_load_vmfb(&mut compiler, path, &all_fns);
    }

    let mut env = Env::with_registry(compiler.registry.clone());
    env.vmfb_sessions = compiler.vmfb_sessions.clone();
    env.tracer = Some(Tracer::from_config(config));
    register_builtins(&mut env);
    let mut last = Value::Nil;
    for c in &compiled {
        if !matches!(c, CompiledExpr::Nil) {
            last = interpreter::eval(c, &mut env)?;
        }
    }
    Ok(last)
}

/// Evaluate source with --blame profiling (and optionally tracing).
pub fn eval_source_with_blame(
    source: &str,
    file_path: Option<&std::path::Path>,
    tracer_config: Option<TracerConfig>,
) -> Result<Value, SheafError> {
    let (val, _) = eval_source_with_blame_internal(source, file_path, tracer_config, true, false)?;
    Ok(val)
}

/// Evaluate source with --blame profiling (and optionally tracing).
/// When `mem_profile` is true, samples RSS at key checkpoints and returns the report.
pub fn eval_source_with_blame_mem(
    source: &str,
    file_path: Option<&std::path::Path>,
    tracer_config: Option<TracerConfig>,
) -> Result<(Value, String), SheafError> {
    let (val, report) = eval_source_with_blame_internal(source, file_path, tracer_config, true, true)?;
    let report_str = report.unwrap_or_default();
    Ok((val, report_str))
}

/// Evaluate source with memory profiling only (no blame).
pub fn eval_source_with_mem(
    source: &str,
    file_path: Option<&std::path::Path>,
) -> Result<(Value, String), SheafError> {
    let (val, report) = eval_source_with_blame_internal(source, file_path, None, false, true)?;
    let report_str = report.unwrap_or_default();
    Ok((val, report_str))
}

fn eval_source_with_blame_internal(
    source: &str,
    file_path: Option<&std::path::Path>,
    tracer_config: Option<TracerConfig>,
    blame_report: bool,
    mem_profile: bool,
) -> Result<(Value, Option<String>), SheafError> {
    let filename = file_path
        .and_then(|p| p.to_str())
        .unwrap_or("<eval>");
    let exprs = crate::core::parse(source, filename)?;
    let mut compiler = CompilerContext::new();
    if let Some(path) = file_path {
        if let Some(dir) = path.parent() {
            compiler.set_current_dir(
                dir.canonicalize().unwrap_or_else(|_| dir.to_path_buf()),
            );
        }
    }
    let mut compiled = Vec::new();
    for expr in &exprs {
        compiled.push(compiler.compile(expr)?);
    }

    #[cfg(iree_runtime)]
    if let Some(path) = file_path {
        let all_fns: Vec<String> = compiler.registry.keys().cloned().collect();
        crate::runtime::vmfb_loader::try_load_vmfb(&mut compiler, path, &all_fns);
    }

    let mut env = Env::with_registry(compiler.registry.clone());
    env.vmfb_sessions = compiler.vmfb_sessions.clone();
    register_builtins(&mut env);
    // Only create blame profiler when caller wants blame report
    if blame_report {
        env.profiler = Some(crate::interpreter::profiler::Profiler::new());
    }
    if let Some(config) = tracer_config {
        env.tracer = Some(Tracer::from_config(config));
    }
    if mem_profile {
        env.mem_profiler = Some(crate::interpreter::mem_profile::MemProfiler::new());
        if let Some(ref mut mp) = env.mem_profiler {
            mp.sample("after init");
        }
    }
    let mut last = Value::Nil;
    for c in &compiled {
        if !matches!(c, CompiledExpr::Nil) {
            last = interpreter::eval(c, &mut env)?;
        }
    }
    if mem_profile {
        if let Some(ref mut mp) = env.mem_profiler {
            mp.sample("after eval");
        }
    }
    #[cfg(iree_runtime)]
    if mem_profile {
        if let Some(ref mut mp) = env.mem_profiler {
            for session in &env.vmfb_sessions {
                if let Some(iree_session) = session.downcast_ref::<crate::runtime::iree_session::IreeSession>() {
                    mp.sample_iree(iree_session.device_allocator_ptr());
                }
            }
        }
    }
    if blame_report {
        if let Some(ref profiler) = env.profiler {
            profiler.report();
        }
    }

    // Release IREE resources before reporting memory peak to capture teardown spikes
    let mp = env.mem_profiler.take();
    drop(env);

    let mem_report = if mem_profile {
        mp.as_ref().map(|m| m.report())
    } else {
        None
    };
    Ok((last, mem_report))
}

/// Stateful interpreter: accumulates definitions and bindings across calls.
/// Used by the REPL so that `(defn f ...)` in one line is visible in the next.
pub struct Interpreter {
    compiler: CompilerContext,
    env: Env,
}

impl Interpreter {
    pub fn new() -> Self {
        let compiler = CompilerContext::new();
        let mut env = Env::with_registry(compiler.registry.clone());
        register_builtins(&mut env);
        Self { compiler, env }
    }

    /// Evaluate one input (expression or definition). Returns the resulting value.
    pub fn eval(&mut self, source: &str) -> Result<Value, SheafError> {
        let exprs = crate::core::parse(source, "<repl>")?;
        let mut last = Value::Nil;
        for expr in &exprs {
            let compiled = self.compiler.compile(expr)?;
            // Sync any newly registered functions and VMFB sessions into env
            self.env.registry = self.compiler.registry.clone();
            self.env.vmfb_sessions = self.compiler.vmfb_sessions.clone();
            if !matches!(compiled, CompiledExpr::Nil) {
                last = interpreter::eval(&compiled, &mut self.env)?;
            }
        }
        Ok(last)
    }

    pub fn registry_names(&self) -> Vec<String> {
        self.compiler.registry.keys().cloned().collect()
    }

    pub fn env(&self) -> &Env {
        &self.env
    }

    pub fn env_mut(&mut self) -> &mut Env {
        &mut self.env
    }

    pub fn load_path(&self) -> &[std::path::PathBuf] {
        &self.compiler.load_path
    }
}
