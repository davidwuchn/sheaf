// Copyright (c) 2025 Damien Boureille
// Licensed under the MIT License.

//! Runtime tracer: function call logging, ring-buffer backtrace, and CLI guards.

use crate::core::expr::GuardCheck;
use crate::sheaf_msg;
use crate::interpreter::value::Value;
use std::collections::{HashSet, VecDeque};
use std::time::Instant;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TraceLevel {
    Fast,
    Normal,
    Verbose,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LogFormat {
    Console,
    Json,
}


pub struct CliGuard {
    pub scope: Option<String>,
    pub check: GuardCheck,
}

pub struct TracerConfig {
    pub enabled: bool,
    pub scope_filter: Option<Vec<String>>,
    pub level: TraceLevel,
    pub format: LogFormat,
    pub cli_guards: Vec<CliGuard>,
}

pub struct Tracer {
    pub enabled: bool,
    pub silent_monitoring: bool,
    depth: usize,
    pub level_mode: TraceLevel,
    pub log_format: LogFormat,
    pub scope_filter: Option<HashSet<String>>,
    ring_buffer: VecDeque<String>,
    ring_buffer_max: usize,
    pub cli_guards: Vec<CliGuard>,
    call_start: Option<Instant>,
}

impl Default for Tracer {
    fn default() -> Self {
        Self::new()
    }
}

impl Tracer {
    pub fn new() -> Self {
        Self {
            enabled: false,
            silent_monitoring: false,
            depth: 0,
            level_mode: TraceLevel::Normal,
            log_format: LogFormat::Console,
            scope_filter: None,
            ring_buffer: VecDeque::new(),
            ring_buffer_max: 100,
            cli_guards: Vec::new(),
            call_start: None,
        }
    }

    pub fn from_config(config: TracerConfig) -> Self {
        let silent = !config.enabled && !config.cli_guards.is_empty();
        Self {
            enabled: config.enabled,
            silent_monitoring: silent,
            depth: 0,
            level_mode: config.level,
            log_format: config.format,
            scope_filter: config.scope_filter.map(|v| v.into_iter().collect()),
            ring_buffer: VecDeque::new(),
            ring_buffer_max: 100,
            cli_guards: config.cli_guards,
            call_start: None,
        }
    }

    pub fn should_trace(&self, name: &str) -> bool {
        if !self.enabled {
            return false;
        }
        match &self.scope_filter {
            Some(filter) => filter.contains(name),
            None => true,
        }
    }

    pub fn is_active(&self, name: &str) -> bool {
        self.should_trace(name) || self.silent_monitoring
    }

    /// Lightweight trace for compiled (IREE) dispatch: name + tag only, no format_value.
    pub fn log_compiled_dispatch(&mut self, name: &str) {
        if !matches!(self.level_mode, TraceLevel::Normal | TraceLevel::Verbose) {
            return;
        }
        let indent = "│ ".repeat(self.depth);
        let line = format!("{}├─ [{}] (compiled)", indent, name);
        self.push_ring(&line);
        if self.should_trace(name) {
            match self.log_format {
                LogFormat::Console => sheaf_msg!("{}", line),
                LogFormat::Json => {
                    eprintln!("{{\"type\":\"call\",\"fn\":\"{}\",\"dispatch\":\"compiled\",\"depth\":{}}}", name, self.depth);
                }
            }
        }
    }

    pub fn log_call(&mut self, name: &str, args: &[Value]) {
        let indent = "│ ".repeat(self.depth);
        let args_str: Vec<String> = args.iter().map(|a| format_value(a, self.level_mode)).collect();
        let line = format!("{}├─ [{}] {}", indent, name, args_str.join(", "));

        self.push_ring(&line);
        if self.should_trace(name) {
            match self.log_format {
                LogFormat::Console => sheaf_msg!("{}", line),
                LogFormat::Json => {
                    eprintln!("{{\"type\":\"call\",\"fn\":\"{}\",\"depth\":{}}}", name, self.depth);
                }
            }
        }
        self.depth += 1;
        self.call_start = Some(Instant::now());
    }

    pub fn log_return(&mut self, name: &str, result: &Value) {
        self.depth = self.depth.saturating_sub(1);
        let elapsed = self.call_start.take()
            .map(|s| s.elapsed())
            .unwrap_or_default();
        let elapsed_str = if elapsed.as_micros() < 10 {
            format!("{:.1}μs", elapsed.as_nanos() as f64 / 1000.0)
        } else if elapsed.as_millis() < 1 {
            format!("{:.2}ms", elapsed.as_micros() as f64 / 1000.0)
        } else {
            format!("{:.0}ms", elapsed.as_millis())
        };

        let indent = "│ ".repeat(self.depth);
        let result_str = format_value(result, self.level_mode);
        let line = format!("{}└─ <- {} ({})", indent, result_str, elapsed_str);

        self.push_ring(&line);
        if self.should_trace(name) {
            match self.log_format {
                LogFormat::Console => sheaf_msg!("{}", line),
                LogFormat::Json => {
                    eprintln!("{{\"type\":\"return\",\"fn\":\"{}\",\"elapsed_us\":{}}}", name, elapsed.as_micros());
                }
            }
        }

        self.check_cli_guards(name, result);
    }

    pub fn check_cli_guards(&self, fn_name: &str, val: &Value) {
        for guard in &self.cli_guards {
            if let Some(ref scope) = guard.scope
                && scope != fn_name {
                    continue;
                }
            if let Err(msg) = super::apply_guard_check(&guard.check, val) {
                sheaf_msg!("/!\\ Guard Breached: {:?}", guard.check);
                sheaf_msg!("Function: {}", fn_name);
                sheaf_msg!("{}", msg);
                self.dump_ring_buffer();
                std::process::exit(1);
            }
        }
    }

    pub fn dump_ring_buffer(&self) {
        if self.ring_buffer.is_empty() {
            sheaf_msg!("\n(No backtrace available)\n");
            return;
        }
        sheaf_msg!("\nBacktrace (last {} operations):\n", self.ring_buffer.len());
        for entry in &self.ring_buffer {
            sheaf_msg!("{}", entry);
        }
        sheaf_msg!("\n--- End of Backtrace ---\n");
    }

    fn push_ring(&mut self, line: &str) {
        self.ring_buffer.push_back(line.to_string());
        if self.ring_buffer.len() > self.ring_buffer_max {
            self.ring_buffer.pop_front();
        }
    }
}

fn format_value(val: &Value, level: TraceLevel) -> String {
    match val {
        Value::Tensor { data, .. } => {
            let shape: Vec<usize> = data.shape().to_vec();
            let shape_str = shape.iter().map(|d| d.to_string()).collect::<Vec<_>>().join("x");
            let bytes = data.len() * 4;
            let mem = if bytes < 1024 {
                format!("{}B", bytes)
            } else if bytes < 1024 * 1024 {
                format!("{:.1}KB", bytes as f64 / 1024.0)
            } else {
                format!("{:.1}MB", bytes as f64 / (1024.0 * 1024.0))
            };

            match level {
                TraceLevel::Fast => format!("f32[{}] ({})", shape_str, mem),
                TraceLevel::Normal => {
                    let v_min = data.iter().cloned().fold(f32::INFINITY, f32::min);
                    let v_max = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    let finite = data.iter().all(|x| x.is_finite());
                    let mut s = format!("f32[{}] [min:{:.2e} max:{:.2e}] ({})", shape_str, v_min, v_max, mem);
                    if !finite {
                        s.push_str(" [NaN DETECTED]");
                    }
                    s
                }
                TraceLevel::Verbose => {
                    let v_min = data.iter().cloned().fold(f32::INFINITY, f32::min);
                    let v_max = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    let v_mean: f32 = data.iter().sum::<f32>() / data.len().max(1) as f32;
                    let finite = data.iter().all(|x| x.is_finite());
                    let mut s = format!("f32[{}] [μ:{:.2e} min:{:.2e} max:{:.2e}] ({})", shape_str, v_mean, v_min, v_max, mem);
                    if !finite {
                        s.push_str(" [NaN DETECTED]");
                    }
                    s
                }
            }
        }
        Value::Int(n) => format!("{}", n),
        Value::Float(f) => format!("{:.6}", f),
        Value::Bool(b) => format!("{}", b),
        Value::Nil => "nil".to_string(),
        Value::String(s) => {
            if s.len() > 50 { format!("\"{}...\"", &s[..47]) } else { format!("\"{}\"", s) }
        }
        Value::List(items) => format!("list(len:{})", items.len()),
        Value::Tuple(items) => format!("tuple(len:{})", items.len()),
        Value::Dict(map) => format!("dict(keys:{:?})", map.keys().collect::<Vec<_>>()),
        Value::Function { params, .. } => format!("fn/{}", params.len()),
        Value::BuiltinFn { name, .. } => format!("<builtin:{}>", name),
        Value::Keyword(k) => format!(":{}", k),
        Value::DeviceBuffer(db) => {
            let shape_str = db.shape.iter().map(|d| d.to_string()).collect::<Vec<_>>().join("x");
            let n_elems: usize = db.shape.iter().product::<usize>().max(1);
            let bytes = n_elems * 4; // f32
            let mem = if bytes < 1024 {
                format!("{}B", bytes)
            } else if bytes < 1024 * 1024 {
                format!("{:.1}KB", bytes as f64 / 1024.0)
            } else {
                format!("{:.1}MB", bytes as f64 / (1024.0 * 1024.0))
            };
            format!("f32[{}] ({}, device)", shape_str, mem)
        }
    }
}
