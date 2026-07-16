// Copyright (c) 2025 Damien Boureille
// Licensed under the MIT License.

//! Sheaf V2 - Rust implementation
//!
//! A functional language for differentiable programming,
//! compiling directly to StableHLO and running on IREE.

// Build-time stamped version. Releases use the Cargo package version;
// nightly builds override it via SHEAF_BUILD_VERSION (see build.rs).
include!(concat!(env!("OUT_DIR"), "/generated_version.rs"));

pub mod autodiff;
pub mod core;
pub mod forms;
pub mod interpreter;
pub mod lowering;
pub mod runtime;

// Re-export main types
pub use core::ast::SheafValue;
pub use lowering::{collect_hof_calls, CodeGenerator, StableHLOEmitter, StableHLOType};
pub use core::expr::{BindingPattern, CompiledExpr, CompilerContext};
pub use core::error::{SheafError, SheafResult, SourceLocation};
pub use core::parse;
