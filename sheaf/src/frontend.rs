// Copyright (c) 2026 Damien Boureille
// Licensed under the MIT License.

//! Bootstrap frontend used to compile the embedded standard library.
//!
//! This crate root deliberately excludes the interpreter, lowering pipeline,
//! runtime, CLI, and IREE bindings.

#[path = "core/mod.rs"]
pub mod core;
#[path = "forms/mod.rs"]
pub mod forms;
#[allow(dead_code)]
#[path = "lowering/stablehlo/types.rs"]
mod stablehlo_types;

pub mod lowering {
    pub mod stablehlo {
        pub use crate::stablehlo_types::StableHLOType;
    }
}

pub use core::ast::SheafValue;
pub use core::error::{SheafError, SheafResult, SourceLocation};
pub use core::expr::{BindingPattern, CompiledExpr, CompilerContext};
pub use core::parse;
