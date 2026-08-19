// Copyright (c) 2025 Damien Boureille
// Licensed under the MIT License.

//! Core compiler components

pub mod ast;
pub mod color;
pub mod config;
pub mod error;
pub mod error_format;
pub mod expr;
pub mod inference;
pub mod macro_engine;
pub mod parser;
#[cfg(not(sheaf_frontend))]
pub mod trace;

pub use expr::{CompiledExpr, CompilerContext, FunctionDef};
pub use error::{SheafError, SheafResult};
pub use inference::{FunctionSignature, infer_function_signature};
pub use parser::parse;
