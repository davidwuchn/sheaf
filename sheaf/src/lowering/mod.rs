// Copyright (c) 2025 Damien Boureille
// Licensed under the MIT License.

//! Lowering: translates Sheaf IR (CompiledExpr) to StableHLO MLIR.
//!
//! - `codegen`: instruction selection -> maps Sheaf builtins to StableHLO ops.
//! - `stablehlo`: MLIR emission -> serializes ops, types, and broadcasts.

pub mod call_graph;
pub mod codegen;
pub mod config;
pub mod effects;
pub mod stablehlo;
pub mod transforms;

pub use call_graph::jit_eligibility;
pub use codegen::CodeGenerator;
pub use config::{build_index_map, json_to_stablehlo_type, layout_to_index_map, lower_get_calls};
pub use effects::{collect_effects, collect_hof_calls, format_effects, has_side_effects};
pub use stablehlo::{StableHLOEmitter, StableHLOType};
