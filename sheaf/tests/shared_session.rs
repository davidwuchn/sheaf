// Verifies process-wide IREE session initialization.
// Run with: cargo test --test shared_session -- --ignored --nocapture

#![cfg(iree_runtime)]

use sheaf_compiler::core::config;
use sheaf_compiler::runtime::iree_session::{
    session_creation_attempt_count, shared_session,
};
use std::sync::Arc;

#[test]
#[ignore = "requires the IREE runtime and a CPU device"]
fn shared_session_has_process_wide_identity() -> Result<(), Box<dyn std::error::Error>> {
    config::init(0, Some("cpu".to_string()), false);
    let before = session_creation_attempt_count();

    let first = shared_session()?;
    let second = shared_session()?;

    assert!(Arc::ptr_eq(&first, &second));
    assert!(session_creation_attempt_count().saturating_sub(before) <= 1);
    Ok(())
}
