// Verifies that JIT variants are reused across independent evaluations.

#![cfg(iree_runtime)]

use sheaf_compiler::core::config;
use sheaf_compiler::interpreter::eval::eval_source;
use sheaf_compiler::interpreter::value::Value;
use sheaf_compiler::runtime::iree_session::{
    initialized_shared_session, session_creation_attempt_count,
};
use sheaf_compiler::runtime::toolchain::ensure_toolchain;
use std::ffi::OsString;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

struct TempDir(PathBuf);

impl TempDir {
    fn new(label: &str) -> Self {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock before Unix epoch")
            .as_nanos();
        let path =
            std::env::temp_dir().join(format!("sheaf-{label}-{}-{nonce}", std::process::id()));
        fs::create_dir(&path).expect("create temporary test directory");
        Self(path)
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

struct EnvVarGuard {
    name: &'static str,
    previous: Option<OsString>,
}

impl EnvVarGuard {
    fn set(name: &'static str, value: impl AsRef<std::ffi::OsStr>) -> Self {
        let previous = std::env::var_os(name);
        // This binary contains one test; no parallel test can observe this process environment.
        unsafe { std::env::set_var(name, value) };
        Self { name, previous }
    }
}

impl Drop for EnvVarGuard {
    fn drop(&mut self) {
        unsafe {
            match &self.previous {
                Some(value) => std::env::set_var(self.name, value),
                None => std::env::remove_var(self.name),
            }
        }
    }
}

fn compiler_invocations(counter: &Path) -> usize {
    match fs::read_to_string(counter) {
        Ok(contents) => contents.lines().count(),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => 0,
        Err(error) => panic!("read compiler invocation counter: {error}"),
    }
}

fn source(function_name: &str) -> String {
    format!(
        "(defn {function_name} [x] (sum (+ x 1.0)))\n[({function_name} (zeros [2])) ({function_name} (zeros [3]))]\n"
    )
}

fn assert_variant_results(value: Value) {
    let Value::Tensor { data, .. } = value else {
        panic!("expected the two scalar results as a tensor");
    };
    assert_eq!(data.shape(), &[2]);
    assert_eq!(data.iter().copied().collect::<Vec<_>>(), vec![2.0, 3.0]);
}

#[test]
fn reuses_polymorphic_jit_variants_across_independent_environments() {
    config::init(0, Some("cpu".to_string()), false);

    let real_compiler = ensure_toolchain().expect("resolve IREE compiler through Sheaf toolchain");
    let temp = TempDir::new("polymorphic-jit-reuse");
    let counter = temp.path().join("iree-compile-count");
    let wrapper = temp.path().join("iree-compile-wrapper");
    fs::write(
        &wrapper,
        "#!/bin/sh\nprintf '%s\\n' 1 >> \"$SHEAF_DAY8_COUNTER\"\nexec \"$SHEAF_DAY8_REAL_COMPILER\" \"$@\"\n",
    )
    .expect("write compiler wrapper");
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(&wrapper, fs::Permissions::from_mode(0o755))
            .expect("make compiler wrapper executable");
    }

    let _counter_env = EnvVarGuard::set("SHEAF_DAY8_COUNTER", &counter);
    let _real_compiler_env = EnvVarGuard::set("SHEAF_DAY8_REAL_COMPILER", &real_compiler);
    let _compiler_env = EnvVarGuard::set("IREE_COMPILE", &wrapper);

    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock before Unix epoch")
        .as_nanos();
    let function_name = format!("day8_poly_{}_{}", std::process::id(), nonce);
    let program = source(&function_name);
    let sessions_before = session_creation_attempt_count();

    // eval_source constructs a fresh CompilerContext and Env for each call.
    let first_result = eval_source(&program).expect("first independent environment");
    assert_variant_results(first_result);
    assert_eq!(
        compiler_invocations(&counter),
        2,
        "environment A must compile exactly one VMFB for each shape variant"
    );
    let first_session = initialized_shared_session().expect("session initialized by environment A");

    let invocations_after_first_environment = compiler_invocations(&counter);
    let second_result = eval_source(&program).expect("second independent environment");
    assert_variant_results(second_result);
    assert_eq!(
        compiler_invocations(&counter),
        invocations_after_first_environment,
        "environment B must reuse both cached variants without iree-compile"
    );

    let second_session = initialized_shared_session().expect("shared session reused by environment B");
    assert!(
        Arc::ptr_eq(&first_session, &second_session),
        "independent environments must use the same process-wide IREE session"
    );
    assert!(
        session_creation_attempt_count().saturating_sub(sessions_before) <= 1,
        "the two environments must not create separate IREE sessions"
    );
}
