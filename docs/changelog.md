## Version history

v2.0.0 — 2026-03-18

Complete rewrite in Rust, with no Python in the execution path. The language semantics remain unchanged: all V1 code runs in V2.

- New architecture: Sheaf source compiles directly to StableHLO MLIR, IREE runtime statically linked and called through FFI.
- Transparent JIT compilation: pure functions compile automatically on first call, with content-hash caching in **sheaf**/
- Automatic differentiation via `value-and-grad`
- DeviceBuffer: compiled functions pass tensors between IREE calls without host round-trips
- Multiple dtype support: f32 (default), bf16, i32 via cast or literal annotation

v1.2.0 — 2026-02-06

- Sheaf programs are now mostly independent from Python, most missing primitives for imperative control have been added.
- I/O module: `(io "load" ...)` / `(io "save" ...)` with safetensors and JSON. Entropy source `(io "entropy")`
- Support f-strings: `(print "loss={:.4f}" loss)`
- Support string escape sequences: `\n`, `\t`, `\"`, `\\`
- New primitives: `filter`, `find`, `index-of`, `argmax`, `argmin`, `arange`, `eye`, `index-update`, `int`, `float`, `sort`, `chars`, `rms-norm`, `do`, `while`
- Error messages: suggestions for common mistakes (`def` -> `defn`, `lambda` -> `fn`, `import` -> `use`), paren balancer with culprit detection
- More bugfixes
- REPL has `--trace` and `--guard` modes for standalone tracing and debugging
- All examples are now standalone and do not require Python

v1.1.0 — 2026-01-24

- Syntax cleanup: quoted arrays (`'[]`) are now the canonical way to distinguish lists from tensors. Legacy `list` form is deprecated.
- More syntax purity: also deprecate `lambda` (alias for `fn`) and `dict`
- Protection for special forms (`fn`, `let`, `get`...)
- Many bugfixes in the compiler

v1.0.0 — 2026-01-13

- First stable release
