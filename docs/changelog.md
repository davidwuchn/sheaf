## Version history

### v2.1.0 — 2026-05-18

This release brings a few refinements to the language, as well as many bug fixes and performance improvements. The CLEVR example is now fully differentiable and includes a new web visualizer.

**Language improvments**

- `def` for global constants, like in Clojure
- `reduce` now also supports the 2-argument form used in Clojure
- `sort` now accepts tensor arguments
- Macros can now use quasiquoted templates
- Better error hints for shapes and indexing errors

**Runtime**

- Distinguish unsupported codegen cases from real bugs when a JIT error happens
- Gracefully bail out when graph exceeds 10K nodes
- `--mem-profile` flag
- Removed unneeded dependencies

**Bugfix**

- Fixed reduce backward pass by reusing scan_vjp (1ea2fba)
- Dict key loss fix in JIT path: dict keys no longer dropped through the JIT codegen path
- VAG lambda preprocessing: static constants resolved inside lambda bodies before JIT (a496238, 150c663)
- get backward for scalar indices: reverse-mode AD now handles scalar indices correctly (30fe862)
- Slice shape propagation: `:axis` keyword now properly propagates shape in autodiff (4a779e5)
- Batched matmul gradients: fixed panic with batched tensor inputs (101b290)
- Einsum dimension mismatch: no longer panics on subscript/operand shape mismatch (123ed91)
- No longer panics on 0-dimensional tensor slicing (4b99ff1)
- One-hot bounds check: out-of-bounds indices now return proper error (d6b5883)
- nth out-of-bounds: tensor indexing no longer panics (f39da67)
- Deep layout propagation in scan VJP: nested dicts handled correctly (ed32b3f)
- Lambda scoping in resolve_constants_rec: fixed incorrect constant resolution across lambda boundaries (699148e)
- Macro expansion errors: fixed error location and body evaluation for non-quasiquoted templates (9abf9aa, ceecb71)
- Direct param unquotes: no longer eval'd at compile time in macro engine (9b4deab)
- False error for unsupported codegen calls: no longer spurious errors (add7c46)
- Fixed an REPL inline help overflow (inline help occasionally didn't show the proper help section) (f77d06c)

**Performance**

- Fused value-and-grad IREE dispatch (single IREE dispatch instead of separate forward/backward calls)
- Dot-general optimization (reshape + matmul2D + reshape instead of generic contraction)
- Backward pass optimization (simplify + CSE + double transpose elimination)
- Forward-pass binding reuse in backward to avoid recomputing values already available (ad43163)
- Fixes cache misses for closures capturing scalars (45697e5)
- Scalar constant extraction: singleton tensors extracted as scalars in codegen (5475c2f)
- Apple Accelerate enabled for ndarray on macOS (47c682f, 5c80080)

### v2.0.0 — 2026-04-02

Complete rewrite in Rust, with no Python in the execution path. The language semantics remain unchanged: all V1 code runs in V2.

- New architecture: Sheaf source compiles directly to StableHLO MLIR, IREE runtime statically linked and called through FFI.
- Transparent JIT compilation: pure functions compile automatically on first call, with content-hash caching in **sheaf**/
- Automatic differentiation via `value-and-grad`
- DeviceBuffer: compiled functions pass tensors between IREE calls without host round-trips
- Multiple dtype support: f32 (default), bf16, i32 via cast or literal annotation

### v1.2.0 — 2026-02-06

- Sheaf programs are now mostly independent from Python, most missing primitives for imperative control have been added.
- I/O module: `(io "load" ...)` / `(io "save" ...)` with safetensors and JSON. Entropy source `(io "entropy")`
- Support f-strings: `(print "loss={:.4f}" loss)`
- Support string escape sequences: `\n`, `\t`, `\"`, `\\`
- New primitives: `filter`, `find`, `index-of`, `argmax`, `argmin`, `arange`, `eye`, `index-update`, `int`, `float`, `sort`, `chars`, `rms-norm`, `do`, `while`
- Error messages: suggestions for common mistakes (`def` -> `defn`, `lambda` -> `fn`, `import` -> `use`), paren balancer with culprit detection
- More bugfixes
- REPL has `--trace` and `--guard` modes for standalone tracing and debugging
- All examples are now standalone and do not require Python

### v1.1.0 — 2026-01-24

- Syntax cleanup: quoted arrays (`'[]`) are now the canonical way to distinguish lists from tensors. Legacy `list` form is deprecated.
- More syntax purity: also deprecate `lambda` (alias for `fn`) and `dict`
- Protection for special forms (`fn`, `let`, `get`...)
- Many bugfixes in the compiler

### v1.0.0 — 2026-01-13

- First stable release
