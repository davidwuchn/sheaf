## Version history

### v2.2.0 — 2026-07-14

This release improves language correctness, brings new training features to the
standard library, adds `let` destructuring, fixes runtime bugs, and improves the
REPL experience.

**Language improvements**

- `stop-gradient` blocks gradient flow while passing the value through
  unchanged, similar to `jax.lax.stop_gradient` (b77e79a)
- `sin`, `cos`, `tan`: fully differentiable trigonometric primitives, for RoPE
  and sinusoidal position encodings (8e0a2c3)
- Destructuring bindings: `(let [[a b] expr] ...)` destructures vectors and
  tuples directly, including typed function parameters, with no interpreter
  fallback (aaa6629, 4016b87, 5a009c3)
- `flip` reverses a tensor or list along an axis (6b3905b)
- `transpose` accepts a quoted permutation such as `(transpose x '[0 2 1])` in
  the `value-and-grad` path (dd0830d)

**Standard library additions**

- `adamw-step` and `sgd-momentum-step` optimizers (0acd028)
- Learning-rate schedulers: `linear-warmup`, `inverse-sqrt-warmup` (Noam),
  `exponential-decay`, `step-decay`, `cosine-decay` (f1724ad)
- `dropout`: inverted, differentiable dropout (cce366d)

**Runtime**

- Removed unnecessary unwrap() calls on infallible writeln! in the MLIR emitter
  (9485d03)
- A warning is now emitted when a scalar baked into a compiled graph changes
  value between calls, instead of failing silently with stale results (4d90969)
- Add a build script to build the IREE runtime without depending on the GitHub
  workflow (e6a366c)

**Performance**

- Host tensors are no longer redundantly cloned in the arithmetic builtins and
  IREE dispatch path (`ensure_host_cow`, `Cow<ArrayD>`) (5c95e8b, 156e00b)
- `random-uniform` now lowers to StableHLO, so compiled functions using it no
  longer fall back to the interpreter (c24c1c1)

**Bug fixes**

- `io save` now matches `io load` when using .safetensors files (184707b)
- JIT no longer bakes dict scalars unless required for tensor shapes (ee0c2dc)
- `defn` no longer infinite-loops when a parameter is referenced in its own body
  (01f3187)
- `tril` no longer panics on a missing argument (3acee0b)
- `scan` rejects ambiguous 2-element tensor returns with a clear error instead
  of misreading them as a carry/output pair (ad6b4cf)
- Runtime vectors of tensors are no longer implicitly stacked into a single
  tensor (d1574d9)
- The YAML test loader no longer drops multi-line test cases (c599ac7)

**REPL**

- Bracket matching with highlightning (bfe4d97)
- `:show <name>` now pretty-prints a function's source instead of a placeholder
  (1c0a35a)
- New `:list` command and a shared inline documentation module (18c8c18)
- Verbatim preservation of multi-line history instead of flattening it into a
  single line (1c0a35a)

### v2.1.0 — 2026-05-18

This release brings a few refinements to the language, as well as many bug fixes
and performance improvements. The CLEVR example is now fully differentiable and
includes a new web visualizer.

**Language improvements**

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

**Bug fixes**

- Fixed reduce backward pass by reusing scan_vjp (1ea2fba)
- Dict key loss fix in JIT path: dict keys no longer dropped through the JIT
  codegen path
- VAG lambda preprocessing: static constants resolved inside lambda bodies
  before JIT (a496238, 150c663)
- get backward for scalar indices: reverse-mode AD now handles scalar indices
  correctly (30fe862)
- Slice shape propagation: `:axis` keyword now properly propagates shape in
  autodiff (4a779e5)
- Batched matmul gradients: fixed panic with batched tensor inputs (101b290)
- Einsum dimension mismatch: no longer panics on subscript/operand shape
  mismatch (123ed91)
- No longer panics on 0-dimensional tensor slicing (4b99ff1)
- One-hot bounds check: out-of-bounds indices now return proper error (d6b5883)
- nth out-of-bounds: tensor indexing no longer panics (f39da67)
- Deep layout propagation in scan VJP: nested dicts handled correctly (ed32b3f)
- Lambda scoping in resolve_constants_rec: fixed incorrect constant resolution
  across lambda boundaries (699148e)
- Macro expansion errors: fixed error location and body evaluation for
  non-quasiquoted templates (9abf9aa, ceecb71)
- Direct param unquotes: no longer eval'd at compile time in macro engine
  (9b4deab)
- False error for unsupported codegen calls: no longer spurious errors (add7c46)
- Fixed an REPL inline help overflow (inline help occasionally didn't show the
  proper help section) (f77d06c)

**Performance**

- Fused value-and-grad IREE dispatch (single IREE dispatch instead of separate
  forward/backward calls)
- Dot-general optimization (reshape + matmul2D + reshape instead of generic
  contraction)
- Backward pass optimization (simplify + CSE + double transpose elimination)
- Forward-pass binding reuse in backward to avoid recomputing values already
  available (ad43163)
- Fixes cache misses for closures capturing scalars (45697e5)
- Scalar constant extraction: singleton tensors extracted as scalars in codegen
  (5475c2f)
- Apple Accelerate enabled for ndarray on macOS (47c682f, 5c80080)

### v2.0.0 — 2026-04-02

Complete rewrite in Rust, with no Python in the execution path. The language
semantics remain unchanged: all V1 code runs in V2.

- New architecture: Sheaf source compiles directly to StableHLO MLIR, IREE
  runtime statically linked and called through FFI.
- Transparent JIT compilation: pure functions compile automatically on first
  call, with content-hash caching in **sheaf**/
- Automatic differentiation via `value-and-grad`
- DeviceBuffer: compiled functions pass tensors between IREE calls without host
  round-trips
- Multiple dtype support: f32 (default), bf16, i32 via cast or literal
  annotation

### v1.2.0 — 2026-02-06

- Sheaf programs are now mostly independent from Python, most missing primitives
  for imperative control have been added.
- I/O module: `(io "load" ...)` / `(io "save" ...)` with safetensors and JSON.
  Entropy source `(io "entropy")`
- Support f-strings: `(print "loss={:.4f}" loss)`
- Support string escape sequences: `\n`, `\t`, `\"`, `\\`
- New primitives: `filter`, `find`, `index-of`, `argmax`, `argmin`, `arange`,
  `eye`, `index-update`, `int`, `float`, `sort`, `chars`, `rms-norm`, `do`,
  `while`
- Error messages: suggestions for common mistakes (`def` -> `defn`, `lambda` ->
  `fn`, `import` -> `use`), paren balancer with culprit detection
- More bugfixes
- REPL has `--trace` and `--guard` modes for standalone tracing and debugging
- All examples are now standalone and do not require Python

### v1.1.0 — 2026-01-24

- Syntax cleanup: quoted arrays (`'[]`) are now the canonical way to distinguish
  lists from tensors. Legacy `list` form is deprecated.
- More syntax purity: also deprecate `lambda` (alias for `fn`) and `dict`
- Protection for special forms (`fn`, `let`, `get`...)
- Many bugfixes in the compiler

### v1.0.0 — 2026-01-13

- First stable release
