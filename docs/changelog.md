## Version history

### v2.3.0 — 2026-09-25

This release delivers major performance and memory-use improvements. The
language, compiler, and runtime have been reworked to scale to larger models,
both for inference and training. The runtime has been performance-tested with up
to 1.5 billion parameters, and a Sheaf port of Gemma 4 is already underway to
push performance further. The release also adds mixed-precision training, more
robust autodiff, broader support for compiled operations, and numerous bug fixes
across the compiler and runtime.

**Language improvements**

- The new `dynamic-update-slice` function writes a tensor into another tensor
  without changing the input to simplify KV-cache updates (383edfa)
- `dynamic-slice` now accepts one runtime start index per axis (383edfa)
- `concat` now supports separate tensor arguments and an optional `:axis`
  keyword in compiled functions, in addition to the vector form (a343c8b)

**Examples**

- A new `macros` example uses macros to derive models from a template (d06fff3)

**Runtime**

- Reverse-mode autodiff supports differently shaped vector and matrix products
  (d5ea38e)
- The JIT can compile separate versions of a function for different shapes and
  dtypes. If one version fails to compile, the others still work (9d8b6c1,
  624c258, 66654ce)
- f16 tensors can now be copied to and from the device without being converted
  to f32 (f2e39d3)
- All evaluations in a process now share one IREE session. If a function has
  already been compiled, a new evaluation can call it without running
  `iree-compile` again (5ff2db5, de1cbdf, 624c258)
- One IREE session can load several VMFB modules without mixing up functions
  that have the same name (4780bb6, 16a7e78)
- JIT cache keys now use the parsed function instead of its formatted text.
  Different functions can no longer receive the same cache key (d05b9a2)
- Before compiling a function, the JIT now checks every function it calls.
  Recursive functions and unsupported higher-order calls fall back before they
  reach the compiler (1979746, 9c73d90)
- Dot products, reductions, elementwise functions, indexing, and shape changes
  no longer turn f16 or bf16 tensors into f32 tensors. Compiled functions can
  also cast values to f16 (08d6441, ef34c7f)
- Autodiff now reports an error when an operation has no gradient rule. It used
  to return zero without warning. Constants, unused values, and `stop-gradient`
  still produce zero gradients as expected
  ([#3](https://github.com/sheaf-lang/sheaf/issues/3))
- A failed recompilation can be retried on the next call. Errors while
  converting buffers are returned normally instead of leaking IREE values or
  panicking (45b1d8a, 32a1052)
- Rank-zero tensors no longer lose their dtype when copied back from the device
  (417331b)

**Performance**

- Compiled modules now share cached IREE buffers, so they can reuse device
  copies of the same model weights (29fc51e, f63c667)
- JIT calls look up compiled versions directly instead of rebuilding a cache key
  first (e2bdb0a, cf17884)
- The compiler remembers which scalar arguments affect tensor shapes instead of
  repeating the same analysis on every call (ddc24c3)
- Compiling the same function twice now generates the same ANF and gradient
  names (b98e7e2)
- Code generation no longer copies the full function registry for every
  compilation (a5b0bf5)

**Bug fixes**

- The interpreter and compiler now use the same dtype and broadcasting rules.
  Arithmetic with f16 and bf16 tensors no longer widens them to f32 without
  warning (88cde2c, 2cfcb9e, 1913129, 24da544, 84064fa, 4a98c7f, 913e86e,
  120f7e3)
- Unused leaves in tuple parameters receive explicit zero gradients, and missing
  gradient results are reported instead of being treated as zero (48e4467,
  9efa57e)
- Shape variables no longer leak from one expression into another and produce
  wrong reshape dimensions (14d3b88)
- Unrolled `reduce` calls no longer lose tuple bindings or mix up fields in
  nested parameter tuples (deaacc6, 1e1abcd)
- Tuple destructuring now works when the tuple comes from a `let` or `do`
  expression (f4033be)
- Compiled functions no longer turn dictionaries with string keys, or results
  from `assoc`, into raw tuples (c22a702, 1a20907)
- `value-and-grad` can now compile nested reductions over dictionary parameters
  (2f5f987)
- Shared embedding and output weights now receive gradients with the correct
  shape (34e8a8a)
- Scalar i32 outputs from IREE are decoded as integers rather than f32 values
  (d96fa0f)
- Converting a compiled rank-zero tensor with `float` returns a scalar again,
  including when the value is still stored in a device buffer (f144062)
- Unsupported calls nested inside tuples and other compound expressions are
  detected before JIT compilation (8181609)
- Two JIT compilations running at the same time can no longer write or load the
  same cache file concurrently (9532561)
- The JIT no longer loses static dimensions when a tensor is passed directly to
  a function (3b5e122)
- The buffer cache no longer releases a device buffer while a returned
  `DeviceBuffer` still uses it (77a2eae)
- Symbolic bindings created while tracing a `let` expression no longer escape
  their lexical scope (d6fe8f4)

**REPL and CLI**

- Grouped inline documentation handles comma-separated headings correctly, and
  internal primitives are hidden from completion (341bb41)
- Registry listings are now sorted (8608565)
- Multiline values align continuation lines with their first line (d84e4d5)
- `--jit-profile` now works without `--blame`. It cannot be combined with
  `--trace`, and `--blame` warns when tracing has disabled the JIT (71a1dcd)
- `--mem-profile` no longer reports cumulative Metal allocations as live memory
  (60efeeb)

**Build and release**

- The former build system, which used a mix of Bash, Python, Cargo, CMake and
  Ninja, has been replaced with Bazel. Bazel now hermetically builds Sheaf,
  IREE, the standard library, tests, examples, and release binaries
  ([#9](https://github.com/sheaf-lang/sheaf/issues/9)).
- The standard library is now compiled as an artefact during the build process
  instead of every time Sheaf starts (1a6462f, f9cb32d)
- Example archives published with releases are now built automatically (31ba3b6)
- Nightly builds report a distinct date-stamped version rather than the latest
  stable release (e828841)
- Linux builds now include Vulkan support by default and automatically enable
  CUDA when a compatible toolkit is available (8f42f59, 1ac9523)
- Release builds through Bazel use link-time optimization (21ae97c)
- Nightly and release binaries use the same Bazel build as local installations.
  The Linux x86-64 binary is tested on CUDA before it is published (90fe08b,
  b27e0eb)
- Nightly releases now include Linux aarch64 binaries, alongside Linux x86-64
  and Apple Silicon builds (fe5ea2e)
- The test suite now runs the MLP, Hydra, CLEVR, and NanoGPT examples separately
  on each platform. It also ensures that model forward and training functions
  are JIT-compiled and do not fall back to the interpreter (599de6c, 5e62abe,
  4814986)

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
