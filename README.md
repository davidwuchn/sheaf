<p align="center">
<img width="522" height="129" alt="Sheaf logo" src="https://github.com/user-attachments/assets/d2b63f55-534d-41dc-bdcf-0a950c995031" />
</p>

[![Release](https://img.shields.io/github/v/release/sheaf-lang/sheaf)](https://github.com/sheaf-lang/sheaf/releases)
[![License](https://img.shields.io/github/license/sheaf-lang/sheaf)](https://github.com/sheaf-lang/sheaf/blob/main/LICENSE)
[![Stars](https://img.shields.io/github/stars/sheaf-lang/sheaf?style=social)](https://github.com/sheaf-lang/sheaf/stargazers)

Sheaf is a functional language for machine learning that combines the
expressiveness of Lisp with the performance of modern ML compilers.

### Highlights

- **Clojure for tensors**: homoiconicity, immutability, minimalist syntax,
  threading macros
- **REPL-driven development**: immediate tensor shapes and dtypes, inline
  documentation, and environment inspection
- **GPU-first**: compiles to [StableHLO](https://github.com/openxla/stablehlo)
  and runs on CUDA, Metal, Vulkan, and CPUs through
  [IREE](https://github.com/iree-org/iree)
- **Reverse-mode autodiff**: differentiates supported tensor programs before
  StableHLO code generation
- **JIT compilation**: eligible pure functions are automatically compiled from
  runtime shapes
- **Single native binary**: the Sheaf runtime ships as one executable for Linux
  and Apple Silicon, with no Python environment required
- **LLM-native**: built-in context generation for AI assistants

### Sample code

Define a model, differentiate, get gradients:

```clojure
sheaf> (def x (tensor [[1.0 0.0] [0.0 1.0]]))    ;; inputs
sheaf> (def y (tensor [[1.0 1.0 0.0 0.0]         ;; targets
                       [0.0 0.0 1.0 1.0]]))
sheaf> (def W (zeros '[2 4]))                    ;; weights

sheaf> ((value-and-grad
           (fn [W] (mse-loss (@ x W) y)))        ;; loss + gradients
         W)
=> [0.5 [[-0.25 -0.25 0.0 0.0]
        [0.0 0.0 -0.25 -0.25]]]
```

Transformer block with residual connections:

```clojure
(defn transformer-block [x layer-p config]
  (as-> x h
    ;; 1. Self-Attention
    (-> h
        (layer-norm (get layer-p :ln1) 2)
        (multi-head-attention layer-p config)
        (first) ;; Get the attention output, ignore weights
        (+ h))  ;; Residual 1

    ;; 2. MLP
    (-> h
        (layer-norm (get layer-p :ln2) 2)
        (mlp (get layer-p :mlp))
        (+ h)))) ;; Residual 2
```

Use macros to derive three compiled graphs from the same layer list:

```clojure
(defmacro defresidual [name args & layers] ...)   ;; generates residual graph
(defmacro definspect  [name args & layers] ...)   ;; generates monitoring graph

(defmodel    net         (x) [linear :h1 gelu] [linear :h2 gelu])
(defresidual res-net     (x) [linear :h1 gelu] [linear :h2 gelu])
(definspect  inspect-net (x) [linear :h1 gelu] [linear :h2 gelu])
```

Check out the [examples](https://github.com/sheaf-lang/sheaf/tree/main/examples)
for more code samples.

### Install

Note: Sheaf is under active development. The langage and compiler interfaces may
still change between releases.

- Download the binary tarball from https://github.com/sheaf-lang/sheaf/releases
- Download the examples:
  https://github.com/sheaf-lang/sheaf/releases/download/v2.2.0/sheaf-examples.tar.gz

On macOS, the Sheaf binary might be blocked by Gatekeeper. Unlock it with:

```bash
xattr -dr com.apple.quarantine /path/to/sheaf
```

### Build from source

Sheaf statically links against the [IREE](https://iree.dev) runtime, whose
libraries are not vendored in this repository, so a fresh `git clone` requires
one extra step before `cargo build` will work:

```bash
cd sheaf
./build-iree.sh                          # builds IREE into iree-runtime/
cargo build --release                    # builds Sheaf
cp target/release/sheaf ~/.local/bin     # (or /usr/local/bin)
```

The following tools are required on both Linux and macOS:

`cmake ninja git g++ rustc cargo`

On Linux, you will need to install `cuda-nvcc` and/or the Vulkan headers.

On macOS, the Xcode Command Line Tools are required. Install them with:

```bash
xcode-select --install
```

Sheaf requires Rust 1.85 or later. If your distribution ships an older version,
either build from a container, or [upgrade your compiler](https://rustup.rs).

### Links

- [Website](https://sheaf-lang.org/)
- [Quick Start](https://sheaf-lang.org/quickstart/)
- [Documentation](https://sheaf-lang.org/starting/)
- [Reference](https://sheaf-lang.org/reference/)
- [Examples](https://github.com/sheaf-lang/sheaf/tree/main/examples)
