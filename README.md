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
- **Reverse-mode autodiff**: ahead-of-time automatic differentiation
- **JIT compilation**: pure functions are automatically JIT-compiled
- **Single native binary**: runs on Linux (x86_64, aarch64) and macOS (Apple
  Silicon) with no runtime dependencies
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

- Download the binary tarball from https://github.com/sheaf-lang/sheaf/releases
- Download the examples:
  https://github.com/sheaf-lang/sheaf/releases/download/v2.1.0/sheaf-examples.tar.gz

### Links

- [Website](https://sheaf-lang.org/)
- [Quick Start](https://sheaf-lang.org/quickstart/)
- [Documentation](https://sheaf-lang.org/starting/)
- [Reference](https://sheaf-lang.org/reference/)
- [Examples](https://github.com/sheaf-lang/sheaf/tree/main/examples)
