<p align="center">
<img width="522" height="129" alt="Sheaf-logo" src="https://github.com/user-attachments/assets/d2b63f55-534d-41dc-bdcf-0a950c995031" />
</p>

Sheaf is a functional language for machine learning.
Inspired by Clojure, it compiles to [StableHLO](https://github.com/openxla/stablehlo) and runs on GPU via [IREE](https://github.com/iree-org/iree).

Sheaf ships as a **single native binary** with zero runtime dependencies, for Linux (x86_64, aarch64) and macOS.

### Quick start

See the [Quick Start guide](https://sheaf-lang.org/quickstart/) for installation and first steps.

### Goals

- **Clojure paradigm**: homoiconicity, immutability, minimalist syntax
- **Native hardware performance**: compiles to StableHLO, executes via IREE on CUDA, Metal GPU and CPU
- **JIT compilation**: pure functions are automatically compiled and dispatched to the best available device
- **Reverse-mode autodiff**: automatic differentiation on the expression graph, before compilation

### Sample code

```clojure
(defn transformer-block [x layer-p config]
  (as-> x h
    (-> h
        (layer-norm (get layer-p :ln1) 2)
        (multi-head-attention layer-p config)
        (first)
        (+ h))
    (-> h
        (layer-norm (get layer-p :ln2) 2)
        (mlp (get layer-p :mlp))
        (+ h)))) ;; residual
```

### Links

- [Website](https://sheaf-lang.org/)
- [Documentation](https://sheaf-lang.org/starting/)
- [Reference](https://sheaf-lang.org/reference/)
