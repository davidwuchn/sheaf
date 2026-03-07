<img width="479" height="154" alt="Sheaf" src="https://github.com/user-attachments/assets/b79b4f7f-dc77-459e-a3a6-987bda4e70f2" />

Sheaf is a functional language for differentiable computation.
Inspired by Clojure, it compiles to [StableHLO](https://github.com/openxla/stablehlo) and runs on CPU, Metal GPU, and CUDA via [IREE](https://github.com/iree-org/iree).

Sheaf ships as a **single native binary** with zero runtime dependencies.

> **Note:** This is Sheaf V2, a ground-up rewrite in Rust replacing the original Python/JAX implementation while keeping the language and syntax. The [website](http://sheaf-lang.org/) will be updated as V2 stabilizes.

### Goals

- **Clojure paradigm**: homoiconicity, immutability, minimalist syntax
- **Native hardware performance**: compiles to StableHLO, executes via IREE on CPU/GPU/TPU
- **JIT compilation**: pure functions are automatically compiled and dispatched to the best available device
- **Symbolic autodiff**: reverse-mode automatic differentiation on the AST, before compilation

### Sample code

```clojure
(defn transformer-block [x layer-p config]
  (as-> x h
    (-> h   ;; 1. Self-Attention
        (layer-norm (get layer-p :ln1) 2)
        (multi-head-attention layer-p config)
        (first)  ;; Attention output
        (+ h))   ;; Residual 1

    (-> h   ;; 2. MLP
        (layer-norm (get layer-p :ln2) 2)
        (mlp (get layer-p :mlp))
        (+ h)))) ;; Residual 2
```

### Architecture

```
Sheaf source (.shf)
  --> Parser --> AST
  --> Type inference
  --> StableHLO codegen (MLIR)
  --> IREE compiler (VMFB)
  --> IREE runtime (CPU / Metal / CUDA)
```

The interpreter handles effectful operations (I/O, randomness) while pure numerical functions are JIT-compiled to IREE for hardware-accelerated execution.

### Links

- [Website](http://sheaf-lang.org/)
- [Documentation](http://sheaf-lang.org/starting/)
- [Reference](http://sheaf-lang.org/reference/)
