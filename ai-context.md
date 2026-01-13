---
title: Sheaf Language Reference
version: 0.9-RC
target: AI Assistants & LLM Agents
last_updated: 2026-01-12
keywords: [sheaf, jax, lisp, neural-networks, ml, dsl, differentiable]
---

# Sheaf Language Reference (AI Assistant Context)

> **Quick reference for AI assistants helping with Sheaf development.**
>
> Sheaf is a differentiable Lisp dialect for machine learning.
> Philosophy: "Clojure for Tensors". It compiles pure S expressions into accelerated execution backends.
> The current backend targets JAX.

---

## Quick Start Example

```sheaf
(use nn)
(use optim)

;; Simple MLP forward pass with parameter destructuring
(defn forward (x p)
  (as-> x h
    (with-params (get p :l1)    ;; Auto-bind W and b from layer 1
      (relu (+ (@ h W) b)))
    (with-params (get p :l2)    ;; Auto-bind W and b from layer 2
      (sigmoid (+ (@ h W) b)))))

;; Training step with Adam optimizer
(defn train-step (p m v t x y lr)
  (let (loss-fn (lambda (params) (mse-loss params x y))
        [loss grads] ((value-and-grad loss-fn) p)
        [new-p new-m new-v new-t] (adam-step p grads m v t lr 0.9 0.999 1e-8))
    (dict :p new-p :m new-m :v new-v :t new-t :loss loss)))
```

---

## Core Concepts

### 1. Everything is Differentiable

- All operations compile to JAX primitives
- Full gradient support via `(value-and-grad func)`
- Works with JAX transformations: `jax.jit`, `jax.vmap`, `jax.grad`

### 2. Dictionary-Based Parameters

Neural network parameters are stored in nested HashableDicts:

```sheaf
(dict :l1 (dict :W weights :b biases)
      :l2 (dict :W weights :b biases))
```

### 3. Python Integration

```python
from sheaf import Sheaf

shf = Sheaf()
shf.load("(defn add-five [x] (+ x 5))")
result = shf.add_five(10)  # Direct Python call

# Access compiled JAX function
jax_func = shf.registry["add-five"]
```

### 4. Macro System

Sheaf supports compile-time macros with quasiquote syntax:

```sheaf
(defmacro when [cond body]
  `(if ~cond ~body nil))

;; Usage:
(when (> x 0)
  (print "positive"))

;; Expands to:
(if (> x 0)
  (print "positive")
  nil)
```

**Quote and quasiquote operators:**

- `'expr` or `(quote expr)` - quote: prevent evaluation, return data as-is
- `` `expr `` - quasiquote: create template with selective evaluation
- `~expr` - unquote: evaluate expression inside quasiquote
- `~@expr` - unquote-splicing: splice list into quasiquote

**Quote examples:**

```sheaf
'foo              ; => foo (symbol, not evaluated)
'(+ 1 2)          ; => (+ 1 2) (list, not evaluated)
`(+ 1 ~(* 2 3))   ; => (+ 1 6) (quasiquote with unquote)
```

**Standard macros** (from `lib/macros.shf`):

- `when` - conditional execution (returns nil if false)
- `unless` - inverse conditional
- `comment` - comment out code blocks

#### Advanced Macros: `defmodel`

Sheaf macros can manipulate code at compile-time using Lisp functions. The `defmodel` macro (from `lib/defmodel.shf`) demonstrates this:

```sheaf
(use defmodel)

(defmodel my-mlp [x]
  (layer :l1 (linear 128) relu)
  (layer :l2 (linear 10) softmax))
```

**Expands to:**

```sheaf
(defn my-mlp [x p]
  (as-> x _
    (with-params (get p :l1) (relu (+ (@ _ W) b)))
    (with-params (get p :l2) (softmax (+ (@ _ W) b)))))
```

**How it works:**

- The macro uses `map` and `transform-layer` to process each layer specification
- Functions like `map`, `first`, `nth` operate on **S-expressions as data** at compile-time
- The `~@` (unquote-splicing) operator injects the transformed layers into the template

**Key insight:** Macros distinguish between:

- **Symbols as data** (e.g., `'x'` remains a symbol in the AST)
- **Functions as code** (e.g., `(first params)` executes `first` at macro expansion time)
- **First-class functions** (e.g., `transform-layer` in `(map transform-layer layers)`)

This allows powerful metaprogramming while maintaining code clarity.

---

## Syntax at a Glance

### Literals

```sheaf
[1 2 3]          ; Vector (JAX array)
:keyword         ; Keyword (evaluates to string "keyword")
True / False     ; Booleans
nil              ; None
...              ; Ellipsis (for indexing/einsum)
```

### Core Operators

**Function Definition**

```sheaf
(defn name [args] body)              ; Standard function
(defn :jit name [args] body)         ; JIT-compiled (faster, limited control flow)
(lambda [args] body)                 ; Anonymous function
```

**Binding & Scope**

```sheaf
(let [x val y val2] body)            ; Sequential local bindings
(with-params params body)            ; Auto-destructure dict (:W, :b, etc.)
```

**Control Flow**

```sheaf
(if cond then else)                  ; Branching (avoid in JIT functions)
(where cond true-val false-val)      ; Differentiable select (use in JIT)
(repeat [i n] [acc init] body)       ; Loop with accumulator
(static expr)                        ; Force static evaluation in JIT
```

**Macros**

```sheaf
(defmacro name [params] template)    ; Define compile-time macro
'expr / (quote expr)                 ; Quote: prevent evaluation
`expr                                ; Quasiquote: create template
~expr                                ; Unquote: evaluate in template
~@expr                               ; Unquote-splicing: splice list in template

;; Example:
(defmacro when [cond body]
  `(if ~cond ~body nil))
```

**Functional Operations**

```sheaf
(map func coll)                      ; Apply function to each element
(reduce func acc coll)               ; Reduce collection with accumulator
```

**Threading Macros**

```sheaf
(-> x (f1) (f2 a))                   ; Thread-first: (f2 (f1 x) a)
(as-> init name step1 step2)         ; Thread-as: bind value at each step
```

**Module System**

```sheaf
(use nn)                             ; Import from lib/nn.shf
(use optim)                          ; Import from lib/optim.shf
(use macros)                         ; Import standard macros (when, unless, comment)
```

---

## Common Patterns

### Pattern 1: Multi-Layer Perceptron

```sheaf
(defn forward (x p)
  (as-> x h
    (with-params (get p :l1)
      (relu (+ (@ h W) b)))
    (with-params (get p :l2)
      (sigmoid (+ (@ h W) b)))))
```

### Pattern 2: Transformer Block (from BareGPT)

```sheaf
(defn transformer-block (x layer-p config)
  (let (;; Self-Attention + Residual 1
        ln1_x (layer-norm x (get layer-p :ln1) 2)
        attn_out (first (multi-head-attention ln1_x layer-p config))
        x1 (+ x attn_out)

        ;; MLP + Residual 2
        ln2_x1 (layer-norm x1 (get layer-p :ln2) 2)
        mlp_out (mlp ln2_x1 (get layer-p :mlp))
        x2 (+ x1 mlp_out))
    x2))
```

### Pattern 3: Training Loop with Adam

```sheaf
(defn train-step (params m v t inputs targets config)
  (let (lr (get config :lr)
        loss-fn (lambda (p) (cross-entropy-loss (model inputs p config) targets))
        [loss grads] ((value-and-grad loss-fn) params)
        [new-params new-m new-v new-t] (adam-step params grads m v t lr 0.9 0.999 1e-8))
    (dict :loss loss :params new-params :m new-m :v new-v :t new-t)))
```

### Pattern 4: Einsum for Multi-Head Attention

```sheaf
;; Q, K, V projections [Batch, Heads, Time, Head_dim]
(let (Qh (einsum "... t d, d h k -> ... h t k" X Wq_multi)
      Kh (einsum "... t d, d h k -> ... h t k" X Kh_multi)
      Vh (einsum "... t d, d h k -> ... h t k" X Vh_multi))
  ...)
```

---

## Gotchas & Common Mistakes

### ⚠️ Scalars vs Arrays

Sheaf uses JAX arrays as the fundamental data type. Python scalars (int, float) are automatically converted to JAX arrays when needed:

```sheaf
; Scalars are auto-converted
(+ 5 x)              ; 5 becomes a JAX scalar array
(* 3.14 tensor)      ; 3.14 becomes a JAX scalar array

; Vectors/arrays are JAX arrays from the start
[1 2 3]              ; JAX array of shape [3]

; To create a scalar tensor explicitly (rarely needed):
(reshape 42 )        ; Scalar tensor (shape [])
(+ 0 scalar-value)   ; Force conversion to array
```

**When you need explicit conversion:**

- Most operations auto-convert Python scalars
- If a function requires a tensor but you have a scalar, wrap it: `(reshape value)` or add 0: `(+ 0 value)`

### ⚠️ Unary Negation

```sheaf
; WRONG: (- x)          ; Not supported!
; RIGHT: (* -1.0 x)     ; Use multiplication
; RIGHT: (- 0 x)        ; Subtract from zero
```

### ⚠️ Equality Operators

```sheaf
(= a b)      ; Global equality → True/False (use in `if`)
(== a b)     ; Element-wise equality → tensor of bools (use for masks)
(!= a b)     ; Element-wise inequality
```

### ⚠️ Control Flow in JIT Functions

```sheaf
(defn :jit fast-func [x]
  ; WRONG: (if (> x 0) x 0)     ; Dynamic branching breaks JIT
  ; RIGHT: (where (> x 0) x 0)  ; Differentiable select
  ...)
```

### ⚠️ Shape Inference with Static

```sheaf
(defn :jit model [x config]
  (let (D (static (get config :d_model)))  ; Force static evaluation
    (reshape x -1 D)))
```

### ⚠️ Dictionary Access

```sheaf
(get dict :key)              ; Single key access
(get-in dict [:path :to :key])  ; Nested access
(with-params dict body)      ; Auto-bind :W, :b, etc. as variables
```

---

## Quick Reference Tables

### Math Operations

| Operator                      | Description                          | Example                    |
| ----------------------------- | ------------------------------------ | -------------------------- |
| `+`, `-`, `*`, `/`            | Arithmetic (variadic, broadcastable) | `(+ a b c)`                |
| `@`                           | Matrix multiplication                | `(@ W x)`                  |
| `**`                          | Exponentiation                       | `(** x 2)`                 |
| `(einsum pattern ...tensors)` | Einstein summation                   | `(einsum "ij,jk->ik" A B)` |
| `(sum t :axis i)`             | Reduction                            | `(sum logits :axis -1)`    |
| `(mean t :axis i)`            | Mean                                 | `(mean loss)`              |

### Tensor Shaping

| Function                    | Description        | Example                        |
| --------------------------- | ------------------ | ------------------------------ |
| `(shape t)`                 | Get shape tuple    | `(shape x)` → `[B, T, D]`      |
| `(shape t axis)`            | Get dimension      | `(shape x -1)` → `D`           |
| `(reshape t ...dims)`       | Reshape tensor     | `(reshape x -1 D)`             |
| `(transpose t ...axes)`     | Permute axes       | `(transpose x 1 0 2)`          |
| `(swapaxes t a1 a2)`        | Swap two axes      | `(swapaxes x -1 -2)`           |
| `(tensor-split t n [axis])` | Split into n parts | `(tensor-split x 3)` → `[...]` |

### Activations

| Function              | Description           |
| --------------------- | --------------------- |
| `(relu x)`            | ReLU activation       |
| `(gelu x)`            | GELU (used in GPT)    |
| `(sigmoid x)`         | Sigmoid (0-1 range)   |
| `(tanh x)`            | Hyperbolic tangent    |
| `(softmax x :axis i)` | Softmax normalization |
| `(silu x)`            | Swish / SiLU          |

### List Construction & Manipulation

| Function           | Description                      | Example                       |
| ------------------ | -------------------------------- | ----------------------------- |
| `(list ...items)`  | Create list from arguments       | `(list 1 2 3)` → `[1, 2, 3]`  |
| `(cons head tail)` | Prepend element to list          | `(cons 1 (list 2))` → `[1 2]` |
| `(first coll)`     | Get first element (nil if empty) | `(first (list 1 2))` → `1`    |
| `(rest coll)`      | All except first ([] if empty)   | `(rest (list 1 2))` → `[2]`   |
| `(empty? coll)`    | Check if empty                   | `(empty? (list))` → `True`    |
| `(count coll)`     | Number of elements               | `(count (list 1 2))` → `2`    |

### Symbol Manipulation

| Function           | Description               | Example                    |
| ------------------ | ------------------------- | -------------------------- |
| `(symbol? obj)`    | Check if object is symbol | `(symbol? 'foo)` → `True`  |
| `(gensym prefix?)` | Generate unique symbol    | `(gensym)` → `"G__abc123"` |

### Random Numbers (JAX PRNG)

JAX uses explicit PRNG keys (not global state). Always create a key first, then split for independent samples.

| Function                     | Description                       |
| ---------------------------- | --------------------------------- |
| `(random-key seed)`          | Create PRNG key from integer seed |
| `(random-split key)`         | Split key → 2 independent keys    |
| `(random-split key n)`       | Split into n independent keys     |
| `(random-normal key shape)`  | Sample from N(0,1)                |
| `(random-uniform key shape)` | Sample from U(0,1)                |
| `(choice key n :p probs)`    | Sample categorical                |

**Quick example:**

```sheaf
(let (key (random-key 42)
      keys (random-split key 3))
  (random-normal (first keys) [10 10]))
```

### Weight Initialization

| Function                           | Best For        |
| ---------------------------------- | --------------- |
| `(init-xavier-uniform key shape)`  | Tanh, Sigmoid   |
| `(init-kaiming-uniform key shape)` | ReLU networks   |
| `(init-orthogonal key shape)`      | RNNs, deep nets |

---

## Standard Library (lib/)

### nn.shf

```sheaf
(layer-norm x p axis)              ; Layer normalization
(linear x w b)                     ; Dense layer: x @ w + b
(cross-entropy-loss labels logits) ; Cross-entropy
```

### optim.shf

```sheaf
(sgd-step p g lr)                           ; SGD update
(adam-step p g m v t lr b1 b2 eps)          ; Adam optimizer
(global-norm pytree)                        ; Compute L2 norm of gradients
(clip-by-global-norm pytree max-norm)       ; Gradient clipping
```

---

## Observability & Debugging

### Interactive REPL (Console)

Launch the Sheaf interactive console for exploration and debugging:

```bash
python -m sheaf.repl
```

Features:

- **Expression evaluation**: Test functions and inspect results
- **Tensor statistics**: Automatic μ/min/max for large tensors
- **Tracing control**: `:trace verbose`, `:scope function-name`
- **Environment inspection**: `:env` to see all functions/variables
- **Auto-completion**: Tab-complete commands, functions, and variables
- **Command history**: Saved to `~/.sheaf_history`

Example session:

```sheaf
sheaf> (defn double (x) (* x 2))
sheaf> (double 21)
⇒ Tensor i32[] = 42

sheaf> :trace verbose
sheaf> (reshape (arange 100) 10 10)
⇒ Tensor i32[10x10] (μ=49.500 min=0.000 max=99.000)
```

### Tracing

Configure via `Sheaf` instance:

```python
shf = Sheaf(trace='normal', scope='forward', log='console')
```

**Trace Levels:**

- `fast`: Function names, shapes, memory, time
- `normal`: + min/max ranges, NaN detection
- `verbose`: + mean values

### Guards (Runtime Assertions)

```sheaf
(guard :no-nan x)                  ; Fail if NaN/Inf
(guard :range [-10.0 10.0] x)      ; Fail if outside range
(guard :shape [64 256] x)          ; Fail if shape mismatch
```

Guards automatically convert Python scalars to JAX arrays, so you can use them with both scalars and tensors:

```sheaf
(guard :range [0.0 100.0] 42)      ; Works with scalar
(guard :range [0.0 100.0] tensor)  ; Works with array
```

Guards trigger an **Emergency Backtrace** (last 100 ops with shapes/stats) when violated.

---

## Architecture Overview

```
sheaf/
├── core/
│   ├── compiler.py       # S-expr → JAX compiler
│   ├── parser.py         # Lisp parser
│   ├── trace.py          # Tracing & guards
│   └── error_handler.py  # Error reporting
├── runtime/
│   ├── core_ops.py       # defn, let, if, etc.
│   ├── jax_ops.py        # einsum, reshape, etc.
│   ├── math_ops.py       # +, -, *, /, @, etc.
│   ├── nn_ops.py         # relu, softmax, etc.
│   └── string_ops.py     # str, concat, etc.
└── lib/
    ├── nn.shf            # Neural network stdlib
    └── optim.shf         # Optimizers stdlib
```

---

## Best Practices

### 1. Use `sigmoid` for independent probabilities

```sheaf
(sigmoid attention-scores)  ; Each object can be selected independently
```

### 2. Use `softmax` for exclusive selection

```sheaf
(softmax logits :axis -1)  ; Only one class selected
```

### 3. Use `...` (ellipsis) for rank-agnostic code

```sheaf
(einsum "... i j, ... j k -> ... i k" A B)  ; Works for any batch dims
(get tensor ... -1)  ; Index last element regardless of rank
```

### 4. Use `if` for control-flow, `where` for data-flow

```sheaf
; Control-flow (non-differentiable):
(if (= epoch 0) init-lr (* lr 0.1))

; Data-flow (differentiable, JIT-safe):
(where (> x 0) x (* x 0.01))  ; Leaky ReLU
```

### 5. Leverage `with-params` for cleaner code

```sheaf
; Instead of:
(let (W (get params :W)
      b (get params :b))
  (+ (@ x W) b))

; Write:
(with-params params
  (+ (@ x W) b))
```

---

## Common Tasks

### Load and compile Sheaf code

```python
from sheaf import Sheaf

shf = Sheaf()
shf.load_file("model.shf")
shf.compile("forward")  # Optional: explicit compilation
```

### Apply JAX transforms

```python
import jax

# Direct access to compiled functions
batched_forward = jax.vmap(shf.forward)
fast_forward = jax.jit(shf.forward)
```

### Training loop

```python
state = {"p": params, "m": m, "v": v, "t": 1}
for epoch in range(epochs):
    state = shf.train_step(state["p"], state["m"], state["v"],
                           state["t"], X, y, lr)
    print(f"Loss: {state['loss']}")
```

### Checkpointing with pytrees

```python
# Convert Sheaf state to pytree
tree = shf.to_pytree(state)

# Save (using Python, not Sheaf)
import safetensors
safetensors.save_file(tree, "checkpoint.safetensors")

# Load and restore
loaded_tree = safetensors.load_file("checkpoint.safetensors")
restored_state = shf.from_pytree(loaded_tree)

# Continue training
state = shf.train_step(restored_state, data)
```

**Properties:**

- `to_pytree` converts Sheaf values (dict, list, tensor, scalar) to JAX pytrees
- `from_pytree` reconstructs Sheaf values from pytrees
- Invertible: `from_pytree(to_pytree(x)) == x`
- Compatible with `jax.tree_util` operations
- Rejects non-serializable types (functions, symbols)

---

## Version & Compatibility

- **Sheaf Version:** 0.9-RC
- **Python:** 3.8+
- **JAX:** Latest stable
- **File Extension:** `.shf`
- **Modeline:** `;; -*- mode: clojure -*-` (use Clojure syntax highlighting)

---

**End of AI Context Reference** • For full details, see `SPECS.md`
