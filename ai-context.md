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
(defn forward [x p]
  (as-> x h
    (with-params [p :l1]    ;; Auto-bind W and b from layer 1
      (relu (+ (@ h W) b)))
    (with-params [p :l2]    ;; Auto-bind W and b from layer 2
      (sigmoid (+ (@ h W) b)))))

;; Training step with Adam optimizer
(defn train-step [p m v t x y lr]
  (let [loss-fn (fn [params] (mse-loss params x y))
        [loss grads] ((value-and-grad loss-fn) p)
        [new-p new-m new-v new-t] (adam-step p grads m v t lr 0.9 0.999 1e-8)]
    {:p new-p :m new-m :v new-v :t new-t :loss loss}))
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
{:l1 {:W weights :b biases}
 :l2 {:W weights :b biases}}
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
    (with-params [p :l1] (relu (+ (@ _ W) b)))
    (with-params [p :l2] (softmax (+ (@ _ W) b)))))
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
[1 2 3]          ; Vector literal (JAX array when numeric, tuple otherwise)
[D D]            ; Shape literal (evaluates to tuple when symbols present)
[[1 2] [3 4]]    ; Nested vectors (2D JAX array)
{:a 1 :b 2}      ; Dictionary (like Clojure/Python)
{"key" value}    ; Dictionary with string keys
:keyword         ; Keyword (evaluates to string "keyword")
true / false     ; Booleans (lowercase)
nil              ; None
...              ; Ellipsis (for indexing/einsum)
```

**Bracket contexts:**

- `[]` in **binding context** (first arg to defn, let, fn): destructuring pattern
- `[]` in **expression context** (function arguments, return values): data literal

### Core Operators

**Function Definition**

```sheaf
(defn name [args] body)              ; Standard function
(defn :jit name [args] body)         ; JIT-compiled (faster, limited control flow)
(fn [args] body)                     ; Anonymous function (preferred)
(lambda [args] body)                 ; Anonymous function (legacy alias for fn)
```

**Binding & Scope**

```sheaf
(let [x val y val2] body)            ; Sequential local bindings
(with-params params body)            ; Auto-destructure dict (:W, :b, etc.)
```

**Control Flow**

```sheaf
(if cond then else)                  ; Branching (avoid in JIT functions)
(case expr clause1 clause2 ...)      ; Pattern matching (each clause: (pattern result))
(where cond true-val false-val)      ; Differentiable select (use in JIT)
(repeat [i n] [acc init] body)       ; Loop with accumulator
(scan fn init xs)                    ; Fold with intermediate results (like Haskell scanl)
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
(apply func args-list)               ; Apply function to list of arguments
(vmap func)                          ; Vectorized map (JAX vmap) - auto-batch over axis 0
(vmap func axis)                     ; Vectorized map over specified axis
(vmap func [0 nil])                  ; vmap first arg on axis 0, keep second fixed
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
(defn forward [x p]
  (as-> x h
    (with-params [p :l1]
      (relu (+ (@ h W) b)))
    (with-params [get p :l2]
      (sigmoid (+ (@ h W) b)))))
```

### Pattern 2: Transformer Block (from BareGPT)

```sheaf
(defn transformer-block [x layer-p config]
  (let [;; Self-Attention + Residual 1
        ln1_x (layer-norm x (get layer-p :ln1) 2)
        attn_out (first (multi-head-attention ln1_x layer-p config))
        x1 (+ x attn_out)

        ;; MLP + Residual 2
        ln2_x1 (layer-norm x1 (get layer-p :ln2) 2)
        mlp_out (mlp ln2_x1 (get layer-p :mlp))
        x2 (+ x1 mlp_out)]
    x2))
```

### Pattern 3: Training Loop with Adam

```sheaf
(defn train-step [params m v t inputs targets config]
  (let [lr (get config :lr)
        loss-fn (fn [p] (cross-entropy-loss (model inputs p config) targets))
        [loss grads] ((value-and-grad loss-fn) params)
        [new-params new-m new-v new-t] (adam-step params grads m v t lr 0.9 0.999 1e-8)]
    {:loss loss :params new-params :m new-m :v new-v :t new-t}))
```

### Pattern 4: Einsum for Multi-Head Attention

```sheaf
;; Q, K, V projections [Batch, Heads, Time, Head_dim]
(let [Qh (einsum "... t d, d h k -> ... h t k" X Wq_multi)
      Kh (einsum "... t d, d h k -> ... h t k" X Kh_multi)
      Vh (einsum "... t d, d h k -> ... h t k" X Vh_multi)]
  ...)
```

### Pattern 5: Batching with vmap

```sheaf
;; Define single-sample forward pass
(defn forward-single [x params]
  (with-params [params]
    (sigmoid (+ (@ x W) b))))

;; Batch it automatically with vmap
(let [batch-forward (vmap forward-single [0 nil])  ; vmap over axis 0 of inputs, keep params fixed
      batch-results (batch-forward X params)]
  batch-results)

;; Or use the defbatch macro (from lib/macros.shf) for cleaner syntax:
(defbatch linear-layer [x w b] [0 nil nil]
  (+ (@ x w) b))
; Now (linear-layer batch-x W b) automatically batches over first axis of x
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

### ⚠️ Shape Inference with Static & Reshape

```sheaf
(defn :jit model [x config]
  (let [D (static (get config :d_model))]  ; Force static evaluation
    (reshape x '(-1 D))))  ; -1 infers dimension automatically (like NumPy)

;; Examples:
(reshape (arange 12) '(3 -1))    ; => shape (3 4) - infers 4
(reshape (arange 12) '(2 -1 2))  ; => shape (2 3 2) - infers 3
```

### ⚠️ Dictionary Access

```sheaf
(get dict :key)                      ; Single key access
(get dict :missing 99)               ; With default value
(get-in dict [:path :to :key])       ; Nested access
(get-in dict [:path :missing] 99)    ; Nested with default
(assoc dict :k1 v1 :k2 v2)          ; Add/update keys (functional)
(merge dict1 dict2)                  ; Merge dicts (later overrides)
(keys dict)                          ; Get all keys as list
(vals dict)                          ; Get all values as list
(with-params [dict] body)            ; Auto-bind :W, :b, etc. as variables
(with-params [dict :key] body)       ; Shorthand for (get dict :key)
```

**Dictionary manipulation examples:**

```sheaf
;; Functional dict update for multi-task learning
(let [model {:head old-head :layers layers}
      heads {:task1 head1 :task2 head2}]
  (assoc model :head (get heads :task1)))

;; Merge config with defaults
(merge {:lr 0.001 :epochs 100} user-config)

;; Iterate over dict
(let [params {:W w :b b}]
  (map (fn [k] (str k)) (keys params)))  ; => ["W" "b"]
```

---

## Quick Reference Tables

### Math Operations

| Operator                          | Description                                | Example                                                        |
| --------------------------------- | ------------------------------------------ | -------------------------------------------------------------- |
| `+`, `-`, `*`, `/`                | Arithmetic (variadic, broadcastable)       | `(+ a b c)`                                                    |
| `//`                              | Integer division                           | `(// 7 2)` → `3`                                               |
| `@`                               | Matrix multiplication                      | `(@ W x)`                                                      |
| `**`                              | Exponentiation                             | `(** x 2)`                                                     |
| `(einsum pattern ...tensors)`     | Einstein summation                         | `(einsum "ij,jk->ik" A B)`                                     |
| `(sum t :axis i [:keepdims])`     | Reduction (keep dims as 1 if :keepdims)    | `(sum logits :axis -1)` or `(sum logits :axis 1 :keepdims)`    |
| `(product t :axis i [:keepdims])` | Product reduction (keep dims if :keepdims) | `(product weights :axis 0)` or `(product x :axis 0 :keepdims)` |
| `(mean t :axis i [:keepdims])`    | Mean (keep dims as 1 if :keepdims)         | `(mean loss)` or `(mean x :axis 1 :keepdims)`                  |
| `(var t :axis i [:keepdims])`     | Variance (keep dims if :keepdims)          | `(var x :axis -1)` or `(var x :axis 1 :keepdims)`              |
| `(min t :axis i [:keepdims])`     | Minimum (keep dims if :keepdims)           | `(min x :axis 0)` or `(min x :axis 0 :keepdims)`               |
| `(max t :axis i [:keepdims])`     | Maximum (keep dims if :keepdims)           | `(max x :axis 0)` or `(max x :axis 0 :keepdims)`               |
| `(minimum a b)`                   | Element-wise minimum                       | `(minimum x y)`                                                |
| `(maximum a b)`                   | Element-wise maximum                       | `(maximum x y)`                                                |
| `(abs x)`                         | Absolute value                             | `(abs x)`                                                      |
| `(exp x)`                         | Exponential                                | `(exp x)`                                                      |
| `(log x)`                         | Natural logarithm                          | `(log x)`                                                      |
| `(sqrt x)`                        | Square root                                | `(sqrt x)`                                                     |

### Tensor Shaping

| Function                         | Description              | Example                        |
| -------------------------------- | ------------------------ | ------------------------------ |
| `(shape t)`                      | Get shape tuple          | `(shape x)` → `[B, T, D]`      |
| `(shape t axis)`                 | Get dimension            | `(shape x -1)` → `D`           |
| `(ndim t)`                       | Number of dimensions     | `(ndim x)` → `3`               |
| `(reshape t ...dims)`            | Reshape tensor           | `(reshape x '(-1 D))`          |
| `(transpose t ...axes)`          | Permute axes             | `(transpose x '(1 0 2))`       |
| `(swapaxes t a1 a2)`             | Swap two axes            | `(swapaxes x -1 -2)`           |
| `(concat ...seq [:axis i])`      | Concatenate lists/arrays | `(concat '[1] '[2] :axis 0)`   |
| `(tensor-split t n [axis])`      | Split into n parts       | `(tensor-split x 3)` → `[...]` |
| `(slice t start end)`            | Slice along first axis   | `(slice x 0 10)`               |
| `(dynamic-slice t starts sizes)` | Dynamic slice            | `(dynamic-slice x [i] [n])`    |
| `(roll t shift :axis i)`         | Roll elements along axis | `(roll x 1 :axis 0)`           |
| `(tril m [k])`                   | Lower triangular matrix  | `(tril x)` or `(tril x -1)`    |

### Tensor Creation

| Function                | Description      | Example                           |
| ----------------------- | ---------------- | --------------------------------- |
| `(zeros shape)`         | Tensor of zeros  | `(zeros '(3 4))` → shape `[3, 4]` |
| `(ones shape)`          | Tensor of ones   | `(ones '(D D))` → shape `[D, D]`  |
| `(random-normal k s)`   | Normal samples   | `(random-normal key '(64 128))`   |
| `(xavier-init k s)`     | Xavier init      | `(xavier-init key '(D D))`        |
| `(arange n)`            | 0 to n-1         | `(arange 5)` → `[0 1 2 3 4]`      |
| `(range n)`             | Alias for arange | `(range 5)` → `[0 1 2 3 4]`       |
| `(one-hot idx n)`       | One-hot encoding | `(one-hot 2 5)` → `[0 0 1 0 0]`   |
| `(normalize x :axis i)` | L2 normalization | `(normalize x :axis -1)`          |

**Shape syntax:**

- **Quoted shapes** `'[...]` for static/literal shapes (no variables):

  ```sheaf
  (zeros '[3 4])       ; Static shape (3, 4) - quote prevents JAX array creation
  (ones '[1])          ; Single dimension - must quote to get tuple, not array
  ```

- **Unquoted shapes** `[...]` for dynamic shapes with variables:
  ```sheaf
  (let [D 128 H 8]
    (ones [D H]))      ; Variables evaluated → shape (128, 8)
  ```

**Why quote?** Without quote, `[1]` becomes a JAX array. With quote, `'[1]` becomes a Python tuple `(1,)` - which is what shape functions expect.

### Activations

| Function                  | Description                      |
| ------------------------- | -------------------------------- |
| `(relu x)`                | ReLU activation                  |
| `(leaky-relu x)`          | Leaky ReLU (slope 0.01)          |
| `(gelu x)`                | GELU (used in GPT)               |
| `(selu x)`                | Scaled ELU                       |
| `(celu x)`                | Continuous ELU                   |
| `(sigmoid x)`             | Sigmoid (0-1 range)              |
| `(tanh x)`                | Hyperbolic tangent               |
| `(softmax x :axis i)`     | Softmax normalization            |
| `(log-softmax x :axis i)` | Log-softmax (numerically stable) |
| `(silu x)`                | Swish / SiLU                     |

### List/Vector Operations

| Function                   | Description                      | Example                               |
| -------------------------- | -------------------------------- | ------------------------------------- |
| `[1 2 3]`                  | Vector literal                   | `[1 2 3]` → JAX array or tuple        |
| `(cons head tail)`         | Prepend element to list          | `(cons 1 [2 3])` → `[1 2 3]`          |
| `(append coll x)`          | Append element to list           | `(append [1 2] 3)` → `[1 2 3]`        |
| `(append-and-roll coll x)` | Append and remove first (FIFO)   | `(append-and-roll [1 2] 3)` → `[2 3]` |
| `(first coll)`             | Get first element (nil if empty) | `(first [1 2])` → `1`                 |
| `(second coll)`            | Get second element               | `(second [1 2 3])` → `2`              |
| `(last coll)`              | Get last element                 | `(last [1 2 3])` → `3`                |
| `(nth coll n)`             | Get nth element (0-indexed)      | `(nth [1 2 3] 1)` → `2`               |
| `(rest coll)`              | All except first ([] if empty)   | `(rest [1 2])` → `[2]`                |
| `(len coll)`               | Number of elements               | `(len [1 2])` → `2`                   |
| `(count coll)`             | Alias for len                    | `(count [1 2])` → `2`                 |
| `(empty? coll)`            | Check if empty                   | `(empty? [])` → `true`                |

### Symbol Manipulation

| Function           | Description               | Example                    |
| ------------------ | ------------------------- | -------------------------- |
| `(symbol? obj)`    | Check if object is symbol | `(symbol? 'foo)` → `True`  |
| `(gensym prefix?)` | Generate unique symbol    | `(gensym)` → `"G__abc123"` |

### PyTree Operations

Pytrees are nested structures (dicts, lists) that JAX and Sheaf use for parameter storage.

**Functions:**

| Function                | Description                   | Example                   |
| ----------------------- | ----------------------------- | ------------------------- |
| `(tree-map fn tree)`    | Apply function to all leaves  | `(tree-map relu params)`  |
| `(tree-map-zeros tree)` | Replace all leaves with zeros | `(tree-map-zeros params)` |

**Example - Transform nested parameters:**

```sheaf
;; Square all elements in a nested structure
(let [params {:layer1 {:w [2.0 4.0] :b 0.5}
              :layer2 {:w [10.0]}}]
  (tree-map (fn [x] (* x x)) params))
; => {:layer1 {:w [4.0 16.0], :b 0.25}, :layer2 {:w [100.0]}}

;; Scale all gradients by learning rate
(tree-map (fn [g] (* g -0.001)) gradients)

;; Initialize optimizer state (zeros everywhere)
(let [state (tree-map-zeros params)]
  state)  ; Same structure as params, all zeros
```

### Utilities

| Function                     | Description                  | Example               |
| ---------------------------- | ---------------------------- | --------------------- |
| `(top_k x k)`                | Top-k values and indices     | `(top_k scores 5)`    |
| `(probe label x)`            | Print value during execution | `(probe "debug" x)`   |
| `(str x)`                    | Convert to string            | `(str 42)` → `"42"`   |
| `(str-call fn-name ...args)` | Call string-named function   | `(str-call "relu" x)` |

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
(let [[k1 k2 k3] (random-split (random-key 42) 3)]
  (random-normal k1 [10 10]))
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
(layer-norm x p axis)                    ; Layer normalization
(linear x w b)                           ; Dense layer: x @ w + b
(cross-entropy-loss labels logits)       ; Cross-entropy (one-hot labels)
(sparse-cross-entropy labels logits)     ; Cross-entropy (integer labels)
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
sheaf> (defn double [x] (* x 2))
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
├── __init__.py           # Main Sheaf class, to_pytree/from_pytree
├── __main__.py           # CLI entry point
├── core/
│   ├── compiler.py       # S-expr → JAX compiler, HashableDict, special forms dispatch
│   ├── parser.py         # Lisp tokenizer & parser, SheafList with metadata
│   ├── macro_engine.py   # Macro expansion, quote/quasiquote, compile-time eval
│   ├── special_forms.py  # Special forms: defn, let, if, defmacro, etc.
│   ├── tracer.py         # Execution tracing, guards (:no-nan, :range, :shape)
│   └── error_handler.py  # Error reporting & Emergency Backtrace
├── runtime/
│   ├── core_ops.py       # defn, let, if, get, with-params, etc.
│   ├── jax_ops.py        # einsum, reshape, transpose, swapaxes, tensor-split, etc.
│   ├── math_ops.py       # +, -, *, /, @, **, sum, mean, etc.
│   ├── nn_ops.py         # relu, gelu, sigmoid, tanh, softmax, silu, layer-norm, etc.
│   └── string_ops.py     # str, concat, symbol?, gensym
├── repl/
│   ├── __init__.py       # REPL session management
│   ├── __main__.py       # REPL entry point (python -m sheaf.repl)
│   └── help.py           # Interactive help & documentation
└── lib/
    ├── macros.shf        # Standard macros: when, unless, comment
    ├── nn.shf            # Neural network stdlib: layer-norm, linear, cross-entropy-loss
    ├── optim.shf         # Optimizers: sgd-step, adam-step, gradient clipping
    └── repl.shf          # REPL-specific helpers
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
(let [W (get params :W)
      b (get params :b)]
  (+ (@ x W) b))

; Write (legacy syntax):
(with-params params
  (+ (@ x W) b))

; Or with brackets (recommended):
(with-params [params]
  (+ (@ x W) b))

; Or with key shorthand (most elegant):
(with-params [params :l1]
  (+ (@ x W) b))

; With complex expression (compute params on the fly):
(with-params [(get layer-params layer-id)]
  (+ (@ x W) b))

; Or with transformation:
(with-params [(tree-map (fn [x] (* x scale)) (get params :l1))]
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
