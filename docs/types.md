# Sheaf Data Types

Sheaf uses tensors for numerical computation. A number written on its own is a
scalar. An array of numbers written in square brackets is a tensor.

The default tensor dtype is `f32`.

## Type Summary Table

| Value        | Example         | Notes                             |
| ------------ | --------------- | --------------------------------- |
| Scalar       | `42`, `3.14`    | Integer or floating-point value   |
| Tensor       | `[1 2 3]`       | Numeric array (`f32` by default)  |
| Typed tensor | `[1 2 3] :f16`  | Explicit dtype                    |
| List         | `'[1 2 3]`      | Quoted data, not a tensor         |
| Dictionary   | `{:x 1}`        | Keys and values (can be nested)   |
| Keyword      | `:weight`       | Commonly used as a dictionary key |
| Boolean      | `true`, `false` | Scalar truth values               |
| String       | `"hello"`       | Text                              |
| Nil          | `nil`           | Absence of a value                |

In Sheaf code, scalar and tensor literals are expressed like this:

```clojure
42                    ; integer scalar
3.14                  ; floating-point scalar
[1 2 3]               ; f32 tensor with shape [3]
[[1 2] [3 4]]         ; f32 tensor with shape [2 2]
```

## Dtypes and casts

A tensor’s dtype determines how its numbers are stored. Tensor literals use `f32`
by default. Both `f16` and `bf16` use half as much memory per number as `f32`,
but they make different trade-offs: `f16` keeps more precision, while `bf16` can
represent much larger and smaller values.

The dtype can be set by appending a keyword to a tensor literal:

```clojure
[1 2 3] :f32
[1 2 3] :f16
[1 2 3] :bf16
[1 2 3] :i32
[1 0] :bool
```

**Note**: `bf16` is not supported on Apple Metal.

`cast` changes the dtype of an existing tensor. It accepts `:f32`, `:f16`,
`:bf16`, and `:i32`:

```clojure
(cast (ones '[3]) :f16) ; => tensor f16[3] = [1. 1. 1.]

(let [x [1 2 3] :bf16]
  (cast x :f32))        ; => tensor f32[3] = [1. 2. 3.]
```

`int` and `float` convert scalars. They also convert tensors: a rank-zero tensor
becomes a scalar, while a tensor with more axes keeps its shape.

```clojure
(int 3.9)           ; => 3
(float 3)           ; => 3.0
(int [1.2 2.8])     ; => tensor i32[2] = [1 2]
```

Arithmetic with a scalar can adopt the dtype of a typed tensor.
However, if two tensors have different dtypes, Sheaf does not silently convert one to match the other but rejects the operation with a `dtype mismatch` error:

```clojure
(+ [1 2] :f16 3)       ; => tensor f16[2] = [4. 5.]
                       ; '3' was automatically cast to f16

(let [x [1 2] :f16     ; f16 tensor
      y [3 4]]         ; no dtype specified, so y is a f32 tensor
  (+ x y))             ; => error: dtype mismatch: f16 and f32
```

To add them, cast one tensor to the other's dtype:

```clojure
(let [x [1 2] :f16
      y [3 4]]
  (+ x (cast y :f16))) ; => tensor f16[2] = [4. 6.]
```

### Lists and tensor conversion

The quote `'` keeps a vector as a list instead of turning it into a tensor.
This is useful for shapes: `[2 3]` creates a tensor, while `'[2 3]` keeps the
dimensions as a list. The list can then be used to create or reshape a tensor:

```clojure
(zeros '[2 3])                      ; => tensor f32[2x3]
(reshape (arange 6) '[2 3])         ; => tensor i32[2x3]
```

`tensor` converts a list of numbers to an `f32` tensor, even when the list
contains integers:

```clojure
(tensor '[1 2 3])                   ; => tensor f32[3] = [1. 2. 3.]
```

## Brackets in syntax

Brackets do not always create a tensor. In a function definition they name the
parameters. In a `let` binding they can name the parts of a value. A quoted
vector stays a list, as in the shape examples above:

```clojure
(defn add [x y] (+ x y))  ; [x y] names the parameters
(let [[a b] [1 2]] a)     ; [a b] binds the two elements to a and b
'[3 4]                    ; a list, not a tensor
```

Brackets can also form a list without a quote. For example, `[:x :y]` contains
keywords rather than numbers, so it is a list. A list of numbers such as `[1 2]`
needs the quote to stay a list rather than become a tensor.

## Dictionaries

Dictionaries group related values together and associate each value with a name.

A model layer, for example, has weights and biases. Keeping both in one dictionary makes
the layer a single value, with each part accessible by name.
Those names are called _keys_, which can be keywords such as `:weight` or strings.

A dictionary is written with braces, and `get` reads a value by its key:

```clojure
; Defining a dictionary
(def point {:x 1 :y 2})
; Reading the value for ':y':
(get point :y)                     ; => 2
```

The `keys` function returns the keys as a list of strings, even when they were
written as keywords. `vals` returns a list of the values. In Sheaf source, those
two lists would be written `'["x" "y"]` and `'[1 2]`. The `=>` comments below show
how the REPL displays them, with commas between items and single quotes around
strings.

To create a different version of a dictionary, `assoc` adds or replaces a
value and `dissoc` removes keys. `dissoc` takes a list, so it can remove several
keys at once. Both return a new dictionary, leaving `point` unchanged:

```clojure
(keys point)                       ; => ['x', 'y']
(vals point)                       ; => [1, 2]
(assoc point :z 3)                 ; => {:x 1 :y 2 :z 3}
(dissoc point [:y])                ; => {:x 1} (point is unchanged)
(dissoc point [:x :y])             ; => {}
```

Dictionaries can also be passed to and returned from functions:

```clojure
(defn with-z [point]
  (assoc point :z 3))

(with-z point)                     ; => {:x 1 :y 2 :z 3}
```

Since data structures often have more than one layer, dictionaries can contain other dictionaries, which can hold tensors and lists.

`get-in` reads a value from a nested dictionary by following a path of keys:

```clojure
; A dictionary containing a dictionary of tensors:
(def params {:layer {:weight [[1 2]
                              [3 4]]
                     :bias [0.1 0.2]}})

(get params :layer)                ; => {:bias [0.1 0.2], :weight [[1. 2.] [3. 4.]]}
(get-in params [:layer :bias])     ; => tensor f32[2] = [0.1 0.2]
```

## Booleans and strings

`true` and `false` are scalar booleans. Element-wise comparisons return boolean
tensors, while `=` compares complete values:

```clojure
(> [1 2 3] 2)                     ; [false false true]
(== [1 2 1] 1)                    ; [true false true]
(= [1 2 1] 1)                     ; false
```

Strings are text values, not tensors. String operations run in the interpreter.
Strings are not tensor arguments to compiled functions.

For function signatures and more examples, see the [Function Reference](reference.md).
