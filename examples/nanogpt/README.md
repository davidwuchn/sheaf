# NanoGPT

A char-level GPT trained on Shakespeare, after Karpathy's
[nanoGPT](https://github.com/karpathy/nanoGPT), using the same 10.65M-parameter
model and weights.

This port explores how Transformer training looks when expressed in a language
built around pure functions, program transforms, and explicit state.

## Quick start

The examples archive published with Sheaf releases includes pretrained
weights. Run inference directly from the extracted NanoGPT directory:

```bash
sheaf sample.shf
```

To train the model from scratch and run inference with the resulting checkpoint:

```bash
sheaf train.shf
sheaf sample.shf
```

From a source checkout, Bazel downloads the pretrained weights from
[Hugging Face](https://huggingface.co/dbrll/sheaf-nanogpt-shakespeare) and runs
inference automatically:

```bash
bazel run //examples/nanogpt
```

## Training model: implicit vs explicit

The main difference between Sheaf and PyTorch is where the state lives.

In PyTorch, training state is attached to objects. Parameters live inside the
model, while gradients are stored on parameter tensors and optimizer state is
held by the optimizer.

```python
def train_step(model, optimizer, X, Y, lr, betas, eps, weight_decay, grad_clip):
    logits, loss = model(X, Y)                                   # forward
    loss.backward()                                              # backward, accumulates into .grad

    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()                                             # reads .grad, mutates params
    optimizer.zero_grad(set_to_none=True)                        # clear .grad buffers
# the Adam moments (m, v) and step count (t) live inside optimizer.state
```

In Sheaf, the same computation is expressed as a transformation of explicit
values:

```clojure
(defn train-step [state x y lr config]
  (let [loss-fn (fn [p]
                  (cross-entropy-loss (gpt-forward x p config) y))

        [loss g] ((value-and-grad loss-fn)
                  (get state :params))

        [p m v t] (adamw-step (get state :params) g
                              (get state :m)
                              (get state :v)
                              (get state :t)
                              lr 0.9 0.95 1e-8 0.1)]
    (assoc state
           :params p
           :m m
           :v v
           :t t
           :loss loss)))
```

A training step is therefore a pure function: parameters, gradients, and
optimizer statistics are ordinary values. Nothing is attached to a model object,
and no mutation occurs between steps.

### Autodiff

In Sheaf, `value-and-grad` is a _program transform_. It takes a function and
returns a transformed function that computes both the function value and its
gradient.

Unlike PyTorch, there is no equivalent of `zero_grad()`: gradients are not
accumulated in mutable buffers. They are values returned by the transformed
program.

### Parameters and state

`gpt-forward` is a pure function of `(idx params config)`. Parameters are a
nested dict, not an object with registered submodules. The optimizer state (`m`,
`v`, `t`) is another dict, threaded explicitly through each step.

A stack of Transformer blocks is just a list processed with reduce:

```clojure
(reduce (fn [h bp] (transformer-block h bp config)) x blocks)
```

## Sheaf-isms in the code

A few idioms recur throughout `model.shf`, for readers coming from Python:

- `(@ x W)` is matmul, like Python's `x @ W`.
- `(tr W)` transposes. HuggingFace stores `Linear` weights as `[out, in]`, so we
  transpose for the matmul: `(@ x (tr W))` reads as `x @ W.T`.
- `(with-params [p :key] body)` binds each field of a param sub-dict as a local
  (`weight`, `b`), the Sheaf equivalent of reaching into `params["key"]`.
- Shapes with symbols are shape tuples: `(zeros [n])`, `(ones [T T])`. Quote
  them only when fully static literals: `'[3 4]`.
