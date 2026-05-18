## CLEVR: Neuro-Symbolic Visual Reasoning

Traditional approaches in AI have complementary strengths and weaknesses.

Neural networks are good at perception and pattern recognition but opaque "black boxes" with sometimes unexplainable decisions.

Symbolic systems are good at logical reasoning and explicit rule-based inference but brittle when handling noisy or uncertain data.

Neuro-symbolic AI combines both approaches:

- Neural networks handle perception: extracting objects, colors, shapes and positions from images
- Symbolic logic handles reasoning: filtering, selection, composition of operations

### How this example works

Unlike a transformer that predicts answers based on statistical patterns, this model uses neural networks only for perception (identifying colors and shapes) while the query dictates which neural operations to call and in what order.

This model cannot hallucinate. A Transformer might guess an answer because it saw similar training data, but a neuro-symbolic model cannot answer "red" unless the query function extracts that specific logit from the selected object.

#### Process

Input (visual scene) -> Query (symbolic) -> Neural Network

1. The input contains random colored geometric objects at different positions, encoded as one-hots and positions in an array.

2. The query is a symbolic question expressed as an S-expression.
   It acts as a sequence of dynamic attention masks on the neural network.

Each symbolic operation (like "filter-shape") computes a mask that narrows the neural network's focus until only the relevant information remains in the "search space". See `utils.shf`for their implementation.

```
   ["query-color", ["leftmost", ["filter-shape", ":circle"]]]
```

3. The query assembles and runs a dynamic pipeline of neural modules to filter the focus:

   ```
    Input (Scene Tensor)
        |
   [Filter] "select circles"  -> (Dot product + embedding + Sigmoid)
        |
   [Select] "pick the leftmost" -> (Softmax + attention)
        |
   [Extract] "get its color"  -> (Extraction + Logits)
        |
    Answer (Argmax)
   ```

Note: CLEVR is fully differentiable. Queries can be trained with `train.shf`.

### Sheaf’s Role in the Architecture

In PyTorch or JAX, the symbolic query is one data structure, the neural modules are Python objects, and a dispatcher sits between them to translate one into the other.

In Sheaf, code and data are the same. For this reason, a symbolic query such as `["query-color" ["leftmost" ["filter-shape" ":circle"]]]` is a list, and that list also directly is the neural pipeline.

Since the query vocabulary and the function vocabulary are one and the same, adding a new operation is just a matter of defining a function, which automatically extends the query language.

The computation graph is also represented as data in the source language. This allows macros to generate and modify neural architectures at compile time. For example, the `defspatial` macro in `utils.shf` creates four spatial-attention functions from a single template, which can be traced and inspected.

Also specific to Sheaf: the entire neuro-symbolic pipeline (query execution, soft attention, and embedding lookup) is differentiable with a single call to `value-and-grad`. Homoiconicity makes the function the graph, and differentiating the function differentiates the graph.

### Visualizer

GUI: Start the training and query dashboard with `viz/run.sh`.

CLI:

```bash
sheaf train.shf # Optional, trained weights are already included.
sheaf run.shf
```
