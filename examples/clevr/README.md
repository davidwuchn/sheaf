# CLEVR: Neuro-Symbolic Visual Reasoning

## What is Neuro-Symbolic AI?

Traditional approaches in AI have complementary strengths and weaknesses:

- **Neural networks**: Good at perception and pattern recognition but opaque "black boxes" with unexplainable decisions
- **Symbolic systems**: Good at logical reasoning and explicit rule-based inference but brittle when handling noisy or uncertain data

Neuro-symbolic AI combines both approaches:

- Neural networks handle perception: extracting objects, colors, shapes and positions from images
- Symbolic logic handles reasoning: filtering, selection, composition of operations

## How this example works

Unlike a transformer that predicts answers based on statistical patterns, this model uses neural networks only for perception (identifying colors and shapes) while the query dictates which neural operations to call and in what order.

This model cannot hallucinate. A Transformer might guess an answer because it saw similar training data, but a neuro-symbolic model cannot answer "red" unless the query function extracts that specific logit from the selected object.

### Process

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
        ↓
   [Filter] "select circles"  -> (Dot product + embedding + Sigmoid)
        ↓
   [Select] "pick the leftmost" -> (Softmax + attention)
        ↓
   [Extract] "get its color"  -> (Extraction + Logits)
        ↓
    Answer (Argmax)
   ```

## What does Sheaf have to do with this?

Sheaf queries are data structures (S-expressions) that map 1:1 to the model's call stack. Sheaf can then transform a symbolic tree into a JAX computational graph through simple recursion, without the overhead of an intermediate parser or the limitations of Python’s fixed syntax.

## Files

- `data.py`: Generates random scenes (4 colors, 3 shapes, up to 5 objects) and questions (filter, select, spatial relations, attribute queries)
- `model.shf`: Core model with soft filters, selections, spatial relations, and attribute extraction
- `utils.shf`: Differentiable operations for filtering, selection, intersection, existence checking
- `train.py`: Training loop (100 epochs, Adam optimizer, cross-entropy loss)
- `run.py`: Evaluation on 10 random test cases
- `dashboard/app.py`: Interactive visualization
