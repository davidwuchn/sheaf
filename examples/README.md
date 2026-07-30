## Examples

This directory contains five reference implementations in Sheaf.

### Clevr

Clevr is a neuro-symbolic model. It uses a neural network to understand and
recognize shapes in a synthetic scene. Symbolic rules then guide its attention
mechanism to answer queries.

### Hydra

Hydra starts training an underpowered model (one that cannot solve the problem)
then grows a new layer at runtime when it detects a loss plateau, resuming
training without restarting.

In JAX or PyTorch, this requires retracing the computation graph and
reinitializing the optimizer. Because Sheaf is a homoiconic language, the model
is data itself: changing its shape mid-loop is no different from updating a
dictionary.

### Macros

Use macros to generate two models from the same layer specification. This shows
Sheaf's homoiconicity: macros can transform Sheaf code into new Sheaf programs.

### NanoGPT

A port of [Karpathy's nanoGPT](https://github.com/karpathy/nanoGPT) in Sheaf.
The architecture follows the original faithfully, keeping the same layer
structure, weight layout, and numerical results.

### XOR MLP

The "Hello World" of Sheaf. A tiny Multi-Layer Perceptron to solve the XOR
non-linear problem.
