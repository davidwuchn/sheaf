## Examples

This directory contains four reference implementations showcasing Sheaf's syntax.

### BareGPT

BareGPT is a small generative Transformer that is trained on a corpus of Shakespeare’s works, although it can be trained on any textual data. It then generates text character-by-character.

### Clevr

Clevr is a neuro-symbolic model. It uses a neural network to understand and recognize shapes in a synthetic scene. Symbolic rules then shape its attention mechanism to answer a request.

### Hydra

Hydra is a self-evolving model.

It starts training on the XOR problem without a hidden layer, which is linearly inseparable.
After detecting a loss plateau, it dynamically inserts a hidden layer at runtime without interrupting the training loop, enabling learning to resume and converge.

### XOR MLP

The "Hello World" of Sheaf. A tiny Multi-Layer Perceptron to solve the XOR non-linear problem.
