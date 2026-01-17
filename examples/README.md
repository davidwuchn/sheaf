## Examples

This directory contains four reference implementations in Sheaf.

### BareGPT

BareGPT is a small generative Transformer trained on a corpus of Shakespeare’s works. When started, it will generate text character-by-character.
BareGPT ships with pre-trained weights but can be re-trained with any training material.

### Clevr

Clevr is a neuro-symbolic model. It uses a neural network to understand and recognize shapes in a synthetic scene. Symbolic rules then guide its attention mechanism to answer queries.

### Hydra

Hydra is a self-evolving model. It starts training on the XOR problem without a hidden layer, making it linearly inseparable.
After detecting a loss plateau, it dynamically inserts a hidden layer at runtime without interrupting the training loop, enabling learning to resume and converge.

### XOR MLP

The "Hello World" of Sheaf. A tiny Multi-Layer Perceptron to solve the XOR non-linear problem.
