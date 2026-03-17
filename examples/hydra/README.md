# Hydra

A self-evolving neural network implemented in Sheaf.

## How it works

The network starts with a single linear head (no hidden layers), which cannot
solve XOR. Every 20 epochs, the training loop checks for a loss plateau
(progress < 0.003). When one is detected, `grow-hydra` appends a new hidden
layer.

In Sheaf, a model is a dictionary. Growing a layer is appending to a list.
Autodiff and the optimizer follow the updated structure on the next call.
