# Hydra

A self-evolving neural network implemented in Sheaf.

## How it works

The network starts with a single linear head (no hidden layers), which cannot
solve XOR. Every 20 epochs, the training loop checks for a loss plateau
(progress < 0.003). When one is detected, `grow-hydra` appends a new hidden
layer.

## Zero recompilation at grow

`grow-hydra` appends a new layer and reinitialises the head. This changes the
parameter structure, but the JIT compiler caches compiled functions by content
hash. All matmul shapes that appear after a grow (`[4,32]`, `[32,1]`, etc.)
were already compiled during the initial training phase, so they hit the VMFB
cache. Growing the network requires zero recompilation.
