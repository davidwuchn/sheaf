Hydra is a self-evolving model.

It starts training on the XOR problem without a hidden layer, which is impossible to solve.
After detecting a loss plateau, it dynamically inserts a hidden layer, enabling learning to resume and converge.

In PyTorch or JAX, mutating a model during training would force recompilation or requires interpreted "eager execution", sacrificing performance.

In Sheaf, the model is a data structure (S-expression), we can evolve it live without stopping the JIT engine because nothing breaks the execution graph.
