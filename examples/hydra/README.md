Hydra is a self-evolving model.

It starts training on the XOR problem without a hidden layer, which is impossible to solve.

After detecting a loss plateau, it dynamically inserts a hidden layer, enabling learning to resume and converge.

In PyTorch or JAX, doing this kind of structural mutation mid-training either triggers a recompilation or forces into eager mode, with a non-trivial performance hit.

In Sheaf, the model is a data structure (S-expression), we can evolve it live without stopping the JIT engine because nothing breaks the execution graph.
