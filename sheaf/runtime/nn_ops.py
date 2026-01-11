# 2025 Damien Boureille | MIT License
# Part of the Sheaf Language - https://github.com/sheaf/sheaf

"""
Exposes high-level neural primitives and loss functions.
Integrates JAX neural network functional API.
"""

import jax
import jax.nn.initializers as init
import jax.numpy as jnp


def sparse_cross_entropy(logits, targets):
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    targets_expanded = jnp.expand_dims(targets, axis=-1)
    target_log_probs = jnp.take_along_axis(log_probs, targets_expanded, axis=-1)
    return -jnp.mean(target_log_probs)


def get_nn_env():
    return {
        "celu": jax.nn.celu,  # alpha (default 1.0)
        "gelu": jax.nn.gelu,
        "init-kaiming-normal": init.kaiming_normal(),
        "init-kaiming-uniform": init.kaiming_uniform(),
        "init-lecun-normal": init.lecun_normal(),
        "init-lecun-uniform": init.lecun_uniform(),
        "init-ones": init.ones,
        "init-orthogonal": init.orthogonal(),
        "init-xavier-normal": init.xavier_normal(),
        "init-xavier-uniform": init.xavier_uniform(),
        "init-zeros": init.zeros,
        "leaky-relu": jax.nn.leaky_relu,  # alpha (default 0.01)
        "log-softmax": jax.nn.log_softmax,
        "relu": jax.nn.relu,
        "selu": jax.nn.selu,  # Scaled Exponential Linear Unit
        "sigmoid": jax.nn.sigmoid,
        "silu": jax.nn.silu,  # Swish function (x * sigmoid(x))
        "softmax": jax.nn.softmax,
        "sparse-cross-entropy": sparse_cross_entropy,
        "value-and-grad": jax.value_and_grad,
    }
