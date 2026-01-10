# 2025 Damien Boureille | MIT License
# Part of the Sheaf Language - https://github.com/sheaf/sheaf

"""
Maps Sheaf primitive symbols to JAX/LAX numerical implementations.
Defines the foreign function interface (FFI) for tensor operations.
"""

import jax
import jax.numpy as jnp


def sheaf_transpose(tensor, *axes):
    if not axes:
        return jnp.transpose(tensor)
    # If the user provides axes, we use them as a permutation
    return jnp.transpose(tensor, axes=axes)


def sheaf_reshape(a, *shape_args):
    # Reshape tensor by flattening into a single dimension
    flat_shape = []
    for item in shape_args:
        if isinstance(item, (tuple, list)):
            flat_shape.extend(item)
        else:
            flat_shape.append(item)
    return jnp.reshape(a, tuple(flat_shape))


def sheaf_shape(tensor, axis=None):
    """
    Handles JAX arrays
    """
    try:
        s = tensor.shape
        if axis is not None:
            # axis can be negative (like -1), so we check against range
            return s[axis]
        return s
    except (AttributeError, IndexError, TypeError) as e:
        # If it's not a tensor or axis is wrong, we provide context
        if not hasattr(tensor, "shape"):
            raise TypeError(
                f"Object has no 'shape' attribute. Type: {type(tensor).__name__}"
            )
        raise IndexError(
            f"Dimension index {axis} is out of range for shape {tensor.shape}"
        )


def sheaf_tree_map(f, *trees):
    def safe_f(*args):
        # Check if any argument passed to the lambda is a module
        import types

        for i, arg in enumerate(args):
            if isinstance(arg, types.ModuleType):
                raise TypeError(
                    f"Leaf in tree-map at position {i} is a module! Type: {type(arg)}"
                )
        return f(*args)

    return jax.tree_util.tree_map(safe_f, *trees)


def get_jax_env():
    return {
        "arange": jnp.arange,
        "choice": jax.random.choice,
        "einsum": jnp.einsum,
        "minimum": jnp.minimum,
        "maximum": jnp.maximum,
        "ndim": lambda x: x.ndim,
        "normalize": lambda x: x / (jnp.sum(x, axis=-1, keepdims=True) + 1e-12),
        "one-hot": jax.nn.one_hot,
        "ones": jnp.ones,
        "product": jnp.prod,
        "random-normal": jax.random.normal,
        "random-uniform": jax.random.uniform,
        "range": lambda *args: jnp.arange(*args),
        "reshape": sheaf_reshape,
        "roll": jnp.roll,
        "shape": sheaf_shape,
        "split": jax.random.split,
        "swapaxes": jnp.swapaxes,
        "tanh": jnp.tanh,
        "top_k": jax.lax.top_k,
        "transpose": sheaf_transpose,
        # "tree-map": jax.tree_util.tree_map,
        "tree-map": sheaf_tree_map,
        "tril": jnp.tril,
        "value-and-grad": lambda f: jax.value_and_grad(f),
        "var": jnp.var,
        "where": jnp.where,
        "zeros": jnp.zeros,
        # "reshape": lambda a, *shape: jnp.reshape(a, shape),
        # "transpose": lambda a, *axes: jnp.transpose(a, axes if axes else None),
    }
