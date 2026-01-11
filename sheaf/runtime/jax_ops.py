# 2025 Damien Boureille | MIT License
# Part of the Sheaf Language - https://github.com/sheaf/sheaf

"""
Maps Sheaf primitive symbols to JAX/LAX numerical implementations.
Defines the foreign function interface (FFI) for tensor operations.
"""

import jax
import jax.numpy as jnp


def sheaf_append(lst, x):
    """
    Appends an element to a list or a JAX array.
    Used for accumulating generated tokens.
    """
    if isinstance(lst, list):
        return lst + [x]
    # Fallback for JAX arrays (note: this creates a new array)
    return jnp.append(lst, x)


def sheaf_append_and_roll(window, new_id):
    """
    Efficiently updates a rolling context window for autoregressive inference.
    Shifts the window to the left and adds the new ID at the end.
    """
    # Ensure new_id is an array to allow concatenation
    new_id_arr = jnp.atleast_1d(jnp.array(new_id, dtype=jnp.int32))
    # Concatenate the window (minus the first element) with the new ID
    return jnp.concatenate([window[1:], new_id_arr])


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


def sheaf_slice(x, start, length):
    """
    Wraps JAX dynamic_slice for simpler S-expression syntax.
    Enables efficient batch sampling directly on device.
    """
    # We assume slicing on the first dimension for 1D data (sequences)
    # or use dynamic_slice_in_dim for more flexibility
    return jax.lax.dynamic_slice_in_dim(x, start, length, axis=0)


def sheaf_transpose(tensor, *axes):
    if not axes:
        return jnp.transpose(tensor)
    # If the user provides axes, we use them as a permutation
    return jnp.transpose(tensor, axes=axes)


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
        "append": sheaf_append,
        "append-and-roll": sheaf_append_and_roll,
        "arange": jnp.arange,
        "choice": jax.random.choice,
        "einsum": jnp.einsum,
        "maximum": jnp.maximum,
        "minimum": jnp.minimum,
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
        "slice": sheaf_slice,
        "split": jax.random.split,
        "swapaxes": jnp.swapaxes,
        "tanh": jnp.tanh,
        "top_k": jax.lax.top_k,
        "transpose": sheaf_transpose,
        "tree-map": sheaf_tree_map,
        "tril": jnp.tril,
        "value-and-grad": lambda f: jax.value_and_grad(f),
        "var": jnp.var,
        "where": jnp.where,
        "zeros": jnp.zeros,
        # "reshape": lambda a, *shape: jnp.reshape(a, shape),
        # "transpose": lambda a, *axes: jnp.transpose(a, axes if axes else None),
        # "tree-map": jax.tree_util.tree_map,
    }
