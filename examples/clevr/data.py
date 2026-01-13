"""
CLEVR-style scene generation for neuro-symbolic reasoning.

Scenes contain colored shapes at 2D positions.
"""

import jax
import jax.numpy as jnp

# Vocabulary
COLORS = ["red", "green", "blue", "yellow"]
SHAPES = ["circle", "square", "triangle"]

# Scene dimensions
MAX_OBJECTS = 5
FEATURE_DIM = 9  # color(4) + shape(3) + position(2)


def generate_scene(key, n_objects=4):
    """
    Generate a random scene with n_objects.

    Returns dict with 'objects' list and 'n_objects' count.
    Each object has: shape, color, x, y (positions in [0.1, 0.9])
    """
    keys = jax.random.split(key, n_objects * 3)
    objects = []

    for i in range(n_objects):
        shape = SHAPES[int(jax.random.randint(keys[i * 3], (), 0, len(SHAPES)))]
        color = COLORS[int(jax.random.randint(keys[i * 3 + 1], (), 0, len(COLORS)))]
        x = float(jax.random.uniform(keys[i * 3 + 2], minval=0.1, maxval=0.9))
        y = float(jax.random.uniform(keys[i * 3 + 2], minval=0.1, maxval=0.9))
        objects.append({"shape": shape, "color": color, "x": x, "y": y})

    return {"objects": objects, "n_objects": n_objects}
