"""
CLEVR-style scene generation for neuro-symbolic reasoning.

Scenes contain colored shapes at 2D positions.
"""

import jax.numpy as jnp

# Vocabulary
COLORS = ["red", "green", "blue", "yellow"]
SHAPES = ["circle", "square", "triangle"]

# Scene dimensions
MAX_OBJECTS = 5
FEATURE_DIM = 9  # color(4) + shape(3) + position(2)
