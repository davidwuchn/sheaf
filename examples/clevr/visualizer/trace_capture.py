"""
Trace Capture System for CLEVR Debugger

Captures intermediate values during query execution for visualization.
This is a specialized tracer for the dashboard, separate from the core trace system.
"""

import jax
import jax.numpy as jnp


class AttentionCapture:
    """
    Capture attention masks and intermediate values during query execution.

    Usage:
        capture = AttentionCapture()
        with capture:
            result = execute_query(scene, query, params)

        # Access captured values
        for step in capture.steps:
            print(step['operation'], step['attention'])
    """

    def __init__(self):
        self.steps = []
        self.active = False

    def __enter__(self):
        self.active = True
        self.steps = []
        return self

    def __exit__(self, *args):
        self.active = False

    def capture(self, operation, attention=None, value=None, metadata=None):
        """
        Capture an intermediate step.

        Args:
            operation: Name of the operation (e.g., "filter-color", "leftmost")
            attention: Attention weights [N_objects] or [Batch, N_objects]
            value: Output value of this operation
            metadata: Additional info (dict)
        """
        if not self.active:
            return

        step = {
            "operation": operation,
            "attention": attention,
            "value": value,
            "metadata": metadata or {},
        }

        self.steps.append(step)

    def get_step(self, index):
        """Get a specific step by index."""
        if 0 <= index < len(self.steps):
            return self.steps[index]
        return None

    def get_by_operation(self, operation_name):
        """Get all steps with a specific operation name."""
        return [s for s in self.steps if s["operation"] == operation_name]

    def clear(self):
        """Clear all captured steps."""
        self.steps = []


# Global capture instance
_global_capture = AttentionCapture()


def get_capture():
    """Get the global capture instance."""
    return _global_capture


def capture_step(operation, attention=None, value=None, **metadata):
    """
    Convenience function to capture a step.

    Example:
        from trace_capture import capture_step

        attention_weights = soft_filter(scene, query_vec)
        capture_step("filter-color", attention=attention_weights, color="red")
    """
    _global_capture.capture(operation, attention, value, metadata)


# Wrapper functions for visual operations that capture attention
def captured_soft_filter(scene, query_vec, temperature, operation_name="filter"):
    """Soft filter with attention capture."""
    from sheaf.runtime.visual_ops import soft_filter

    filtered, attention = soft_filter(scene, query_vec, temperature)

    # Capture attention weights
    capture_step(
        operation_name, attention=attention, value=filtered, temperature=temperature
    )

    return filtered, attention


def captured_soft_unique(scene, temperature, operation_name="unique"):
    """Soft unique with attention capture."""
    from sheaf.runtime.visual_ops import soft_unique

    # Calculate objectness scores before selection
    objectness = jnp.sum(jnp.abs(scene), axis=-1)  # [B, N]
    selection_weights = jax.nn.softmax(objectness / temperature, axis=-1)

    selected = soft_unique(scene, temperature)

    # Capture selection weights as "attention"
    capture_step(
        operation_name,
        attention=selection_weights,
        value=selected,
        temperature=temperature,
    )

    return selected


def captured_soft_leftmost(scene, temperature, operation_name="leftmost"):
    """Soft leftmost with attention capture."""
    from sheaf.runtime.visual_ops import soft_leftmost

    # Extract x-coordinates and compute selection scores
    x_coords = scene[..., -2]  # [B, N]
    scores = -x_coords / temperature
    selection_weights = jax.nn.softmax(scores, axis=-1)

    selected = soft_leftmost(scene, temperature)

    # Capture selection weights
    capture_step(
        operation_name,
        attention=selection_weights,
        value=selected,
        temperature=temperature,
    )

    return selected


def captured_spatial_left_of(scene, reference_obj, threshold, temperature):
    """Spatial left-of with attention capture."""
    from sheaf.runtime.visual_ops import spatial_relation_left_of

    filtered = spatial_relation_left_of(scene, reference_obj, threshold, temperature)

    # Compute attention scores (sigmoid of relative position)
    scene_x = scene[..., -2]
    ref_x = reference_obj[..., -2]
    relative_x = scene_x - jnp.expand_dims(ref_x, axis=-1)
    attention = jax.nn.sigmoid((-relative_x - threshold) / temperature)

    capture_step(
        "spatial-left-of",
        attention=attention,
        value=filtered,
        threshold=threshold,
        temperature=temperature,
    )

    return filtered
