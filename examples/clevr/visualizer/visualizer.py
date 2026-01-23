"""
CLEVR Visualizer

Dependencies: streamlit, matplotlib
Run: streamlit run app.py
"""

import os
import sys

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import streamlit as st

# Add parent directory to path for data module
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_dir)

# Add sheaf root to path
sheaf_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "/../.."))
sys.path.insert(0, sheaf_root)

import data

from sheaf import Sheaf

# Pre-configured queries

example_queries = {
    "Color of leftmost circle": [
        "query-color",
        ["leftmost", ["filter-shape", ":circle"]],
    ],
    "Exists red square?": [
        "exists?",
        ["intersect", ["filter-color", ":red"], ["filter-shape", ":square"]],
    ],
    "Shape left of blue object": [
        "query-shape",
        ["left-of", ["unique", ["filter-color", ":blue"]]],
    ],
    "Color of rightmost object": [
        "query-color",
        ["rightmost", ["filter-shape", ":square"]],
    ],
    "Exists yellow triangle?": [
        "exists?",
        ["intersect", ["filter-color", ":yellow"], ["filter-shape", ":triangle"]],
    ],
}


# Color mappings
COLOR_MAP = {
    "red": "#FF4444",
    "green": "#44FF44",
    "blue": "#4444FF",
    "yellow": "#FFFF44",
}

SHAPE_SYMBOLS = {
    "circle": "o",
    "square": "s",
    "triangle": "^",
}


def load_sheaf_model():
    shf = Sheaf()
    model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    model_path = os.path.join(model_dir, "model.shf")
    old_cwd = os.getcwd()
    os.chdir(model_dir)
    try:
        with open(model_path, "r") as f:
            shf.load(f.read())
    finally:
        os.chdir(old_cwd)

    # Load trained parameters
    params_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "weights.pkl")
    )
    if os.path.exists(params_path):
        import pickle

        with open(params_path, "rb") as f:
            params = pickle.load(f)
    else:
        # Initialize random parameters if no trained weights exist
        init_key = jax.random.PRNGKey(42)
        params = shf.init_clevr_params(init_key)

    return shf, params


def plot_scene(scene_dict, attention_weights=None, title="Scene"):
    """
    Plot 2D scene with objects as colored shapes.

    Args:
        scene_dict: Scene dictionary from data.py
        attention_weights: Optional [N_objects] array for highlighting
        title: Plot title

    Returns:
        matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    objects = scene_dict["objects"]
    n_objects = len(objects)

    # Normalize attention weights if provided
    if attention_weights is not None:
        # Convert JAX array to numpy
        attention_weights = jnp.array(attention_weights)
        # Normalize to [0, 1]
        att_min = jnp.min(attention_weights)
        att_max = jnp.max(attention_weights)
        if att_max > att_min:
            attention_norm = (attention_weights - att_min) / (att_max - att_min)
        else:
            attention_norm = jnp.ones_like(attention_weights)
    else:
        attention_norm = jnp.ones(n_objects)

    # Plot each object
    for i, obj in enumerate(objects):
        x, y = obj["x"], obj["y"]
        color = COLOR_MAP[obj["color"]]
        shape = obj["shape"]
        marker = SHAPE_SYMBOLS[shape]

        # Alpha based on attention weight
        alpha = float(attention_norm[i]) * 0.8 + 0.2  # Min alpha = 0.2
        size = 200 + float(attention_norm[i]) * 400  # Size based on attention

        # Plot marker
        ax.scatter(
            x,
            y,
            c=color,
            marker=marker,
            s=size,
            alpha=alpha,
            edgecolors="black",
            linewidths=2,
        )

        # Add label
        label = f"{obj['color']}\n{obj['shape']}"
        if attention_weights is not None:
            label += f"\n{float(attention_weights[i]):.2f}"

        ax.text(
            x,
            y - 0.08,
            label,
            ha="center",
            va="top",
            fontsize=8,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
        )

    return fig


def decode_answer(prediction, query_type):
    """
    Decode prediction to human-readable answer.

    Args:
        prediction: Model output [Batch, D]
        query_type: "color", "shape", or "boolean"

    Returns:
        answer string, probabilities dict
    """
    pred = prediction[0]  # Remove batch dimension

    if query_type == "color":
        probs = jax.nn.softmax(pred)
        colors = data.COLORS
        answer_idx = int(jnp.argmax(probs))
        answer = colors[answer_idx]
        prob_dict = {colors[i]: float(probs[i]) for i in range(len(colors))}
        return answer, prob_dict

    elif query_type == "shape":
        probs = jax.nn.softmax(pred)
        shapes = data.SHAPES
        answer_idx = int(jnp.argmax(probs))
        answer = shapes[answer_idx]
        prob_dict = {shapes[i]: float(probs[i]) for i in range(len(shapes))}
        return answer, prob_dict

    elif query_type == "boolean":
        # pred is already a probability (0-1) from sigmoid in the model
        prob_yes = float(pred)
        answer = "Yes" if prob_yes > 0.5 else "No"
        prob_dict = {"Yes": prob_yes, "No": 1 - prob_yes}
        return answer, prob_dict

    return "Unknown", {}


def generate_fixed_scene():
    """Generate a scene with 5 objects: all colors and shapes represented."""
    import random
    import time

    # Fixed positions that are well-distributed and visible
    base_positions = [
        (0.2, 0.75),
        (0.5, 0.8),
        (0.8, 0.75),
        (0.35, 0.3),
        (0.65, 0.25),
    ]

    # Ensure we have all colors and shapes represented
    colors = data.COLORS  # [red, green, blue, yellow]
    shapes = data.SHAPES  # [circle, square, triangle]

    # Shuffle colors and shapes to create different combinations
    seed = int(time.time() * 1000) % 100000
    random.seed(seed)
    colors_shuffled = colors.copy()
    shapes_shuffled = shapes.copy()
    random.shuffle(colors_shuffled)
    random.shuffle(shapes_shuffled)

    # Create objects with shuffled colors and shapes
    objects = []
    colors_cycle = (colors_shuffled + colors_shuffled)[:5]  # Repeat to get 5
    shapes_cycle = (shapes_shuffled + shapes_shuffled)[:5]  # Repeat to get 5

    for i, (x, y) in enumerate(base_positions):
        objects.append(
            {
                "color": colors_cycle[i],
                "shape": shapes_cycle[i],
                "x": x,
                "y": y,
            }
        )

    # Add small random jitter to positions
    key = jax.random.PRNGKey(seed)
    keys = jax.random.split(key, 5)
    for i, obj in enumerate(objects):
        x_noise = float(jax.random.uniform(keys[i], minval=-0.05, maxval=0.05))
        y_noise = float(jax.random.uniform(keys[i], minval=-0.05, maxval=0.05))
        obj["x"] = max(0.1, min(0.9, obj["x"] + x_noise))
        obj["y"] = max(0.1, min(0.9, obj["y"] + y_noise))

    scene = {"objects": objects, "n_objects": 5}
    scene_tensor = data.scene_to_tensor(scene)
    return scene, scene_tensor


def infer_query_type(query):
    op = query[0]
    if op == "query-color":
        return "color"
    elif op == "query-shape":
        return "shape"
    elif op == "exists?":
        return "boolean"
    return "color"  # default


def plot_probability_distribution(prob_dict, title="Answer Probabilities"):
    fig, ax = plt.subplots(figsize=(8, 4))

    labels = list(prob_dict.keys())
    values = list(prob_dict.values())

    bars = ax.bar(labels, values, color="skyblue", edgecolor="black")

    # Highlight max
    max_idx = values.index(max(values))
    bars[max_idx].set_color("lightcoral")

    ax.set_ylabel("Probability")
    ax.set_title(title)
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.3)

    # Add value labels on bars
    for i, (label, value) in enumerate(zip(labels, values)):
        ax.text(i, value + 0.02, f"{value:.3f}", ha="center", va="bottom")

    return fig


# STREAMLIT APP


def main():
    st.set_page_config(page_title="CLEVR Visualizer", layout="wide")

    st.title("CLEVR Neuro-Symbolic Visualizer")

    # Load model (cached)
    if "shf" not in st.session_state:
        with st.spinner("Loading Sheaf model..."):
            st.session_state.shf, st.session_state.params = load_sheaf_model()
        st.success("Model loaded!")

    shf = st.session_state.shf
    params = st.session_state.params

    # Generate initial scene if not exists
    if "scene" not in st.session_state:
        st.session_state.scene, st.session_state.scene_tensor = generate_fixed_scene()

    scene = st.session_state.scene
    scene_tensor = st.session_state.scene_tensor

    # Main layout: Scene on left, Query and probability on right
    scene_col, query_col = st.columns(2)

    with scene_col:
        fig_scene = plot_scene(scene, title="")
        st.pyplot(fig_scene, use_container_width=True)
        plt.close()

    with query_col:
        st.subheader("Query")

        example_name = st.selectbox(
            "Choose a query:",
            ["Custom"] + list(example_queries.keys()),
            key="query_selector",
        )

        if example_name != "Custom":
            default_query = str(example_queries[example_name])
        else:
            default_query = '["query-color", ["leftmost", ["filter-shape", ":circle"]]]'

        query_str = st.text_area(
            "Sheaf Query:", value=default_query, height=80, label_visibility="collapsed"
        )

        if st.button("Execute Query", type="primary", use_container_width=True):
            st.session_state.execute_query = True

        st.divider()

        if st.session_state.get("execute_query", False):
            try:
                query = eval(query_str)

                # Execute query
                scene_batch = jnp.expand_dims(scene_tensor, axis=0)

                with st.spinner("Executing query..."):
                    prediction = shf.execute_query(scene_batch, query, params)

                # Decode answer
                query_type = infer_query_type(query)
                answer, prob_dict = decode_answer(prediction, query_type)

                # Store in session for display
                st.session_state.last_query = query
                st.session_state.last_answer = answer
                st.session_state.last_prob_dict = prob_dict

                # Show probability chart
                fig_probs = plot_probability_distribution(
                    prob_dict, title="Answer Probabilities"
                )
                st.pyplot(fig_probs, use_container_width=True)
                plt.close()

                st.session_state.execute_query = False

            except Exception as e:
                st.error(f"Error executing query: {e}")
                import traceback

                st.text(traceback.format_exc())
                st.session_state.execute_query = False

    # Generate scene button below scene
    with scene_col:
        if st.button(
            "🎲 Generate Random Scene", type="primary", use_container_width=True
        ):
            st.session_state.scene, st.session_state.scene_tensor = (
                generate_fixed_scene()
            )
            st.rerun()

    # Query result banner at the bottom
    if st.session_state.get("last_query"):
        st.divider()
        query_result = st.session_state.last_query
        result = st.session_state.last_answer
        st.info(f"📊 Query: `{query_result}` → **Result: {result}**")


if __name__ == "__main__":
    main()
