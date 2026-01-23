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

    # ========================================================================
    # SIDEBAR: Scene Generation
    # ========================================================================

    st.sidebar.header("Scene Configuration")

    seed = st.sidebar.number_input("Random Seed", min_value=0, max_value=9999, value=42)
    n_objects = st.sidebar.slider(
        "Number of Objects", min_value=2, max_value=5, value=4
    )

    if st.sidebar.button("Generate New Scene") or "scene" not in st.session_state:
        key = jax.random.PRNGKey(seed)
        st.session_state.scene = data.generate_scene(key, n_objects=n_objects)
        st.session_state.scene_tensor = data.scene_to_tensor(st.session_state.scene)

    scene = st.session_state.scene
    scene_tensor = st.session_state.scene_tensor

    # ========================================================================
    # MAIN: Query Interface
    # ========================================================================

    col1, col2 = st.columns([1, 1])

    with col1:
        st.header("Query Input")

        # Example selector
        example_name = st.selectbox(
            "Choose a query:", ["Custom"] + list(example_queries.keys())
        )

        if example_name != "Custom":
            default_query = str(example_queries[example_name])
        else:
            default_query = '["query-color", ["leftmost", ["filter-shape", ":circle"]]]'

        # Query input
        query_str = st.text_area(
            "Sheaf Query (Python list format):", value=default_query, height=100
        )

        execute_button = st.button("Execute Query", type="primary")

    with col2:
        st.header("Scene Visualization")

        # Display scene
        fig_scene = plot_scene(scene, title=f"Scene (Seed: {seed})")
        st.pyplot(fig_scene)
        plt.close()

    # Execution and results

    if execute_button:
        try:
            # Parse query
            query = eval(query_str)  # Safe in this context

            st.success(f"✓ Query parsed: `{query}`")

            # Execute query
            scene_batch = jnp.expand_dims(scene_tensor, axis=0)

            with st.spinner("Executing query..."):
                prediction = shf.execute_query(scene_batch, query, params)

            # Decode answer
            query_type = infer_query_type(query)
            answer, prob_dict = decode_answer(prediction, query_type)

            st.markdown("---")
            st.header("Results:")

            result_col1, result_col2 = st.columns([1, 1])

            with result_col1:
                st.subheader("Answer")
                st.markdown(f"### **{answer}**")
                st.caption(f"Query type: {query_type}")

            with result_col2:
                st.subheader("Probability Distribution")
                fig_probs = plot_probability_distribution(prob_dict)
                st.pyplot(fig_probs)
                plt.close()

        except Exception as e:
            st.error(f"Error executing query: {e}")
            st.exception(e)


if __name__ == "__main__":
    main()
