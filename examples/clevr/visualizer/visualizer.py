"""
CLEVR Neuro-Symbolic Visualizer (V2)

Uses Sheaf V2 Rust backend via subprocess.
Dependencies: streamlit, matplotlib, numpy
Run: cd examples/clevr/visualizer && streamlit run visualizer.py
"""

import os
import shutil
import subprocess
import random

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

COLORS = ["red", "green", "blue", "yellow"]
SHAPES = ["circle", "square", "triangle"]
N_OBJECTS = 5
N_FEATURES = 9

COLOR_HEX = {
    "red": "#FF4444",
    "green": "#44FF44",
    "blue": "#4444FF",
    "yellow": "#FFFF44",
}
SHAPE_MARKER = {"circle": "o", "square": "s", "triangle": "^"}

EXAMPLE_QUERIES = {
    "Color of leftmost circle": [
        "query-color", ["leftmost", ["filter-shape", ":circle"]],
    ],
    "Exists red square?": [
        "exists?", ["intersect", ["filter-color", ":red"], ["filter-shape", ":square"]],
    ],
    "Shape left of blue object": [
        "query-shape", ["left-of", ["unique", ["filter-color", ":blue"]]],
    ],
    "Color of rightmost square": [
        "query-color", ["rightmost", ["filter-shape", ":square"]],
    ],
    "Exists yellow triangle?": [
        "exists?", ["intersect", ["filter-color", ":yellow"], ["filter-shape", ":triangle"]],
    ],
}

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CLEVR_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
PROJECT_ROOT = os.path.abspath(os.path.join(CLEVR_DIR, "..", ".."))


def find_sheaf_binary():
    path_bin = shutil.which("sheaf")
    if path_bin:
        return path_bin
    cargo_bin = os.path.join(PROJECT_ROOT, "sheaf", "target", "release", "sheaf")
    if os.path.exists(cargo_bin):
        return cargo_bin
    return None


def generate_scene():
    objects = []
    for _ in range(N_OBJECTS):
        objects.append({
            "color": random.choice(COLORS),
            "shape": random.choice(SHAPES),
            "x": round(random.uniform(0.1, 0.9), 4),
            "y": round(random.uniform(0.1, 0.9), 4),
        })
    return {"objects": objects, "n_objects": N_OBJECTS}


def scene_to_tensor(scene):
    tensor = np.zeros((N_OBJECTS, N_FEATURES), dtype=np.float32)
    for i, obj in enumerate(scene["objects"]):
        tensor[i, COLORS.index(obj["color"])] = 1.0
        tensor[i, 4 + SHAPES.index(obj["shape"])] = 1.0
        tensor[i, 7] = obj["x"]
        tensor[i, 8] = obj["y"]
    return tensor


def format_query_sheaf(query):
    """Convert Python query list to Sheaf vector notation."""
    if isinstance(query, str):
        return f'"{query}"'
    elif isinstance(query, list):
        return "[" + " ".join(format_query_sheaf(e) for e in query) + "]"
    return str(query)


def run_sheaf_query(sheaf_bin, scene_tensor, query):
    tensor_str = " ".join(f"{x:.6f}" for x in scene_tensor.flatten())
    query_str = format_query_sheaf(query)
    code = f'(use ./bridge.shf) (run-bridge (tensor [{tensor_str}]) {query_str})'

    proc = subprocess.run(
        [sheaf_bin, "-c", code],
        capture_output=True, text=True, cwd=CLEVR_DIR, timeout=30,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"Sheaf error:\n{proc.stderr}\n{proc.stdout}")

    return parse_output(proc.stdout)


def parse_output(text):
    result = {"steps": []}
    for line in text.strip().split("\n"):
        line = line.strip()
        if line.startswith("RESULT:"):
            _, rtype, answer = line.split(":", 2)
            result["type"] = rtype
            result["answer"] = answer
        elif line.startswith("LOGITS:"):
            result["logits"] = [float(x) for x in line[7:].split(",") if x]
        elif line.startswith("STEP:"):
            result["steps"].append([float(x) for x in line[5:].split(",") if x])
    return result


def extract_op_names(query):
    """Extract operation names in depth-first order (matching step order)."""
    ops = []

    def walk(q):
        if not isinstance(q, list) or len(q) == 0:
            return
        for arg in q[1:]:
            if isinstance(arg, list) and len(arg) > 1 and not str(arg[0]).startswith(":"):
                walk(arg)
        ops.append(q[0])

    walk(query)
    return ops


def infer_query_type(query):
    op = query[0]
    if op == "query-color":
        return "color"
    elif op == "query-shape":
        return "shape"
    return "boolean"


def plot_scene(scene, title=""):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")

    for obj in scene["objects"]:
        ax.scatter(
            obj["x"], obj["y"],
            c=COLOR_HEX[obj["color"]], marker=SHAPE_MARKER[obj["shape"]],
            s=400, alpha=0.9, edgecolors="black", linewidths=2,
        )
        ax.text(
            obj["x"], obj["y"] - 0.07,
            f"{obj['color']}\n{obj['shape']}",
            ha="center", va="top", fontsize=8,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
        )

    return fig


def plot_probabilities(prob_dict, title="Probabilities"):
    fig, ax = plt.subplots(figsize=(6, 3))
    labels = list(prob_dict.keys())
    values = list(prob_dict.values())
    bars = ax.bar(labels, values, color="skyblue", edgecolor="black")
    max_idx = values.index(max(values))
    bars[max_idx].set_color("lightcoral")
    ax.set_ylim(0, 1.1)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    for i, v in enumerate(values):
        ax.text(i, v + 0.02, f"{v:.3f}", ha="center", va="bottom")
    return fig


def plot_attention_steps(scene, op_names, steps):
    n = len(steps)
    if n == 0:
        return None

    fig, axes = plt.subplots(1, n, figsize=(3.5 * n, 4))
    if n == 1:
        axes = [axes]

    objects = scene["objects"]

    for i, (op, att_values) in enumerate(zip(op_names, steps)):
        ax = axes[i]

        is_per_object = len(att_values) >= len(objects) and "query" not in op

        if not is_per_object:
            # Bar chart for logits or aggregated outputs
            if "color" in op:
                names = COLORS
            elif "shape" in op:
                names = SHAPES
            else:
                names = [f"d{j}" for j in range(len(att_values))]
            ax.barh(names, att_values, color="salmon", edgecolor="black")
            ax.set_xlim(0, max(max(att_values), 0.01) * 1.2)
        else:
            # Scene attention heatmap
            att = np.array(att_values[:len(objects)])
            att_min, att_max = att.min(), att.max()
            if att_max > att_min:
                att_norm = (att - att_min) / (att_max - att_min)
            else:
                att_norm = np.ones(len(objects))

            for j, obj in enumerate(objects):
                alpha = float(att_norm[j]) * 0.8 + 0.2
                size = 100 + float(att_norm[j]) * 350
                ax.scatter(
                    obj["x"], obj["y"],
                    c=COLOR_HEX[obj["color"]], marker=SHAPE_MARKER[obj["shape"]],
                    s=size, alpha=alpha, edgecolors="black", linewidths=1.5,
                )
                ax.text(
                    obj["x"], obj["y"] - 0.07, f"{att_values[j]:.2f}",
                    ha="center", va="top", fontsize=8,
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
                )

            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_aspect("equal")
            ax.grid(True, alpha=0.2)
            ax.set_xticks([])
            ax.set_yticks([])

        ax.set_title(op, fontsize=10, fontweight="bold")

    plt.tight_layout()
    return fig


OP_DETAILS = {
    "filter-color": {
        "desc": "Soft attribute filter",
        "math": "scores = einsum('...nd,d->...n', scene, W_color[idx])\n"
                "         att = sigmoid((scores - 0.5) / temperature)\n"
                "         out = scene * att",
    },
    "filter-shape": {
        "desc": "Soft attribute filter",
        "math": "scores = einsum('...nd,d->...n', scene, W_shape[idx])\n"
                "         att = sigmoid((scores - 0.5) / temperature)\n"
                "         out = scene * att",
    },
    "unique": {
        "desc": "Soft single-object selection",
        "math": "objectness = sum(|scene|, axis=-1)\n"
                "         att = softmax(objectness / temperature, axis=-1)\n"
                "         out = einsum('...n,...nd->...d', att, scene)",
    },
    "leftmost": {
        "desc": "Argmin-x via softmax",
        "math": "x = scene[..., 7];  mask = sum(|scene|, axis=-1) > 1e-3\n"
                "         att = softmax(where(mask, -x / temperature, -1e10), axis=-1)\n"
                "         out = einsum('...n,...nd->...d', att, scene)",
    },
    "rightmost": {
        "desc": "Argmax-x via softmax",
        "math": "x = scene[..., 7];  mask = sum(|scene|, axis=-1) > 1e-3\n"
                "         att = softmax(where(mask, x / temperature, -1e10), axis=-1)\n"
                "         out = einsum('...n,...nd->...d', att, scene)",
    },
    "left-of": {
        "desc": "Spatial relation (x < ref_x)",
        "math": "relative = scene_x - ref_x\n"
                "         scores = sigmoid(-(relative + threshold) / temperature)\n"
                "         out = scene * scores",
    },
    "right-of": {
        "desc": "Spatial relation (x > ref_x)",
        "math": "relative = scene_x - ref_x\n"
                "         scores = sigmoid((relative - threshold) / temperature)\n"
                "         out = scene * scores",
    },
    "intersect": {
        "desc": "Element-wise intersection",
        "math": "out = minimum(scene_a, scene_b)",
    },
    "exists?": {
        "desc": "Existence test",
        "math": "objectness = sum(|scene|, axis=-1)\n"
                "         total = sum(objectness, axis=-1)\n"
                "         out = sigmoid(5.0 * (total - threshold))",
    },
    "query-color": {
        "desc": "Attribute extraction [0:4]",
        "math": "aggregated = sum(scene, axis=-2)  (if batched)\n"
                "         out = aggregated[..., 0:4]",
    },
    "query-shape": {
        "desc": "Attribute extraction [4:7]",
        "math": "aggregated = sum(scene, axis=-2)  (if batched)\n"
                "         out = aggregated[..., 4:7]",
    },
}


def format_pipeline_text(scene, query, op_names, steps, answer, scene_tensor):
    objects = scene["objects"]

    lines = [f"Query: {query}\n"]

    # Raw tensor
    lines.append(f"Input: Scene tensor [1, {N_OBJECTS}, {N_FEATURES}]")
    lines.append(f"  Axes: [red, green, blue, yellow, circle, square, triangle, x, y]")
    for i in range(N_OBJECTS):
        row = scene_tensor[i]
        vals = "  ".join(f"{v:5.2f}" for v in row)
        lines.append(f"  [{vals}]")

    # Decoded
    lines.append("")
    lines.append("Decoded:")
    for i, obj in enumerate(objects):
        lines.append(
            f"  Object {i+1}: {obj['color']:7s} {obj['shape']:8s}  "
            f"x={obj['x']:.2f}  y={obj['y']:.2f}"
        )
    lines.append("    \u2193")

    # Steps
    for op, att in zip(op_names, steps):
        detail = OP_DETAILS.get(op, {})
        desc = detail.get("desc", op)
        math = detail.get("math", "")

        lines.append(f"[{op}] {desc}")
        if math:
            lines.append(f"  Math: {math}")

        is_per_object = len(att) >= len(objects) and "query" not in op

        if not is_per_object:
            if "color" in op:
                names = COLORS
            elif "shape" in op:
                names = SHAPES
            else:
                names = [f"d{j}" for j in range(len(att))]
            val_str = ", ".join(f"{n}: {v:.2f}" for n, v in zip(names, att))
            lines.append(f"  Output: [{val_str}]")
        else:
            att_str = ", ".join(
                f"{obj['color']}_{obj['shape']}: {att[j]:.2f}"
                for j, obj in enumerate(objects)
            )
            lines.append(f"  Attention: [{att_str}]")

        lines.append("    \u2193")

    lines.append(f"[Decision] argmax(logits): {answer.upper()}")
    return "\n".join(lines)


def main():
    st.set_page_config(page_title="CLEVR Visualizer", layout="wide")
    st.title("CLEVR Neuro-Symbolic Visualizer")

    sheaf_bin = find_sheaf_binary()
    if not sheaf_bin:
        st.error(
            "Sheaf binary not found. "
            "Build with: `cd sheaf && cargo build --release`"
        )
        return

    if "scene" not in st.session_state:
        st.session_state.scene = generate_scene()
        st.session_state.scene_tensor = scene_to_tensor(st.session_state.scene)

    scene = st.session_state.scene
    scene_tensor = st.session_state.scene_tensor

    query_col, scene_col = st.columns(2)

    with query_col:
        st.subheader("Query")

        example_name = st.selectbox(
            "Choose a query:",
            ["Custom"] + list(EXAMPLE_QUERIES.keys()),
        )

        if example_name != "Custom":
            default_query = str(EXAMPLE_QUERIES[example_name])
        else:
            default_query = '["query-color", ["leftmost", ["filter-shape", ":circle"]]]'

        query_str = st.text_area(
            "Query:", value=default_query, height=80,
            label_visibility="collapsed",
        )

        col1, col2 = st.columns(2)
        with col1:
            execute = st.button("Execute Query", type="primary", use_container_width=True)
        with col2:
            if st.button("New Scene", use_container_width=True):
                st.session_state.scene = generate_scene()
                st.session_state.scene_tensor = scene_to_tensor(st.session_state.scene)
                st.session_state.pop("last_result", None)
                st.session_state.pop("last_query", None)
                st.rerun()

        if execute:
            try:
                query = eval(query_str)
                with st.spinner("Running Sheaf query..."):
                    result = run_sheaf_query(sheaf_bin, scene_tensor, query)
                st.session_state.last_result = result
                st.session_state.last_query = query
            except Exception as e:
                st.error(f"Error: {e}")

        if "last_result" in st.session_state:
            r = st.session_state.last_result
            st.divider()
            st.info(f"**Result: {r['answer'].upper()}**")

            if r["type"] == "color":
                probs = dict(zip(COLORS, r["logits"]))
            elif r["type"] == "shape":
                probs = dict(zip(SHAPES, r["logits"]))
            else:
                p = r["logits"][0]
                probs = {"Yes": p, "No": 1 - p}

            fig_prob = plot_probabilities(probs)
            st.pyplot(fig_prob)
            plt.close()

    with scene_col:
        st.subheader("Scene")
        fig_scene = plot_scene(scene, title="")
        st.pyplot(fig_scene)
        plt.close()

    if "last_result" in st.session_state and "last_query" in st.session_state:
        st.divider()
        st.subheader("Symbolic Attention Shaping")

        r = st.session_state.last_result
        query = st.session_state.last_query
        op_names = extract_op_names(query)
        steps = r["steps"]

        pipeline = format_pipeline_text(
            scene, query, op_names, steps, r["answer"], scene_tensor,
        )
        st.code(pipeline, language="text")

        if steps:
            fig_att = plot_attention_steps(scene, op_names, steps)
            if fig_att:
                st.pyplot(fig_att)
                plt.close()


if __name__ == "__main__":
    main()
