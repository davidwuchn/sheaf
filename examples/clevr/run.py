"""
CLEVR Neuro-Symbolic Reasoning Demo
"""

import os
import pickle
import sys

import jax
import jax.numpy as jnp

from sheaf import Sheaf

try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    current_dir = os.getcwd()
if current_dir not in sys.path:
    sys.path.append(current_dir)

import data


def load_model():
    shf = Sheaf()
    model_dir = os.path.abspath(os.path.dirname(__file__))
    model_path = os.path.join(model_dir, "model.shf")

    # Change to model directory so relative imports (use ./utils.shf) work
    old_cwd = os.getcwd()
    os.chdir(model_dir)
    try:
        with open(model_path) as f:
            shf.load(f.read())
    finally:
        os.chdir(old_cwd)

    return shf


def load_params(shf):
    params_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "weights.pkl")
    )

    if os.path.exists(params_path):
        print(f"Loading trained parameters from {params_path}\n")
        with open(params_path, "rb") as f:
            params = pickle.load(f)
        return params
    else:
        print(f"No trained parameters found at {params_path}")
        print("To train the model first, run: python train.py")
        print("Using random parameters\n")
        return shf.init_clevr_params(jax.random.PRNGKey(0))


def test_query(shf, scene_dict, query, answer, params):
    # Test a single query and return success/failure
    scene = jnp.expand_dims(data.scene_to_tensor(scene_dict), 0)
    result = shf.execute_query(scene, query, params)

    op = query[0]
    if op == "query-color":
        predicted = data.COLORS[int(jnp.argmax(result[0]))]
        success = predicted == answer
    elif op == "query-shape":
        predicted = data.SHAPES[int(jnp.argmax(result[0]))]
        success = predicted == answer
    elif op == "exists?":
        predicted = float(result[0]) > 0.5
        success = predicted == answer
    else:
        predicted, success = None, False

    return {
        "query": query,
        "expected": answer,
        "predicted": predicted,
        "success": success,
    }


def run_tests(shf, params, num_tests=10):
    # Run random tests and report accuracy
    key = jax.random.PRNGKey(42)
    passed = 0

    for i in range(num_tests):
        key, scene_key, query_key = jax.random.split(key, 3)
        scene = data.generate_scene(scene_key, n_objects=4)
        query, answer = data.generate_query(scene, query_key)

        result = test_query(shf, scene, query, answer, params)
        status = "PASS" if result["success"] else "FAIL"
        print(
            f"[{status}] {result['query']} -> {result['predicted']} (expected: {result['expected']})"
        )

        if result["success"]:
            passed += 1

    print(f"\nAccuracy: {passed}/{num_tests} ({100 * passed / num_tests:.1f}%)")
    return passed / num_tests


def main():
    print("CLEVR Neuro-Symbolic Reasoning\n")

    shf = load_model()
    print(f"Loaded functions: {list(shf.registry.keys())}")

    params = load_params(shf)
    run_tests(shf, params, num_tests=10)


if __name__ == "__main__":
    main()
