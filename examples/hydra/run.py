import os

import jax
import jax.numpy as jnp

from sheaf import Sheaf


def print_struct(params):
    layers = params.get("layers", []) or params.get(":layers", [])
    structure = "Input(2) -> "
    for i, _ in enumerate(layers):
        structure += f"ReLU(32) -> "
    structure += "Linear(1) -> Output"
    print(f"[Model Map]: {structure}\n")


def main():
    shf = Sheaf()
    model_dir = os.path.abspath(os.path.dirname(__file__))
    model_path = os.path.join(model_dir, "hydra.shf")

    old_cwd = os.getcwd()
    os.chdir(model_dir)
    try:
        with open(model_path) as f:
            shf.load(f.read())
    finally:
        os.chdir(old_cwd)

    key = jax.random.PRNGKey(42)
    config = {"d_model": 32, "lr": 1e-2}

    # X = XOR data, Y = expected answers
    X = jnp.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=jnp.float32)
    Y = jnp.array([[0.0], [1.0], [1.0], [0.0]], dtype=jnp.float32)

    # Initial params: just a head and an empty list of hidden layers
    params = {
        "head": {
            "W": jax.random.normal(key, (2, 1)),
            "b": jnp.zeros((1,)),
        },
        "layers": [],  # Start with 0 hidden layers
    }

    print("--- Starting Training ---")
    print_struct(params)
    previous_loss = loss = float("inf")
    epoch = 0

    # We can't reach such a loss with the initial model configuration
    while loss > 0.008:
        # Call Sheaf for the training
        state = shf.train_step(params, X, Y, config["lr"])
        params = state["p"]
        loss = state["loss"]
        epoch += 1

        if epoch % 20 == 0:
            print(f"Epoch {epoch:3} | Loss: {state['loss']:.6f}")
            diff = previous_loss - loss

            # --- Self-evolution ---
            # If we hit a loss plateau
            if epoch > 0 and 0 <= diff < 0.003:
                print("\n[Evolution] Hit loss plateau. Adding a new dense layer...")
                key, subkey = jax.random.split(key)
                # Call Sheaf to grow the model "live", without stopping the JAX XLA engine
                params = shf.grow_hydra(params, subkey, config)
                print_struct(params)
            else:
                previous_loss = loss

    print("--- Training Successful ---")


if __name__ == "__main__":
    main()
