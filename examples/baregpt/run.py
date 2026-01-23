import os
import sys
import time

import jax
import jax.numpy as jnp

from sheaf import Sheaf

try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    current_dir = os.getcwd()
if current_dir not in sys.path:
    sys.path.append(current_dir)

import utils

# --- HYPERPARAMETERS ---
CONFIG = {
    "d_model": 256,
    "n_layers": 4,
    "n_heads": 4,
    "batch_size": 32,
    "block_size": 128,
    "lr": 1e-3,
}
EPOCHS = 500
WEIGHTS_PATH = os.path.join(current_dir, "out", "weights.pkl")


def prepare_data(path):
    with open(path, "r") as f:
        text = f.read().strip()
    vocab = sorted(list(set(text)))
    encoded = utils.encode(text, vocab)
    return text, encoded, vocab


def save_checkpoint(shf, params, path):
    import pickle

    tree = shf.to_pytree(params)
    with open(path, "wb") as f:
        pickle.dump(tree, f)


def load_checkpoint(shf, path):

    import pickle

    with open(path, "rb") as f:
        tree = pickle.load(f)
    return shf.from_pytree(tree)


def load_or_train_params(shf, encoded_data, config):
    if os.path.exists(WEIGHTS_PATH):
        print(f"Loading weights from {WEIGHTS_PATH}...")
        params = load_checkpoint(shf, WEIGHTS_PATH)
        return jax.tree_util.tree_map(lambda x: jnp.array(x, dtype=jnp.float32), params)

    print("Weights not found. Initializing and training...")
    params = shf.init_gpt_params(jax.random.PRNGKey(42), config)
    return run_training(shf, params, encoded_data, config)


def run_training(shf, params, encoded_data, config):
    # Adam states
    m = jax.tree_util.tree_map(jnp.zeros_like, params)
    v = jax.tree_util.tree_map(jnp.zeros_like, params)
    t = 0

    print(f"Starting training for {EPOCHS} epochs...")
    last_time = time.perf_counter()

    for step in range(1, EPOCHS + 1):
        x_ids, y_ids = utils.get_batch(encoded_data, config["block_size"], config["batch_size"])

        # New short call: shf.train_step direct execution
        res = shf.train_step(params, m, v, t, x_ids, y_ids, config)

        loss, params, m, v, t = res["loss"], res["params"], res["m"], res["v"], res["t"]

        if step == 1 or step % 10 == 0:
            now = time.perf_counter()
            print(f"Step {step} | Loss: {loss:.4f} | Time: {now - last_time:.2f}s")
            last_time = now

    os.makedirs(os.path.dirname(WEIGHTS_PATH), exist_ok=True)
    save_checkpoint(shf, params, WEIGHTS_PATH)
    return params


def run_inference(
    shf,
    params,
    vocab,
    config,
    length=512,
    prompt="FIRST CITIZEN:",
    trace=False,
    scope=None,
):
    print(f"\nGenerating text (Prompt: '{prompt}')...")
    initial_ids = utils.encode(prompt, vocab)
    ids = jnp.zeros(config["block_size"], dtype=jnp.int32)
    ids = ids.at[-len(initial_ids) :].set(jnp.array(initial_ids[-config["block_size"]:]))
    key = jax.random.PRNGKey(time.perf_counter_ns() % 2**32)

    for _ in range(length):
        # New short call: shf.generate_token
        res = shf.generate_token(
            ids, params, config, key, 10, 0.8, trace=trace, scope=scope
        )
        ids = jnp.roll(ids, -1).at[-1].set(int(res["next_id"]))
        key = res["key"]
        print(utils.decode([int(res["next_id"])], vocab), end="", flush=True)
    print("\n")


def main():
    shf = Sheaf()
    with open(os.path.join(current_dir, "model.shf"), "r") as f:
        shf.load(f.read())

    text, encoded_data, vocab = prepare_data(
        os.path.join(current_dir, "data", "shakespeare.txt")
    )
    config = CONFIG
    config["vocab_size"] = len(vocab)
    params = load_or_train_params(shf, encoded_data, config)
    run_inference(shf, params, vocab, config)

    # DEBUG - trace one token
    # run_inference(shf, params, vocab, config, length=1, trace="normal")


if __name__ == "__main__":
    main()
