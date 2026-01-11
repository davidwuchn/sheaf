import os
import pickle
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
D_MODEL = 256
N_LAYERS = 4
N_HEADS = 4
BATCH_SIZE = 32
BLOCK_SIZE = 128
EPOCHS = 1000
LEARNING_RATE = 1e-3

WEIGHTS_PATH = os.path.join(current_dir, "out", "weights.pkl")


def prepare_data(path):
    with open(path, "r") as f:
        text = f.read().strip()
    vocab = sorted(list(set(text)))
    encoded = utils.encode(text, vocab)
    return text, encoded, vocab


def load_or_train_params(shf, encoded_data, config):
    if os.path.exists(WEIGHTS_PATH):
        print(f"Loading weights from {WEIGHTS_PATH}...")
        with open(WEIGHTS_PATH, "rb") as f:
            params = pickle.load(f)

        # params = run_training(shf, params, encoded_data, config)
        return jax.tree_util.tree_map(lambda x: jnp.array(x, dtype=jnp.float32), params)

    print("Weights not found. Initializing and training...")
    # New short call: shf.init_params instead of shf.registry["init-gpt-params"]
    params = shf.init_gpt_params(jax.random.PRNGKey(42), config)
    return run_training(shf, params, encoded_data, config)


def run_training(shf, params, encoded_data, config):
    # Adam states
    m = jax.tree_util.tree_map(jnp.zeros_like, params)
    v = jax.tree_util.tree_map(jnp.zeros_like, params)
    t = jnp.array(0, dtype=jnp.int32)
    t = 0

    print(f"Starting training for {EPOCHS} epochs...")
    last_time = time.perf_counter()

    for step in range(1, EPOCHS + 1):
        x_ids, y_ids = utils.get_batch(encoded_data, BLOCK_SIZE, BATCH_SIZE)

        # New short call: shf.train_step direct execution
        res = shf.train_step(params, m, v, t, x_ids, y_ids, config)

        loss, params, m, v, t = res["loss"], res["params"], res["m"], res["v"], res["t"]

        if step == 1 or step % 10 == 0:
            now = time.perf_counter()
            print(f"Step {step} | Loss: {loss:.4f} | Time: {now - last_time:.2f}s")
            last_time = now

    os.makedirs(os.path.dirname(WEIGHTS_PATH), exist_ok=True)
    with open(WEIGHTS_PATH, "wb") as f:
        pickle.dump(params, f)
    return params


def run_inference(
    shf, params, vocab, config, length=512, prompt="FIRST CITIZEN:", trace=False
):
    print(f"\nGenerating text (Prompt: '{prompt}')...")
    initial_ids = utils.encode(prompt, vocab)
    ids = jnp.zeros(BLOCK_SIZE, dtype=jnp.int32)
    ids = ids.at[-len(initial_ids) :].set(jnp.array(initial_ids[-BLOCK_SIZE:]))
    key = jax.random.PRNGKey(time.perf_counter_ns() % 2**32)

    for _ in range(length):
        # New short call: shf.generate_token
        res = shf.generate_token(ids, params, config, key, 10, 0.8, trace=trace)
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
    config = {
        "d_model": D_MODEL,
        "n_layers": N_LAYERS,
        "n_heads": N_HEADS,
        "batch_size": BATCH_SIZE,
        "block_size": BLOCK_SIZE,
        "vocab_size": len(vocab),
        "lr": LEARNING_RATE,
    }

    params = load_or_train_params(shf, encoded_data, config)
    run_inference(shf, params, vocab, config)

    # DEBUG - trace one token
    # run_inference(shf, params, vocab, config, length=1, trace="normal")


if __name__ == "__main__":
    main()
