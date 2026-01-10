import jax
import jax.numpy as jnp

from sheaf import Sheaf

shf = Sheaf("mlp.shf")

X = jnp.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=jnp.float32)
Y = jnp.array([[0], [1], [1], [0]], dtype=jnp.float32)

params = shf.init_params(jax.random.PRNGKey(42))
m = jax.tree_util.tree_map(jnp.zeros_like, params)
v = jax.tree_util.tree_map(jnp.zeros_like, params)

state = {
    "p": params,
    "m": jax.tree_util.tree_map(jnp.zeros_like, params),
    "v": jax.tree_util.tree_map(jnp.zeros_like, params),
    "t": 0,
    "loss": 0.0,
}

print("Training XOR for 100 steps")
for epoch in range(10):
    state = shf.train_n_steps(state, X, Y, 0.1, 10)
    print(f"Step {10 + epoch * 10:3} | Loss: {state['loss']:.6f}")

preds = shf.forward(X, state["p"])
print("\nPredictions (X -> Y):")
for inp, pred in zip(X, preds):
    print(f"{inp} -> {float(pred[0]):.4f}")
