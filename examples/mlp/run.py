import jax
import jax.numpy as jnp

from sheaf import Sheaf

shf = Sheaf("mlp.shf")

X = jnp.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=jnp.float32)
Y = jnp.array([[0], [1], [1], [0]], dtype=jnp.float32)

params = shf.init_params(jax.random.PRNGKey(42))

for epoch in range(10):
    state = shf.train(params, X, Y, 0.5, 100)
    params = state["p"]
    print(f"Epoch {epoch + 1:2} | Loss: {state['loss']:.6f}")

print("\nPredictions:")
for x, y in zip(X, shf.forward(X, params)):
    print(f"  {x} -> {y[0]:.3f}")
