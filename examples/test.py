from collections import namedtuple

import jax
import jax.numpy as jnp

from sheaf import Sheaf

sheaf = Sheaf()

# On définit une fonction de prédiction factice pour que generate-next-id puisse compiler
sheaf.env["predict-fn"] = lambda ids, p: jnp.ones((1, 64, 65))

jax_params = {"w": jnp.array([[1.0, 2.0], [3.0, 4.0]]), "b": jnp.array([0.5, 0.5])}

with open("baregpt.shf", "r") as file:
    # Read the entire content and assign it to the 'content' variable
    shf_code = file.read()


# Compilation du module .shf
module = sheaf.load(shf_code)

# Test de la layernorm
ln_fn = jax.jit(module["layer-norm"])
print("Layernorm Output:", ln_fn(jnp.array([[1.0, 2.0, 3.0]])))

# Test de simple-layer
sl_params = {
    "w": jnp.ones((3, 10)),  # (dim_in, dim_out)
    "b": jnp.zeros(10),
}
sl_fn = jax.jit(module["simple-layer"])
input_data = jnp.array([[1.0, 2.0, 3.0]])  # (1, 3)
print("simple-layer Output:", sl_fn(input_data, sl_params))

# Test de la génération (id et nouvelle clé)
key = jax.random.PRNGKey(0)
config = {"vocab_size": 65}
res = module["generate-next-id"](jnp.zeros(64), {}, config, key, 10, 0.8)
print("Generated Token ID:", res[0])

# Test de Adam :

params = {"layer1": {"w": jnp.array([1.0, 2.0]), "b": jnp.array([0.5])}}
grads = {"layer1": {"w": jnp.array([0.1, -0.1]), "b": jnp.array([0.01])}}


# Initialisation des moments (m et v) à zéro avec la même structure que params
def adam_init(p):
    return jax.tree_util.tree_map(jnp.zeros_like, p)


m = adam_init(params)
v = adam_init(params)
t = 0  # Compteur de pas

# 2. Récupération de la fonction compilée
# On suppose que 'adam-step' est dans ton fichier .shf chargé par sheaf.load()
adam_step_fn = jax.jit(module["adam-step"])

# 3. L'appel
# Note : on passe les hyperparamètres (lr, beta1, beta2, eps)
new_params, new_m, new_v, new_t = adam_step_fn(
    params, grads, m, v, t, 1e-3, 0.9, 0.999, 1e-8
)

print(f"Pas de temps : {new_t}")
print(f"Nouveaux paramètres : {new_params}")

multi_head_attention_fn = jax.jit(module["multi-head-attention"])
transformer_block = jax.jit(module["transformer-block"])
gpt_model = jax.jit(module["gpt-model"])

####

# --- Simulation d'une structure complète pour 2 couches ---
D = 32  # n_embd
H = 4  # n_heads
L = 2  # n_layers
V = 65  # vocab_size

config = {"n_embd": D, "n_heads": H, "n_layers": L, "vocab_size": V}

# Structure de paramètres "Lisp-friendly"
mock_layer = {
    "ln1": {"w": jnp.ones(D), "b": jnp.zeros(D)},  # Si ta layernorm utilise des poids
    "ln2": {"w": jnp.ones(D), "b": jnp.zeros(D)},
    "attn": {
        "Wq": jnp.ones((D, D)),
        "Wk": jnp.ones((D, D)),
        "Wv": jnp.ones((D, D)),
        "Wo": jnp.ones((D, D)),
    },
    "mlp": {
        "w1": jnp.ones((D, D * 4)),
        "b1": jnp.zeros((D * 4,)),  # Vecteur de taille 4D
        "w2": jnp.ones((D * 4, D)),
        "b2": jnp.zeros((D,)),  # Vecteur de taille D
    },
}

params = {
    "layers": [mock_layer for _ in range(L)],
    "emb": {"token": jnp.ones((V, D)), "pos": jnp.ones((1024, D))},
    "ln_f": {"w": jnp.ones(D), "b": jnp.zeros(D)},
    "head": {"w": jnp.ones((D, V))},
}

# On compile le gros morceau
try:
    # 1. On définit la structure
    Config = namedtuple("Config", ["n_embd", "n_heads", "n_layers", "vocab_size"])
    # 2. On crée l'instance
    config = Config(n_embd=32, n_heads=4, n_layers=2, vocab_size=65)
    # 3. L'appel JIT fonctionnera maintenant avec static_argnums=(2,)
    gpt_fn = jax.jit(module["gpt-model"], static_argnums=(2,))
    input_ids = jnp.zeros((1, 8), dtype=jnp.int32)  # Batch 1, Seq 8
    output = gpt_fn(input_ids, params, config)
    print("GPT Model Output Shape:", output.shape)
except Exception as e:
    print(f"Erreur de compilation GPT: {e}")

mlp_fn = jax.jit(module["mlp"])
# Test du MLP
mlp_p = params["layers"][0]["mlp"]
# Création d'un vecteur latent bidon (Batch=1, Seq=8, Emb=32)
h_in = jnp.ones((1, 8, 32))
mlp_out = mlp_fn(h_in, mlp_p)
print("MLP Output Shape:", mlp_out.shape)  # Doit être (1, 8, 32)

# Entrainement

# Initialisation des moments (on peut le faire via Sheaf ou Python)
m = jax.tree_util.tree_map(jnp.zeros_like, params)
v = jax.tree_util.tree_map(jnp.zeros_like, params)
t = 0

# On récupère la fonction d'entraînement compilée
sheaf_train_step = jax.jit(module["train-step"], static_argnums=(6,))

# Boucle d'entraînement

d_model = 256
n_layers = 2
n_heads = 4
batch_size = 16
block_size = 128
epochs = 300
learning_rate = 0.25e-3

# 1. On récupère la fonction d'entraînement
# static_argnums=6 correspond à l'argument 'config'
train_step_fn = jax.jit(module["train-step"], static_argnums=(6,))

# 2. On prépare des données "Shakespeare" factices
# Batch=2, Seq=16
X_ids = jnp.zeros((2, 16), dtype=jnp.int32)
Y_ids = jnp.zeros((2, 16), dtype=jnp.int32)

# 3. On lance un pas d'entraînement
try:
    params, m, v, t, loss = train_step_fn(params, m, v, t, X_ids, Y_ids, config, 1e-3)
    print(f"Success! Loss initiale: {loss}")
except Exception as e:
    print(f"Erreur lors du train-step: {e}")
