# debug_directional_derivative.py
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

# -------------------------
# Toy dataset: 2D binary
# -------------------------
X_pos = jnp.array([[2.0, 2.0], [3.0, 2.5], [2.5, 3.0]])  # class 1
X_neg = jnp.array([[-2.0, -2.0], [-3.0, -2.5], [-2.5, -3.0]])  # class 0
X = jnp.vstack([X_pos, X_neg])
Y = jnp.array([1, 1, 1, 0, 0, 0])

# Directions K (pretend these are counterfactual shifts)
K = jnp.array([
    [1.0, 0.0],   # move in +x
    [0.0, 1.0],   # move in +y
    [-1.0, 0.0],  # move in -x
    [0.0, -1.0],  # move in -y
    [1.0, 1.0],
    [-1.0, -1.0]
])


# -------------------------
# Tiny model
# -------------------------
def init_params(key, hidden_dim=4):
    k1, k2, k3 = jax.random.split(key, 3)
    W1 = jax.random.normal(k1, (2, hidden_dim)) * 0.1
    b1 = jnp.zeros((hidden_dim,))
    W2 = jax.random.normal(k2, (hidden_dim, 2)) * 0.1
    b2 = jnp.zeros((2,))
    return {"W1": W1, "b1": b1, "W2": W2, "b2": b2}


def model(params, x):
    h = jnp.tanh(x @ params["W1"] + params["b1"])
    logits = h @ params["W2"] + params["b2"]
    return jax.nn.softmax(logits)


# -------------------------
# Jacobian wrt input
# -------------------------
def predict_wrapper(params, model, x, rng=None):
    return model(params, x)

jac_fn = jax.jacobian(predict_wrapper, argnums=2)  # ∂f/∂x
jac_map = jax.vmap(jac_fn, in_axes=(None, None, 0, None), out_axes=0)

# -------------------------
# Debug directional derivative
# -------------------------
def debug_directional(params, model, X, Y, K):
    rng = jax.random.PRNGKey(0)
    g_y = jac_map(params, model, X, rng)  # (N, num_classes, D)

    # Pick gradient for true class
    g_y_true = jnp.array([g[y] for g, y in zip(g_y, Y)])  # (N,D)

    # Project along K
    directional_derivative = jnp.einsum("nd,nd->n", g_y_true, K)

    # Desired: negative means pointing "towards" class of interest
    desired_sign = -1
    violation = directional_derivative * desired_sign

    print("Directional derivatives:", directional_derivative)
    print("Violation:", violation)

    return g_y_true, directional_derivative


# -------------------------
# Run + Visualize
# -------------------------
if __name__ == "__main__":
    params = init_params(jax.random.PRNGKey(42))
    g_y_true, dd = debug_directional(params, model, X, Y, K)

    # Plot
    plt.figure(figsize=(6, 6))
    plt.scatter(X[:3, 0], X[:3, 1], c="blue", label="Class 1")
    plt.scatter(X[3:, 0], X[3:, 1], c="red", label="Class 0")

    for i, (x, g, k) in enumerate(zip(X, g_y_true, K)):
        plt.arrow(x[0], x[1], g[0], g[1], color="green", head_width=0.1, label="grad" if i == 0 else "")
        plt.arrow(x[0], x[1], k[0], k[1], color="orange", head_width=0.1, label="K" if i == 0 else "")

    plt.legend()
    plt.title("Gradients (green) vs Directions K (orange)")
    plt.axis("equal")
    plt.show()
