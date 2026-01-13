import jax
import jax.numpy as jnp
from jax import nn

class GeometricDirectionalLoss:
    def __init__(self, predict_fn, variant='hinge', margin=1.0, scale=20.0, distance_gamma=0.5):
        """
        Args:
            predict_fn: Function (params, x) -> logits
            variant: One of ['hinge', 'cosine', 'distance_energy', 'hessian']
            margin: Margin for Hinge loss
            scale: Scaling factor for Softplus/Cosine (alpha)
            distance_gamma: Decay rate for distance weighting
        """
        self.predict_fn = predict_fn
        self.variant = variant
        self.margin = margin
        self.scale = scale
        self.distance_gamma = distance_gamma

    def _compute_directional_data(self, params, x, k, y):
        """
        Computes the directional derivative (tangent) and auxiliary data.
        Uses jax.jvp for O(1) efficiency relative to forward pass.
        """
        # 1. Define the vector v = K - X
        # We stop_gradient because we want to update params, not X or K.
        v = jax.lax.stop_gradient(k - x)
        dist = jnp.linalg.norm(v) + 1e-6
        
        # 2. Define the scalar function for the specific class y
        # We only care about the slope of the *true class* logit
        def class_score_fn(input_x):
            logits = self.predict_fn(params, input_x)
            return logits[y]

        # 3. Compute Value and Directional Derivative (Gradient dot Vector)
        # primals = f(x), tangents = \nabla f(x) \cdot v
        _, dir_derivative = jax.jvp(class_score_fn, (x,), (v,))
        
        return dir_derivative, dist, v

    def _compute_full_gradient_norm(self, params, x, y):
        """Helper for Cosine variant: requires full gradient norm."""
        def class_score_fn(input_x):
            return self.predict_fn(params, input_x)[y]
        grads = jax.grad(class_score_fn)(x)
        return jnp.linalg.norm(grads)

    def _compute_hessian_vector_product(self, params, x, v, y):
        """Helper for Hessian variant: computes v^T * H * v."""
        def gradient_dot_v(input_x):
            # Compute gradient-vector product (directional derivative)
            # This is the first order derivative
            g_fn = jax.grad(lambda s: self.predict_fn(params, s)[y])
            return jnp.dot(g_fn(input_x), v)
        
        # Differentiate the directional derivative along v again
        _, hvp_scalar = jax.jvp(gradient_dot_v, (x,), (v,))
        return hvp_scalar

    def sample_loss(self, params, x, k, y):
        """Computes loss for a single sample (to be vmapped)."""
        
        # Get fundamental geometric data
        # dir_deriv is the "slope" of the model along K-X
        dir_deriv, dist, v = self._compute_directional_data(params, x, k, y)
        
        # --- Variant A: Geometric Hinge (Robust Slope) ---
        # Penalizes if slope > -margin. 
        # Enforces a minimum steepness of descent toward counterfactual.
        if self.variant == 'hinge':
            return nn.relu(dir_deriv + self.margin)

        # --- Variant B: Cosine Alignment (Scale Invariant) ---
        # Penalizes angular misalignment regardless of gradient magnitude.
        # Target: Cosine Sim = -1. Loss is Softplus of alignment error.
        elif self.variant == 'cosine':
            g_norm = self._compute_full_gradient_norm(params, x, y)
            # Cosine = (g. v) / (|g| |v|)
            # dir_deriv is already (g. v)
            cosine_sim = dir_deriv / (g_norm * dist + 1e-7)
            
            # Loss = Softplus( scale * (cosine_sim - target) ) 
            # We want cosine_sim to be -1, so we minimize cosine_sim + 1
            return nn.softplus(self.scale * (cosine_sim + 1.0))

        # --- Variant C: Distance-Weighted Potentials ---
        # Enforces stricter constraints on nearby counterfactuals, looser on distant ones.
        # Uses Softplus energy instead of Hinge to avoid dead zones.
        elif self.variant == 'distance_energy':
            # Energy increases as slope becomes less negative (more positive)
            energy = nn.softplus(self.scale * dir_deriv) 
            
            # Weight decays with distance (Potential Theory)
            weight = 1.0 / (1.0 + self.distance_gamma * dist)
            return weight * energy

        # --- Variant D: Hessian Stabilizer (Second Order) ---
        # Hinge Loss + Penalty on the change of the slope (Curvature).
        # Ensures the decision boundary is locally linear/stable.
        elif self.variant == 'hessian':
            hinge_loss = nn.relu(dir_deriv + self.margin)
            
            # Compute curvature along the path v
            curvature = self._compute_hessian_vector_product(params, x, v, y)
            
            # Penalize absolute curvature
            return hinge_loss + 0.1 * jnp.abs(curvature)
        
        return 0.0

    def batch_loss(self, params, batch):
        """Vectorized computation over the batch."""
        X, Y, K = batch['X'], batch, batch['K']['vector']
        
        # Vmap the sample_loss function over the batch dimension (axis 0)
        # in_axes: (None, 0, 0, 0) corresponds to (params, X, K, Y)
        batch_loss_fn = jax.vmap(self.sample_loss, in_axes=(None, 0, 0, 0))
        
        losses = batch_loss_fn(params, X, K, Y)
        return jnp.mean(losses)