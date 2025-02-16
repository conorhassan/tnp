import jax
import jax.numpy as jnp 
import gpjax as gpx 
from gpjax.gps import Prior
from gpjax.mean_functions import Zero
from typing import Tuple


class SimpleGPGenerator:

    def __init__(
        self, 
        kernel=gpx.kernels.RBF(), 
        noise_std: float=0.1, 
        context_range: Tuple[float, float]=((-3., 3.),), 
        target_range: Tuple[float, float]=((-3., 3.),), 
        dim: int=1, 
        min_nc: int=3, 
        max_nc: int=50, 
        min_nt: int=3, 
        max_nt: int=50,
        batch_size: int=32
    ):
        self.kernel = kernel
        self.noise_std = noise_std 
        self.prior = Prior(mean_function=Zero(), kernel=self.kernel)
        self.context_range = jnp.array(context_range)
        self.target_range = jnp.array(target_range)
        self.dim = dim 
        self.min_nc, self.max_nc = min_nc, max_nc 
        self.min_nt, self.max_nt = min_nt, max_nt 
        self.batch_size = batch_size 

    def sample_inputs(
        self, 
        key: jax.random.PRNGKey
    ):
        k1, k2, k3, k4 = jax.random.split(key, 4)

        # sample number of context and target points
        nc = jax.random.randint(k1, (), self.min_nc, self.max_nc + 1)
        nt = jax.random.randint(k2, (), self.min_nt, self.max_nt + 1)

        # Sample context points
        xc = (
            jax.random.uniform(k3, (self.batch_size, nc, self.dim))
            * (self.context_range[:, 1] - self.context_range[:, 0])
            + self.context_range[:, 0]
        )
        
        # Sample target points
        xt = (
            jax.random.uniform(k4, (self.batch_size, nt, self.dim))
            * (self.target_range[:, 1] - self.target_range[:, 0])
            + self.target_range[:, 0]
        )

        return xc, xt, nc, nt
    
    def sample_outputs(
        self, 
        x: jnp.ndarray, 
        key: jax.random.PRNGKey
    ):

        k1, k2 = jax.random.split(key, 2)

        # sample single GP prior 
        def sample_(x_, key):
            rv = self.prior.predict(x_)
            f = rv.sample(key, sample_shape=(1,))
            return f.squeeze(0)
        
        # vmap over batch dimension 
        f = jax.vmap(sample_)(x, jax.random.split(k1, x.shape[0]))

        # add observation noise 
        y = f + jax.random.normal(k2, f.shape) * self.noise_std
        return y[..., None]
    
    def generate_batch(self, key: jax.random.PRNGKey):

        k1, k2 = jax.random.split(key, 2)

        # sample inputs (context and target)
        xc, xt, nc, nt = self.sample_inputs(k1)

        # combine for joint sampling 
        x = jnp.concatenate([xc, xt], axis=1)

        # sample outputs 
        y = self.sample_outputs(x, k2)

        # split back into context and target 
        yc, yt = y[:, :nc], y[:, nc:]

        # Create a mask with consistent dimensions
        num_heads = 4  # Adjust this to match your model's configuration
        context_mask = jnp.ones((self.batch_size, num_heads, nc, nc))
        target_to_context_mask = jnp.ones((self.batch_size, num_heads, nt, nc))
        context_to_target_mask = jnp.zeros((self.batch_size, num_heads, nc, nt))
        target_mask = jnp.zeros((self.batch_size, num_heads, nt, nt))

        # Concatenate masks along the correct axis
        mask = jnp.concatenate([
            jnp.concatenate([context_mask, context_to_target_mask], axis=3),
            jnp.concatenate([target_to_context_mask, target_mask], axis=3)
        ], axis=2)
 
        return xc, yc, xt, yt, mask 
