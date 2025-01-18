import jax
import jax.numpy as jnp 
import gpjax as gpx 
from gpjax.gps import Prior
from gpjax.mean_functions import Zero
from typing import Tuple

import matplotlib.pyplot as plt 

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
        batch_size: int=64
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

        return xc, yc, xt, yt
    
    

if __name__ == "__main__":
    key = jax.random.PRNGKey(2)
    generator = SimpleGPGenerator()
    xc, xt, nc, nt = generator.sample_inputs(key)
    yc = generator.sample_outputs(xc, key)
    xc, yc, xt, yt = generator.generate_batch(key)

        # Plot a few examples
    plt.figure(figsize=(10, 5))
    for i in range(3):  # Plot first 3 batch elements
        plt.subplot(1, 3, i+1)
        plt.scatter(xc[i, :, 0], yc[i, :, 0], label='context', alpha=0.6)
        plt.scatter(xt[i, :, 0], yt[i, :, 0], label='target', alpha=0.6)
        plt.legend()
    plt.tight_layout()
    plt.show()