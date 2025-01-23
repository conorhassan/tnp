from abc import ABC, abstractmethod
import jax
import jax.numpy as jnp 
from flax import nnx
import numpyro.distributions as dist
from jaxtyping import Array, Float

class Likelihood(nnx.Module, ABC):
    """Base class for likelihood functions."""
    @abstractmethod 
    def __call__(self, x: Float[Array, "..."]) -> dist.Distribution:
        raise NotImplementedError
    
class NormalLikelihood(Likelihood):
    """Fixed-variance normal likelihood."""
    def __init__(self, noise: float, train_noise: bool = True):
        super().__init__()
        if train_noise:
            self.log_noise = nnx.Param(jnp.log(jnp.array(noise)))
        else:
            self.log_noise = jnp.log(jnp.array(noise))
    
    @property
    def noise(self):
        return jnp.exp(self.log_noise)
    
    def __call__(self, x: Float[Array, "... dim"]) -> dist.Normal:
        return dist.Normal(x, self.noise)

class HeteroscedasticNormalLikelihood(Likelihood):
    """Variable-variance normal likelihood."""
    def __init__(self, min_noise: float = 0.01, rngs: nnx.Rngs = None):
        super().__init__()
        self.min_noise = min_noise
    
    def __call__(self, x: Float[Array, "... dim"]) -> dist.Normal:
        assert x.shape[-1] % 2 == 0
        split_idx = x.shape[-1] // 2
        loc, log_scale = x[..., :split_idx], x[..., split_idx:]
        scale = jax.nn.softplus(log_scale) + self.min_noise
        return dist.Normal(loc, scale)