from abc import ABC, abstractmethod
import jax
import jax.numpy as jnp 
import equinox as eqx 
import numpyro.distributions as dist

class Likelihood(eqx.Module, ABC):
    """Base class for likelihood functions."""
    @abstractmethod 
    def __call__(self, x: jnp.ndarray) -> dist.Distribution:
        raise NotImplementedError
    
class NormalLikelihood(Likelihood):
    """Fixed-variance normal likelihood."""
    log_noise: jnp.ndarray
    train_noise: bool 

    def __init__(self, noise: float, train_noise: bool = True):
        self.log_noise = jnp.log(jnp.array(noise))
        self.train_noise = train_noise
    
    @property
    def noise(self):
        return jnp.exp(self.log_noise)
    
    def __call__(self, x: jnp.ndarray) -> dist.Normal:
        return dist.Normal(x, self.noise)
    
class HeteroscedasticNormalLikelihood(Likelihood):
    """Variable-variance normal likelihood."""
    min_noise: float 

    def __init__(self, min_noise: float = 0.01):
        self.min_noise = min_noise 

    def __call__(self, x: jnp.ndarray) -> dist.Normal: 
        assert x.shape[-1] % 2 == 0 
        split_idx = x.shape[-1] // 2 
        loc, log_scale = x[..., :split_idx], x[..., split_idx:]
        scale = jax.nn.softplus(log_scale) + self.min_noise
        return dist.Normal(loc, scale)