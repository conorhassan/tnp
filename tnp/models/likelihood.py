from abc import ABC, abstractmethod
import jax
import jax.numpy as jnp 
import equinox as eqx 
import tensorflow_probability.substrates.jax as tfp
tfd = tfp.distributions

class Likelihood(eqx.Module, ABC):
    """Base class for likelihood functions."""
    @abstractmethod 
    def __call__(self, x: jnp.ndarray) -> tfd.Distribution:
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
    
    def __call__(self, x: jnp.ndarray) -> tfd.Normal:
        return tfd.Normal(loc=x, scale=self.noise)
    
class HeteroscedasticNormalLikelihood(Likelihood):
    """Variable-variance normal likelihood."""
    min_noise: float 

    def __init__(self, min_noise: float = 0.0):
        self.min_noise = min_noise 

    def __call__(self, x: jnp.ndarray) -> tfd.Normal: 
        assert x.shape[-1] % 2 == 0 
        split_idx = x.shape[-1] // 2 
        loc, log_var = x[..., :split_idx], x[..., split_idx:]
        scale = jnp.sqrt(jax.nn.softplus(log_var)) + self.min_noise
        return tfd.Normal(loc=loc, scale=scale)
