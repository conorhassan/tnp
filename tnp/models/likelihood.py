from abc import ABC, abstractmethod
import jax
import jax.numpy as jnp 
import equinox as eqx 
import numpyro.distributions as dist


class Likelihood(eqx.Module, ABC):
    """Base class for likelihood functions. 
    
    All likelihood implementations should inherit from this 
    class and implement the __call__ method.
    """
    @abstractmethod 
    def __call__(self, x: jnp.ndarray) -> dist.Distribution:
        raise NotImplementedError
    
class NormalLikelihood(Likelihood):
    """Fixed-variance normal likelihood.
    
    Attributes: 
        log_noise: Learnable log noise parameter. 
        train_noise: Whether to update noise during training."""
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
    """Variable-variance normal likelihood. 
    
    Attributes: 
        min_noise: Minimum noise level to add
    """
    min_noise: float 

    def __init__(self, min_noise: float = 0.0):
        self.min_noise = min_noise 

    def __call__(self, x: jnp.ndarray) -> dist.Normal: 
        # Check even number of features for mean/variance pairs.
        assert x.shape[-1] % 2 == 0 

        # Split into location and log variance
        split_idx = x.shape[-1] // 2 
        loc, log_var = x[..., :split_idx], x[..., split_idx:]

        # Compute scale 
        scale = jnp.sqrt(jax.nn.softplus(log_var)) + self.min_noise

        return dist.Normal(loc, scale)