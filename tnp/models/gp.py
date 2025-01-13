from abc import ABC, abstractmethod 
import jax
import jax.numpy as jnp
from gpjax.kernels.stationary.base import StationaryKernel 
from gpjax.kernels.stationary.utils import squared_distance
from jaxtyping import Float 
from gpjax.typing import Array, ScalarFloat 


class RandomHyperparameterKernel(ABC):
    @abstractmethod 
    def sample_hyperparameters(self, key: jax.random.PRNGKey):
        pass


class ScaleKernel(StationaryKernel, RandomHyperparameterKernel):
    def __init__(self, base_kernel, min_log10_outputscale: float, max_log10_outputscale: float, **kwargs):
        super().__init__(**kwargs)
        self.base_kernel = base_kernel
        self.min_log10_outputscale = min_log10_outputscale
        self.max_log10_outputscale = max_log10_outputscale 


    def sample_hyperparameters(self, key: jax.random.PRNGKey):
        key1, key2 = jax.random.split(key)
        log10_outputscale = jax.random.uniform(
            key1, 
            shape=(), 
            minval=self.min_log10_outputscale, 
            maxval=self.max_log10_outputscale
        )
        self.variance = 10.0 ** log10_outputscale
        self.base_kernel.sample_hyperparameters(key2)


class RBFKernel(RBF, RandomHyperparameterKernel):
    def __init__(self, min_log10_lengthscale: float, max_log10_lengthscale: float, **kwargs):
        super().__init__(**kwargs)
        self.min_log10_lengthscale = min_log10_lengthscale
        self.max_log10_lengthscale = max_log10_lengthscale

    def sample_hyperparameters(self, key: jax.random.PRNGKey):
        shape = self.ard_num_dims if hasattr(self, "ard_num_dims") and self.ard_num_dims is not None else ()
        log10_lengthscale = jax.random.uniform(
            key, 
            shape=shape, 
            minval=self.min_log10_lengthscale, 
            maxval=self.max_log10_lengthscale
        )
        self.lengthscale = 10.0 ** log10_lengthscale
