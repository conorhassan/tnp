import jax
import jax.numpy as jnp 
import gpjax as gpx 
from gpjax.gps import Prior
from gpjax.mean_functions import Zero
from typing import Tuple, Any, Callable 
from dataclasses import dataclass

from tnp.data.simple_gp import SimpleGPGenerator

@dataclass
class OptimizationFunction:
    sampler: SimpleGPGenerator
    maximize: bool
    x_range: jnp.ndarray 
    x_opt: jnp.ndarray
    lengthscale: jnp.ndarray
    sigma_f: jnp.ndarray
    f_offset: jnp.ndarray

    def __call__(self, x: jnp.ndarray, key: jax.random.PRNGKey) -> jnp.ndarray:
        
