from abc import ABC, abstractmethod 
from dataclasses import dataclass
from typing import Any, Optional, Iterator 
import jax
import jax.numpy as jnp
import equinox as eqx 

from jaxtyping import Array, Float, Int, PyTree

@dataclass
class BaseBatch(ABC):
    pass

@dataclass
class Batch(BaseBatch):
    x: Float[Array, "batch num_points input_dim"]
    y: Float[Array, "batch num_points output_dim"]
    xt: Float[Array, "batch num_target input_dim"]
    yt: Float[Array, "batch num_target output_dim"]
    xc: Float[Array, "batch num_context input_dim"]
    yc: Float[Array, "batch num_context output_dim"]

class GroundTruthPredictor(eqx.Module):
    @abstractmethod 
    def __call__(
        self, 
        xc: Float[Array, "batch num_context input_dim"], 
        yc: Float[Array, "batch num_context output_dim"], 
        xt: Float[Array, "batch num_target input_dim"], 
        yt: Optional[Float[Array, "batch num_target output_dim"]] = None
    ) -> Float[Array, "batch num_target output_dim"]:
        pass

    @abstractmethod
    def sample_outputs(
        self, 
        x: Float[Array, "batch num_points input_dim"], 
    ) -> Float[Array, "batch num_points output_dim"]:
        pass


class DataGenerator(ABC):
    def __init__(
        self,
        *,
        samples_per_epoch: int,
        batch_size: int,
        seed: int = 0, 
        **kwargs,
    ):
        """Base data generator for JAX-based data generation.

        Arguments:
            samples_per_epoch: Number of samples per epoch.
            batch_size: Batch size.
            deterministic: If True, generates same sequence each time
            deterministic_seed: Seed for deterministic generation
        """
        self.samples_per_epoch = samples_per_epoch
        self.batch_size = batch_size
        self.num_batches = samples_per_epoch // batch_size
        self.seed = seed
        self.batches = None
        self.batch_counter = 0
        self.key = jax.random.PRNGKey(seed)

    def __iter__(self):
        """Reset generator state and return iterator."""
        if self.batches is None:
            # Pre-generate all batches for deterministic case
            keys = jax.random.split(self.key, self.num_batches)
            self.batches = [self.generate_batch(key) for key in keys]
        
        return self

    def __next__(self) -> BaseBatch:
        if self.batch_counter >= self.num_batches:
            raise StopIteration

        if self.batches is not None:
            batch = self.batches[self.batch_counter]
        else:
            self.key, key = jax.random.split(self.key)
            batch = self.generate_batch(key)

        self.batch_counter += 1
        return batch

    @abstractmethod
    def generate_batch(self, key: jax.random.PRNGKey) -> BaseBatch:
        """Generate batch of data.
        
        Args:
            key: PRNG key for random number generation
            
        Returns:
            BaseBatch: Generated batch of data
        """
        pass