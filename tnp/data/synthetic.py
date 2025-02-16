from abc import ABC, abstractmethod 
from dataclasses import dataclass
from typing import Optional, Tuple

import jax
import jax.numpy as jnp 

from jaxtyping import Array, Float 

from .base import Batch, DataGenerator, GroundTruthPredictor

@dataclass
class SyntheticBatch(Batch):
    gt_mean: Optional[Float[Array, "..."]] = None
    gt_field: Optional[Float[Array, "..."]] = None 
    gt_loglik: Optional[Float[Array, "..."]] = None 
    gt_pred: Optional[GroundTruthPredictor] = None 

class SyntheticGenerator(DataGenerator, ABC):
    def __init__(
        self, 
        *, 
        dim: int, 
        min_nc: int,
        max_nc: int, 
        min_nt: int, 
        max_nt: int, 
        **kwargs
    ): 
        super().__init__(**kwargs)

        self.dim = dim 
        self.min_nc = min_nc 
        self.max_nc = max_nc 
        self.min_nt = min_nt 
        self.max_nt = max_nt 

    def generate_batch(self, key: jax.random.PRNGKey) -> Batch: 
        # split key for nc and nt sampling 
        k1, k2 = jax.random.split(key)

        # sample number of context and target points
        nc = jax.random.randint(k1, (), self.min_nc, self.max_nc + 1)
        nt = jax.random.randint(k2, (), self.min_nt, self.max_nt + 1)

        return self.sample_batch(
            nc=nc, 
            nt=nt, 
            batch_shape=(self.batch_size,), 
            key=key
        )
    
    def sample_batch(
        self, 
        nc: int, 
        nt: int, 
        batch_shape: Tuple[int, ...], 
        key: jax.random.PRNGKey
    ) -> SyntheticBatch:
        k1, k2 = jax.random.split(key)
        x = self.sample_inputs(nc, nt=nt, batch_shape=batch_shape, key=k1)
        y, gt_pred = self.sample_outputs(x=x, key=k2)

        xc = x[:, :nc, :]
        yc = y[:, :nc, :]
        xt = x[:, nc:, :]
        yt = y[:, nc:, :]

        return SyntheticBatch(
            x=x, 
            y=y, 
            xc=xc, 
            yc=yc, 
            xt=xt, 
            yt=yt, 
            gt_pred=gt_pred
        )
    
    @abstractmethod 
    def sample_inputs(
        self,
        nc: int, 
        batch_shape: Tuple[int, ...], 
        nt: Optional[int], 
        key: jax.random.PRNGKey
    ) -> Float[Array, "batch nc_plus_nt dim"]:
        pass

    @abstractmethod 
    def sample_outputs(
        self, 
        x: Float[Array, "batch nc_plus_nt dim"], 
        key: jax.random.PRNGKey
    ) -> Tuple[Float[Array, "batch nc_plus_nt 1"], Optional[GroundTruthPredictor]]: 
        pass

class SyntheticGeneratorUniformInput(SyntheticGenerator):
    def __init__(
        self,
        *,
        context_range: Tuple[Tuple[float, float], ...],
        target_range: Tuple[Tuple[float, float], ...],
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.context_range = jnp.array(context_range, dtype=float)
        self.target_range = jnp.array(target_range, dtype=float)

    def sample_inputs(
        self,
        nc: int,
        batch_shape: Tuple[int, ...],
        nt: Optional[int],
        key: jax.random.PRNGKey,
    ) -> Float[Array, "batch nc_plus_nt dim"]:
        k1, k2 = jax.random.split(key)
        
        xc = (
            jax.random.uniform(k1, (*batch_shape, nc, self.dim))
            * (self.context_range[:, 1] - self.context_range[:, 0])
            + self.context_range[:, 0]
        )

        if nt is not None:
            xt = (
                jax.random.uniform(k2, (*batch_shape, nt, self.dim))
                * (self.target_range[:, 1] - self.target_range[:, 0])
                + self.target_range[:, 0]
            )
            return jnp.concatenate([xc, xt], axis=1)

        return xc