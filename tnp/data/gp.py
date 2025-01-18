import jax
import jax.numpy as jnp
from abc import ABC
import einops
from typing import Optional, Union, Dict, Tuple, Iterable

from jaxtyping import Float, Array


from gpjax.dataset import Dataset
from gpjax.gps import ConjugatePosterior, Prior
from gpjax.likelihoods import Gaussian

from .base import GroundTruthPredictor
from .synthetic import SyntheticGeneratorUniformInput, SyntheticBatch
from ..models.gp import RandomHyperparameterKernel


"""
So we have the "models" and all the "generators" for generating different types of GP priors
"""


class GPRegressionModel(ConjugatePosterior):

    def __init__(
        self, 
        prior: Prior,
        likelihood: Gaussian
    ):
        super().__init__(prior, likelihood)

    def forward(self, x: jnp.ndarray, train_data: Optional[Dataset] = None):
        if train_data is None:
            return self.prior(x)
        return self(x, train_data)


class GPGroundTruthPredictor(GroundTruthPredictor):
    prior: Prior
    likelihood: Gaussian
    result_cache: Optional[Dict[str, jnp.ndarray]]

    def __init__(
        self, 
        prior: Prior, 
        likelihood: Gaussian
    ):
        super().__init__()
        self.prior = prior
        self.likelihood = likelihood 
        self.result_cache = None


    def __call__(
        self, 
        xc: jnp.ndarray,
        yc: jnp.ndarray, 
        xt: jnp.ndarray,
        yt: Optional[jnp.ndarray] = None,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, Optional[jnp.ndarray]]:
        
        if yt is not None and self._result_cache is not None:
            return (
                self._result_cache["mean"],
                self._result_cache["std"],
                self._result_cache["gt_loglik"],
            )
        
        mean_list = []
        std_list = []
        gt_loglik_list = []

        for idx, (xc_, yc_, xt_) in enumerate(zip(xc, yc, xt)):
            train_data = Dataset(X=xc_, y=yc_[..., 0])
            gp_model = GPRegressionModel(prior=self.prior, likelihood=self.likelihood)
            dist = gp_model.forward(xt_, train_data)
            if yt is not None:
                gt_loglik = dist.log_prob(yt[idx, ..., 0])
                gt_loglik_list.append(gt_loglik)

            mean_list.append(dist.mean())
            std_list.append(jnp.sqrt(dist.variance()))

        mean = jnp.stack(mean_list, axis=0)
        std = jnp.stack(std_list, axis=0)
        gt_loglik = jnp.stack(gt_loglik_list, axis=0) if gt_loglik_list else None

        return mean, std, gt_loglik

    def sample_outputs(
        self, x: jnp.ndarray, key: jax.random.PRNGKey, sample_shape: tuple = ()
    ) -> jnp.ndarray:
            """sample from prior"""
            gp_model = GPRegressionModel(prior=self.prior, likelihood=self.likelihood)
            dist = gp_model.forward(x)
            f = dist.sample(key, sample_shape)
            y = self.likelihood(f).sample(key)
            return y[..., None]
    

class GPGenerator(ABC):
    def __init__(
        self, 
        *, 
        kernel: Union[
            RandomHyperparameterKernel,
            Tuple[RandomHyperparameterKernel, ...],
        ],
        noise_std: float, 
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.kernel = kernel
        if isinstance(self.kernel, Iterable):
            self.kernel = tuple(self.kernel)

        self.noise_std = noise_std

    def set_up_gp(self, key: jax.random.PRNGKey) -> GPGroundTruthPredictor:
        key1, key2 = jax.random.split(key)

        # sample from different GP priors
        if isinstance(self.kernel, tuple):
            kernel = self.kernel[jax.random.randint(key1, (), 0, len(self.kernel))]
        else:
            kernel = self.kernel

        kernel.sample_hyperparameters(key2)

        likelihood = Gaussian(num_datapoints=1) # update when data is provided
        likelihood.obs_stddev = self.noise_std

        return GPGroundTruthPredictor(prior=kernel, likelihood=likelihood)


class RandomScaleGPGenerator(GPGenerator, SyntheticGeneratorUniformInput):
    pass


class RandomScaleGPGeneratorSameInputs(RandomScaleGPGenerator):
    def sample_inputs(
        self,
        nc: int,
        nt: Optional[int],
        key: jax.random.PRNGKey,
        batch_shape: Tuple[int, ...] = (),
    ) -> Float[Array, "batch nc_plus_nt dim"]:
        # First get base inputs
        x = super().sample_inputs(nc=nc, batch_shape=batch_shape, nt=nt, key=key)

    def sample_outputs(
        self,
        x: Float[Array, "batch nc_plus_nt dim"],  # Added type hint
        key: jax.random.PRNGKey
    ) -> Tuple[Float[Array, "batch nc_plus_nt 1"], Optional[GroundTruthPredictor]]:  # Fixed return type
        gt_pred = self.set_up_gp(key)
        sample_shape = x.shape[:-2]
        return gt_pred.sample_outputs(x[0], key, sample_shape=sample_shape), gt_pred

    def generate_batch(
        self,
        key: jax.random.PRNGKey,
    ) -> SyntheticBatch:  # Added return type and removed batch_size param
        k1, k2, k3 = jax.random.split(key, 3)
        nc = jax.random.randint(k1, (), self.min_nc, self.max_nc + 1)
        nt = jax.random.randint(k2, (), self.min_nt, self.max_nt + 1)
        
        batch = self.sample_batch(nc=nc, nt=nt, batch_shape=(self.batch_size,), key=k3)
        return batch

