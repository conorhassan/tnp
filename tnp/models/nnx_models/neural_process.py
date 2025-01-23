from abc import ABC
import jax
import jax.numpy as jnp
from flax import nnx 
import numpyro.distributions as dist
from jaxtyping import Array, Float
from typing import Tuple, Optional
from einops import pack, unpack, rearrange
from nnx.layers import Identity


class TNPEncoder(nnx.Module):
    def __init__(
        self, 
        transformer_encoder: nnx.Module, 
        xy_encoder: nnx.Module, 
        x_encoder: nnx.Module = Identity(), 
        y_encoder: nnx.Module = Identity()
    ):
        self.transformer_encoder = transformer_encoder
        self.xy_encoder = xy_encoder
        self.x_encoder = x_encoder
        self.y_encoder = y_encoder

    def __call__(
        self, xc: Float[Array, "..."], 
        yc: Float[Array, "..."], 
        xt: Float[Array, "..."], 
        key: jax.random.PRNGKey, 
        enable_dropout: bool = False
    ) -> Float[Array, "num_target latent_dim"]:
        yc, yt = self.preprocess_observations(xt, yc)

        x, ps = pack([xc, xt], "* d")
        x_encoded = self.x_encoder(x)
        xc_encoded, xt_encoded = unpack(x_encoded, ps, "* d")

        y, ps = pack([yc, yt], "* d")
        y_encoded = self.y_encoder(y)
        yc_encoded, yt_encoded = unpack(y_encoded, ps, "* d")

        zc, _ = pack([xc_encoded, yc_encoded], "s *")
        zt, _ = pack([xt_encoded, yt_encoded], "s *")

        zc = self.xy_encoder(zc)
        zt = self.xy_encoder(zt)

        output = self.transformer_encoder(zc, zt, key=key, enable_dropout=enable_dropout)
        return output
    
    def preprocess_observations(
        self, 
        xt: Float[Array, "num_target input_dim"], 
        yc: Float[Array, "num_context output_dim"]
    ) -> Tuple[Float[Array, "num_target output_dim_plus_1"]]:
        yt = jnp.zeros((xt.shape[0], yc.shape[-1]))
        yc = jnp.concatenate([yc, jnp.zeros_like(yc[..., :1])], axis=-1)
        yt = jnp.concatenate([yt, jnp.ones_like(yt[..., :1])], axis=-1)
        return yc, yt


class TNPDecoder(nnx.Module):
    def __init__(self, z_decoder: nnx.Module):
        self.z_decoder = z_decoder

    def __call__(
        self, 
        z: Float[Array, "num_target latent_dim"], 
        xt: Optional[Float[Array, "num_target input_dim"]] = None
    ) -> Float[Array, "num_target output_dim"]:
        if xt is not None:
            num_target = xt.shape[0]
            zt = rearrange(z[-num_target:, :], "t d -> t d")
        else: 
            zt = z
        return self.z_decoder(zt)


class NeuralProcess(nnx.Module, ABC): 
    encoder: nnx.Module 
    decoder: nnx.Module 
    likelihood: Likelihood


class ConditionalNeuralProcess(NeuralProcess):
    def __call__(
        self, 
        xc: Float[Array, "batch num_context input_dim"], 
        yc: Float[Array, "batch num_context output_dim"],
        xt: Float[Array, "batch num_target input_dim"], 
        key: jax.random.PRNGKey, 
        enable_dropout: bool = False
    ) -> dist.Distribution:
        z = self.encoder(xc, yc, xt, key, enable_dropout)
        pred = self.decoder(z, xt)
        return self.likelihood(pred)
    

class TNP(ConditionalNeuralProcess):
    encoder: TNPEncoder
    decoder: TNPDecoder
    likelihood: Likelihood

    def __call__(
        self,
        xc: Float[Array, "batch num_context input_dim"], 
        yc: Float[Array, "batch num_context output_dim"], 
        xt: Float[Array, "batch num_target input_dim"], 
        key: jax.random.PRNGKey, 
        enable_dropout: bool = False
    ) -> dist.Distribution:
        return super().__call__(xc, yc, xt, key, enable_dropout)