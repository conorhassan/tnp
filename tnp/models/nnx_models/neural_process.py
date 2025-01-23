from abc import ABC
import jax
import jax.numpy as jnp
from flax import nnx 
import numpyro.distributions as dist
from jaxtyping import Array, Float
from typing import Tuple, Optional
from einops import pack, unpack, rearrange

from nnx_models.layers import Identity
from nnx_models.likelihood import Likelihood

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
        self, 
        xc: Float[Array, "batch num_context input_dim"], 
        yc: Float[Array, "batch num_context input_dim"], 
        xt: Float[Array, "batch num_target output_dim"],
    ) -> Float[Array, "batch num_target latent_dim"]:
        yc, yt = self.preprocess_observations(xt, yc)

        x, ps = pack([xc, xt], "batch * features")
        x_encoded = self.x_encoder(x)
        xc_encoded, xt_encoded = unpack(x_encoded, ps, "batch * features") 

        y, ps = pack([yc, yt], "batch * features")
        y_encoded = self.y_encoder(y)
        yc_encoded, yt_encoded = unpack(y_encoded, ps, "batch * features")

        zc, _ = pack([xc_encoded, yc_encoded], "batch seq *")
        zt, _ = pack([xt_encoded, yt_encoded], "batch seq *")

        zc = self.xy_encoder(zc)
        zt = self.xy_encoder(zt)

        return self.transformer_encoder(zc, zt)
    
    def preprocess_observations(
        self, 
        xt: Float[Array, "batch num_target input_dim"], 
        yc: Float[Array, "batch num_context output_dim"]
    ) -> Tuple[Float[Array, "batch num_target output_dim_plus_1"]]:
        # create zero tensor for the target points, since they are unknown 
        yt = jnp.zeros((*xt.shape[:-1], yc.shape[-1]))

        # add flag dimensions
        yc = jnp.concatenate([yc, jnp.zeros_like(yc[..., :1])], axis=-1) # add a zero flag 
        yt = jnp.concatenate([yt, jnp.ones_like(yt[..., :1])], axis=-1)

        return yc, yt


class TNPDecoder(nnx.Module):
    def __init__(self, z_decoder: nnx.Module):
        self.z_decoder = z_decoder

    def __call__(
        self, 
        z: Float[Array, "batch num_target latent_dim"], 
        xt: Optional[Float[Array, "batch num_target input_dim"]] = None
    ) -> Float[Array, "batch num_target output_dim"]:
        if xt is not None:
            num_target = xt.shape[0]
            zt = rearrange(z[-num_target:, :], "b t d -> b t d")
        else: 
            zt = z
        return self.z_decoder(zt)


class NeuralProcess(nnx.Module, ABC): 
    def __init__(
        self, 
        encoder: nnx.Module, 
        decoder: nnx.Module, 
        likelihood: Likelihood
    ):
        self.encoder = encoder
        self.decoder = decoder
        self.likelihood = likelihood


class ConditionalNeuralProcess(NeuralProcess):
    def __init__(
        self, 
        encoder: nnx.Module, 
        decoder: nnx.Module, 
        likelihood: Likelihood
    ):
        super().__init__(encoder, decoder, likelihood)

    def __call__(
        self, 
        xc: Float[Array, "batch num_context input_dim"], 
        yc: Float[Array, "batch num_context output_dim"],
        xt: Float[Array, "batch num_target input_dim"], 
    ) -> dist.Distribution:
        z = self.encoder(xc, yc, xt)
        pred = self.decoder(z, xt)
        return self.likelihood(pred)
    

class TNP(ConditionalNeuralProcess):
    def __init__(
        self, 
        transformer_encoder: nnx.Module, 
        xy_encoder: nnx.Module, 
        z_decoder: nnx.Module, 
        likelihood: Likelihood
    ):
        super().__init__(TNPEncoder(transformer_encoder, xy_encoder), TNPDecoder(z_decoder), likelihood)

    def __call__(
        self,
        xc: Float[Array, "batch num_context input_dim"], 
        yc: Float[Array, "batch num_context output_dim"], 
        xt: Float[Array, "batch num_target input_dim"], 
    ) -> dist.Distribution:
        return super().__call__(xc, yc, xt)