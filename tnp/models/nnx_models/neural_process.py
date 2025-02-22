from abc import ABC
import jax
import jax.numpy as jnp
from flax import nnx 
import numpyro.distributions as dist
from jaxtyping import Array, Float
from typing import Tuple, Optional, List
from einops import pack, unpack, rearrange

from tnp.models.nnx_models.layers import Identity, TNPTransformer, MLP
from tnp.models.nnx_models.likelihood import Likelihood, HeteroscedasticNormalLikelihood


class NonLinearEmbedder(nnx.Module):
    def __init__(
        self, 
        xy_embedder: nnx.Module, 
        x_embedder: nnx.Module = Identity(), 
        y_embedder: nnx.Module = Identity()
    ):
        self.xy_embedder = xy_embedder
        self.x_embedder = x_embedder
        self.y_embedder = y_embedder

    def __call__(
        self, 
        xc: Float[Array, "batch num_context input_dim"], 
        yc: Float[Array, "batch num_context input_dim"], 
        xt: Float[Array, "batch num_target output_dim"]
    ) -> Float[Array, "batch num_target latent_dim"]:
        yc, yt = self.preprocess_observations(xt, yc)

        x, ps = pack([xc, xt], "batch * features")
        x_encoded = self.x_embedder(x)
        xc_encoded, xt_encoded = unpack(x_encoded, ps, "batch * features") 

        y, ps = pack([yc, yt], "batch * features")
        y_encoded = self.y_embedder(y)
        yc_encoded, yt_encoded = unpack(y_encoded, ps, "batch * features")

        zc, _ = pack([xc_encoded, yc_encoded], "batch seq *")
        zt, _ = pack([xt_encoded, yt_encoded], "batch seq *")

        zc = self.xy_embedder(zc)
        zt = self.xy_embedder(zt)

        return zc, zt
    
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

# class TNPEncoder(nnx.Module):
#     def __init__(
#         self, 
#         transformer_encoder: nnx.Module, 
#         xy_encoder: nnx.Module, 
#         x_encoder: nnx.Module = Identity(), 
#         y_encoder: nnx.Module = Identity()
#     ):
#         self.transformer_encoder = transformer_encoder
#         self.xy_encoder = xy_encoder
#         self.x_encoder = x_encoder
#         self.y_encoder = y_encoder

#     def __call__(
#         self, 
#         xc: Float[Array, "batch num_context input_dim"], 
#         yc: Float[Array, "batch num_context input_dim"], 
#         xt: Float[Array, "batch num_target output_dim"],
#         mask: Float[Array, "batch seq seq"]
#     ) -> Float[Array, "batch num_target latent_dim"]:
#         yc, yt = self.preprocess_observations(xt, yc)

#         x, ps = pack([xc, xt], "batch * features")
#         x_encoded = self.x_encoder(x)
#         xc_encoded, xt_encoded = unpack(x_encoded, ps, "batch * features") 

#         y, ps = pack([yc, yt], "batch * features")
#         y_encoded = self.y_encoder(y)
#         yc_encoded, yt_encoded = unpack(y_encoded, ps, "batch * features")

#         zc, _ = pack([xc_encoded, yc_encoded], "batch seq *")
#         zt, _ = pack([xt_encoded, yt_encoded], "batch seq *")

#         zc = self.xy_encoder(zc)
#         zt = self.xy_encoder(zt)

#         zt = self.transformer_encoder(zc, zt, mask)
#         return zt
    
#     def preprocess_observations(
#         self, 
#         xt: Float[Array, "batch num_target input_dim"], 
#         yc: Float[Array, "batch num_context output_dim"]
#     ) -> Tuple[Float[Array, "batch num_target output_dim_plus_1"]]:
#         # create zero tensor for the target points, since they are unknown 
#         yt = jnp.zeros((*xt.shape[:-1], yc.shape[-1]))

#         # add flag dimensions
#         yc = jnp.concatenate([yc, jnp.zeros_like(yc[..., :1])], axis=-1) # add a zero flag 
#         yt = jnp.concatenate([yt, jnp.ones_like(yt[..., :1])], axis=-1)

#         return yc, yt
    

class TNPDecoder(nnx.Module):
    def __init__(self, z_decoder: nnx.Module):
        self.z_decoder = z_decoder

    def __call__(
        self, 
        z: Float[Array, "batch num_target latent_dim"], 
        xt: Optional[Float[Array, "batch num_target input_dim"]] = None
    ) -> Float[Array, "batch num_target output_dim"]:
        if xt is not None:
            # Filter the latent representation based on the number of target points
            num_target = xt.shape[1]  # Ensure you use the correct dimension for batch processing
            zt = rearrange(z[:, -num_target:, :], "b t d -> b t d")
        else: 
            zt = z
        return self.z_decoder(zt) 


class PredictionMap(nnx.Module, ABC): 
    def __init__(
        self, 
        embedder: nnx.Module,
        encoder: nnx.Module, 
        decoder: nnx.Module, 
        likelihood: Likelihood
    ):
        self.embedder = embedder
        self.encoder = encoder
        self.decoder = decoder
        self.likelihood = likelihood


class DiagonalPredictionMap(PredictionMap):
    def __init__(
        self, 
        embedder: nnx.Module,
        encoder: nnx.Module, 
        decoder: nnx.Module, 
        likelihood: Likelihood
    ):
        super().__init__(embedder, encoder, decoder, likelihood)

    def __call__(
        self, 
        xc: Float[Array, "batch num_context input_dim"], 
        yc: Float[Array, "batch num_context output_dim"],
        xt: Float[Array, "batch num_target input_dim"], 
        mask: Float[Array, "batch seq seq"]
    ) -> dist.Distribution:
        emb = self.embedder(xc, yc, xt)
        z = self.encoder(*emb, mask)
        pred = self.decoder(z, xt)
        return self.likelihood(pred)
    

class TransformerNeuralProcess(DiagonalPredictionMap):
    def __init__(
        self, 
        input_dim: int,
        output_dim: int,
        latent_dim: int,
        hidden_dim: int,
        num_heads: int,
        dropout_rate: float,
        rngs: nnx.Rngs
    ):
        # define the embedding function
        embedder = NonLinearEmbedder(
            xy_embedder=MLP(in_dim=input_dim + output_dim, out_dim=latent_dim, rngs=rngs),
        )

        # Define the encoder using a transformer
        encoder = TNPTransformer(
            input_dim=latent_dim, 
            hidden_dim=hidden_dim, 
            num_heads=num_heads, 
            dropout_rate=dropout_rate, 
            rngs=rngs
        )

        # Define the decoder
        decoder = TNPDecoder(
            z_decoder=MLP(in_dim=latent_dim, out_dim=output_dim, rngs=rngs)
        )

        # Define the likelihood
        likelihood = HeteroscedasticNormalLikelihood(min_noise=1e-3)

        # Initialize the DiagonalPredictionMap with these components
        super().__init__(embedder, encoder, decoder, likelihood)

    def __call__(
        self,
        xc: Float[Array, "batch num_context input_dim"], 
        yc: Float[Array, "batch num_context output_dim"], 
        xt: Float[Array, "batch num_target input_dim"], 
        mask: Float[Array, "batch seq seq"]
    ) -> dist.Distribution:
        return super().__call__(xc, yc, xt, mask)
    

class AmortizedConditioningEngine(DiagonalPredictionMap):
    pass


# # # Embeddings for the "Amortized Conditioning Engine"
# class EmbedderMarker(nnx.Module):
#     def __init__(
#         self,
#         dim_xc: int,
#         dim_yc: int,
#         num_latent: int,
#         dim_hid: int,
#         dim_out: int,
#         emb_depth: int,
#         name: Optional[str] = None,
#         pos_emb_init: bool = False,
#         use_skipcon_mlp: bool = False,
#         discrete_index: Optional[List[int]] = None,
#     ):
#         self.embedder_marker = nnx.Embed(2 + num_latent, dim_out)
#         self.discrete_index = discrete_index

#         if discrete_index:
#             self.embedder_discrete = nnx.Embed(len(discrete_index), dim_out)

#         if use_skipcon_mlp:
#             self.embedderx = build_mlp_with_linear_skipcon(
#                 dim_xc - 1, dim_hid, dim_out, emb_depth
#             )
#             self.embedderyc = build_mlp_with_linear_skipcon(
#                 dim_yc, dim_hid, dim_out, emb_depth
#             )
#         else:
#             self.embedderx = build_mlp(dim_xc - 1, dim_hid, dim_out, emb_depth)
#             self.embedderyc = build_mlp(dim_yc, dim_hid, dim_out, emb_depth)

#         self.name = name

#         if pos_emb_init:
#             self.embedder_marker.embedding = positional_encoding_init(
#                 2 + num_latent, dim_out, 2 + num_latent
#             )

#     def __call__(self, batch):
#         xc = batch['xc'][:, :, 1:]
#         xce = batch['xc'][:, :, :1]
#         yc = batch['yc']

#         xt = batch['xt'][:, :, 1:]
#         xte = batch['xt'][:, :, :1]
#         yt = batch['yt'][:, :, -1:]

#         # Create the mask
#         if self.discrete_index:
#             discrete_values = jnp.array(self.discrete_index)
#             mask_discrete = jnp.isin(yc, discrete_values).astype(jnp.float32)
#             inverse_mask_discrete = 1 - mask_discrete
#         else:
#             mask_discrete = jnp.zeros_like(yc).astype(jnp.float32)

#         mask_context_x = (xce.astype(jnp.int32) == 1).astype(jnp.float32)

#         if self.discrete_index:
#             context_embedding = (
#                 self.embedderx(xc) * mask_context_x +
#                 self.embedderyc(yc) * inverse_mask_discrete +
#                 self.embedder_discrete((yc * mask_discrete)[:, :, 0].astype(jnp.int32)) * mask_discrete
#             )
#         else:
#             context_embedding = (
#                 self.embedderx(xc) * mask_context_x +
#                 self.embedderyc(yc)
#             )

#         context_embedding += self.embedder_marker(xce[:, :, 0].astype(jnp.int32))

#         # Add xt and marker_? embeddings
#         mask_target_x = (xte.astype(jnp.int32) == 1).astype(jnp.float32)

#         target_embedding = (
#             self.embedderx(xt) * mask_target_x +
#             self.embedder_marker(jnp.zeros_like(yt).astype(jnp.int32)[:, :, 0])
#         )

#         target_embedding += self.embedder_marker(xte[:, :, 0].astype(jnp.int32))

#         res = jnp.concatenate([context_embedding, target_embedding], axis=1)
#         return res

# class EmbedderMarkerPrior(nnx.Module):
#     def __init__(
#         self,
#         dim_xc: int,
#         dim_yc: int,
#         num_latent: int,
#         dim_hid: int,
#         dim_out: int,
#         emb_depth: int,
#         num_bins: int,
#         name: Optional[str] = None,
#         pos_emb_init: bool = False,
#         use_skipcon_mlp: bool = False,
#         discrete_index: Optional[List[int]] = None,
#     ):
#         self.embedder_marker = nnx.Embed(2 + num_latent, dim_out)
#         self.discrete_index = discrete_index

#         if discrete_index:
#             self.embedder_discrete = nnx.Embed(len(discrete_index), dim_out)

#         if use_skipcon_mlp:
#             self.embedderx = build_mlp_with_linear_skipcon(
#                 dim_xc - 1, dim_hid, dim_out, emb_depth
#             )
#             self.embedderyc = build_mlp_with_linear_skipcon(
#                 dim_yc, dim_hid, dim_out, emb_depth
#             )
#             self.embedderbin = build_mlp_with_linear_skipcon(
#                 num_bins, dim_hid, dim_out, emb_depth
#             )
#         else:
#             self.embedderx = build_mlp(dim_xc - 1, dim_hid, dim_out, emb_depth)
#             self.embedderyc = build_mlp(dim_yc, dim_hid, dim_out, emb_depth)
#             self.embedderbin = build_mlp(num_bins, dim_hid, dim_out, emb_depth)

#         self.name = name

#         if pos_emb_init:
#             self.embedder_marker.embedding = positional_encoding_init(
#                 2 + num_latent, dim_out, 2 + num_latent
#             )

#     def __call__(self, batch):
#         xc = batch['xc'][:, :, 1:]
#         xce = batch['xc'][:, :, :1]
#         yc = batch['yc']

#         xt = batch['xt'][:, :, 1:]
#         xte = batch['xt'][:, :, :1]
#         yt = batch['yt'][:, :, -1:]

#         # Create the mask
#         if self.discrete_index:
#             discrete_values = jnp.array(self.discrete_index)
#             mask_discrete = jnp.isin(yc, discrete_values).astype(jnp.float32)
#             inverse_mask_discrete = 1 - mask_discrete
#         else:
#             mask_discrete = jnp.zeros_like(yc).astype(jnp.float32)

#         mask_context_x = (xce.astype(jnp.int32) == 1).astype(jnp.float32)
#         bin_weights_mask = batch['bin_weights_mask']
#         inverse_bin_weight_mask = 1 - bin_weights_mask

#         if self.discrete_index:
#             context_embedding = (
#                 self.embedderx(xc) * mask_context_x +
#                 self.embedderyc(yc) * inverse_bin_weight_mask * inverse_mask_discrete +
#                 self.embedder_discrete((yc * inverse_bin_weight_mask * mask_discrete)[:, :, 0].astype(jnp.int32)) * mask_discrete +
#                 self.embedderbin(batch['latent_bin_weights']) * bin_weights_mask
#             )
#         else:
#             context_embedding = (
#                 self.embedderx(xc) * mask_context_x +
#                 self.embedderyc(yc) * inverse_bin_weight_mask +
#                 self.embedderbin(batch['latent_bin_weights']) * bin_weights_mask
#             )

#         context_embedding += self.embedder_marker(xce[:, :, 0].astype(jnp.int32))

#         # Add xt and marker_? embeddings
#         mask_target_x = (xte.astype(jnp.int32) == 1).astype(jnp.float32)

#         target_embedding = (
#             self.embedderx(xt) * mask_target_x +
#             self.embedder_marker(jnp.zeros_like(yt).astype(jnp.int32)[:, :, 0])
#         )

#         target_embedding += self.embedder_marker(xte[:, :, 0].astype(jnp.int32))

#         res = jnp.concatenate([context_embedding, target_embedding], axis=1)
#         return res

# class EmbedderMarkerPriorInjectionBin(nnx.Module):
#     def __init__(
#         self,
#         dim_xc: int,
#         dim_yc: int,
#         num_latent: int,
#         num_bins: int,
#         dim_hid: int,
#         dim_out: int,
#         emb_depth: int,
#         name: Optional[str] = None,
#         pos_emb_init: bool = False,
#         use_skipcon_mlp: bool = False,
#     ):
#         self.embedder_marker = nnx.Embed(2 + num_latent, dim_out)

#         if use_skipcon_mlp:
#             self.embedderx = build_mlp_with_linear_skipcon(
#                 dim_xc - 1, dim_hid, dim_out, emb_depth
#             )
#             self.embedderyc = build_mlp_with_linear_skipcon(
#                 dim_yc, dim_hid, dim_out, emb_depth
#             )
#             self.embedderbin = build_mlp_with_linear_skipcon(
#                 num_bins, dim_hid, dim_out, emb_depth
#             )
#         else:
#             self.embedderx = build_mlp(dim_xc - 1, dim_hid, dim_out, emb_depth)
#             self.embedderyc = build_mlp(dim_yc, dim_hid, dim_out, emb_depth)
#             self.embedderbin = build_mlp(num_bins, dim_hid, dim_out, emb_depth)

#         self.name = name

#         if pos_emb_init:
#             self.embedder_marker.embedding = positional_encoding_init(
#                 2 + num_latent, dim_out, 2 + num_latent
#             )

#     def __call__(self, batch):
#         # Data embedding
#         embedding_xc_data = self.embedderx(batch['xc_data'][:, :, 1:])
#         embedding_yc_data = self.embedderyc(batch['yc_data'])

#         # Known latent embedding
#         embedding_yc_known = self.embedderyc(batch['yc_latent_known'])

#         # Unknown latent embedding
#         embedding_yc_unknown = self.embedderbin(batch['bins_latent_unknown'])

#         # Concatenate latent embedding
#         embedding_yc_latent = jnp.concatenate([embedding_yc_known, embedding_yc_unknown], axis=1)

#         embedding_data = embedding_xc_data + embedding_yc_data
#         context_embedding = jnp.concatenate([embedding_data, embedding_yc_latent], axis=1)

#         # Add marker_c embedding
#         context_embedding += self.embedder_marker(batch['xc'][:, :, 0].astype(jnp.int32))

#         # Add xt and marker_? embeddings
#         mask_t_latent = (batch['xt'][:, :, :1].astype(jnp.int32) == 1).astype(jnp.float32)
#         target_embedding = (
#             self.embedderx(batch['xt'][:, :, 1:]) * mask_t_latent +
#             self.embedder_marker(jnp.zeros_like(batch['yt']).astype(jnp.int32)[:, :, 0])
#         )

#         # Add marker_t embedding
#         target_embedding += self.embedder_marker(batch['xt'][:, :, 0].astype(jnp.int32))

#         res = jnp.concatenate([context_embedding, target_embedding], axis=1)
#         return res