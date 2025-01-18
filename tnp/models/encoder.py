import jax
import jax.numpy as jnp
import equinox as eqx 
from jaxtyping import Array, Float
from typing import Tuple
from einops import pack, unpack


class TNPEncoder(eqx.Module):
    """Transformer Neural Process Encoder. 
    
    Encodes context and target sets through a series of transformations and a transformer.
    
    Attributes:
    transformer_encoder: eqx.Module
        Transformer-based encoder (TNPTransformer, Perceiver, or IST)
    xy_encoder: eqx.Module
        Network that jointly encodes x and y values
    x_encoder: eqx.Module
        Optional network for encoding x values (defaults to identity)
    y_encoder: eqx.Module
        Optional network for encoding y values (defaults to identity)
    """
    transformer_encoder: eqx.Module
    xy_encoder: eqx.Module 
    x_encoder: eqx.Module = eqx.nn.Identity()
    y_encoder: eqx.Module = eqx.nn.Identity()


    def __call__(
        self, 
        xc: Float[Array, "num_context input_dim"], 
        yc: Float[Array, "num_context output_dim"], 
        xt: Float[Array, "num_target input_dim"],
        *, 
        key: jax.random.PRNGKey, 
        enable_dropout: bool = False
    ) -> Float[Array, "num_target latent_dim"]:
        """Encode context and target sets into latent representations.
        
        Args:
            xc: Context inputs with shape [batch, num_context, input_dim]
            yc: Context outputs with shape [batch, num_context, output_dim]
            xt: Target inputs with shape [batch, num_targets, input_dim]
            
        Returns:
            jnp.ndarray: Encoded representations [num_points, latent_dim]
        """
        yc, yt = self.preprocess_observations(xt, yc)

        # Encode x values
        x, ps = pack([xc, xt], "* d")
        x_encoded = self.x_encoder(x)
        xc_encoded, xt_encoded = unpack(x_encoded, ps, "* d")

        # Encode y values
        y, ps = pack([yc, yt], "* d")
        y_encoded = self.y_encoder(y)
        yc_encoded, yt_encoded = unpack(y_encoded, ps, "* d")

        zc, _ = pack([xc_encoded, yc_encoded], "s *")
        zt, _ = pack([xt_encoded, yt_encoded], "s *")

        # Apply xy encoder
        zc = self.xy_encoder(zc)
        zt = self.xy_encoder(zt)

        # Apply transformer
        output = self.transformer_encoder(zc, zt, key=key, enable_dropout=enable_dropout)
    
        return output
    
    
    def preprocess_observations(
        self,
        xt: Float[Array, "num_target input_dim"], 
        yc: Float[Array, "num_context output_dim"]
        ) -> Tuple[Float[Array, "num_context output_dim_plus_1"], 
                Float[Array, "num_target output_dim_plus_1"]]:
            """Preprocess observations by adding mask channels. 
            
            Args: 
                xt: Target inputs with shape [batch, num_targets, input_dim]
                yc: Context outputs with shape [batch, num_context, output_dim]
                
            Returns: 
                Tuple[jnp.ndarray, jnp.ndarray]: Processed (context_outputs, target_outputs)
                where each has an additional mask channel (0 for context, 1 for targets)
            """
            # Create zero tensor for target outputs matching context shape
            yt = jnp.zeros((xt.shape[0], yc.shape[-1]))
            
            # Add mask channels
            yc = jnp.concatenate([yc, jnp.zeros_like(yc[..., :1])], axis=-1)
            yt = jnp.concatenate([yt, jnp.ones_like(yt[..., :1])], axis=-1)
            
            return yc, yt