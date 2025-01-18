from typing import Optional
import equinox as eqx 
from jaxtyping import Array, Float
from einops import rearrange


class TNPDecoder(eqx.Module):
    """Transformer Neural Process Decoder.
    
    This decoder takes a latent representation and optionally target inputs,
    and produces predictions through a decoder network.
    
    Attributes:
        z_decoder: eqx.Module
            The decoder network that transforms latent representations into predictions.
            Expected to handle the actual decoding of latent variables to output space.
    """
    z_decoder: eqx.Module 

    def __call__(
        self, 
        z: Float[Array, "num_target latent_dim"], 
        xt: Optional[Float[Array, "num_target input_dim"]] = None, 
    )-> Float[Array, "num_target output_dim"]:
        """Process latent representations to make predictions.
        
        TODO: docstring needs an update...
        TODO: also need to update the docstring to say that the batch dimension has been removed because 
        of how the vmap operation works 

        Args:
            z: jnp.ndarray
                Latent representations with shape [batch, ..., num_points, latent_dim]
            xt: Optional[jnp.ndarray], default=None
                Target inputs with shape [batch, num_targets, input_dim]
                If provided, only the corresponding latent variables are processed
        
        Returns:
            jnp.ndarray: Decoded predictions with shape [batch, ..., num_targets, output_dim]
        """
        # 
        if xt is not None:
            # Just take the last num_target points
            num_target = xt.shape[0]
            zt = rearrange(
                z[-num_target:, :], 
                "t d -> t d"
            )
        else: 
            zt = z # Use all latent variables 

        # Decode latent variables to predictions 
        return self.z_decoder(zt)