from typing import Tuple, Optional
from abc import ABC, abstractmethod
import jax
import jax.numpy as jnp 
import equinox as eqx 
import numpyro.distributions as dist
from jaxtyping import Array, Float
from einops import rearrange, pack, unpack

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
        z: Float[Array, "batch num_target latent_dim"], 
        xt: Optional[Float[Array, "batch  num_target input_dim"]] = None, 
    )-> Float[Array, "batch num_target output_dim"]:
        """Process latent representations to make predictions.
        
        TODO: docstring needs an update...

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
            num_target = xt.shape[1]
            zt = rearrange(
                z[:, -num_target:, :], 
                "b t d -> b t d"
            )
        else: 
            zt = z # Use all latent variables 

        # Decode latent variables to predictions 
        return self.z_decoder(zt)


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
        xc: Float[Array, "batch num_context input_dim"], 
        yc: Float[Array, "batch num_context output_dim"], 
        xt: Float[Array, "batch num_target input_dim"],
        *, 
        key: jax.random.PRNGKey, 
        enable_dropout: bool = False
    ) -> Float[Array, "batch num_target latent_dim"]:
        """Encode context and target sets into latent representations.
        
        Args:
            xc: Context inputs with shape [batch, num_context, input_dim]
            yc: Context outputs with shape [batch, num_context, output_dim]
            xt: Target inputs with shape [batch, num_targets, input_dim]
            
        Returns:
            jnp.ndarray: Encoded representations [batch, num_points, latent_dim]
        """
        yc, yt = self.preprocess_observations(xt, yc)

        # Encode x values
        x, ps = pack([xc, xt], "b * d")
        x_encoded = self.x_encoder(x)
        xc_encoded, xt_encoded = unpack(x_encoded, ps, "b * d")
        
        # Encode y values
        y, ps = pack([yc, yt], "b * d")
        y_encoded = self.y_encoder(y)
        yc_encoded, yt_encoded = unpack(y_encoded, ps, "b * d")
        
        # Join encodings
        zc = rearrange([xc_encoded, yc_encoded], "n b s d -> b s (n d)")
        zt = rearrange([xt_encoded, yt_encoded], "n b s d -> b s (n d)")
        
        # Apply xy encoder
        zc = self.xy_encoder(zc)
        zt = self.xy_encoder(zt)
        
        # Apply transformer
        output = self.transformer_encoder(zc, zt, key=key, enable_dropout=enable_dropout)
    
        return output
    
    
    def preprocess_observations(
        self,
        xt: Float[Array, "batch num_target input_dim"], 
        yc: Float[Array, "batch num_context output_dim"]
        ) -> Tuple[Float[Array, "batch num_context output_dim_plus_1"], 
                Float[Array, "batch num_target output_dim_plus_1"]]:
            """Preprocess observations by adding mask channels. 
            
            Args: 
                xt: Target inputs with shape [batch, num_targets, input_dim]
                yc: Context outputs with shape [batch, num_context, output_dim]
                
            Returns: 
                Tuple[jnp.ndarray, jnp.ndarray]: Processed (context_outputs, target_outputs)
                where each has an additional mask channel (0 for context, 1 for targets)
            """
            # Create zero tensor for target outputs matching context shape
            yt = jnp.zeros((xt.shape[0], xt.shape[1], yc.shape[-1]))
            
            # Add mask channels
            yc = jnp.concatenate([yc, jnp.zeros_like(yc[..., :1])], axis=-1)
            yt = jnp.concatenate([yt, jnp.ones_like(yt[..., :1])], axis=-1)
            
            return yc, yt
    

class Likelihood(eqx.Module, ABC):
    """Base class for likelihood functions. 
    
    All likelihood implementations should inherit from this 
    class and implement the __call__ method.
    """
    @abstractmethod 
    def __call__(self, x: jnp.ndarray) -> dist.Distribution:
        raise NotImplementedError
    
class NormalLikelihood(Likelihood):
    """Fixed-variance normal likelihood.
    
    Attributes: 
        log_noise: Learnable log noise parameter. 
        train_noise: Whether to update noise during training."""
    log_noise: jnp.ndarray
    train_noise: bool 

    def __init__(self, noise: float, train_noise: bool = True):
        self.log_noise = jnp.log(jnp.array(noise))
        self.train_noise = train_noise
    
    @property
    def noise(self):
        return jnp.exp(self.log_noise)
    
    def __call__(self, x: jnp.ndarray) -> dist.Normal:
        return dist.Normal(x, self.noise)
    

class HeteroscedasticNormalLikelihood(Likelihood):
    """Variable-variance normal likelihood. 
    
    Attributes: 
        min_noise: Minimum noise level to add
    """
    min_noise: float 

    def __init__(self, min_noise: float = 0.0):
        self.min_noise = min_noise 

    def __call__(self, x: jnp.ndarray) -> dist.Normal: 
        # Check even number of features for mean/variance pairs.
        assert x.shape[-1] % 2 == 0 

        # Split into location and log variance
        split_idx = x.shape[-1] // 2 
        loc, log_var = x[..., :split_idx], x[..., split_idx:]

        # Compute scale 
        scale = jnp.sqrt(jax.nn.softplus(log_var)) + self.min_noise

        return dist.Normal(loc, scale)


class NeuralProcess(eqx.Module, ABC): 
    """Represents a neural process base class.
    
    Attributes:
        encoder: eqx.Module
            Encoder module for processing inputs
        decoder: eqx.Module
            Decoder module for generating predictions
        likelihood: eqx.Module
            Module for modeling output distributions
    """
    encoder: eqx.Module 
    decoder: eqx.Module 
    likelihood: Likelihood


class ConditionalNeuralProcess(NeuralProcess):
    """Conditional Neural Process implementation.
    
    Implements the forward pass for conditional neural processes,
    processing context and target sets to make predictions.
    """
    def __call__(
        self, 
        xc: Float[Array, "batch num_context input_dim"], 
        yc: Float[Array, "batch num_context output_dim"],
        xt: Float[Array, "batch num_target input_dim"], 
        *, 
        key: jax.random.PRNGKey, 
        enable_dropout: bool = False
    ) -> dist.Distribution:
        """Forward pass for CNPs.
        
        Args:
            xc: Context inputs with shape [batch, num_context, input_dim]
            yc: Context outputs with shape [batch, num_context, output_dim]
            xt: Target inputs with shape [batch, num_targets, input_dim]
            
        Returns:
            Distribution over target outputs
        """
        z = self.encoder(xc, yc, xt, key=key, enable_dropout=enable_dropout)
        pred = self.decoder(z, xt)
        return self.likelihood(pred)
    

class TNP(ConditionalNeuralProcess):
    """Transformer Neural Process implementation.
    
    A specific implementation of CNP that uses transformer architecture
    for flexible neural conditioning.
    
    Attributes:
        encoder: TNPEncoder
            Transformer-based encoder for context and target sets
        decoder: TNPDecoder
            Decoder for generating predictions
        likelihood: eqx.Module
            Module that outputs distribution parameters for numpyro
    """
    encoder: TNPEncoder
    decoder: TNPDecoder
    likelihood: Likelihood

    def __call__(
        self,
        xc: Float[Array, "batch num_context input_dim"], 
        yc: Float[Array, "batch num_context output_dim"], 
        xt: Float[Array, "batch num_target input_dim"], 
        *, 
        key: jax.random.PRNGKey, 
        enable_dropout: bool = False
    ) -> dist.Distribution:
        """Forward pass through the TNP.
        
        Args:
            xc: Context inputs with shape [batch, num_context, input_dim]
            yc: Context outputs with shape [batch, num_context, output_dim]
            xt: Target inputs with shape [batch, num_targets, input_dim]
            
        Returns:
            numpyro.distributions.Distribution: Predicted distribution over target outputs
        """
        return super().__call__(xc, yc, xt, key=key, enable_dropout=enable_dropout)
    
# """TODO: want to refactor the remaining code into this `jaxtyping` sort of style"""
class AttentionBlock(eqx.Module):
    layer_norm1: eqx.nn.LayerNorm 
    layer_norm2: eqx.nn.LayerNorm 
    attention: eqx.nn.MultiheadAttention 
    linear1: eqx.nn.Linear 
    linear2: eqx.nn.Linear 
    dropout1: eqx.nn.Dropout 
    dropout2: eqx.nn.Dropout

    def __init__(
        self,
        input_shape: int,
        hidden_dim: int,
        num_heads: int,
        dropout_rate: float,
        key
    ):
        key1, key2, key3 = jax.random.split(key, 3)

        self.layer_norm1 = eqx.nn.LayerNorm(input_shape)
        self.layer_norm2 = eqx.nn.LayerNorm(input_shape)
        self.attention = eqx.nn.MultiheadAttention(num_heads, input_shape, key=key1)

        self.linear1 = eqx.nn.Linear(input_shape, hidden_dim, key=key2)
        self.linear2 = eqx.nn.Linear(hidden_dim, input_shape, key=key3)
        self.dropout1 = eqx.nn.Dropout(dropout_rate)
        self.dropout2 = eqx.nn.Dropout(dropout_rate)

    def __call__(
        self,
        x: jnp.ndarray,
        enable_dropout: bool,
        key) -> jnp.ndarray:
        input_x = jax.vmap(self.layer_norm1)(x)
        x = x + self.attention(input_x, input_x, input_x)

        input_x = jax.vmap(self.layer_norm2)(x)
        input_x = jax.vmap(self.linear1)(input_x)
        input_x = jax.nn.gelu(input_x)

        key1, key2 = jax.random.split(key, num=2)

        input_x = self.dropout1(input_x, inference=not enable_dropout, key=key1)
        input_x = jax.vmap(self.linear2)(input_x)
        input_x = self.dropout2(input_x, inference=not enable_dropout, key=key2)

        x = x + input_x

        return x


class TNPTransformer(eqx.Module):
    """Transformer component for neural process. 

    Applies self-attention mechanism to process context and target sequences. 

    Attributes:
        transformer: Transformer mechanism with layer normalization, multi-head attention,
        and feed-forward layers.
    """
    transformer: AttentionBlock
    
    def __init__(
        self, 
        dim: int,
        hidden_dim: int = 64,
        num_heads: int = 4,
        dropout_rate: float = 0.1,
        key: jax.random.PRNGKey = None,
    ):
        """Initialize transformer.
        
        Args:
            dim: Dimension of input features
            hidden_dim: Dimension of feed-forward hidden layer
            num_heads: Number of attention heads
            dropout_rate: Dropout probability
            key: PRNG key for initialization
        """
        self.transformer = AttentionBlock(
            input_shape=dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout_rate=dropout_rate,
            key=key
        )
    
    def __call__(self, 
                 zc: Float[Array, "batch num_context dim"],
                 zt: Float[Array, "batch num_target dim"], 
                 *, 
                 key: jax.random.PRNGKey, 
                 enable_dropout: bool = False
    ) -> Float[Array, "batch num_target dim"]:
        """Process context and target sequences through transformer.
        
        Args:
            zc: Context encodings
            zt: Target encodings
            
        Returns:
            Processed target encodings
        """
        def transform_batch(
                zc_batch: Float[Array, "batch num_context dim"], 
                zt_batch: Float[Array, "batch num_target dim"], 
                key: jax.random.PRNGKey
        ) -> Float[Array, "batch num_target dim"]:
            """Transform a single batch through the attention block.
            
            Concatenates context and target sequences, applies self-attention,
            and extracts the transformed target representations.
            
            Args:
                zc_batch: Single batch of context encodings
                zt_batch: Single batch of target encodings
                
            Returns:
                Transformed target encodings after attending to context
            """
            z = jnp.concatenate([zc_batch, zt_batch])
            z = self.transformer(z, enable_dropout=enable_dropout, key=key)
            return z[-zt_batch.shape[0]:]
        
        batch_size = zc.shape[0]
        keys = jax.random.split(key, batch_size)
        return jax.vmap(transform_batch)(zc, zt, keys)


def make_mlp(in_dim: int, out_dim: int, key: jax.random.PRNGKey) -> eqx.Module:
    class BatchedMLP(eqx.Module):
        layers: list
        
        def __init__(self, in_dim, out_dim, key, width_size=64, depth=3):
            keys = jax.random.split(key, depth)
            
            # Create dimensions for each layer
            dims = [in_dim] + [width_size] * (depth-1) + [out_dim]
            
            # Create layers
            self.layers = []
            for i in range(depth):
                self.layers.append(
                    eqx.nn.Linear(dims[i], dims[i+1], key=keys[i])
                )
        
        def __call__(self, x):
            # TODO: there is 
            # Double vmap each layer with activation
            for i, layer in enumerate(self.layers):
                batched_layer = jax.vmap(jax.vmap(layer))
                x = batched_layer(x)
                # ReLU on all but last layer
                if i < len(self.layers) - 1:
                    x = jax.nn.relu(x)
                    
            return x
    
    return BatchedMLP(in_dim, out_dim, key)

# in the training / evaluation mode
def train_step(model, batch, key):
    # training mode - dropout is active 
    model = eqx.nn.inference_mode(model)
    # TODO: add training logic here