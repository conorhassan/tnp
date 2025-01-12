import jax
import jax.numpy as jnp 
import equinox as eqx 
from jaxtyping import Array, Float


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