from abc import ABC
import jax
import jax.numpy as jnp
from flax import nnx 
from jaxtyping import Array, Float

class Identity(nnx.Module):
    def __init__(self):
        pass 

    def __call__(self, x: Float[Array, "..."]):
        return x 

class MLP(nnx.Module):
    def __init__(
        self, 
        in_dim: int, 
        out_dim: int, 
        rngs: nnx.Rngs, 
        width_size: int = 64,
        depth: int = 3, 
        dropout: bool = True, 
        dropout_rate: float = 0.2,
    ):
        kernel_init = nnx.initializers.variance_scaling(
            scale=0.3,
            mode='fan_avg',
            distribution='truncated_normal'
        )
        bias_init = nnx.initializers.zeros_init()

        # First layer
        self.linear1 = nnx.Linear(in_dim, width_size, 
                                kernel_init=kernel_init, 
                                bias_init=bias_init, 
                                rngs=rngs)
        self.dropout1 = nnx.Dropout(dropout_rate, rngs=rngs) if dropout else None

        # Middle layer
        self.linear2 = nnx.Linear(width_size, width_size,
                                kernel_init=kernel_init,
                                bias_init=bias_init,
                                rngs=rngs)
        self.dropout2 = nnx.Dropout(dropout_rate, rngs=rngs) if dropout else None

        # Final layer
        self.linear3 = nnx.Linear(width_size, out_dim,
                                kernel_init=kernel_init,
                                bias_init=bias_init,
                                rngs=rngs)
        self.use_dropout = dropout

    def __call__(self, x):
        x = self.linear1(x)
        x = jax.nn.gelu(x)
        if self.use_dropout:
            x = self.dropout1(x)
            
        x = self.linear2(x)
        x = jax.nn.gelu(x)
        if self.use_dropout:
            x = self.dropout2(x)
            
        x = self.linear3(x)
        return x

class TransformerBlock(nnx.Module):
    def __init__(
        self, 
        input_dim: int,
        hidden_dim: int,
        out_dim: int, 
        num_heads: int,
        rngs: nnx.Rngs,
        dropout_rate: float = 0.2,
        deterministic: bool = False,
        use_bias: bool = True,
        decode: bool = False,
    ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.dropout_rate = dropout_rate

        self.pre_layer_norm = nnx.LayerNorm(self.input_dim, rngs=rngs)
        self.post_layer_norm = nnx.LayerNorm(self.input_dim, rngs=rngs)
        self.attention = nnx.MultiHeadAttention(
            num_heads=num_heads,
            in_features=self.input_dim,
            qkv_features=self.input_dim,
            out_features=self.input_dim,
            dropout_rate=self.dropout_rate,
            deterministic=deterministic,
            use_bias=use_bias,
            decode=decode,
            rngs=rngs,
            kernel_init=nnx.initializers.variance_scaling(
                scale=0.2,  # Standard for attention
                mode='fan_in',
                distribution='truncated_normal'
            ),
            out_kernel_init=nnx.initializers.variance_scaling(
                scale=0.2,
                mode='fan_avg',
                distribution='uniform'
            ),
            bias_init=nnx.initializers.zeros_init(),
            out_bias_init=nnx.initializers.zeros_init()
        )
        self.mlp = MLP(
            in_dim=self.input_dim,
            out_dim=self.input_dim,
            rngs=rngs,
            width_size=self.hidden_dim,
            depth=3,
            dropout=True,
            dropout_rate=self.dropout_rate,
        )

    def __call__(
        self, 
        x: Float[Array, "batch seq dim"], 
    ):
        x = self.pre_layer_norm(x)
        x = x + self.attention(x, x, x) # residual + attention
        # do post layer norm and mlp on attention output and apply as residual
        x_residual = self.post_layer_norm(x)
        x_residual = self.mlp(x_residual)
        return x + x_residual
    
class TNPTransformer(nnx.Module):
    def __init__(
        self, 
        input_dim: int,
        hidden_dim: int = 64,
        num_heads: int = 4,
        dropout_rate: float = 0.2,
        deterministic: bool = True,
        rngs: nnx.Rngs = nnx.Rngs(0),
    ):
        self.transformer = TransformerBlock(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            out_dim=input_dim,
            num_heads=num_heads,
            dropout_rate=dropout_rate,
            deterministic=deterministic,
            rngs=rngs
        )
    
    def __call__(self, 
                 zc: Float[Array, "batch num_context dim"],
                 zt: Float[Array, "batch num_target dim"], 
    ) -> Float[Array, "batch num_target dim"]:
        z = jnp.concatenate([zc, zt], axis=1)
        z = self.transformer(z)
        return z[:, -zt.shape[1]:]