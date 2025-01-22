from flax import nnx 
import jax 
import jax.numpy as jnp
from nnx.layers import TransformerBlock, TNPTransformer

# class TNPTransformer(nnx.Module):
#     def __init__(
#         self, 
#         num_heads: int = 8, 
#         input_dim: int = 128, 
#         rngs: nnx.Rngs = None
#     ):
#         self.num_heads = num_heads
#         self.input_dim = input_dim
#         self.rngs = rngs

transformer_encoder = nnx.MultiHeadAttention(
    num_heads=4, 
    in_features=128, 
    qkv_features=16, 
    out_features=16, 
    dropout_rate=0.2,
    deterministic=False,
    use_bias=True,
    decode=False,
    rngs=nnx.Rngs(0)
)

if __name__ == "__main__":
    batch_dim = 5
    seq_dim = 10 
    input_dim = 128 

    key1, key2, key3 = jax.random.split(jax.random.PRNGKey(0), 3)

    q, k, v = (
        jax.random.normal(key1, (batch_dim, seq_dim, input_dim)),
        jax.random.normal(key2, (batch_dim, seq_dim, input_dim)),
        jax.random.normal(key3, (batch_dim, seq_dim, input_dim)),
    )

    print("TESTING MULTIHEAD ATTENTION LAYER")
    print(transformer_encoder(q, k, v))

    print("TESTING TRANSFORMER BLOCK")
    attention_block = TransformerBlock(
        input_dim=input_dim,
        hidden_dim=input_dim,
        out_dim=input_dim,
        num_heads=4,
        rngs=nnx.Rngs(0),
    )

    print(attention_block(q).shape)
    
    print("TESTING THE TRANSFORMER NEURAL PROCESS")
    tnp_transformer = TNPTransformer(
        input_dim=64,
        hidden_dim=64,
        num_heads=4,
        dropout_rate=0.2,
        rngs=nnx.Rngs(0),
    )

    zc = jnp.ones((512, 10, 64))
    zt = jnp.ones((512, 7, 64))

    print(jnp.concatenate([zc, zt], axis=1).shape)
    print(tnp_transformer(zc, zt).shape)