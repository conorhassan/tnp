from flax import nnx 
import jax 
import jax.numpy as jnp
import optax
from typing import Tuple
from collections import deque

from tnp.data.gp_redo import SimpleGPGenerator
from tnp.models.nnx_models.layers import TransformerBlock, TNPTransformer, MLP
from tnp.models.nnx_models.neural_process import TNPEncoder, TNPDecoder, TNP 
from tnp.models.nnx_models.likelihood import HeteroscedasticNormalLikelihood

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

def create_model_and_optimizer(
    input_dim: int, 
    output_dim: int, 
    latent_dim: int, 
    learning_rate: float = 1e-3, 
) -> Tuple[TNP, optax.GradientTransformation]:
    
    transformer_encoder = TNPTransformer(
        input_dim=latent_dim,
        hidden_dim=latent_dim,
        num_heads=4,
        dropout_rate=0.2,
        rngs=nnx.Rngs(0),
    )

    xy_encoder = MLP(
        in_dim=input_dim + output_dim,
        out_dim=latent_dim,
        rngs=nnx.Rngs(0),
    )

    decoder = MLP(
        in_dim=latent_dim,
        out_dim=output_dim,
        rngs=nnx.Rngs(0),
    )

    likelihood = HeteroscedasticNormalLikelihood(min_noise=1e-3)

    model = TNP(
        transformer_encoder=transformer_encoder,
        xy_encoder=xy_encoder,
        z_decoder=decoder,
        likelihood=likelihood,
    )

    optimizer = optax.chain(optax.adamw(learning_rate), optax.clip(0.1))

    return model, optimizer

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

    model, optimizer = create_model_and_optimizer(
        input_dim=1,
        output_dim=2,
        latent_dim=256,
        learning_rate=1e-3,
    )

    optimizer = nnx.Optimizer(model, optimizer)

    generator = SimpleGPGenerator()
    key  = jax.random.PRNGKey(0)

    @nnx.jit
    def train_step(model, optimizer, batch):
        xc, yc, xt, yt = batch 

        def loss_fn(model):
            pred = model(xc, yc, xt)
            return -jnp.mean(pred.log_prob(yt))

        loss, grads = nnx.value_and_grad(loss_fn)(model)
        optimizer.update(grads)

        return loss

    losses = deque(maxlen=100)

    for i in range(1000):
        key, data_key = jax.random.split(key)
        batch = generator.generate_batch(data_key)
        loss = train_step(model, optimizer, batch)
        losses.append(loss)
        avg_loss = sum(losses) / len(losses)
        print(f"Step {i}, Loss: {avg_loss:.4f}")
    print("MODEL RUNS THE FORWARD METHOD MULTIPLE TIMES WITH NO PROBLEMS!")

    import code
    code.interact(local=locals())