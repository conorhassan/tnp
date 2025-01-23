import jax
import jax.numpy as jnp
import optax
import equinox as eqx
from typing import Tuple 

from tnp.models.eqx_models.encoder import TNPEncoder
from tnp.models.eqx_models.likelihood import HeteroscedasticNormalLikelihood
from tnp.models.eqx_models.decoder import TNPDecoder
from tnp.models.eqx_models.neural_process import TNP
from tnp.models.eqx_models.layers import TNPTransformer, make_mlp

from tnp.data.simple_gp import SimpleGPGenerator

from collections import deque

def create_model_and_optimizer(
    input_dim: int, 
    output_dim: int, 
    latent_dim: int, 
    key: jax.random.PRNGKey, 
    learning_rate: float = 1e-3, 
) -> Tuple[TNP, optax.GradientTransformation]:

    key1, key2, key3, key4 = jax.random.split(key, 4)

    # create model components 
    transformer = TNPTransformer(
        dim=latent_dim,
        key=key1
    )
    xy_encoder = make_mlp(
        input_dim + output_dim, 
        latent_dim, 
        key2
    )
    decoder = make_mlp(
        latent_dim, 
        output_dim, 
        key3
    )

    likelihood = HeteroscedasticNormalLikelihood(
        min_noise=1e-3
    )

    # assemble model 
    model = TNP(
        encoder=TNPEncoder(transformer_encoder=transformer, xy_encoder=xy_encoder), 
        decoder=TNPDecoder(z_decoder=decoder), 
        likelihood=likelihood
    )

    # create optimizer
    optimizer = optax.chain(optax.adamw(learning_rate), optax.clip(0.1))

    return model, optimizer


if __name__ == "__main__":

    key = jax.random.PRNGKey(0)

    input_dim = 1 
    output_dim = 1 # NOTE: decoder will output 2 * output_dim for mean/variance
    latent_dim = 32 

    model, optimizer = create_model_and_optimizer(
        input_dim=input_dim, 
        output_dim=2*output_dim, 
        latent_dim=latent_dim, 
        key=key
    )

    # initialize optimizer state 
    params = eqx.filter(model, eqx.is_array)
    opt_state = optimizer.init(params)

    # create data generator
    generator = SimpleGPGenerator()

    # initialize key
    key, subkey = jax.random.split(key)

    def forward(model, xc, yc, xt, key, enable_dropout):  # Add enable_dropout param
        return model(xc, yc, xt, key, enable_dropout)

    batched_forward = jax.vmap(forward, in_axes=(None, 0, 0, 0, None, None))  # None for enable_dropout

    def loss_fn(model, xc, yc, xt, yt, key, enable_dropout=False):  # Add enable_dropout with default
        pred_dist = batched_forward(model, xc, yc, xt, key, enable_dropout)
        log_prob = pred_dist.log_prob(yt)
        return -jnp.mean(log_prob)

    @eqx.filter_jit
    def train_step(model, opt_state, xc, yc, xt, yt, key):
        loss, grads = eqx.filter_value_and_grad(loss_fn)(model, xc, yc, xt, yt, key, True)
        updates, opt_state = optimizer.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss
    
    num_steps = 3000
    
    losses = deque(maxlen=100)

    for step in range(num_steps):
        key, data_key, model_key = jax.random.split(key, 3)
        xc, yc, xt, yt = generator.generate_batch(data_key)
        model, opt_state, loss = train_step(model, opt_state, xc, yc, xt, yt, model_key)
        losses.append(loss)
        avg_loss = sum(losses) / len(losses)
        print(f"Step {step}, Loss: {avg_loss:.4f}")

    
    pred_dist = batched_forward(model, xc, yc, xt, key, False)

    # Sample from distribution
    key, sampling_key = jax.random.split(key)
    samples = jax.vmap(lambda d: d.sample(key=sampling_key, sample_shape=(100,)))(pred_dist)

    print(jnp.mean(samples, axis=1)[0, :, 0])
    print(yt[0, :, 0])