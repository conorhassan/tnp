import jax
import jax.numpy as jnp
import optax
import equinox as eqx
from typing import Tuple 

from tnp.models.encoder import TNPEncoder
from tnp.models.likelihood import HeteroscedasticNormalLikelihood
from tnp.models.decoder import TNPDecoder
from tnp.models.neural_process import TNP
from tnp.models.layers import TNPTransformer, make_mlp

from tnp.models.gp import RBFKernel
from tnp.data.gp import RandomScaleGPGeneratorSameInputs

from tnp.data.gp_redo import SimpleGPGenerator

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
    optimizer = optax.adamw(learning_rate)

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

    generator = SimpleGPGenerator()
    xc, yc, xt, yt = generator.generate_batch(key)

    # evaluate lo
    key, subkey = jax.random.split(key)

    def forward(xc, yc, xt, key):
        return model(xc, yc, xt, key)
    
    batched_forward = jax.vmap(forward, in_axes=(0, 0, 0, None))

    def loss_fn(model):
        pred_dist = batched_forward(xc, yc, xt, key)
        log_prob = pred_dist.log_prob(yt)
        return -jnp.mean(log_prob)

    # def loss_fn(model):
    #     @jax.vmap
    #     def batched_forward(xc, yc, xt):
    #         return model(xc, yc, xt, key=subkey)
    #     pred_dist = batched_forward(xc, yc, xt)
    #     log_prob = pred_dist.log_prob(yt)
    #     return -jnp.mean(log_prob)

    print(loss_fn(model))
    print("Successfully evaluated the loss!")

    # optimizer.init(model)
    loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
    updates, opt_state = optimizer.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)

    loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
    print(loss)

    print("Updated the model!")


