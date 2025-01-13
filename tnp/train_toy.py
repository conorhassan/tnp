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

    # dummy data 
    batch_size = 4 
    nc, nt = 5, 10 

    xc = jnp.zeros((batch_size, nc, input_dim))
    yc = jnp.zeros((batch_size, nc, output_dim))
    xt = jnp.zeros((batch_size, nt, input_dim))
    yt = jnp.ones((batch_size, nt, output_dim)) 

    # try forward pass
    key, subkey = jax.random.split(key)
    pred_dist = model(xc, yc, xt, key=subkey)

    log_prob = jnp.mean(pred_dist.log_prob(yt), axis=(1, 2))

    print(log_prob)
    print(log_prob.shape)

    # optimizer.init(model)
    print("Model success")

    params = eqx.filter(model, eqx.is_array)
    opt_state = optimizer.init(params)

    from functools import partial

    def batched_forward(xc, yc, xt):
        return model(xc, yc, xt, key=subkey)
    
    print("Before model:", xc.shape)
    pred_dist = batched_forward(xc, yc, xt)

    
    def loss_fn(model):

        @jax.vmap 
        def batched_forward(xc, yc, xt):
            return model(xc, yc, xt, key=subkey)
        

        pred_dist = batched_forward(xc, yc, xt)
        log_prob = jnp.mean(pred_dist.log_prob(yt), axis=(1, 2))
        return -log_prob
    
    
    # def loss_fn(model):

    #     def forward_pass(xc, yc, xt, key):
    #         return model(xc, yc, xt, key=key)
        
    #     forward_pass = partial(forward_pass, key=subkey)
        
    #     pred_dist = jax.vmap(forward_pass, in_axes=(0, 0, 0))(xc, yc, xt)
    #     log_prob = jnp.mean(pred_dist.log_prob(yt), axis=(1, 2))
    #     return -log_prob
    
    loss, grads = eqx.filter_value_and_grad(loss_fn)(model)

    updates, opt_state = optimizer.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)

    print(loss)
    print(grads)


    print("Initialized the model :)")


