import jax
import jax.numpy as jnp

from tnp.models.eqx_models.likelihood import HeteroscedasticNormalLikelihood
from tnp.models.eqx_models.layers import make_mlp, TNPTransformer
from tnp.models.eqx_models.encoder import TNPEncoder
from tnp.models.eqx_models.decoder import TNPDecoder
from tnp.models.eqx_models.neural_process import TNP

print("make_mlp:", make_mlp)
print("TNPEncoder:", TNPEncoder)

def main():
    # Initialize random keys
    key = jax.random.PRNGKey(0)
    init_key, data_key, model_key = jax.random.split(key, 3)
    model_keys = jax.random.split(init_key, 5)  # One for each component

    # Model dimensions
    input_dim = 1
    output_dim = 1
    hidden_dim = 32
    latent_dim = 16

    # Create model components
    x_encoder = make_mlp(in_dim=input_dim, out_dim=hidden_dim, key=model_keys[0])
    y_encoder = make_mlp(in_dim=output_dim + 1, out_dim=hidden_dim, key=model_keys[1])  # +1 for mask
    xy_encoder = make_mlp(in_dim=hidden_dim*2, out_dim=latent_dim, key=model_keys[2])
    transformer = TNPTransformer(
        dim=latent_dim,
        hidden_dim=64,
        num_heads=4,
        dropout_rate=0.1,
        key=model_keys[3]
    )
    z_decoder = make_mlp(in_dim=latent_dim, out_dim=output_dim*2, key=model_keys[4])  # *2 for heteroscedastic
    likelihood = HeteroscedasticNormalLikelihood(min_noise=1e-3)

    # Create TNP
    encoder = TNPEncoder(transformer, xy_encoder, x_encoder, y_encoder)

    print("x_encoder before TNP:", type(x_encoder))
    print("encoder.x_encoder after creation:", type(encoder.x_encoder))

    decoder = TNPDecoder(z_decoder)
    model = TNP(encoder, decoder, likelihood)

    # Generate dummy data
    data_key1, data_key2 = jax.random.split(data_key)
    batch_size = 2
    num_context = 5
    num_target = 3

    xc = jax.random.normal(data_key1, (batch_size, num_context, input_dim))
    yc = jnp.sin(xc) + 0.1 * jax.random.normal(data_key2, (batch_size, num_context, output_dim))
    xt = jax.random.normal(data_key2, (batch_size, num_target, input_dim))

    # Forward pass
    dist = model(
        xc, yc, xt,
        key=model_key,
        enable_dropout=False  # Set to True for training
    )
    
    print(f"Output distribution type: {type(dist)}")
    print(f"Mean shape: {dist.loc.shape}")
    print(f"Scale shape: {dist.scale.shape}")

if __name__ == "__main__":
    main()