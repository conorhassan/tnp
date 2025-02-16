import jax
import jax.numpy as jnp
from flax import nnx 
from tnp.models.nnx_models.likelihood import NormalLikelihood, HeteroscedasticNormalLikelihood

def test_likelihoods():
    # Test data
    batch_size = 32
    seq_len = 10
    feature_dim = 4
    
    # Test NormalLikelihood
    print("\nTesting NormalLikelihood:")
    normal_like = NormalLikelihood(noise=0.1, train_noise=True)
    x = jnp.ones((batch_size, seq_len, feature_dim))
    dist = normal_like(x)
    print(f"Input shape: {x.shape}")
    print(f"Output loc shape: {dist.loc.shape}")
    print(f"Output scale shape: {dist.scale.shape}")
    
    # Test trainable parameter
    print(f"Is log_noise a Parameter? {isinstance(normal_like.log_noise, nnx.Param)}")
    
    # Test HeteroscedasticNormalLikelihood
    print("\nTesting HeteroscedasticNormalLikelihood:")
    hetero_like = HeteroscedasticNormalLikelihood(min_noise=0.01)
    # Double feature dim because we need space for mean and scale
    x = jnp.ones((batch_size, seq_len, feature_dim * 2))
    dist = hetero_like(x)
    print(f"Input shape: {x.shape}")
    print(f"Output loc shape: {dist.loc.shape}")
    print(f"Output scale shape: {dist.scale.shape}")
    
    # Test sampling
    print("\nTesting sampling:")
    key = jax.random.PRNGKey(0)
    samples = dist.sample(key)
    print(f"Sample shape: {samples.shape}")

if __name__ == "__main__":
    test_likelihoods()
