import jax
import jax.numpy as jnp
from jaxtyping import Array, Float
from typing import Optional, Tuple
from tnp.data.base import Batch, GroundTruthPredictor
from tnp.data.synthetic import SyntheticBatch, SyntheticGeneratorUniformInput

class SineWaveGenerator(SyntheticGeneratorUniformInput):
    def sample_outputs(
        self,
        x: Float[Array, "batch nc_plus_nt dim"],
        key: jax.random.PRNGKey
    ) -> Tuple[Float[Array, "batch nc_plus_nt 1"], Optional[GroundTruthPredictor]]:
        # Add some noise to sine wave
        noise = 0.1 * jax.random.normal(key, x.shape[:-1] + (1,))
        y = jnp.sin(x) + noise
        return y, None

def test_initialization():
    """Test basic initialization and properties"""
    generator = SineWaveGenerator(
        dim=1,
        min_nc=3,
        max_nc=10,
        min_nt=5,
        max_nt=15,
        samples_per_epoch=100,
        batch_size=32,
        context_range=[(-2., 2.)],
        target_range=[(-4., 4.)],
        seed=42
    )
    
    assert generator.batch_size == 32
    assert generator.num_batches == 3  # 100 // 32 = 3

def test_batch_properties():
    """Test properties of generated batches"""
    generator = SineWaveGenerator(
        dim=1,
        min_nc=3,
        max_nc=10,
        min_nt=5,
        max_nt=15,
        samples_per_epoch=100,
        batch_size=32,
        context_range=[(-2., 2.)],
        target_range=[(-4., 4.)],
        seed=42
    )

    batch = next(generator)
    
    assert isinstance(batch, SyntheticBatch)
    assert batch.x.shape[0] == 32  # batch size
    assert batch.x.shape[2] == 1   # dimension
    
    # Check ranges
    assert jnp.all(batch.xc >= -2.0)
    assert jnp.all(batch.xc <= 2.0)
    assert jnp.all(batch.xt >= -4.0)
    assert jnp.all(batch.xt <= 4.0)

def test_non_deterministic_behavior():
    """Test non-deterministic generation"""
    print("\nStarting non-deterministic test...")
    
    # Create generators with different initial keys
    generator1 = SineWaveGenerator(
        dim=1,
        min_nc=3,
        max_nc=10,
        min_nt=5,
        max_nt=15,
        samples_per_epoch=100,
        batch_size=32,
        context_range=[(-2., 2.)],
        target_range=[(-4., 4.)],
        seed=0
    )
    
    generator2 = SineWaveGenerator(
        dim=1,
        min_nc=3,
        max_nc=10,
        min_nt=5,
        max_nt=15,
        samples_per_epoch=100,
        batch_size=32,
        context_range=[(-2., 2.)],
        target_range=[(-4., 4.)],
        seed=42
    )

    print(f"\nGenerator initial keys:\ngen1: {generator1.key}\ngen2: {generator2.key}")
    
    batches1 = list(generator1)
    batches2 = list(generator2)
    
    print(f"\nNumber of batches: {len(batches1)}")
    
    any_different = False
    for i, (b1, b2) in enumerate(zip(batches1, batches2)):
        diff = jnp.abs(b1.x.max() - b2.x.max())
        print(f"Batch {i} max difference: {diff}")
        if diff > 1e-6:
            any_different = True
            print(f"Found difference in batch {i}")
            break
    
    assert any_different, "Expected at least one batch to be different"


def test_variable_points():
    """Test that number of context/target points varies within specified range"""
    generator = SineWaveGenerator(
        dim=1,
        min_nc=3,
        max_nc=5,
        min_nt=2,
        max_nt=4,
        samples_per_epoch=32,
        batch_size=8,
        context_range=[(-1., 1.)],
        target_range=[(-1., 1.)]
    )

    nc_sizes = []
    nt_sizes = []
    
    for batch in generator:
        nc_sizes.append(batch.xc.shape[1])
        nt_sizes.append(batch.xt.shape[1])
    
    assert min(nc_sizes) >= 3
    assert max(nc_sizes) <= 5
    assert min(nt_sizes) >= 2
    assert max(nt_sizes) <= 4

if __name__ == "__main__":
    print("Running tests...")
    test_initialization()
    test_batch_properties()
    test_non_deterministic_behavior()
    test_variable_points()
    print("All tests passed!")
