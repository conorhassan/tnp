import jax
import jax.numpy as jnp

from tnp.data.base import Batch, DataGenerator

class SimpleDataGenerator(DataGenerator):
    def generate_batch(self, key: jax.random.PRNGKey) -> Batch:
        # Generate dummy data
        x = jax.random.normal(key, (self.batch_size, 10, 1))  # 10 points, 1 dimension
        y = jnp.sin(x)  # Simple sine function
        
        # Split into context and target
        xc, xt = x[:, :5], x[:, 5:]
        yc, yt = y[:, :5], y[:, 5:]
        
        return Batch(x=x, y=y, xc=xc, yc=yc, xt=xt, yt=yt)

def test_data_generator_initialization():
    generator = SimpleDataGenerator(samples_per_epoch=100, batch_size=10)
    assert generator.num_batches == 10
    assert generator.batch_size == 10
    assert generator.samples_per_epoch == 100

def test_iterator_behavior():
    generator = SimpleDataGenerator(samples_per_epoch=30, batch_size=10)
    batches = list(generator)
    assert len(batches) == 3
    
    # Check shapes
    batch = batches[0]
    assert batch.x.shape == (10, 10, 1)  # (batch_size, num_points, dim)
    assert batch.xc.shape == (10, 5, 1)  # (batch_size, num_context, dim)
    assert batch.xt.shape == (10, 5, 1)  # (batch_size, num_target, dim)

def test_deterministic_behavior():
    gen1 = SimpleDataGenerator(samples_per_epoch=20, batch_size=10, seed=42)
    gen2 = SimpleDataGenerator(samples_per_epoch=20, batch_size=10, seed=42)
    
    # Check if batches are identical
    for b1, b2 in zip(gen1, gen2):
        assert jnp.allclose(b1.x, b2.x)

if __name__ == "__main__":
    print("Running tests...")
    test_data_generator_initialization()
    test_iterator_behavior()
    test_deterministic_behavior()
    print("All tests passed!")
