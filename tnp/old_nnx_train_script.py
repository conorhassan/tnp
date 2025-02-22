import logging
import hydra
from omegaconf import DictConfig
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from collections import deque

from tnp.data.simple_gp import SimpleGPGenerator
from tnp.models.nnx_models.neural_process import TransformerNeuralProcess
from tnp.models.nnx_models.likelihood import HeteroscedasticNormalLikelihood

@hydra.main(config_path=None, config_name=None)
def main(cfg: DictConfig):

    logging.getLogger("cola").setLevel(logging.WARNING) 
    logging.getLogger("hydra").setLevel(logging.WARNING)

    # Initialize the model
    model = TransformerNeuralProcess(
        input_dim=cfg.model.input_dim,
        output_dim=cfg.model.output_dim,
        latent_dim=cfg.model.latent_dim,
        hidden_dim=cfg.model.hidden_dim,
        num_heads=cfg.model.num_heads,
        dropout_rate=cfg.model.dropout_rate,
        rngs=nnx.Rngs(cfg.training.rng_seed)
    )

    # Initialize the optimizer
    optimizer = optax.chain(optax.adamw(cfg.model.learning_rate), optax.clip(0.1))
    optimizer = nnx.Optimizer(model, optimizer)

    # Data generator
    generator = SimpleGPGenerator(batch_size=cfg.training.batch_size)
    key = jax.random.PRNGKey(cfg.training.rng_seed)

    @nnx.jit
    def train_step(model, optimizer, batch):
        xc, yc, xt, yt, mask = batch

        def loss_fn(model):
            pred = model(xc, yc, xt, mask)
            return -jnp.mean(pred.log_prob(yt))

        loss, grads = nnx.value_and_grad(loss_fn)(model)
        optimizer.update(grads)

        return loss

    losses = deque(maxlen=100)
    avg_losses = [] 

    for i in range(10):
        key, data_key = jax.random.split(key)
        batch = generator.generate_batch(data_key)
        loss = train_step(model, optimizer, batch)
        losses.append(loss)
        avg_loss = sum(losses) / len(losses)
        avg_losses.append(avg_loss)
        print(f"Step {i}, Loss: {avg_loss:.4f}")

    np.save("average_losses.npy", avg_losses)

if __name__ == "__main__":
    main()