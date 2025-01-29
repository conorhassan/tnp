import wandb

import jax 
import jax.numpy as jnp

from flax import nnx 
import optax
import orbax.checkpoint as ocp

from collections import deque
from pathlib import Path
from typing import Tuple

from tnp.data.simple_gp import SimpleGPGenerator
from tnp.models.nnx_models.layers import TNPTransformer, MLP
from tnp.models.nnx_models.neural_process import TNP 
from tnp.models.nnx_models.likelihood import HeteroscedasticNormalLikelihood

wandb.init(
    project="tnp-training", 
    config={
        "input_dim": 1, 
        "output_dim": 2, 
        "latent_dim": 256, 
        "learning_rate": 1e-3, 
        "batch_size": 32, 
        "num_steps": 1000,
    }
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
        deterministic=True,
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

    return model, nnx.Optimizer(model, optimizer)


# training loop with wandb, metrics, and orbax checkpointing
def train(
    model: TNP, 
    optimizer: nnx.Optimizer, 
    generator: SimpleGPGenerator, 
    num_steps: int, 
    checkpoint_dir: str = "checkpoints", 
    log_every: int = 10, 
    checkpoint_every: int = 500
): 
    # checkpoint 
    checkpoint_dir = Path(checkpoint_dir).absolute()
    checkpoint_dir = ocp.test_utils.erase_and_create_empty(str(checkpoint_dir))
    checkpointer = ocp.StandardCheckpointer()

    # track losses 
    losses = deque(maxlen=100)
    best_loss = float("inf")

    # training loop
    key = jax.random.PRNGKey(0)
    for step in range(num_steps):
        # generate bzatch 
        key, data_key = jax.random.split(key)
        batch = generator.generate_batch(data_key)

        # training step 
        model.train()
        loss = train_step(model, optimizer, batch)
        losses.append(loss)
        avg_loss = sum(losses) / len(losses)

        if step % log_every == 0: 
            wandb.log({
                "step": step, 
                "loss": loss, 
                "avg_loss": avg_loss
            })
            print(f"Step {step}, Loss: {avg_loss:.4f}")

        # save checkpoint
        if step % checkpoint_every == 0:
            checkpoint_path = checkpoint_dir / f"checkpoint_step_{step}"
            model.eval()
            _, params, _ = nnx.split(model, nnx.Param, ...)
            checkpointer.save(checkpoint_path, params)
            print(f"Checkpoint saved at {checkpoint_path}")
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_checkpoint_path = checkpoint_dir / "best_model"
            model.eval()
            _, params, _ = nnx.split(model, nnx.Param, ...)
            checkpointer.save(best_checkpoint_path, params, force=True)
            print(f"New best model saved at {best_checkpoint_path}")


@nnx.jit
def train_step(model, optimizer, batch):
    xc, yc, xt, yt = batch
    
    def loss_fn(model):
        pred = model(xc, yc, xt)
        return -jnp.mean(pred.log_prob(yt))
    
    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(grads)
    return loss

# Main execution
if __name__ == "__main__":
    # Initialize model and optimizer
    model, optimizer = create_model_and_optimizer(
        input_dim=1,
        output_dim=2,
        latent_dim=256,
        learning_rate=1e-3,
    )
    
    _, state = nnx.split(model)
    nnx.display(state)
    # Data generator
    generator = SimpleGPGenerator()
    
    # Train
    train(
        model=model,
        optimizer=optimizer,
        generator=generator,
        num_steps=1000,
        checkpoint_dir="checkpoints",
        log_every=100,
        checkpoint_every=500,
    )