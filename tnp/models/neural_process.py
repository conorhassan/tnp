from abc import ABC
import jax
import equinox as eqx 
import numpyro.distributions as dist
from jaxtyping import Array, Float

from .likelihood import Likelihood 
from .decoder import TNPDecoder
from .encoder import TNPEncoder

class NeuralProcess(eqx.Module, ABC): 
    """Represents a neural process base class.
    
    Attributes:
        encoder: eqx.Module
            Encoder module for processing inputs
        decoder: eqx.Module
            Decoder module for generating predictions
        likelihood: eqx.Module
            Module for modeling output distributions
    """
    encoder: eqx.Module 
    decoder: eqx.Module 
    likelihood: Likelihood


class ConditionalNeuralProcess(NeuralProcess):
    """Conditional Neural Process implementation.
    
    Implements the forward pass for conditional neural processes,
    processing context and target sets to make predictions.
    """
    def __call__(
        self, 
        xc: Float[Array, "batch num_context input_dim"], 
        yc: Float[Array, "batch num_context output_dim"],
        xt: Float[Array, "batch num_target input_dim"], 
        *, 
        key: jax.random.PRNGKey, 
        enable_dropout: bool = False
    ) -> dist.Distribution:
        """Forward pass for CNPs.
        
        Args:
            xc: Context inputs with shape [batch, num_context, input_dim]
            yc: Context outputs with shape [batch, num_context, output_dim]
            xt: Target inputs with shape [batch, num_targets, input_dim]
            
        Returns:
            Distribution over target outputs
        """
        z = self.encoder(xc, yc, xt, key=key, enable_dropout=enable_dropout)
        pred = self.decoder(z, xt)
        return self.likelihood(pred)
    

class TNP(ConditionalNeuralProcess):
    """Transformer Neural Process implementation.
    
    A specific implementation of CNP that uses transformer architecture
    for flexible neural conditioning.
    
    Attributes:
        encoder: TNPEncoder
            Transformer-based encoder for context and target sets
        decoder: TNPDecoder
            Decoder for generating predictions
        likelihood: eqx.Module
            Module that outputs distribution parameters for numpyro
    """
    encoder: TNPEncoder
    decoder: TNPDecoder
    likelihood: Likelihood

    def __call__(
        self,
        xc: Float[Array, "batch num_context input_dim"], 
        yc: Float[Array, "batch num_context output_dim"], 
        xt: Float[Array, "batch num_target input_dim"], 
        *, 
        key: jax.random.PRNGKey, 
        enable_dropout: bool = False
    ) -> dist.Distribution:
        """Forward pass through the TNP.
        
        Args:
            xc: Context inputs with shape [batch, num_context, input_dim]
            yc: Context outputs with shape [batch, num_context, output_dim]
            xt: Target inputs with shape [batch, num_targets, input_dim]
            
        Returns:
            numpyro.distributions.Distribution: Predicted distribution over target outputs
        """
        return super().__call__(xc, yc, xt, key=key, enable_dropout=enable_dropout)