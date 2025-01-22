from flax import nnx 
from jaxtyping import Array, Float

class Likelihood(nnx.Module):
    def __init__(self):
        pass 

    def __call__(self, x: Float[Array, "..."]):
        return x 