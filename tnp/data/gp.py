import jax.numpy as jnp
import gpjax as gpx
from gpjax import Module
from gpjax.kernels import Kernel
from gpjax.likelihoods import Gaussian
from gpjax.mean_functions import Constant
from typing import Optional, Union
from dataclasses import dataclass

@dataclass
class GPRegressionModel(Module):
    kernel: Kernel
    mean: Constant = Constant()
    likelihood: Gaussian = Gaussian()
    train_data: Optional[tuple[jnp.ndarray, jnp.ndarray]] = None
    
    def __post_init__(self):
        if self.train_data is not None:
            self.x_train, self.y_train = self.train_data
    
    def prior(self, x: jnp.ndarray):
        mean = self.mean(x)
        cov = self.kernel(x, x)
        return gpx.Normal(loc=mean, scale=jnp.sqrt(cov))
    
    def posterior(self, x: jnp.ndarray):
        if self.train_data is None:
            raise ValueError("No training data provided")
        
        K_xx = self.kernel(self.x_train, self.x_train)
        K_xx_inv = jnp.linalg.inv(K_xx + self.likelihood.variance * jnp.eye(len(self.x_train)))
        K_xstar_x = self.kernel(x, self.x_train)
        
        mean = self.mean(x) + K_xstar_x @ K_xx_inv @ (self.y_train - self.mean(self.x_train))
        cov = self.kernel(x, x) - K_xstar_x @ K_xx_inv @ self.kernel(self.x_train, x)
        
        return gpx.Normal(loc=mean, scale=jnp.sqrt(cov))

if __name__ == "__main__":
    print("Hello world!")

# Example usage:
# kernel = gpx.kernels.RBF()
# model = GPRegressionModel(kernel=kernel)
# x_train = jnp.array([[1.], [2.], [3.]])
# y_train = jnp.array([1., 4., 9.])
# model.train_data = (x_train, y_train)
# 
# # Make predictions
# x_test = jnp.array([[4.], [5.]])
# posterior = model.posterior(x_test)