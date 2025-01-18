import gpjax as gpx 
import jax 
import jax.numpy as jnp 
import matplotlib.pyplot as plt 

key = jax.random.PRNGKey(42)

kernels = [
    gpx.kernels.Matern12(),
    gpx.kernels.RBF(),
    gpx.kernels.Matern32(),
    gpx.kernels.Matern52(),
]
fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(7, 6), tight_layout=True)

x = jnp.linspace(-3.0, 3.0, num=200).reshape(-1, 1)

meanf = gpx.mean_functions.Zero()

for k, ax in zip(kernels, axes.ravel()):
    prior = gpx.gps.Prior(mean_function=meanf, kernel=k)
    rv = prior(x)
    y = rv.sample(seed=key, sample_shape=(10,))
    ax.plot(x, y.T, alpha=0.7)
    ax.set_title(k.name)

plt.show()