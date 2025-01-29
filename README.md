## Installation

```bash
# Clone the repository
git clone https://github.com/conorhassan/tnp.git
cd tnp

python -m venv .venv
source .venv/bin/activate

# Install using uv (recommended)
uv pip install -r requirements.txt
uv pip install -e .
```

## TODO: 

**Related to training:**
- use `jaxlightning` for training that automatically integrates with *WanDB*
- get the equinox and the nnx backend to work both with the same `yaml` config file
  - Config matching original implementation
    - use MA's config as a guide

**Related to Aalto:**
- get the code running on the Aalto system
    - save down the model checkpoint 

**Extensions:** 

- implement autoregressive sampling
- implement PABBO

**Tidying up to-do:**
- understand how random number generators work in `nnx`
- make the use of `jaxtyping` and the docstrings more homogeneous
- note the two buggy things that you found with `nnx`
