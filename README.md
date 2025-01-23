# Installation guide
```
python -m venv .venv
uv pip install -r requirements.txt
```
# TODO: 

- use `jaxlightning` for training that automatically integrates with *WanDB*
- get the `requirements.txt` thing to *actually* work
- get the equinox and the nnx backend to work both with the same `yaml` config file
  - Config matching original implementation
    - use MA's config as a guide
- get the code running on the Aalto system
    - save down the model checkpoint 

**Nice things to do on top:** 

- understand the way that random number generators work in `nnx`
- make the use of `jaxtyping` and the docstrings more homogeneous
- note the two buggy things that you found with `nnx`


**Notes for transition to CLUSTER:**

Add it in your job submission script (e.g., your .sh file):

bashCopyexport PYTHONPATH=$PYTHONPATH:/path/to/transformer_neural_process

Also set up the computer so that you can `.ssh` into it from your MAC and test GPU-enabled code :) 
