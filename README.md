# Installation guide
```
python -m venv .venv
uv pip install -r requirements.txt
```
# TODO: 

- training loop 
- use the jaxtyping library for types!
- try and refactor code / understand. make it "more equinoxy"
- proper example (i.e., fit the model, replicate the yaml)

**Training loop would involve:**

- Loss function (negative log likelihood)
- Optimizer (optax)
- Batch generation
- *jitted* training step
- *WanDB* integration?

"More equinoxy" refactoring would include:
- Using dataclasses properly
- Better module composition
- Proper filtering for training/inference
  - really interested in this! 
Config management
Proper example would involve:

Heteroskedastic regression demo
- Visualization
- Config matching original implementation
  - use MA's config as a guide
Visualization
Config matching original implementation
Which area would you like to explore first?

- need to adjust the file structure 

- need to have data loader code

- need to have the data 

- need to have training code 

- set up training on WANDB 

- need to "manage" the package better


### Notes for transition to CLUSTER: 

Add it in your job submission script (e.g., your .sh file):

bashCopyexport PYTHONPATH=$PYTHONPATH:/path/to/transformer_neural_process

Also set up the computer so that you can `.ssh` into it from your MAC and test GPU-enabled code :) 
