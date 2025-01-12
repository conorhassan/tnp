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
