# TODO: 

- training loop 
- use the jaxtyping library for types!
- try and refactor code / understand. make it "more equinoxy"
- proper example (i.e., fit the model, replicate the yaml)

#### BOT suggestions:

Great list! Let's tackle these one at a time. Which would you like to start with?

Training loop would involve:

Loss function (negative log likelihood)
Optimizer (optax)
Batch generation
Training step with JIT
Jaxtyping integration would make the code more type-safe:

from jaxtyping import Array, Float, Int
Copy
And add proper shape annotations

"More equinoxy" refactoring would include:

Using dataclasses properly
Better module composition
Proper filtering for training/inference
Config management
Proper example would involve:

Heteroskedastic regression demo
Visualization
Config matching original implementation
Which area would you like to explore first?