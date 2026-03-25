# Files

- `__init__.py` — Public API: exports all prior classes (MLPPrior, GPPrior, BagPrior, etc.)
- `boolean_conjunctions.py` — Prior that generates synthetic datasets from random boolean conjunction formulas
- `classification_adapter.py` — Wraps a regression prior, adding discretization, NaN injection, and categorical features
- `fast_gp.py` — Prior that samples synthetic datasets from exact Gaussian Processes via gpytorch
- `mlp.py` — Prior that generates synthetic datasets by sampling and evaluating random MLP networks
- `prior_bag.py` — Meta-prior that randomly selects from a weighted bag of base priors per batch
- `prototyping_datasets.py` — Utilities for generating prototype-based binary datasets and benchmarking classifiers
- `step_function_prior.py` — Prior that generates synthetic datasets from random axis-aligned step functions
- `utils.py` — Shared helpers: class randomization, y-based ordering, and categorical activation
