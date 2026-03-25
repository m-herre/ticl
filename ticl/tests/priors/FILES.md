# Files

- `test_boolean_data.py` — Tests BooleanConjunctionPrior output shapes, feature activity masks, and uninformative feature handling across CPU and CUDA.
- `test_classification_prior.py` — Tests ClassificationAdapterPrior and ClassificationAdapter with feature sampling, curriculum, double sampling, zero-padding, and NaN injection.
- `test_differentiable_hyperparameter.py` — Tests differentiable hyperparameter distributions (uniform, meta_beta, meta_gamma, meta_trunc_norm_log_scaled, meta_choice, meta_choice_mixed) for deterministic sampling.
- `test_mlp_prior.py` — Tests MLPPrior batch generation with sampled and fixed hyperparameters, verifying output shapes and deterministic values.
- `test_multiclass_step.py` — Tests MulticlassSteps produces correct class counts and covers all class indices across varying max_classes and max_steps.
- `test_step_function_prior.py` — Tests StepFunctionPrior generates data with exactly one step per active feature and zeros for inactive features.
