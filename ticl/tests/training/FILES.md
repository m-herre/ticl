# Files

- `__init__.py` — Empty package init.
- `test_train_additive.py` — Tests MotherNetAdditive training with various decoder types, factorized output, bin embeddings, shape attention, and variable features.
- `test_train_baam.py` — Tests GAMformer (BAAM) training with shape attention, binning, NaN handling, categorical embeddings, marginal residuals, regression, and decoder variants.
- `test_train_batabpfn.py` — Tests BiAttentionTabPFN training with linear, random, and fourier feature embeddings, with and without NaN injection.
- `test_train_mothernet.py` — Tests MotherNet training with decoder types (special_token, class_tokens, class_average, average), hidden layers, low-rank weights, checkpoint reload, and activation functions.
- `test_train_perceiver.py` — Tests TabPerceiver training with default settings, two hidden layers, and low-rank weight factorization.
- `test_train_tabflex.py` — Tests TabFlex training with various feature maps (identity, hedgehog, hedgehog_shared), custom feature counts, and sample sizes.
- `test_train_tabpfn.py` — Tests TabPFN training with custom features, weight initialization, stepped multiclass, boolean priors, and uninformative features.
