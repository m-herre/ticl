# Files

- `test_predict_mothernet.py` — Tests MotherNet prediction with pretrained models: single inference, ensemble meta-learner, categorical preprocessing, one-hot encoding, and feature pruning.
- `test_predict_mothernet_init_mlp.py` — Tests MotherNetInitMLPClassifier with zero learning rate (pure extraction) and with fine-tuning on Iris.
- `test_predict_tabflex.py` — Tests TabFlex prediction accuracy across varying sample counts and feature dimensions, including high-dimensional inputs.
- `test_predict_tabpfn.py` — Tests TabPFN handles more than 10 classes by bucketing rare classes and verifying correct label assignment.
