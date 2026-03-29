# Files

- `mothernet_configs.py` — Generates a grid of MotherNet hyperparameter configurations (learning rate, num steps, batch size) and exports them to CSV
- `ssm_mothernet_configs.py` — Generates a grid of SSM/linear-attention MotherNet hyperparameter configurations (learning rate, fixed num steps and batch size) and exports them to CSV
- `tabpfn_configs.py` — Generates a grid of TabPFN hyperparameter configurations (learning rate, num steps, batch size) and exports them to CSV
- `grande_hpo.py` — Random search HPO sampler for factorized GRANDE MotherNet: defines search space, samples configs with constraint filtering, splits into chunks for parallel SLURM jobs
