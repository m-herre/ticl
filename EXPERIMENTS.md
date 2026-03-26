# Experiments Log

Track hypotheses, experiments, and results here. Goal: close the performance gap between GRANDE MotherNet and MLP MotherNet.

## Template

```
### [YYYY-MM-DD] Experiment title
**Hypothesis**: ...
**Test**: ...
**Config**: `python ticl/fit_model.py mothernet --child-model grande ...`
**Result**: ...
**Conclusion**: ...
```

---

## Experiments

### [2026-03-25] Diagnostic: Root causes of GRANDE underperformance
**Hypothesis**: Multiple structural issues prevent GRANDE MotherNet from matching MLP MotherNet: near-random feature selection, seed mismatch, split clustering, low diversity, uniform weighting, miscalibration, gradient vanishing at depth.
**Test**: Load existing checkpoints (factorized epoch 270, depthwise epoch 190) and quantify each hypothesis on breast-w, credit-approval, credit-g datasets. No new training.
**Config**: See `experiments/grande_diagnostics.ipynb`
**Result**: (run notebook to populate)
**Conclusion**: (pending notebook execution)
