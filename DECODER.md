# Decoders

This document explains the decoders in this repository at a conceptual level. It focuses on the decoder families that are wired into current model paths and exercised by current tests:

- `MLPModelDecoder`
- `AdditiveModelDecoder`
- `FactorizedAdditiveModelDecoder`
- `GradTreeDecoder`
- `GrandeDecoder`

`LinearModelDecoder` exists in the codebase, but it is not part of the active supported training paths, so it is not treated as a current decoder family here.

## The Core Idea

In this project, the transformer is usually not the final predictor.

Instead, the workflow is:

1. The encoder and transformer read the training examples of the current task.
2. A decoder turns that training-set representation into the parameters of a smaller child model.
3. That child model is then applied to the test examples.

So the decoder is best understood as a parameter generator. It translates "what this dataset looks like" into "what model should solve this dataset."

## Shared Readout Modes

Before a child-model-specific decoder can predict parameters, it needs a summary of the training set. The repo currently supports these working readout modes through the shared summary layer. Not every model uses every mode, but these are the supported conceptual options.

### `output_attention`

A learned query asks the training set for the summary vector it needs. Conceptually, this says: "look across the whole context and pull out the information that matters for parameter prediction."

This is the most flexible generic readout.

### `special_token`

A dedicated token is prepended to the sequence, and the model learns to write the dataset summary into that token. Conceptually, this is similar to a CLS token in language or vision models.

### `special_token_simple`

This also uses a prepended token, but with a simpler readout. The model is expected to store enough task information directly in that token without a second attention-based read step.

### `class_tokens`

The model gets one token per class. Conceptually, this encourages the decoder to build separate class-specific summaries instead of one global summary.

This is only meaningful for classification-style tasks.

### `class_average`

The training examples are averaged separately for each class. Conceptually, the decoder receives a compact "prototype" for each class.

This gives the decoder a strong class-aware inductive bias and is the default summary style for several MotherNet setups.

### `average`

The simplest option: average all training representations into one vector. Conceptually, this assumes the whole task can be summarized by a single global mean.

It is cheap and stable, but less expressive than the class-aware or attention-based options.

## Decoder Families

### `MLPModelDecoder`

This decoder predicts the weights and biases of a small MLP that will process the test points.

Conceptually:

- The transformer studies the task.
- The decoder emits an entire neural network specialized to that task.
- Prediction on test examples then happens inside that generated MLP, not directly inside the transformer.

Why this works well:

- MLP parameters define a smooth function class, so small parameter changes usually lead to small behavior changes.
- The child model is expressive enough to capture nonlinear patterns.
- In the low-rank/shared-weight setup, the decoder only has to generate task-specific coefficients, while some weight structure is shared across tasks. That reduces the burden of predicting a full network from scratch.

This is the main decoder behind standard MotherNet, and also the decoder used by the linear-attention and perceiver MotherNet variants.

### `AdditiveModelDecoder`

This decoder predicts an additive model: each feature contributes its own one-dimensional shape function, and the final prediction is the sum of those feature contributions plus a bias term.

Conceptually:

- Each feature is discretized into bins.
- The decoder predicts the effect of each bin for each feature.
- At inference time, the model looks up the active bin for every feature and adds the corresponding contributions.

This gives a GAM-like child model:

- easy to interpret,
- naturally feature-wise,
- and biased toward simpler, decomposable relationships.

It is used in the additive model families, including `MotherNetAdditive` and `GAMformer`.

### `FactorizedAdditiveModelDecoder`

This is the lower-rank version of the additive decoder. It still produces additive models, but it does not predict every shape function independently.

Conceptually:

- The decoder first predicts a small set of coefficients per feature or per class-feature pair.
- Those coefficients are combined with shared global shape templates.
- The final per-feature shape functions are reconstructed from that smaller latent representation.

The point of factorization is reuse. Instead of learning every shape function from scratch for every task, the model can reuse a common library of shapes.

#### Shape Attention

When shape attention is enabled, the shared templates behave like a prototype library. Features do not just linearly mix fixed basis vectors; they attend to reusable canonical shapes.

Conceptually, this says:

- "many feature effects look like variations of a smaller set of recurring shapes,"
- and the decoder should select and combine those recurring shapes rather than inventing a fresh one every time.

This is the most structured and parameter-efficient additive decoder in the repo.

### `GradTreeDecoder`

This decoder predicts a differentiable decision tree ensemble in the older "GradTree" style.

Conceptually, it predicts three things for every tree:

- which feature each internal node should inspect,
- what threshold that node should split on,
- what prediction each leaf should emit.

The resulting tree is axis-aligned and uses straight-through decisions:

- in the forward pass it behaves like a hard tree,
- in the backward pass it still allows gradient-based training.

This decoder is working, but it is legacy in this repo. It has been superseded conceptually by the GRANDE path, which is the current tree-based focus.

### `GrandeDecoder`

This decoder predicts the parameters of a GRANDE-style differentiable tree ensemble. It is the current tree-oriented decoder family and the main research direction of the project.

The key conceptual difference from `GradTreeDecoder` is that GRANDE is estimator-local:

- each tree works with its own sampled subset of candidate features,
- the decoder is told summary statistics about those local candidate features,
- and it predicts the tree relative to that local context.

That makes the prediction problem more structured. Instead of always choosing among all input features globally, the decoder works within a smaller per-estimator candidate set.

The GRANDE child model still represents:

- split values,
- feature-choice logits for each split,
- leaf predictions,
- and estimator weights for combining trees.

But the decoder is explicitly conditioned on feature statistics such as:

- mean,
- standard deviation,
- missing-value rate,
- and, for multiclass settings, class-conditional means.

This matters because a useful split threshold is not an abstract number by itself; it only makes sense relative to the local scale and distribution of a feature.

#### `baseline`

One MLP predicts all GRANDE parameters for each estimator at once.

Conceptually, this is the least opinionated version: the decoder gets the task summary and local estimator context, then emits the entire tree in one shot.

#### `factorized_stats`

This version separates the prediction problem into specialized heads:

- one head for split values,
- one for feature-choice logits,
- one for estimator weights,
- one for leaf outputs.

The important conceptual bias is in the split values: they are predicted relative to feature statistics, not as unconstrained free-floating thresholds. In effect, the decoder learns an offset around the local feature distribution instead of inventing thresholds from scratch.

This gives the model a cleaner decomposition of the task and makes split prediction more data-aware.

#### `depthwise_factorized_stats`

This is the most tree-structured GRANDE decoder. It grows the tree level by level instead of predicting every node independently in one flat shot.

Conceptually:

- the root is predicted first,
- the state of a parent node is passed to its children,
- children know their depth and whether they are left or right branches,
- and later splits are conditioned on earlier decisions.

This gives the decoder a recursive tree-building bias. Rather than treating a tree as a bag of unrelated nodes, it treats it as a hierarchy where downstream decisions should depend on upstream ones.

## How To Think About The Whole Design

Across all decoder families, the repo is exploring the same general strategy:

- use a large contextual model to understand the task,
- then emit a smaller, structured predictor that is easier to run, inspect, or constrain.

The main differences between decoders are therefore not about implementation details. They are about what kind of child model the repo wants to generate:

- an MLP when flexibility matters most,
- an additive model when interpretability and feature-wise decomposition matter,
- a tree ensemble when crisp axis-aligned decisions matter.

## Current Practical Status

- `MLPModelDecoder`: mature and broadly used.
- `AdditiveModelDecoder`: working and used for additive/GAM-style models.
- `FactorizedAdditiveModelDecoder`: working and used when a lower-rank or prototype-based additive model is desired.
- `GradTreeDecoder`: working, but legacy.
- `GrandeDecoder`: working and the active tree-decoder direction, with three supported variants: `baseline`, `factorized_stats`, and `depthwise_factorized_stats`.
