# Model selection rationale

## Implementation choices

### XGBoost

The baseline uses shallow, regularized trees; class weight derived from the training split; native missing values; early stopping on validation AUPRC; and a small, declared search over 3/6/12-hour summary windows.

### GRU-D

GRU-D is included because missingness and measurement frequency are informative in ICU data. The implementation uses feature-wise input decay. Values, masks, and elapsed time are explicit inputs.

### GRU-Attention and BiGRU-Attention

Both are kept, but their roles are different. The unidirectional model is the cleaner sequential baseline. The bidirectional model may represent the completed 48-hour window better, but it cannot be described as real-time within-window prediction because its representation sees later observations in both directions. Attention weights are useful diagnostics, not causal explanations.

### Observation-triplet embedding Transformer

The embedding branch is inspired by STraTS. Each observed value becomes a token formed from:

- a learned variable embedding;
- a learned continuous value embedding;
- a learned continuous time embedding.

A compact Transformer contextualizes the tokens, and learned fusion attention produces a patient representation. Unlike a grid Transformer, this branch does not add fake observations through forward-fill. The current code implements supervised mortality prediction only; self-supervised forecasting pretraining is a separately testable extension.

## Why not choose a larger foundation model

Set A has 4,000 labeled stays. Large sequence models add compute and interview buzzwords but can weaken the scientific story through overfitting and irreproducible tuning. The present model set creates a defensible progression:

1. strong tabular baseline;
2. explicit missingness mechanism;
3. recurrent temporal attention;
4. sparse observation embeddings;
5. validation-learned ensemble.

This produces a transparent comparison without making an unvalidated claim that a larger architecture is “state of the art.”
