# Model card

## Intended use

Educational research and methodological comparison for leakage-safe risk modeling on a public benchmark.

## Out of scope

- bedside triage or treatment decisions;
- deployment in a hospital;
- claims of patient benefit, safety, fairness, or transportability;
- causal interpretation of feature importance or attention weights.

## Data limitations

The challenge data were collected from ICU stays represented in a 2012 benchmark. The labeled public Set A is small, lacks a contemporary external cohort, and may reflect site-specific measurement and care patterns. Measurement absence can encode clinician behavior as well as physiology.

## Model risks

- subgroup performance may be unstable because some strata are small;
- calibration may drift across hospitals and time;
- broad plausibility filters can remove genuine extreme observations;
- missingness-aware models can learn site workflow rather than portable physiology;
- a threshold targeting sensitivity changes false-positive workload and requires clinical utility analysis.

## Required evidence before stronger claims

External validation, prospective workflow evaluation, subgroup analysis, decision-curve/utility analysis, data-governance review, clinical review of preprocessing, and post-deployment monitoring.
