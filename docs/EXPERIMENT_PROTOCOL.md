# Experiment protocol

## Question

Using measurements collected during the first 48 hours of an adult ICU stay, how well can each model rank and calibrate the risk of in-hospital death on unseen patients?

## Cohort and split

- Source: PhysioNet/CinC Challenge 2012 Set A, 4,000 labeled stays.
- Unit of analysis: one ICU stay / patient record.
- Split: stratified 80% train, 10% validation, 10% sealed test with seed 42.
- The saved `split_manifest.csv` is the source of truth for every rerun.
- No row-, hour-, or observation-level random splitting is permitted.

## Data processing

- Parse only observations in `[00:00, 47:59]`; malformed timestamps are rejected instead of being silently assigned to hour zero.
- Replace broad implausible values with missing values. These are engineering QC bounds, not diagnostic reference ranges.
- Merge invasive and non-invasive systolic/mean/diastolic blood pressure by hour, preferring invasive readings when both sources occur.
- Fit log transforms, observed-value means/standard deviations, static medians, and static scaling on training patients only.
- Preserve observation masks and elapsed time. Missing standardized values become zero only after the observed-value mean has been subtracted, so zero represents the training mean rather than a physiological zero.

## Model selection

- Primary selection metric: validation AUPRC because mortality is uncommon.
- Secondary metrics: AUROC, Brier score, log loss, F1, sensitivity, specificity, precision, and accuracy.
- XGBoost searches only the declared window sizes and selects by validation AUPRC.
- Neural models use class-weighted binary cross-entropy, AdamW, gradient clipping, and validation-AUPRC early stopping.
- A model's operating threshold is selected on validation predictions to reach at least 0.80 sensitivity.

## Ensemble

Let `p_im` be patient `i`'s probability from model `m`. The ensemble is

`p_i = sum_m w_m p_im`, where `w_m >= 0` and `sum_m w_m = 1`.

Before blending, each model is placed on a comparable probability scale using five-fold out-of-fold sigmoid calibration inside the validation set. Each calibrator is then refitted on the full validation set and applied unchanged to test probabilities. Weights minimize log loss on the OOF-calibrated validation probabilities. Test labels are not passed to either optimizer. For a stronger publication-grade version, generate base-model out-of-fold predictions on the entire 90% development cohort, learn calibration, weights, and thresholds from those OOF predictions, refit each branch on the full development cohort, and evaluate the test split once.

## Repetition and uncertainty

The benchmark uses one frozen seed to establish an auditable baseline. A stronger follow-up experiment should report:

- five training seeds for every neural model;
- bootstrap 95% confidence intervals on test AUROC, AUPRC, and Brier score;
- model-vs-model paired bootstrap differences;
- subgroup checks by sex, ICU type, and age band, with small-cell warnings;
- calibration intercept/slope and a reliability plot;
- external validation before any claim of clinical generalizability.
