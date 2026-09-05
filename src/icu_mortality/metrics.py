"""Consistent discrimination, calibration, and operating-point metrics."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


def select_threshold(y_true: np.ndarray, probabilities: np.ndarray, target_recall: float = 0.80) -> float:
    """Highest validation threshold reaching the requested sensitivity."""

    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    candidates = np.unique(np.r_[0.0, probabilities, 1.0])[::-1]
    feasible = [
        threshold
        for threshold in candidates
        if recall_score(y_true, probabilities >= threshold, zero_division=0) >= target_recall
    ]
    return float(max(feasible)) if feasible else 0.5


def binary_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float | int]:
    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.clip(np.asarray(probabilities, dtype=float), 1e-7, 1 - 1e-7)
    predicted = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()
    return {
        "n": len(y_true),
        "prevalence": float(y_true.mean()),
        "auroc": float(roc_auc_score(y_true, probabilities)),
        "auprc": float(average_precision_score(y_true, probabilities)),
        "brier": float(brier_score_loss(y_true, probabilities)),
        "log_loss": float(log_loss(y_true, probabilities, labels=[0, 1])),
        "threshold": float(threshold),
        "f1": float(f1_score(y_true, predicted, zero_division=0)),
        "sensitivity": float(recall_score(y_true, predicted, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if tn + fp else 0.0,
        "precision": float(precision_score(y_true, predicted, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, predicted)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def bootstrap_confidence_intervals(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    *,
    n_bootstrap: int = 2000,
    seed: int = 42,
) -> dict[str, dict[str, float]]:
    """Stratified patient bootstrap CIs for key threshold-free metrics."""

    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    class_indices = [np.flatnonzero(y_true == label) for label in (0, 1)]
    if any(len(indices) == 0 for indices in class_indices):
        raise ValueError("Both outcome classes are required for bootstrap intervals.")
    rng = np.random.default_rng(seed)
    samples = {"auroc": [], "auprc": [], "brier": []}
    for _ in range(n_bootstrap):
        indices = np.concatenate(
            [rng.choice(group, size=len(group), replace=True) for group in class_indices]
        )
        y_sample = y_true[indices]
        p_sample = probabilities[indices]
        samples["auroc"].append(roc_auc_score(y_sample, p_sample))
        samples["auprc"].append(average_precision_score(y_sample, p_sample))
        samples["brier"].append(brier_score_loss(y_sample, p_sample))
    estimates = {
        "auroc": roc_auc_score(y_true, probabilities),
        "auprc": average_precision_score(y_true, probabilities),
        "brier": brier_score_loss(y_true, probabilities),
    }
    return {
        name: {
            "estimate": float(estimates[name]),
            "lower_95": float(np.percentile(values, 2.5)),
            "upper_95": float(np.percentile(values, 97.5)),
        }
        for name, values in samples.items()
    }


def paired_bootstrap_difference(
    y_true: np.ndarray,
    candidate: np.ndarray,
    reference: np.ndarray,
    *,
    n_bootstrap: int = 2000,
    seed: int = 42,
) -> dict[str, dict[str, float]]:
    """Paired CIs for candidate-minus-reference metric differences."""

    y_true = np.asarray(y_true, dtype=int)
    candidate = np.asarray(candidate, dtype=float)
    reference = np.asarray(reference, dtype=float)
    class_indices = [np.flatnonzero(y_true == label) for label in (0, 1)]
    rng = np.random.default_rng(seed)
    differences = {"auroc": [], "auprc": [], "brier": []}
    for _ in range(n_bootstrap):
        indices = np.concatenate(
            [rng.choice(group, size=len(group), replace=True) for group in class_indices]
        )
        labels = y_true[indices]
        candidate_sample = candidate[indices]
        reference_sample = reference[indices]
        differences["auroc"].append(
            roc_auc_score(labels, candidate_sample) - roc_auc_score(labels, reference_sample)
        )
        differences["auprc"].append(
            average_precision_score(labels, candidate_sample)
            - average_precision_score(labels, reference_sample)
        )
        differences["brier"].append(
            brier_score_loss(labels, candidate_sample)
            - brier_score_loss(labels, reference_sample)
        )
    point = {
        "auroc": roc_auc_score(y_true, candidate) - roc_auc_score(y_true, reference),
        "auprc": average_precision_score(y_true, candidate)
        - average_precision_score(y_true, reference),
        "brier": brier_score_loss(y_true, candidate)
        - brier_score_loss(y_true, reference),
    }
    return {
        name: {
            "estimate": float(point[name]),
            "lower_95": float(np.percentile(values, 2.5)),
            "upper_95": float(np.percentile(values, 97.5)),
        }
        for name, values in differences.items()
    }
