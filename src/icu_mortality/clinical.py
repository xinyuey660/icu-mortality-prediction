"""Clinical-use analyses: decision curves, alert workload, risk-adjusted benchmarking.

These helpers turn predicted probabilities into the quantities a clinical team
asks about: how many alerts a threshold creates, whether acting on the model
beats acting on everyone, and whether a unit has more deaths than its case mix
predicts (standardised mortality ratio, SMR).
"""

from __future__ import annotations

import numpy as np
from scipy.stats import chi2
from sklearn.linear_model import LogisticRegression

from .data import DYNAMIC_FEATURES, ICUDataset
from .metrics import select_threshold

GCS_BANDS = ("3–8", "9–12", "13–15")


def net_benefit(y_true: np.ndarray, probabilities: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """Decision-curve net benefit: TP/n - FP/n * pt / (1 - pt) at each threshold pt."""

    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    n = len(y_true)
    output = []
    for threshold in np.asarray(thresholds, dtype=float):
        flagged = probabilities >= threshold
        true_positive = np.sum(flagged & (y_true == 1))
        false_positive = np.sum(flagged & (y_true == 0))
        output.append(true_positive / n - false_positive / n * threshold / (1.0 - threshold))
    return np.asarray(output)


def treat_all_net_benefit(y_true: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """Net benefit of flagging every patient."""

    prevalence = float(np.mean(y_true))
    thresholds = np.asarray(thresholds, dtype=float)
    return prevalence - (1.0 - prevalence) * thresholds / (1.0 - thresholds)


def alert_summary(y_true: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, float]:
    """Workload of alerting at ``threshold``: alerts per 100 patients, sensitivity, PPV."""

    y_true = np.asarray(y_true, dtype=int)
    flagged = np.asarray(probabilities, dtype=float) >= threshold
    alerts = int(flagged.sum())
    true_positive = int(np.sum(flagged & (y_true == 1)))
    ppv = true_positive / alerts if alerts else float("nan")
    return {
        "threshold": float(threshold),
        "alerts_per_100": 100.0 * alerts / len(y_true),
        "sensitivity": true_positive / max(int(y_true.sum()), 1),
        "ppv": ppv,
        "alerts_per_death_found": 1.0 / ppv if ppv and np.isfinite(ppv) else float("nan"),
    }


def alerts_for_sensitivity(
    y_true: np.ndarray, probabilities: np.ndarray, target: float = 0.80
) -> dict[str, float]:
    """Alert workload at the highest threshold that still finds ``target`` of deaths."""

    return alert_summary(y_true, probabilities, select_threshold(y_true, probabilities, target))


def standardized_mortality_ratio(
    observed: float, expected: float, alpha: float = 0.05
) -> dict[str, float]:
    """Observed / expected deaths with an exact Poisson confidence interval."""

    if expected <= 0:
        raise ValueError("Expected deaths must be positive.")
    lower = chi2.ppf(alpha / 2, 2 * observed) / 2 if observed > 0 else 0.0
    upper = chi2.ppf(1 - alpha / 2, 2 * observed + 2) / 2
    return {
        "observed": float(observed),
        "expected": float(expected),
        "smr": float(observed / expected),
        "lower": float(lower / expected),
        "upper": float(upper / expected),
    }


def funnel_limits(expected: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    """SMR control limits around 1 for a unit with ``expected`` deaths."""

    expected = np.asarray(expected, dtype=float)
    lower = chi2.ppf(alpha / 2, 2 * expected) / 2 / expected
    upper = chi2.ppf(1 - alpha / 2, 2 * expected + 2) / 2 / expected
    return lower, upper


def calibration_summary(y_true: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    """Observed/expected ratio and calibration slope (logistic regression on logit p)."""

    y_true = np.asarray(y_true, dtype=int)
    p = np.clip(np.asarray(probabilities, dtype=float), 1e-6, 1 - 1e-6)
    logit = np.log(p / (1.0 - p)).reshape(-1, 1)
    # C=inf fits an unpenalised regression.
    slope = LogisticRegression(C=np.inf, solver="lbfgs").fit(logit, y_true).coef_[0, 0]
    return {
        "observed_over_expected": float(y_true.sum() / p.sum()),
        "calibration_slope": float(slope),
    }


def last_observed(data: ICUDataset, feature: str, start_hour: int, end_hour: int) -> np.ndarray:
    """Last recorded value of ``feature`` in hours [start_hour, end_hour); NaN if none."""

    index = DYNAMIC_FEATURES.index(feature)
    count = data.count[:, start_hour:end_hour, index]
    values = data.value_last[:, start_hour:end_hour, index]
    observed = count > 0
    last_hour = observed.shape[1] - 1 - observed[:, ::-1].argmax(axis=1)
    output = values[np.arange(len(data)), last_hour].astype(np.float64)
    output[~observed.any(axis=1)] = np.nan
    return output


def gcs_band(values: np.ndarray) -> np.ndarray:
    """Severity band index for GCS: 0 = 3–8, 1 = 9–12, 2 = 13–15, -1 = missing."""

    values = np.asarray(values, dtype=float)
    bands = np.full(values.shape, -1, dtype=int)
    bands[values <= 8] = 0
    bands[(values > 8) & (values <= 12)] = 1
    bands[values > 12] = 2
    bands[~np.isfinite(values)] = -1
    return bands


def mechanically_ventilated(data: ICUDataset) -> np.ndarray:
    """True when mechanical ventilation was recorded at any point in the 48 hours."""

    index = DYNAMIC_FEATURES.index("MechVent")
    recorded = (data.count[:, :, index] > 0) & (np.nan_to_num(data.value_max[:, :, index]) >= 1)
    return recorded.any(axis=1)
