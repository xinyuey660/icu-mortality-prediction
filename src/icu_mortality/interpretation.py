"""Variable-level explanations for the window-summary XGBoost model.

XGBoost sees one column per window, variable, and summary statistic. Clinicians
think in variables (GCS, urine output, BUN), so SHAP contributions are summed
per patient across all columns that belong to the same variable. TreeSHAP
values are additive, so a grouped contribution is still an exact share of the
model's log-odds output.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import DYNAMIC_FEATURES, ICUDataset

VARIABLE_LABELS = {
    "BP_Dias": "Diastolic BP", "BP_MAP": "Mean arterial pressure", "BP_Sys": "Systolic BP",
    "FiO2": "FiO2", "HCO3": "Bicarbonate", "HCT": "Haematocrit", "HR": "Heart rate",
    "ICUType": "ICU type", "K": "Potassium", "MechVent": "Mechanical ventilation",
    "Mg": "Magnesium", "Na": "Sodium", "RespRate": "Respiratory rate", "Temp": "Temperature",
    "Urine": "Urine output", "WBC": "White cell count",
}

STAT_LABELS = {
    "mean": "mean",
    "last": "last",
    "range": "range",
    "trend": "trend",
    "observed_rate": "observed rate",
    "delta": "hours since last",
}


def variable_name(feature_name: str) -> str:
    """Map a model column such as ``w03__GCS__last`` to its clinical variable."""

    parts = feature_name.split("__")
    if len(parts) == 3:
        return parts[1]
    if len(parts) == 2 and parts[1] == "missing":
        return parts[0]
    return feature_name


def variable_label(name: str) -> str:
    """Plain-language label for a clinical variable."""

    return VARIABLE_LABELS.get(name, name)


def readable_feature_name(feature_name: str, window_hours: int) -> str:
    """Turn ``w03__BP_Sys__last`` into ``Systolic BP last (36–48h)`` for plots."""

    parts = feature_name.split("__")
    if len(parts) == 3:
        window = int(parts[0].lstrip("w"))
        start, end = window * window_hours, (window + 1) * window_hours
        stat = STAT_LABELS.get(parts[2], parts[2])
        return f"{variable_label(parts[1])} {stat} ({start}–{end}h)"
    if len(parts) == 2 and parts[1] == "missing":
        return f"{variable_label(parts[0])} missing"
    return variable_label(feature_name)


def xgboost_contributions(model, x: np.ndarray) -> tuple[np.ndarray, float]:
    """Exact TreeSHAP values at the early-stopped iteration, plus the base value."""

    from xgboost import DMatrix

    best = getattr(model, "best_iteration", None)
    iteration_range = (0, int(best) + 1) if best is not None else (0, 0)
    contributions = model.get_booster().predict(
        DMatrix(x), pred_contribs=True, iteration_range=iteration_range
    )
    return contributions[:, :-1], float(contributions[0, -1])


def group_by_variable(contributions: np.ndarray, feature_names: list[str]) -> pd.DataFrame:
    """Sum signed per-column contributions into one column per clinical variable."""

    if contributions.shape[1] != len(feature_names):
        raise ValueError("Contribution columns and feature names must align.")
    frame = pd.DataFrame(contributions, columns=feature_names)
    groups = [variable_name(name) for name in feature_names]
    return frame.T.groupby(groups, sort=False).sum().T


def variable_importance(grouped: pd.DataFrame) -> pd.Series:
    """Mean absolute grouped contribution, largest first."""

    return grouped.abs().mean(axis=0).sort_values(ascending=False)


def binned_trajectories(
    data: ICUDataset,
    variables: tuple[str, ...],
    bin_hours: int = 6,
) -> dict[str, np.ndarray]:
    """Per-patient mean of observed values in each time bin; NaN when unobserved."""

    if 48 % bin_hours:
        raise ValueError("bin_hours must divide 48.")
    bins = 48 // bin_hours
    output: dict[str, np.ndarray] = {}
    for name in variables:
        index = DYNAMIC_FEATURES.index(name)
        count = data.count[:, :, index].astype(np.float64)
        total = data.value_sum[:, :, index].astype(np.float64)
        count = count.reshape(len(data), bins, bin_hours).sum(axis=2)
        total = total.reshape(len(data), bins, bin_hours).sum(axis=2)
        output[name] = np.divide(total, count, out=np.full_like(total, np.nan), where=count > 0)
    return output


@dataclass(frozen=True)
class GroupSummary:
    median: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    observed: np.ndarray


def summarize_group(values: np.ndarray) -> GroupSummary:
    """Median and interquartile range per bin over patients with an observation."""

    observed = np.isfinite(values).sum(axis=0)
    # Bins where no patient in the group was measured stay NaN.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        median = np.nanmedian(values, axis=0)
        lower = np.nanpercentile(values, 25, axis=0)
        upper = np.nanpercentile(values, 75, axis=0)
    return GroupSummary(median=median, lower=lower, upper=upper, observed=observed)


def select_cases(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, int]:
    """Pick the most confident correct alert, false alarm, and missed death."""

    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    flagged = probabilities >= threshold
    cases: dict[str, int] = {}
    candidates = {
        "correct_alert": np.flatnonzero(flagged & (y_true == 1)),
        "false_alarm": np.flatnonzero(flagged & (y_true == 0)),
        "missed_death": np.flatnonzero(~flagged & (y_true == 1)),
    }
    for name, indices in candidates.items():
        if len(indices) == 0:
            continue
        if name == "missed_death":
            cases[name] = int(indices[np.argmin(probabilities[indices])])
        else:
            cases[name] = int(indices[np.argmax(probabilities[indices])])
    return cases
