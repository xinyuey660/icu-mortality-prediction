#!/usr/bin/env python3
"""Clinical-use analysis of the XGBoost risk model on all 4,000 Set A patients.

Every patient receives an out-of-fold risk from 5-fold cross-validation: in each
fold, XGBoost (12h windows) is trained on the other four folds, early-stopped and
Platt-calibrated on an inner validation split, then applied to the held-out
fold. The admission severity scores shipped with Set A (SAPS-I and SOFA) are
turned into out-of-fold probabilities with the same folds.

Outputs (results/clinical/):
- score_comparison.png: ROC and calibration against SAPS-I and SOFA
- decision_curve.png: net benefit across alert thresholds
- funnel_plot.png: risk-adjusted mortality (SMR) by ICU type
- subgroup_performance.png: AUROC and observed/expected deaths by subgroup
- gcs_recovery.png: mortality by GCS band in the first and last 12 hours
- clinical_analysis.json: every number in the report
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold, train_test_split

from icu_mortality.clinical import (
    GCS_BANDS,
    alert_summary,
    alerts_for_sensitivity,
    calibration_summary,
    funnel_limits,
    gcs_band,
    last_observed,
    mechanically_ventilated,
    net_benefit,
    standardized_mortality_ratio,
    treat_all_net_benefit,
)
from icu_mortality.data import STATIC_FEATURES, PhysioNet2012Loader
from icu_mortality.ensemble import SigmoidCalibrator
from icu_mortality.metrics import bootstrap_confidence_intervals, paired_bootstrap_difference
from icu_mortality.preprocessing import XGBoostPreprocessor
from icu_mortality.training import train_xgboost

COLORS = {"XGBoost": "#2A6F97", "SAPS-I": "#D97706", "SOFA": "#6B7280"}
ICU_TYPES = {1: "Coronary care", 2: "Cardiac surgery recovery", 3: "Medical", 4: "Surgical"}


def out_of_fold_predictions(data, scores: pd.DataFrame, args) -> pd.DataFrame:
    """Calibrated XGBoost risk and score-based risk for every patient, never in-sample."""

    n = len(data)
    output = pd.DataFrame({"XGBoost": np.nan, "SAPS-I": np.nan, "SOFA": np.nan}, index=range(n))
    folds = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    for fold, (dev_idx, held_idx) in enumerate(folds.split(np.zeros(n), data.y)):
        train_idx, val_idx = train_test_split(
            dev_idx, test_size=args.inner_validation, stratify=data.y[dev_idx],
            random_state=args.seed + fold,
        )
        train, validation, held = (data.subset(i) for i in (train_idx, val_idx, held_idx))
        preprocessor = XGBoostPreprocessor(args.window).fit(train)
        x_train, y_train = preprocessor.transform(train)
        x_val, y_val = preprocessor.transform(validation)
        x_held, _ = preprocessor.transform(held)
        model = train_xgboost(x_train, y_train, x_val, y_val, seed=args.seed + fold)
        calibrator = SigmoidCalibrator().fit(y_val, model.predict_proba(x_val)[:, 1])
        output.loc[held_idx, "XGBoost"] = calibrator.predict(model.predict_proba(x_held)[:, 1])

        for score in ("SAPS-I", "SOFA"):
            values = scores[score].to_numpy()
            fit_idx = dev_idx[np.isfinite(values[dev_idx])]
            apply_idx = held_idx[np.isfinite(values[held_idx])]
            regression = LogisticRegression().fit(values[fit_idx].reshape(-1, 1), data.y[fit_idx])
            output.loc[apply_idx, score] = regression.predict_proba(
                values[apply_idx].reshape(-1, 1)
            )[:, 1]
        print(f"[fold {fold + 1}/{args.folds}] held-out AUROC="
              f"{roc_auc_score(data.y[held_idx], output.loc[held_idx, 'XGBoost']):.4f}")
    return output


def plot_score_comparison(y, risks: pd.DataFrame, metrics: dict, path: Path) -> None:
    figure, (roc_axis, cal_axis) = plt.subplots(1, 2, figsize=(12, 5.2))
    for name, color in COLORS.items():
        fpr, tpr, _ = roc_curve(y, risks[name])
        auroc = metrics[name]["auroc"]
        roc_axis.plot(fpr, tpr, color=color, linewidth=2,
                      label=f"{name}: AUROC {auroc['estimate']:.3f} "
                            f"({auroc['lower_95']:.3f}–{auroc['upper_95']:.3f})")
        observed, predicted = calibration_curve(y, risks[name], n_bins=10, strategy="quantile")
        cal_axis.plot(predicted, observed, color=color, marker="o", linewidth=2, label=name)
    roc_axis.plot([0, 1], [0, 1], color="black", linestyle=":", linewidth=1)
    roc_axis.set(xlabel="1 − specificity", ylabel="Sensitivity")
    roc_axis.set_title("Discrimination", fontweight="bold")
    roc_axis.legend(loc="lower right", fontsize=9)
    limit = 0.0
    for name in COLORS:
        observed, predicted = calibration_curve(y, risks[name], n_bins=10, strategy="quantile")
        limit = max(limit, predicted.max(), observed.max())
    limit *= 1.15
    cal_axis.plot([0, limit], [0, limit], color="black", linestyle=":", linewidth=1)
    cal_axis.set_xlim(0, limit)
    cal_axis.set_ylim(0, limit)
    cal_axis.set(xlabel="Predicted risk (decile mean)", ylabel="Observed mortality")
    cal_axis.set_title("Calibration (deciles of predicted risk)", fontweight="bold")
    cal_axis.legend(loc="upper left", fontsize=9)
    for axis in (roc_axis, cal_axis):
        axis.grid(alpha=0.2)
    figure.suptitle("XGBoost vs admission severity scores, out-of-fold risk for every patient",
                    fontweight="bold")
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_decision_curve(y, risks: pd.DataFrame, thresholds, path: Path) -> dict:
    curves = {name: net_benefit(y, risks[name], thresholds) for name in COLORS}
    curves["Flag everyone"] = treat_all_net_benefit(y, thresholds)
    figure, axis = plt.subplots(figsize=(8.5, 5.2))
    for name, color in COLORS.items():
        axis.plot(thresholds, curves[name], color=color, linewidth=2.2, label=name)
    axis.plot(thresholds, curves["Flag everyone"], color="black", linestyle="--", linewidth=1.2,
              label="Flag everyone")
    axis.axhline(0, color="black", linewidth=1, label="Flag no one")
    axis.set_ylim(-0.02, float(np.mean(y)) + 0.02)
    axis.set(xlabel="Risk threshold for an alert", ylabel="Net benefit (true deaths flagged per patient)")
    axis.set_title("Decision curve: is alerting on the model worth the false alarms?",
                   fontweight="bold")
    axis.grid(alpha=0.2)
    axis.legend(fontsize=9)
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return {name: [round(float(v), 4) for v in values] for name, values in curves.items()}


def plot_funnel(units: list[dict], path: Path) -> None:
    figure, axis = plt.subplots(figsize=(8.5, 5.5))
    largest = max(unit["expected"] for unit in units) * 1.25
    grid = np.linspace(15, largest, 300)
    for alpha, style, label in ((0.05, "--", "95% limits"), (0.002, ":", "99.8% limits")):
        lower, upper = funnel_limits(grid, alpha)
        axis.plot(grid, lower, color="black", linestyle=style, linewidth=1, label=label)
        axis.plot(grid, upper, color="black", linestyle=style, linewidth=1)
    axis.axhline(1, color="black", linewidth=0.8)
    for position, unit in enumerate(sorted(units, key=lambda item: item["expected"])):
        axis.scatter(unit["expected"], unit["smr"], s=90, color=COLORS["XGBoost"], zorder=3)
        # Alternate labels above and below so neighbouring units stay legible.
        offset = (8, -30) if position % 2 == 0 else (8, 10)
        axis.annotate(f"{unit['unit']}\nSMR {unit['smr']:.2f} ({unit['lower']:.2f}–{unit['upper']:.2f})",
                      (unit["expected"], unit["smr"]), textcoords="offset points",
                      xytext=offset, fontsize=9,
                      bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 1})
    axis.set_ylim(0.4, 1.8)
    axis.set(xlabel="Expected deaths (sum of model risk)", ylabel="SMR (observed / expected)")
    axis.set_title("Risk-adjusted mortality by ICU type (funnel plot)", fontweight="bold")
    axis.grid(alpha=0.2)
    axis.legend(fontsize=9, loc="upper right")
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_subgroups(rows: list[dict], overall: float, path: Path) -> None:
    figure, (auc_axis, oe_axis) = plt.subplots(
        1, 2, figsize=(12, 0.42 * len(rows) + 1.8), sharey=True,
        gridspec_kw={"width_ratios": [3, 2]},
    )
    y = np.arange(len(rows))[::-1]
    labels = [f"{row['group']}: {row['level']} (n={row['n']}, {row['deaths']} deaths)" for row in rows]
    estimates = np.array([row["auroc"] for row in rows])
    lower = estimates - np.array([row["auroc_lower"] for row in rows])
    upper = np.array([row["auroc_upper"] for row in rows]) - estimates
    auc_axis.errorbar(estimates, y, xerr=[lower, upper], fmt="o", color=COLORS["XGBoost"], capsize=3)
    auc_axis.axvline(overall, color="black", linestyle=":", linewidth=1)
    auc_axis.set_yticks(y, labels)
    auc_axis.set_xlabel("AUROC (95% bootstrap CI)")
    auc_axis.set_title("Discrimination by subgroup", fontweight="bold")
    oe = np.array([row["observed_over_expected"] for row in rows])
    oe_axis.barh(y, oe - 1, left=1, color=[COLORS["SAPS-I"] if v > 1 else COLORS["XGBoost"] for v in oe])
    oe_axis.axvline(1, color="black", linewidth=1)
    for position, value in zip(y, oe):
        oe_axis.text(value + (0.02 if value >= 1 else -0.02), position, f"{value:.2f}",
                     va="center", ha="left" if value >= 1 else "right", fontsize=9)
    oe_axis.set_xlim(min(0.5, oe.min() - 0.15), max(1.5, oe.max() + 0.15))
    oe_axis.set_xlabel("Observed / expected deaths")
    oe_axis.set_title("Calibration by subgroup", fontweight="bold")
    for axis in (auc_axis, oe_axis):
        axis.grid(axis="x", alpha=0.2)
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_gcs_recovery(tables: dict[str, dict], path: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw={"wspace": 0.25})
    for axis, (title, table) in zip(axes, tables.items()):
        rate = np.array(table["mortality"], dtype=float)
        counts = np.array(table["n"])
        image = axis.imshow(np.where(counts >= 10, rate, np.nan), cmap="Oranges", vmin=0, vmax=0.6)
        for i in range(3):
            for j in range(3):
                text = "n<10" if counts[i, j] < 10 else f"{rate[i, j]:.0%}\nn={counts[i, j]}"
                axis.text(j, i, text, ha="center", va="center", fontsize=10,
                          color="white" if counts[i, j] >= 10 and rate[i, j] > 0.35 else "black")
        axis.set_xticks(range(3), GCS_BANDS)
        axis.set_yticks(range(3), GCS_BANDS)
        axis.set_xlabel("GCS at 36–48h")
        axis.set_title(title, fontweight="bold")
    axes[0].set_ylabel("GCS at 0–12h")
    figure.colorbar(image, ax=axes, shrink=0.8, label="In-hospital mortality")
    figure.suptitle("Mortality by GCS band at the start and end of the first 48 hours",
                    fontweight="bold")
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def gcs_recovery_table(y, start, end, members) -> dict:
    n = np.zeros((3, 3), dtype=int)
    deaths = np.zeros((3, 3), dtype=int)
    for i in range(3):
        for j in range(3):
            cell = members & (start == i) & (end == j)
            n[i, j], deaths[i, j] = int(cell.sum()), int(y[cell].sum())
    mortality = np.divide(deaths, n, out=np.full((3, 3), np.nan), where=n > 0)
    return {"n": n.tolist(), "deaths": deaths.tolist(),
            "mortality": [[None if np.isnan(v) else round(float(v), 4) for v in row]
                          for row in mortality]}


def run(args: argparse.Namespace) -> Path:
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    loader = PhysioNet2012Loader(args.data_dir, args.outcomes_path)
    data = loader.load()
    _, outcome_path = loader._discover()
    outcomes = pd.read_csv(outcome_path, dtype={"RecordID": str}).set_index("RecordID")
    scores = outcomes.loc[data.patient_ids, ["SAPS-I", "SOFA"]].reset_index(drop=True)
    scores = scores.where(scores >= 0)  # -1 marks a missing score
    y = data.y

    risks = out_of_fold_predictions(data, scores, args)
    complete = risks.notna().all(axis=1).to_numpy()
    y_cc, risks_cc = y[complete], risks.loc[complete].reset_index(drop=True)

    metrics = {name: bootstrap_confidence_intervals(y_cc, risks_cc[name], n_bootstrap=args.bootstrap,
                                                    seed=args.seed) for name in COLORS}
    differences = {f"XGBoost minus {name}": paired_bootstrap_difference(
        y_cc, risks_cc["XGBoost"], risks_cc[name], n_bootstrap=args.bootstrap, seed=args.seed)
        for name in ("SAPS-I", "SOFA")}
    calibration = {name: calibration_summary(y_cc, risks_cc[name]) for name in COLORS}
    plot_score_comparison(y_cc, risks_cc, metrics, output_dir / "score_comparison.png")

    thresholds = np.round(np.arange(0.02, 0.61, 0.01), 2)
    curves = plot_decision_curve(y_cc, risks_cc, thresholds, output_dir / "decision_curve.png")
    workload = {name: alerts_for_sensitivity(y_cc, risks_cc[name], args.target_sensitivity)
                for name in COLORS}
    alert_table = [alert_summary(y, risks["XGBoost"], t) for t in (0.1, 0.2, 0.3, 0.4, 0.5)]

    # Risk-adjusted benchmarking uses the model on every patient.
    icu_type = data.static[:, STATIC_FEATURES.index("ICUType")]
    units = []
    for code, name in ICU_TYPES.items():
        members = icu_type == code
        if members.sum() == 0:
            continue
        smr = standardized_mortality_ratio(int(y[members].sum()), float(risks["XGBoost"][members].sum()))
        units.append({"unit": name, "n": int(members.sum()), **smr})
    plot_funnel(units, output_dir / "funnel_plot.png")

    age = data.static[:, STATIC_FEATURES.index("Age")]
    gender = data.static[:, STATIC_FEATURES.index("Gender")]
    ventilated = mechanically_ventilated(data)
    subgroup_masks = [("ICU type", name, icu_type == code) for code, name in ICU_TYPES.items()]
    subgroup_masks += [
        ("Age", "under 65", age < 65), ("Age", "65–79", (age >= 65) & (age < 80)),
        ("Age", "80 and over", age >= 80),
        ("Sex", "female", gender == 0), ("Sex", "male", gender == 1),
        ("Ventilation", "ventilated", ventilated), ("Ventilation", "not ventilated", ~ventilated),
    ]
    subgroups = []
    for group, level, members in subgroup_masks:
        if members.sum() < 50 or y[members].sum() < 5 or y[members].sum() == members.sum():
            continue
        ci = bootstrap_confidence_intervals(y[members], risks["XGBoost"][members],
                                            n_bootstrap=args.bootstrap, seed=args.seed)["auroc"]
        subgroups.append({
            "group": group, "level": level, "n": int(members.sum()), "deaths": int(y[members].sum()),
            "auroc": ci["estimate"], "auroc_lower": ci["lower_95"], "auroc_upper": ci["upper_95"],
            "observed_over_expected": float(y[members].sum() / risks["XGBoost"][members].sum()),
        })
    overall_auroc = float(roc_auc_score(y, risks["XGBoost"]))
    plot_subgroups(subgroups, overall_auroc, output_dir / "subgroup_performance.png")

    start = gcs_band(last_observed(data, "GCS", 0, 12))
    end = gcs_band(last_observed(data, "GCS", 36, 48))
    gcs_tables = {
        "Mechanically ventilated": gcs_recovery_table(y, start, end, ventilated),
        "Not ventilated": gcs_recovery_table(y, start, end, ~ventilated),
    }
    plot_gcs_recovery(gcs_tables, output_dir / "gcs_recovery.png")

    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "design": {
            "patients": len(data), "deaths": int(y.sum()), "folds": args.folds,
            "inner_validation_fraction": args.inner_validation, "xgboost_window_hours": args.window,
            "seed": args.seed, "bootstrap_resamples": args.bootstrap,
            "score_comparison_patients": int(complete.sum()),
            "score_comparison_deaths": int(y_cc.sum()),
        },
        "overall_out_of_fold": {
            "auroc": overall_auroc, **calibration_summary(y, risks["XGBoost"]),
        },
        "score_comparison": {"metrics": metrics, "paired_differences": differences,
                             "calibration": calibration},
        "decision_curve": {"thresholds": thresholds.tolist(), "net_benefit": curves},
        "alert_workload": {"target_sensitivity": args.target_sensitivity, "by_model": workload,
                           "xgboost_thresholds": alert_table},
        "smr_by_icu_type": units,
        "subgroups": subgroups,
        "gcs_recovery": {"bands": list(GCS_BANDS), "start_window": "0–12h",
                         "end_window": "36–48h", "tables": gcs_tables,
                         "patients_with_both_windows": int(((start >= 0) & (end >= 0)).sum())},
    }
    path = output_dir / "clinical_analysis.json"
    path.write_text(json.dumps(summary, indent=2, default=float) + "\n", encoding="utf-8")
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--outcomes-path")
    parser.add_argument("--output-dir", default="results/clinical")
    parser.add_argument("--window", type=int, default=12)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--inner-validation", type=float, default=0.125)
    parser.add_argument("--target-sensitivity", type=float, default=0.80)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    print(run(build_parser().parse_args()))


if __name__ == "__main__":
    main()
