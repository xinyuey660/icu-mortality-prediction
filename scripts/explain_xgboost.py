#!/usr/bin/env python3
"""Explain the selected XGBoost model on the sealed test split.

Refits XGBoost with the benchmark's seed, split, and selected window, then
writes four figures and a JSON summary:

- variable_importance.png: mean |SHAP| per clinical variable
- shap_beeswarm.png: per-patient SHAP values for the strongest model columns
- risk_trajectories.png: 48-hour trajectories of GCS, urine output, BUN, and
  heart rate for the highest- and lowest-risk test quartiles
- case_explanations.png: variable contributions for a correct alert, a false
  alarm, and a missed death
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from icu_mortality.data import PhysioNet2012Loader
from icu_mortality.interpretation import (
    binned_trajectories,
    group_by_variable,
    readable_feature_name,
    select_cases,
    summarize_group,
    variable_importance,
    variable_label,
    xgboost_contributions,
)
from icu_mortality.metrics import select_threshold
from icu_mortality.preprocessing import XGBoostPreprocessor
from icu_mortality.splits import stratified_split
from icu_mortality.training import train_xgboost

BLUE = "#2A6F97"
ORANGE = "#D97706"
TRAJECTORY_VARIABLES = ("GCS", "Urine", "BUN", "HR")
TRAJECTORY_LABELS = {
    "GCS": "Glasgow Coma Scale (points)",
    "Urine": "Urine output (mL per record)",
    "BUN": "Blood urea nitrogen (mg/dL)",
    "HR": "Heart rate (bpm)",
}
CASE_TITLES = {
    "correct_alert": "Correct alert (died)",
    "false_alarm": "False alarm (survived)",
    "missed_death": "Missed death (died)",
}


def plot_variable_importance(importance: pd.Series, top: int, path: Path) -> None:
    shown = importance.head(top)[::-1]
    figure, axis = plt.subplots(figsize=(8, 0.42 * len(shown) + 1.4))
    axis.barh([variable_label(name) for name in shown.index], shown.values, color=BLUE, alpha=0.92)
    axis.set_xlim(0, shown.max() * 1.12)
    for y, value in enumerate(shown.values):
        axis.text(value * 1.01, y, f"{value:.2f}", va="center", fontsize=9)
    axis.set_xlabel("Mean |SHAP| per patient (log-odds), summed over each variable's columns")
    axis.set_title("What drives the XGBoost mortality risk", fontweight="bold")
    axis.grid(axis="x", alpha=0.2)
    axis.set_axisbelow(True)
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_beeswarm(contributions: np.ndarray, x: np.ndarray, names: list[str], top: int, path: Path) -> None:
    import shap

    shap.summary_plot(
        contributions,
        features=pd.DataFrame(x, columns=names),
        max_display=top,
        show=False,
        plot_size=(9, 0.42 * top + 1.6),
    )
    figure = plt.gcf()
    figure.axes[0].set_title("SHAP values on the sealed test set", fontweight="bold")
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_trajectories(groups: dict[str, dict], bin_hours: int, path: Path) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(11, 7.5), sharex=True)
    for axis, variable in zip(axes.ravel(), TRAJECTORY_VARIABLES):
        for label, color in (("high", ORANGE), ("low", BLUE)):
            summary = groups[label]["summaries"][variable]
            centres = np.arange(len(summary.median)) * bin_hours + bin_hours / 2
            axis.plot(centres, summary.median, color=color, marker="o", linewidth=2,
                      label=groups[label]["legend"])
            axis.fill_between(centres, summary.lower, summary.upper, color=color, alpha=0.15)
        axis.set_title(TRAJECTORY_LABELS[variable], fontweight="bold")
        axis.grid(alpha=0.2)
        axis.set_xticks(np.arange(0, 49, 12))
    for axis in axes[1]:
        axis.set_xlabel("Hours since ICU admission")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=2, fontsize=10, frameon=False)
    figure.suptitle(
        "Highest- vs lowest-risk quartile on the test set (median and interquartile range)",
        fontweight="bold",
    )
    figure.tight_layout(rect=(0, 0.05, 1, 1))
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_cases(cases: dict[str, dict], path: Path) -> None:
    figure, axes = plt.subplots(1, len(cases), figsize=(5.2 * len(cases), 4.6))
    axes = np.atleast_1d(axes)
    for axis, (name, case) in zip(axes, cases.items()):
        items = list(case["top_contributions"].items())[::-1]
        labels = [variable_label(label) for label, _ in items]
        values = [value for _, value in items]
        colors = [ORANGE if value > 0 else BLUE for value in values]
        axis.barh(labels, values, color=colors, alpha=0.92)
        axis.axvline(0, color="black", linewidth=0.8)
        axis.set_title(f"{CASE_TITLES[name]}\npredicted risk {case['probability']:.2f}",
                       fontweight="bold")
        axis.set_xlabel("Contribution to log-odds")
        axis.grid(axis="x", alpha=0.2)
        axis.set_axisbelow(True)
    figure.suptitle("Why the model scored three test patients the way it did "
                    "(orange raises risk, blue lowers it)", fontweight="bold")
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def run(args: argparse.Namespace) -> Path:
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    data = PhysioNet2012Loader(args.data_dir, args.outcomes_path).load()
    split = stratified_split(data.y, args.test_size, args.validation_size, args.seed)
    train = data.subset(split.train)
    validation = data.subset(split.validation)
    test = data.subset(split.test)

    preprocessor = XGBoostPreprocessor(args.window).fit(train)
    x_train, y_train = preprocessor.transform(train)
    x_validation, y_validation = preprocessor.transform(validation)
    x_test, y_test = preprocessor.transform(test)
    model = train_xgboost(x_train, y_train, x_validation, y_validation, seed=args.seed)
    validation_probability = model.predict_proba(x_validation)[:, 1]
    test_probability = model.predict_proba(x_test)[:, 1]
    print(
        f"[xgboost] {args.window}h window | test AUROC={roc_auc_score(y_test, test_probability):.4f} "
        f"| test AUPRC={average_precision_score(y_test, test_probability):.4f}"
    )

    contributions, base_value = xgboost_contributions(model, x_test)
    margin = model.predict(x_test, output_margin=True)
    additivity_error = float(np.max(np.abs(contributions.sum(axis=1) + base_value - margin)))
    if additivity_error > 1e-3:
        raise AssertionError(f"SHAP values do not add up to the model output ({additivity_error}).")

    names = preprocessor.feature_names
    readable = [readable_feature_name(name, args.window) for name in names]
    grouped = group_by_variable(contributions, names)
    importance = variable_importance(grouped)
    column_importance = pd.Series(np.abs(contributions).mean(axis=0), index=readable)
    column_importance = column_importance.sort_values(ascending=False)

    plot_variable_importance(importance, args.top, output_dir / "variable_importance.png")
    plot_beeswarm(contributions, x_test, readable, args.top, output_dir / "shap_beeswarm.png")

    # Risk quartiles come from the model's own test predictions.
    low_cut, high_cut = np.quantile(test_probability, [0.25, 0.75])
    trajectories = binned_trajectories(test, TRAJECTORY_VARIABLES, args.bin_hours)
    groups: dict[str, dict] = {}
    for label, members in (
        ("high", test_probability >= high_cut),
        ("low", test_probability <= low_cut),
    ):
        mortality = float(y_test[members].mean())
        groups[label] = {
            "n": int(members.sum()),
            "observed_mortality": mortality,
            "legend": (
                f"{'Highest' if label == 'high' else 'Lowest'} 25% predicted risk "
                f"(n={int(members.sum())}, {mortality:.0%} died)"
            ),
            "summaries": {
                name: summarize_group(values[members]) for name, values in trajectories.items()
            },
        }
    plot_trajectories(groups, args.bin_hours, output_dir / "risk_trajectories.png")

    threshold = select_threshold(y_validation, validation_probability, args.target_recall)
    picked = select_cases(y_test, test_probability, threshold)
    cases: dict[str, dict] = {}
    for name, index in picked.items():
        row = grouped.iloc[index]
        top = row.reindex(row.abs().sort_values(ascending=False).index).head(args.case_top)
        cases[name] = {
            "probability": float(test_probability[index]),
            "died": bool(y_test[index]),
            "top_contributions": {key: round(float(value), 4) for key, value in top.items()},
        }
    plot_cases(cases, output_dir / "case_explanations.png")

    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "window_hours": args.window,
        "n_test": len(test),
        "test_auroc": float(roc_auc_score(y_test, test_probability)),
        "test_auprc": float(average_precision_score(y_test, test_probability)),
        "shap": {
            "method": "XGBoost native TreeSHAP at the early-stopped iteration (log-odds)",
            "base_value": base_value,
            "max_additivity_error": additivity_error,
            "variable_importance_top": {
                key: round(float(value), 4) for key, value in importance.head(args.top).items()
            },
            "column_importance_top": {
                key: round(float(value), 4)
                for key, value in column_importance.head(args.top).items()
            },
        },
        "risk_groups": {
            label: {
                "n": block["n"],
                "observed_mortality": round(block["observed_mortality"], 4),
                "median_by_bin": {
                    name: [None if np.isnan(v) else round(float(v), 2) for v in s.median]
                    for name, s in block["summaries"].items()
                },
            }
            for label, block in groups.items()
        },
        "trajectory_bin_hours": args.bin_hours,
        "case_threshold": {
            "rule": f"highest validation threshold with sensitivity >= {args.target_recall:.2f}",
            "value": float(threshold),
        },
        "cases": cases,
    }
    report_path = output_dir / "interpretation.json"
    report_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return report_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--outcomes-path")
    parser.add_argument("--output-dir", default="results/interpretation")
    parser.add_argument("--window", type=int, default=12,
                        help="XGBoost window selected by validation AUPRC in the benchmark.")
    parser.add_argument("--test-size", type=float, default=0.10)
    parser.add_argument("--validation-size", type=float, default=0.10)
    parser.add_argument("--target-recall", type=float, default=0.80)
    parser.add_argument("--bin-hours", type=int, default=6)
    parser.add_argument("--top", type=int, default=15)
    parser.add_argument("--case-top", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    print(run(build_parser().parse_args()))


if __name__ == "__main__":
    main()
