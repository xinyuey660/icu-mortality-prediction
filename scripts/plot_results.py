#!/usr/bin/env python3
"""Create GitHub-ready model comparison and calibration figures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve

DISPLAY_NAMES = {
    "xgboost": "XGBoost",
    "grud": "GRU-D",
    "gru_attention": "GRU-Attention",
    "bigru_attention": "BiGRU-Attention",
    "triplet_transformer": "Triplet Transformer",
    "ensemble": "Learned Ensemble",
}


def extract_test_metrics(report: dict) -> dict:
    if "test_metrics" in report:
        return report["test_metrics"]
    return {name: block["test"] for name, block in report["metrics"].items()}


def plot_comparison(report: dict, output_path: Path) -> None:
    metrics = extract_test_metrics(report)
    order = [name for name in DISPLAY_NAMES if name in metrics]
    labels = [DISPLAY_NAMES[name] for name in order]
    y = np.arange(len(order))
    colors = ["#2A6F97" if name != "ensemble" else "#D97706" for name in order]

    figure, axes = plt.subplots(1, 3, figsize=(14, 5.5), sharey=True)
    definitions = [
        ("auroc", "AUROC ↑", (0.75, 0.92)),
        ("auprc", "AUPRC ↑", (0.35, 0.68)),
        ("brier", "Brier score ↓", (0.075, 0.105)),
    ]
    for axis, (key, title, limits) in zip(axes, definitions):
        values = [metrics[name][key] for name in order]
        bars = axis.barh(y, values, color=colors, alpha=0.92)
        axis.set_title(title, fontweight="bold")
        axis.set_xlim(*limits)
        axis.grid(axis="x", alpha=0.2)
        axis.set_axisbelow(True)
        for bar, value in zip(bars, values):
            axis.text(
                value + (limits[1] - limits[0]) * 0.012,
                bar.get_y() + bar.get_height() / 2,
                f"{value:.3f}",
                va="center",
                fontsize=9,
            )
    axes[0].set_yticks(y, labels)
    axes[0].invert_yaxis()
    figure.suptitle("PhysioNet 2012 — sealed test set (n=400)", fontsize=14, fontweight="bold")
    figure.text(
        0.5,
        0.01,
        "Five-fold OOF sigmoid calibration on validation; single split, seed 42",
        ha="center",
        fontsize=9,
        color="#4B5563",
    )
    figure.tight_layout(rect=(0, 0.04, 1, 0.94))
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_calibration(predictions_path: Path, output_path: Path) -> None:
    frame = pd.read_csv(predictions_path)
    y_true = frame["label"].to_numpy(int)
    figure, axis = plt.subplots(figsize=(7, 6))
    axis.plot([0, 1], [0, 1], linestyle="--", color="#6B7280", label="Perfect calibration")
    for name, color in (("xgboost", "#2A6F97"), ("ensemble", "#D97706")):
        probabilities = frame[f"probability__{name}"].to_numpy(float)
        observed, predicted = calibration_curve(
            y_true, probabilities, n_bins=8, strategy="quantile"
        )
        axis.plot(
            predicted,
            observed,
            marker="o",
            linewidth=2,
            color=color,
            label=DISPLAY_NAMES[name],
        )
    axis.set_xlabel("Mean predicted probability")
    axis.set_ylabel("Observed mortality rate")
    axis.set_title("Calibration on sealed test set", fontweight="bold")
    axis.grid(alpha=0.2)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    report = json.loads(Path(args.report).read_text(encoding="utf-8"))
    plot_comparison(report, output_dir / "model_comparison.png")
    plot_calibration(Path(args.predictions), output_dir / "calibration_curve.png")
    print(output_dir)


if __name__ == "__main__":
    main()
