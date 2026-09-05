#!/usr/bin/env python3
"""Combine independently completed model runs using validation-only weights."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from icu_mortality.ensemble import (
    ConvexProbabilityEnsemble,
    CrossValidatedSigmoidCalibration,
)
from icu_mortality.metrics import (
    binary_metrics,
    bootstrap_confidence_intervals,
    paired_bootstrap_difference,
    select_threshold,
)


def load_split(run_dir: Path, split: str) -> pd.DataFrame:
    path = run_dir / f"{split}_predictions.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}")
    return pd.read_csv(path, dtype={"patient_id": str})


def collect(run_dirs: list[Path], split: str) -> tuple[pd.DataFrame, list[str]]:
    merged: pd.DataFrame | None = None
    model_names: list[str] = []
    for run_dir in run_dirs:
        frame = load_split(run_dir, split)
        probability_columns = [
            column
            for column in frame.columns
            if column.startswith("raw_probability__")
        ]
        if not probability_columns:
            probability_columns = [
                column
                for column in frame.columns
                if column.startswith("probability__") and column != "probability__ensemble"
            ]
        if len(probability_columns) != 1:
            raise ValueError(
                f"{run_dir} must contain exactly one base-model probability column; "
                f"found {probability_columns}."
            )
        column = probability_columns[0]
        model_name = column.removeprefix("raw_probability__").removeprefix("probability__")
        if model_name in model_names:
            raise ValueError(f"Duplicate model run: {model_name}")
        model_names.append(model_name)
        selected = frame[["patient_id", "label", column]].copy()
        selected = selected.rename(columns={column: f"raw_probability__{model_name}"})
        if merged is None:
            merged = selected
        else:
            merged = merged.merge(
                selected, on=["patient_id", "label"], how="inner", validate="one_to_one"
            )
    if merged is None:
        raise ValueError("No run directories supplied.")
    return merged, model_names


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", nargs="+")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--target-recall", type=float, default=0.80)
    args = parser.parse_args()
    run_dirs = [Path(path).expanduser().resolve() for path in args.run_dirs]
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    validation, validation_names = collect(run_dirs, "validation")
    test, test_names = collect(run_dirs, "test")
    if validation_names != test_names:
        raise ValueError("Validation and test model columns do not match.")
    model_names = validation_names
    raw_columns = [f"raw_probability__{name}" for name in model_names]
    raw_validation_matrix = validation[raw_columns].to_numpy(float)
    raw_test_matrix = test[raw_columns].to_numpy(float)
    calibration = CrossValidatedSigmoidCalibration(len(model_names), n_splits=5, seed=42)
    validation_matrix = calibration.fit_transform(
        validation["label"].to_numpy(int), raw_validation_matrix
    )
    test_matrix = calibration.transform(raw_test_matrix)
    for model_index, model_name in enumerate(model_names):
        validation[f"probability__{model_name}"] = validation_matrix[:, model_index]
        test[f"probability__{model_name}"] = test_matrix[:, model_index]
    ensemble = ConvexProbabilityEnsemble(model_names).fit(
        validation["label"].to_numpy(int), validation_matrix
    )
    validation["probability__ensemble"] = ensemble.predict(validation_matrix)
    test["probability__ensemble"] = ensemble.predict(test_matrix)

    metrics = {}
    thresholds = {}
    for name in model_names + ["ensemble"]:
        column = f"probability__{name}"
        threshold = select_threshold(
            validation["label"].to_numpy(int),
            validation[column].to_numpy(float),
            args.target_recall,
        )
        thresholds[name] = threshold
        metrics[name] = {
            "validation": binary_metrics(
                validation["label"].to_numpy(int), validation[column].to_numpy(float), threshold
            ),
            "test": binary_metrics(
                test["label"].to_numpy(int), test[column].to_numpy(float), threshold
            ),
        }

    validation.to_csv(output_dir / "validation_predictions.csv", index=False)
    test.to_csv(output_dir / "test_predictions.csv", index=False)
    report = {
        "status": "benchmark_combined",
        "source_runs": [str(path) for path in run_dirs],
        "models": model_names,
        "ensemble_weights": ensemble.weight_dict(),
        "calibration": calibration.description,
        "thresholds": thresholds,
        "metrics": metrics,
        "test_confidence_intervals": {
            name: bootstrap_confidence_intervals(
                test["label"].to_numpy(int),
                test[f"probability__{name}"].to_numpy(float),
                n_bootstrap=2000,
                seed=42,
            )
            for name in model_names + ["ensemble"]
        },
    }
    if "xgboost" in model_names:
        report["ensemble_minus_xgboost_paired_ci"] = paired_bootstrap_difference(
            test["label"].to_numpy(int),
            test["probability__ensemble"].to_numpy(float),
            test["probability__xgboost"].to_numpy(float),
            n_bootstrap=2000,
            seed=42,
        )
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(output_dir / "report.json")


if __name__ == "__main__":
    main()
