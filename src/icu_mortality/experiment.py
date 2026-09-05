"""Run the benchmark from raw PhysioNet files to a sealed test report."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from .data import DYNAMIC_FEATURES, PhysioNet2012Loader
from .ensemble import ConvexProbabilityEnsemble, CrossValidatedSigmoidCalibration
from .metrics import binary_metrics, select_threshold
from .preprocessing import SequencePreprocessor, XGBoostPreprocessor
from .splits import stratified_split
from .training import train_torch_model, train_xgboost

MODEL_CHOICES = ("xgboost", "grud", "gru_attention", "bigru_attention", "triplet_transformer")


def _json_default(value):
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Cannot serialize {type(value)!r}")


def _model(model_name: str, input_dim: int, static_dim: int):
    from .models import (
        BiGRUAttentionClassifier,
        GRUAttentionClassifier,
        GRUDClassifier,
        TripletTransformerClassifier,
    )

    if model_name == "grud":
        return GRUDClassifier(input_dim, static_dim)
    if model_name == "gru_attention":
        return GRUAttentionClassifier(input_dim, static_dim)
    if model_name == "bigru_attention":
        return BiGRUAttentionClassifier(input_dim, static_dim)
    if model_name == "triplet_transformer":
        return TripletTransformerClassifier(input_dim, static_dim)
    raise ValueError(model_name)


def run(args: argparse.Namespace) -> Path:
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    data = PhysioNet2012Loader(args.data_dir, args.outcomes_path).load(limit=args.limit)
    split = stratified_split(data.y, args.test_size, args.validation_size, args.seed)
    train = data.subset(split.train)
    validation = data.subset(split.validation)
    test = data.subset(split.test)
    print(
        f"Loaded {len(data)} patients | train={len(train)} | "
        f"validation={len(validation)} | sealed_test={len(test)}"
    )

    split_frame = pd.DataFrame(
        {
            "patient_id": data.patient_ids,
            "label": data.y,
            "split": np.full(len(data), "", dtype=object),
        }
    )
    split_frame.loc[split.train, "split"] = "train"
    split_frame.loc[split.validation, "split"] = "validation"
    split_frame.loc[split.test, "split"] = "test"
    split_frame.to_csv(output_dir / "split_manifest.csv", index=False)

    validation_predictions: dict[str, np.ndarray] = {}
    test_predictions: dict[str, np.ndarray] = {}
    training_details: dict[str, object] = {}

    if "xgboost" in args.models:
        candidates = []
        for window in args.xgb_windows:
            print(f"[xgboost] fitting {window}h window candidate")
            preprocessor = XGBoostPreprocessor(window).fit(train)
            x_train, y_train = preprocessor.transform(train)
            x_validation, y_validation = preprocessor.transform(validation)
            model = train_xgboost(
                x_train, y_train, x_validation, y_validation, seed=args.seed
            )
            val_probability = model.predict_proba(x_validation)[:, 1]
            candidates.append(
                (average_precision_score(y_validation, val_probability), window, preprocessor, model, val_probability)
            )
            print(f"[xgboost] {window}h validation AUPRC={candidates[-1][0]:.4f}")
        score, window, preprocessor, model, val_probability = max(candidates, key=lambda item: item[0])
        x_test, _ = preprocessor.transform(test)
        validation_predictions["xgboost"] = val_probability
        test_predictions["xgboost"] = model.predict_proba(x_test)[:, 1]
        training_details["xgboost"] = {
            "selected_window_hours": int(window),
            "validation_auprc": float(score),
            "best_iteration": int(getattr(model, "best_iteration", 0)),
        }
        print(f"[xgboost] selected {window}h window")

    neural_models = [name for name in args.models if name != "xgboost"]
    if neural_models:
        sequence_preprocessor = SequencePreprocessor().fit(train)
        train_batch = sequence_preprocessor.transform(train)
        validation_batch = sequence_preprocessor.transform(validation)
        test_batch = sequence_preprocessor.transform(test)
        input_dim = len(DYNAMIC_FEATURES)
        static_dim = train_batch.static.shape[1]
        for offset, model_name in enumerate(neural_models):
            print(f"[{model_name}] training started")
            result = train_torch_model(
                _model(model_name, input_dim, static_dim),
                train_batch,
                validation_batch,
                test_batch,
                triplet=model_name == "triplet_transformer",
                max_tokens=args.max_tokens,
                epochs=args.epochs,
                patience=args.patience,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                seed=args.seed + offset,
                device=args.device,
            )
            validation_predictions[model_name] = result.validation_probabilities
            test_predictions[model_name] = result.test_probabilities
            training_details[model_name] = {
                "best_epoch": result.best_epoch,
                "history": result.history,
            }
            print(
                f"[{model_name}] best_epoch={result.best_epoch} | "
                f"validation AUPRC={average_precision_score(validation.y, result.validation_probabilities):.4f}"
            )

    names = list(validation_predictions)
    raw_validation_matrix = np.column_stack([validation_predictions[name] for name in names])
    raw_test_matrix = np.column_stack([test_predictions[name] for name in names])
    calibration = CrossValidatedSigmoidCalibration(
        len(names), n_splits=5, seed=args.seed
    )
    validation_matrix = calibration.fit_transform(validation.y, raw_validation_matrix)
    test_matrix = calibration.transform(raw_test_matrix)
    for model_index, name in enumerate(names):
        validation_predictions[name] = validation_matrix[:, model_index]
        test_predictions[name] = test_matrix[:, model_index]
    ensemble = ConvexProbabilityEnsemble(names).fit(validation.y, validation_matrix)
    validation_predictions["ensemble"] = ensemble.predict(validation_matrix)
    test_predictions["ensemble"] = ensemble.predict(test_matrix)
    print(f"[ensemble] validation-learned weights={ensemble.weight_dict()}")

    metrics: dict[str, dict[str, dict[str, float | int]]] = {}
    thresholds: dict[str, float] = {}
    for name in names + ["ensemble"]:
        threshold = select_threshold(
            validation.y, validation_predictions[name], target_recall=args.target_recall
        )
        thresholds[name] = threshold
        metrics[name] = {
            "validation": binary_metrics(validation.y, validation_predictions[name], threshold),
            "test": binary_metrics(test.y, test_predictions[name], threshold),
        }

    prediction_frame = pd.DataFrame(
        {"patient_id": test.patient_ids, "label": test.y}
    )
    validation_prediction_frame = pd.DataFrame(
        {"patient_id": validation.patient_ids, "label": validation.y}
    )
    for name in names + ["ensemble"]:
        if name in names:
            model_index = names.index(name)
            prediction_frame[f"raw_probability__{name}"] = raw_test_matrix[:, model_index]
            validation_prediction_frame[f"raw_probability__{name}"] = (
                raw_validation_matrix[:, model_index]
            )
        prediction_frame[f"probability__{name}"] = test_predictions[name]
        validation_prediction_frame[f"probability__{name}"] = validation_predictions[name]
    prediction_frame.to_csv(output_dir / "test_predictions.csv", index=False)
    validation_prediction_frame.to_csv(
        output_dir / "validation_predictions.csv", index=False
    )

    report = {
        "status": "benchmark_complete",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "dataset": {
            "n_patients": len(data),
            "n_train": len(train),
            "n_validation": len(validation),
            "n_test": len(test),
            "mortality_rate": float(data.y.mean()),
            "dynamic_features": len(DYNAMIC_FEATURES),
            "static_features": 4,
        },
        "models": names,
        "ensemble_weights": ensemble.weight_dict(),
        "calibration": calibration.description,
        "threshold_selection": f"validation sensitivity >= {args.target_recall:.2f}",
        "thresholds": thresholds,
        "metrics": metrics,
        "training": training_details,
        "limitations": [
            "Single public benchmark; no external clinical validation.",
            "Validation split is used for early stopping and ensemble selection.",
            "Predictions are research outputs and not clinical decision support.",
        ],
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )
    return output_dir / "report.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--outcomes-path")
    parser.add_argument("--output-dir", default="results/runs/benchmark_v1")
    parser.add_argument("--models", nargs="+", choices=MODEL_CHOICES, default=list(MODEL_CHOICES))
    parser.add_argument("--xgb-windows", nargs="+", type=int, default=[3, 6, 12])
    parser.add_argument("--test-size", type=float, default=0.10)
    parser.add_argument("--validation-size", type=float, default=0.10)
    parser.add_argument("--target-recall", type=float, default=0.80)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--limit", type=int)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    print(run(args))


if __name__ == "__main__":
    main()
