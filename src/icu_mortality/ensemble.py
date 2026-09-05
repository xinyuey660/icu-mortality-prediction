"""Validation/OOF-only probability calibration and convex ensembling."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold


def _logit(probabilities: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(probabilities, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1.0 - p))


@dataclass
class SigmoidCalibrator:
    model: LogisticRegression | None = None

    def fit(self, y_true: np.ndarray, probabilities: np.ndarray) -> SigmoidCalibrator:
        self.model = LogisticRegression(C=10.0, solver="lbfgs", random_state=42)
        self.model.fit(_logit(probabilities).reshape(-1, 1), np.asarray(y_true, dtype=int))
        return self

    def predict(self, probabilities: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Calibrator has not been fitted.")
        return self.model.predict_proba(_logit(probabilities).reshape(-1, 1))[:, 1]


class CrossValidatedSigmoidCalibration:
    """OOF Platt scaling for model selection plus full-fit test transforms."""

    def __init__(self, n_models: int, n_splits: int = 5, seed: int = 42):
        self.n_models = n_models
        self.n_splits = n_splits
        self.seed = seed
        self.calibrators: list[SigmoidCalibrator | None] = []
        self.description = "not fitted"

    def fit_transform(self, y_true: np.ndarray, probability_matrix: np.ndarray) -> np.ndarray:
        y_true = np.asarray(y_true, dtype=int)
        matrix = np.asarray(probability_matrix, dtype=float)
        if matrix.shape != (len(y_true), self.n_models):
            raise ValueError("Probability matrix has the wrong shape.")
        output = np.zeros_like(matrix, dtype=float)
        self.calibrators = []
        class_counts = np.bincount(y_true, minlength=2)
        minority_count = int(class_counts.min())
        if minority_count == 0:
            self.calibrators = [None] * self.n_models
            self.description = "identity calibration (validation contains one class)"
            return matrix.copy()
        if minority_count == 1:
            for model_index in range(self.n_models):
                calibrator = SigmoidCalibrator().fit(y_true, matrix[:, model_index])
                self.calibrators.append(calibrator)
                output[:, model_index] = calibrator.predict(matrix[:, model_index])
            self.description = "full-validation sigmoid (too few minority samples for OOF)"
            return output

        effective_splits = min(self.n_splits, minority_count)
        splitter = StratifiedKFold(
            n_splits=effective_splits, shuffle=True, random_state=self.seed
        )
        for model_index in range(self.n_models):
            probabilities = matrix[:, model_index]
            for train_index, holdout_index in splitter.split(probabilities, y_true):
                fold_calibrator = SigmoidCalibrator().fit(
                    y_true[train_index], probabilities[train_index]
                )
                output[holdout_index, model_index] = fold_calibrator.predict(
                    probabilities[holdout_index]
                )
            self.calibrators.append(SigmoidCalibrator().fit(y_true, probabilities))
        self.description = (
            f"{effective_splits}-fold OOF sigmoid on validation; "
            "full-validation refit applied to test"
        )
        return output

    def transform(self, probability_matrix: np.ndarray) -> np.ndarray:
        matrix = np.asarray(probability_matrix, dtype=float)
        if len(self.calibrators) != self.n_models:
            raise RuntimeError("Calibration has not been fitted.")
        if matrix.ndim != 2 or matrix.shape[1] != self.n_models:
            raise ValueError("Probability matrix has the wrong number of columns.")
        return np.column_stack(
            [
                (
                    matrix[:, model_index]
                    if calibrator is None
                    else calibrator.predict(matrix[:, model_index])
                )
                for model_index, calibrator in enumerate(self.calibrators)
            ]
        )


class ConvexProbabilityEnsemble:
    """Non-negative weights summing to one, learned without test labels."""

    def __init__(self, model_names: list[str]):
        if not model_names:
            raise ValueError("At least one model is required.")
        self.model_names = list(model_names)
        self.weights: np.ndarray | None = None

    def fit(self, y_true: np.ndarray, probability_matrix: np.ndarray) -> ConvexProbabilityEnsemble:
        matrix = np.asarray(probability_matrix, dtype=float)
        y_true = np.asarray(y_true, dtype=int)
        if matrix.shape != (len(y_true), len(self.model_names)):
            raise ValueError("Probability matrix must have shape (patients, models).")

        def objective(weights: np.ndarray) -> float:
            blended = np.clip(matrix @ weights, 1e-7, 1 - 1e-7)
            return float(log_loss(y_true, blended, labels=[0, 1]))

        n_models = len(self.model_names)
        result = minimize(
            objective,
            x0=np.full(n_models, 1.0 / n_models),
            method="SLSQP",
            bounds=[(0.0, 1.0)] * n_models,
            constraints={"type": "eq", "fun": lambda w: float(w.sum() - 1.0)},
            options={"maxiter": 1000, "ftol": 1e-10},
        )
        if not result.success:
            raise RuntimeError(f"Ensemble optimization failed: {result.message}")
        weights = np.maximum(result.x, 0.0)
        self.weights = weights / weights.sum()
        return self

    def predict(self, probability_matrix: np.ndarray) -> np.ndarray:
        if self.weights is None:
            raise RuntimeError("Ensemble has not been fitted.")
        matrix = np.asarray(probability_matrix, dtype=float)
        if matrix.ndim != 2 or matrix.shape[1] != len(self.weights):
            raise ValueError("Probability matrix has the wrong number of columns.")
        return np.clip(matrix @ self.weights, 0.0, 1.0)

    def weight_dict(self) -> dict[str, float]:
        if self.weights is None:
            raise RuntimeError("Ensemble has not been fitted.")
        return {
            name: float(weight) for name, weight in zip(self.model_names, self.weights)
        }
