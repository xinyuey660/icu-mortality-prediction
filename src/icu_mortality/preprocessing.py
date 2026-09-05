"""Training-only preprocessing for tree, recurrent, and triplet models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data import DYNAMIC_FEATURES, STATIC_FEATURES, ICUDataset

LOG_FEATURES = {
    "ALP", "ALT", "AST", "Bilirubin", "BUN", "Creatinine", "Glucose", "Lactate",
    "Mg", "PaCO2", "PaO2", "TroponinI", "TroponinT", "Urine", "WBC", "Weight",
}


def compute_delta(mask: np.ndarray) -> np.ndarray:
    """Hours since each feature's previous observation; first gaps count from admission."""

    mask = np.asarray(mask, dtype=np.float32)
    delta = np.zeros_like(mask, dtype=np.float32)
    for time_index in range(mask.shape[1]):
        if time_index == 0:
            delta[:, time_index] = 1.0 - mask[:, time_index]
        else:
            delta[:, time_index] = np.where(
                mask[:, time_index] > 0,
                0.0,
                delta[:, time_index - 1] + 1.0,
            )
    return delta


@dataclass(frozen=True)
class SequenceBatch:
    values: np.ndarray
    mask: np.ndarray
    delta: np.ndarray
    static: np.ndarray
    y: np.ndarray
    patient_ids: np.ndarray


class SequencePreprocessor:
    """Prepare hourly tensors; all fitted values come from the training split."""

    def __init__(self):
        self.value_mean: np.ndarray | None = None
        self.value_std: np.ndarray | None = None
        self.static_median: np.ndarray | None = None
        self.static_mean: np.ndarray | None = None
        self.static_std: np.ndarray | None = None

    @staticmethod
    def _log_transform(values: np.ndarray) -> np.ndarray:
        output = values.copy()
        for index, name in enumerate(DYNAMIC_FEATURES):
            if name in LOG_FEATURES:
                observed = np.isfinite(output[:, :, index])
                output[:, :, index] = np.where(
                    observed, np.log1p(np.maximum(output[:, :, index], 0.0)), np.nan
                )
        return output

    def fit(self, train: ICUDataset) -> SequencePreprocessor:
        values = self._log_transform(train.value_last.astype(np.float64))
        finite = np.isfinite(values)
        counts = finite.sum(axis=(0, 1))
        sums = np.where(finite, values, 0.0).sum(axis=(0, 1))
        self.value_mean = np.divide(
            sums, counts, out=np.zeros_like(sums), where=counts > 0
        )
        centered = np.where(finite, values - self.value_mean, 0.0)
        variance = np.divide(
            np.square(centered).sum(axis=(0, 1)),
            counts,
            out=np.ones_like(sums),
            where=counts > 0,
        )
        self.value_std = np.sqrt(variance)
        self.value_std = np.where(self.value_std > 1e-6, self.value_std, 1.0)

        static = train.static.astype(np.float64)
        with np.errstate(invalid="ignore"):
            self.static_median = np.nanmedian(static, axis=0)
        self.static_median = np.nan_to_num(self.static_median, nan=0.0)
        filled = np.where(np.isfinite(static), static, self.static_median)
        self.static_mean = filled.mean(axis=0)
        self.static_std = filled.std(axis=0)
        self.static_std = np.where(self.static_std > 1e-6, self.static_std, 1.0)
        return self

    def transform(self, data: ICUDataset) -> SequenceBatch:
        if any(
            value is None
            for value in (
                self.value_mean, self.value_std, self.static_median,
                self.static_mean, self.static_std,
            )
        ):
            raise RuntimeError("Call fit(train) before transform().")
        mask = data.mask
        values = self._log_transform(data.value_last.astype(np.float64))
        values = (values - self.value_mean) / self.value_std
        values = np.where(mask > 0, values, 0.0).astype(np.float32)
        delta = np.log1p(compute_delta(mask)).astype(np.float32)

        static_missing = (~np.isfinite(data.static)).astype(np.float32)
        static = np.where(np.isfinite(data.static), data.static, self.static_median)
        static = ((static - self.static_mean) / self.static_std).astype(np.float32)
        static = np.concatenate([static, static_missing], axis=1).astype(np.float32)
        return SequenceBatch(
            values=values, mask=mask, delta=delta, static=static,
            y=data.y.astype(np.float32), patient_ids=data.patient_ids.copy(),
        )

    def fit_transform(self, train: ICUDataset) -> SequenceBatch:
        return self.fit(train).transform(train)


class XGBoostPreprocessor:
    """Create compact window summaries while preserving native missing values."""

    stats = ("mean", "last", "range", "trend", "observed_rate", "delta")

    def __init__(self, window_hours: int = 6):
        if window_hours not in {1, 3, 6, 12, 24} or 48 % window_hours:
            raise ValueError("window_hours must be one of 1, 3, 6, 12, 24.")
        self.window_hours = window_hours
        self.static_median: np.ndarray | None = None
        self.feature_names: list[str] = []

    def fit(self, train: ICUDataset) -> XGBoostPreprocessor:
        with np.errstate(invalid="ignore"):
            self.static_median = np.nanmedian(train.static, axis=0)
        self.static_median = np.nan_to_num(self.static_median, nan=0.0)
        self.feature_names = self._make_names()
        return self

    def _make_names(self) -> list[str]:
        names: list[str] = []
        n_windows = 48 // self.window_hours
        for window in range(n_windows):
            for feature in DYNAMIC_FEATURES:
                for stat in self.stats:
                    names.append(f"w{window:02d}__{feature}__{stat}")
        names.extend(STATIC_FEATURES)
        names.extend(f"{name}__missing" for name in STATIC_FEATURES)
        return names

    def transform(self, data: ICUDataset) -> tuple[np.ndarray, np.ndarray]:
        if self.static_median is None:
            raise RuntimeError("Call fit(train) before transform().")
        n, _, f = data.count.shape
        w = self.window_hours
        k = 48 // w
        count = data.count.reshape(n, k, w, f)
        value_sum = data.value_sum.reshape(n, k, w, f)
        value_min = data.value_min.reshape(n, k, w, f)
        value_max = data.value_max.reshape(n, k, w, f)
        first = data.value_first.reshape(n, k, w, f)
        last = data.value_last.reshape(n, k, w, f)

        count_w = count.sum(axis=2)
        sum_w = value_sum.sum(axis=2)
        mean_w = np.divide(
            sum_w, count_w, out=np.full_like(sum_w, np.nan), where=count_w > 0
        )
        min_w = np.min(np.where(np.isfinite(value_min), value_min, np.inf), axis=2)
        max_w = np.max(np.where(np.isfinite(value_max), value_max, -np.inf), axis=2)
        observed = count > 0
        first_pos = observed.argmax(axis=2)
        last_pos = w - 1 - observed[:, :, ::-1, :].argmax(axis=2)
        first_w = np.take_along_axis(first, first_pos[:, :, None, :], axis=2).squeeze(2)
        last_w = np.take_along_axis(last, last_pos[:, :, None, :], axis=2).squeeze(2)
        empty = count_w == 0
        min_w[empty] = np.nan
        max_w[empty] = np.nan
        first_w[empty] = np.nan
        last_w[empty] = np.nan
        range_w = max_w - min_w
        trend_w = last_w - first_w
        observed_rate = np.minimum(count_w / float(w), 1.0)

        hourly_delta = compute_delta(data.mask)
        delta_w = hourly_delta.reshape(n, k, w, f)[:, :, -1, :]
        blocks = [mean_w, last_w, range_w, trend_w, observed_rate, delta_w]
        temporal = np.stack(blocks, axis=-1).reshape(n, -1)

        static_missing = (~np.isfinite(data.static)).astype(np.float32)
        static = np.where(np.isfinite(data.static), data.static, self.static_median)
        matrix = np.concatenate([temporal, static, static_missing], axis=1).astype(np.float32)
        if matrix.shape[1] != len(self.feature_names):
            raise AssertionError("Feature-name and matrix dimensions diverged.")
        return matrix, data.y.astype(np.int64)

    def fit_transform(self, train: ICUDataset) -> tuple[np.ndarray, np.ndarray]:
        return self.fit(train).transform(train)


def build_triplets(
    batch: SequenceBatch,
    max_tokens: int = 512,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Convert a dense hourly batch to padded (feature, value, time) tokens.

    If a patient has more observations than ``max_tokens``, deterministic
    evenly-spaced sampling preserves coverage of the full 48-hour horizon.
    ``len(DYNAMIC_FEATURES)`` is a no-observation token and the next id is
    padding. The explicit token prevents all-padding Transformer rows.
    """

    n, hours, features = batch.values.shape
    no_observation_id = features
    pad_id = features + 1
    feature_ids = np.full((n, max_tokens), pad_id, dtype=np.int64)
    values = np.zeros((n, max_tokens), dtype=np.float32)
    times = np.zeros((n, max_tokens), dtype=np.float32)
    padding_mask = np.ones((n, max_tokens), dtype=bool)

    for patient in range(n):
        positions = np.argwhere(batch.mask[patient] > 0)
        if len(positions) > max_tokens:
            choice = np.linspace(0, len(positions) - 1, max_tokens).round().astype(int)
            positions = positions[choice]
        length = len(positions)
        if not length:
            feature_ids[patient, 0] = no_observation_id
            padding_mask[patient, 0] = False
            continue
        time_index = positions[:, 0]
        feature_index = positions[:, 1]
        feature_ids[patient, :length] = feature_index
        values[patient, :length] = batch.values[patient, time_index, feature_index]
        times[patient, :length] = time_index.astype(np.float32) / float(hours - 1)
        padding_mask[patient, :length] = False
    return feature_ids, values, times, padding_mask
