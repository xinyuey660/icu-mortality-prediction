"""Deterministic patient-level train/validation/test splitting."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class SplitIndices:
    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray

    def validate(self, n_samples: int) -> None:
        sets = [set(x.tolist()) for x in (self.train, self.validation, self.test)]
        if sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2]:
            raise ValueError("Patient leakage detected between splits.")
        combined = sets[0] | sets[1] | sets[2]
        if combined != set(range(n_samples)):
            raise ValueError("Split indices do not cover every patient exactly once.")


def stratified_split(
    y: np.ndarray,
    test_size: float = 0.10,
    validation_size: float = 0.10,
    seed: int = 42,
) -> SplitIndices:
    """Create an 80/10/10-style split using only patient-level labels."""

    y = np.asarray(y, dtype=int)
    if test_size <= 0 or validation_size <= 0 or test_size + validation_size >= 1:
        raise ValueError("test_size and validation_size must be positive and sum to less than one.")
    all_idx = np.arange(len(y))
    dev_idx, test_idx = train_test_split(
        all_idx, test_size=test_size, stratify=y, random_state=seed,
    )
    validation_fraction_of_dev = validation_size / (1.0 - test_size)
    train_idx, validation_idx = train_test_split(
        dev_idx,
        test_size=validation_fraction_of_dev,
        stratify=y[dev_idx],
        random_state=seed + 1,
    )
    split = SplitIndices(
        train=np.sort(train_idx), validation=np.sort(validation_idx), test=np.sort(test_idx)
    )
    split.validate(len(y))
    return split

