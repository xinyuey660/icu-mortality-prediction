from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from icu_mortality.data import DYNAMIC_FEATURES, PhysioNet2012Loader
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
from icu_mortality.preprocessing import SequencePreprocessor, XGBoostPreprocessor, build_triplets
from icu_mortality.splits import stratified_split


class CorePipelineTests(unittest.TestCase):
    def _dataset(self, root: Path, patients: int = 20):
        set_dir = root / "set-a"
        set_dir.mkdir(parents=True)
        outcomes = []
        for index in range(patients):
            patient_id = str(100000 + index)
            mortality = index % 2
            outcomes.append({"RecordID": patient_id, "In-hospital_death": mortality})
            frame = pd.DataFrame(
                [
                    ("00:00", "RecordID", int(patient_id)),
                    ("00:00", "Age", 45 + index),
                    ("00:00", "Gender", index % 2),
                    ("00:00", "Height", 170),
                    ("00:00", "ICUType", 2),
                    ("00:10", "NISysABP", 120),
                    ("00:20", "SysABP", 110),
                    ("01:00", "HR", 70 + mortality * 30),
                    ("bad", "HR", 999),
                ],
                columns=["Time", "Parameter", "Value"],
            )
            frame.to_csv(set_dir / f"{patient_id}.txt", index=False)
        pd.DataFrame(outcomes).to_csv(root / "Outcomes-a.txt", index=False)
        return PhysioNet2012Loader(root).load()

    def test_loader_merges_bp_and_rejects_bad_time(self):
        with tempfile.TemporaryDirectory() as temp:
            data = self._dataset(Path(temp))
        sys_index = DYNAMIC_FEATURES.index("BP_Sys")
        hr_index = DYNAMIC_FEATURES.index("HR")
        self.assertEqual(data.value_last[0, 0, sys_index], 110)
        self.assertEqual(data.count[0, 0, sys_index], 1)
        self.assertEqual(data.count[0, 0, hr_index], 0)

    def test_split_and_preprocessors(self):
        with tempfile.TemporaryDirectory() as temp:
            data = self._dataset(Path(temp), patients=40)
        split = stratified_split(data.y, seed=7)
        split.validate(len(data))
        train, validation = data.subset(split.train), data.subset(split.validation)
        sequence = SequencePreprocessor().fit(train)
        train_batch = sequence.transform(train)
        validation_batch = sequence.transform(validation)
        self.assertFalse(np.isnan(train_batch.values).any())
        self.assertEqual(train_batch.static.shape[1], 8)
        self.assertEqual(validation_batch.values.shape[1:], (48, len(DYNAMIC_FEATURES)))

        tree = XGBoostPreprocessor(window_hours=6).fit(train)
        matrix, labels = tree.transform(validation)
        self.assertEqual(matrix.shape[0], len(labels))
        self.assertEqual(matrix.shape[1], len(tree.feature_names))

        feature_ids, values, times, padding = build_triplets(validation_batch, max_tokens=16)
        self.assertEqual(feature_ids.shape, (len(validation), 16))
        self.assertEqual(values.shape, times.shape)
        self.assertEqual(padding.dtype, bool)

        empty_batch = validation_batch.__class__(
            values=np.zeros_like(validation_batch.values[:1]),
            mask=np.zeros_like(validation_batch.mask[:1]),
            delta=np.zeros_like(validation_batch.delta[:1]),
            static=validation_batch.static[:1],
            y=validation_batch.y[:1],
            patient_ids=validation_batch.patient_ids[:1],
        )
        empty_ids, _, _, empty_padding = build_triplets(empty_batch, max_tokens=4)
        self.assertEqual(empty_ids[0, 0], len(DYNAMIC_FEATURES))
        self.assertFalse(empty_padding[0, 0])

    def test_threshold_metrics_and_ensemble(self):
        y = np.array([0, 0, 0, 1, 1, 1])
        first = np.array([0.1, 0.2, 0.4, 0.5, 0.8, 0.9])
        second = np.array([0.2, 0.3, 0.2, 0.7, 0.6, 0.8])
        threshold = select_threshold(y, first, target_recall=2 / 3)
        metrics = binary_metrics(y, first, threshold)
        self.assertGreaterEqual(metrics["sensitivity"], 2 / 3)
        ensemble = ConvexProbabilityEnsemble(["a", "b"]).fit(y, np.column_stack([first, second]))
        prediction = ensemble.predict(np.column_stack([first, second]))
        self.assertEqual(prediction.shape, y.shape)
        self.assertAlmostEqual(sum(ensemble.weight_dict().values()), 1.0, places=6)

        y_cal = np.tile(y, 4)
        matrix_cal = np.column_stack([np.tile(first, 4), np.tile(second, 4)])
        calibration = CrossValidatedSigmoidCalibration(2, n_splits=3).fit_transform(
            y_cal, matrix_cal
        )
        self.assertEqual(calibration.shape, matrix_cal.shape)
        self.assertTrue(np.isfinite(calibration).all())

        intervals = bootstrap_confidence_intervals(y_cal, matrix_cal[:, 0], n_bootstrap=20)
        self.assertIn("lower_95", intervals["auprc"])
        difference = paired_bootstrap_difference(
            y_cal, matrix_cal[:, 0], matrix_cal[:, 1], n_bootstrap=20
        )
        self.assertIn("estimate", difference["brier"])


if __name__ == "__main__":
    unittest.main()
