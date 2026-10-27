from __future__ import annotations

import unittest

import numpy as np

from icu_mortality.data import DYNAMIC_FEATURES, ICUDataset
from icu_mortality.interpretation import (
    binned_trajectories,
    group_by_variable,
    readable_feature_name,
    select_cases,
    summarize_group,
    variable_importance,
    variable_name,
)


class InterpretationTests(unittest.TestCase):
    def test_variable_and_readable_names(self):
        self.assertEqual(variable_name("w03__GCS__last"), "GCS")
        self.assertEqual(variable_name("Age__missing"), "Age")
        self.assertEqual(variable_name("ICUType"), "ICUType")
        self.assertEqual(readable_feature_name("w03__GCS__last", 12), "GCS last (36–48h)")
        self.assertEqual(
            readable_feature_name("w00__HR__delta", 6), "Heart rate hours since last (0–6h)"
        )
        self.assertEqual(readable_feature_name("Height__missing", 12), "Height missing")
        self.assertEqual(readable_feature_name("ICUType", 12), "ICU type")

    def test_grouping_preserves_each_patients_total(self):
        names = ["w00__GCS__mean", "w01__GCS__last", "w00__HR__mean", "Age", "Age__missing"]
        contributions = np.array(
            [[0.5, -0.2, 0.1, 0.3, -0.1], [-0.4, -0.1, 0.0, 0.2, 0.0]]
        )
        grouped = group_by_variable(contributions, names)
        self.assertEqual(list(grouped.columns), ["GCS", "HR", "Age"])
        np.testing.assert_allclose(grouped.sum(axis=1), contributions.sum(axis=1))
        np.testing.assert_allclose(grouped["GCS"], [0.3, -0.5])
        importance = variable_importance(grouped)
        self.assertEqual(importance.index[0], "GCS")
        with self.assertRaises(ValueError):
            group_by_variable(contributions[:, :4], names)

    def test_binned_trajectories_average_only_observed_hours(self):
        n, f = 2, len(DYNAMIC_FEATURES)
        count = np.zeros((n, 48, f), dtype=np.float32)
        value_sum = np.zeros((n, 48, f), dtype=np.float32)
        gcs = DYNAMIC_FEATURES.index("GCS")
        count[0, 0, gcs], value_sum[0, 0, gcs] = 1, 15
        count[0, 3, gcs], value_sum[0, 3, gcs] = 2, 20
        empty = np.full((n, 48, f), np.nan, dtype=np.float32)
        data = ICUDataset(
            count=count, value_sum=value_sum, value_min=empty, value_max=empty,
            value_first=empty, value_last=empty, static=np.zeros((n, 4)),
            patient_ids=np.array(["1", "2"]), y=np.array([0, 1]),
        )
        trajectories = binned_trajectories(data, ("GCS",), bin_hours=6)
        self.assertEqual(trajectories["GCS"].shape, (2, 8))
        self.assertAlmostEqual(trajectories["GCS"][0, 0], 35 / 3)
        self.assertTrue(np.isnan(trajectories["GCS"][0, 1]))
        self.assertTrue(np.isnan(trajectories["GCS"][1]).all())
        summary = summarize_group(trajectories["GCS"])
        self.assertEqual(summary.observed[0], 1)
        self.assertTrue(np.isnan(summary.median[1]))
        with self.assertRaises(ValueError):
            binned_trajectories(data, ("GCS",), bin_hours=5)

    def test_select_cases_picks_most_confident_examples(self):
        y = np.array([1, 1, 0, 0, 1, 0])
        p = np.array([0.9, 0.7, 0.8, 0.2, 0.1, 0.05])
        cases = select_cases(y, p, threshold=0.5)
        self.assertEqual(cases, {"correct_alert": 0, "false_alarm": 2, "missed_death": 4})
        self.assertNotIn("missed_death", select_cases(y, p, threshold=0.05))


if __name__ == "__main__":
    unittest.main()
