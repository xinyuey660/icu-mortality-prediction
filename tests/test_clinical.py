from __future__ import annotations

import unittest

import numpy as np

from icu_mortality.clinical import (
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
from icu_mortality.data import DYNAMIC_FEATURES, ICUDataset


def _dataset(n: int = 3) -> ICUDataset:
    f = len(DYNAMIC_FEATURES)
    count = np.zeros((n, 48, f), dtype=np.float32)
    nan = np.full((n, 48, f), np.nan, dtype=np.float32)
    return ICUDataset(
        count=count, value_sum=np.zeros((n, 48, f), dtype=np.float32),
        value_min=nan.copy(), value_max=nan.copy(), value_first=nan.copy(),
        value_last=nan.copy(), static=np.zeros((n, 4)),
        patient_ids=np.array([str(i) for i in range(n)]), y=np.zeros(n, dtype=int),
    )


class ClinicalAnalysisTests(unittest.TestCase):
    def test_net_benefit_matches_hand_calculation(self):
        y = np.array([1, 1, 0, 0, 0])
        p = np.array([0.9, 0.4, 0.6, 0.1, 0.05])
        # At 0.5: TP = 1, FP = 1 -> 1/5 - 1/5 * 1 = 0.
        np.testing.assert_allclose(net_benefit(y, p, np.array([0.5])), [0.0])
        # At 0.2: TP = 2, FP = 1 -> 2/5 - 1/5 * 0.25 = 0.35.
        np.testing.assert_allclose(net_benefit(y, p, np.array([0.2])), [0.35])
        np.testing.assert_allclose(treat_all_net_benefit(y, np.array([0.2])), [0.4 - 0.6 * 0.25])

    def test_alert_workload(self):
        y = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 1])
        p = np.array([0.9, 0.8, 0.7, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.3])
        summary = alert_summary(y, p, 0.5)
        self.assertAlmostEqual(summary["alerts_per_100"], 30.0)
        self.assertAlmostEqual(summary["sensitivity"], 2 / 3)
        self.assertAlmostEqual(summary["ppv"], 2 / 3)
        self.assertAlmostEqual(summary["alerts_per_death_found"], 1.5)
        found = alerts_for_sensitivity(y, p, target=1.0)
        self.assertAlmostEqual(found["sensitivity"], 1.0)
        self.assertAlmostEqual(found["threshold"], 0.3)

    def test_smr_interval_and_funnel(self):
        result = standardized_mortality_ratio(10, 10)
        self.assertAlmostEqual(result["smr"], 1.0)
        self.assertAlmostEqual(result["lower"], 0.4795, places=3)
        self.assertAlmostEqual(result["upper"], 1.8390, places=3)
        self.assertEqual(standardized_mortality_ratio(0, 5)["lower"], 0.0)
        with self.assertRaises(ValueError):
            standardized_mortality_ratio(3, 0)
        lower, upper = funnel_limits(np.array([5.0, 50.0]), 0.05)
        self.assertTrue(np.all(lower < 1) and np.all(upper > 1))
        self.assertGreater(upper[0] - lower[0], upper[1] - lower[1])

    def test_calibration_summary_on_calibrated_data(self):
        rng = np.random.default_rng(0)
        p = rng.uniform(0.02, 0.6, size=20000)
        y = rng.binomial(1, p)
        summary = calibration_summary(y, p)
        self.assertAlmostEqual(summary["observed_over_expected"], 1.0, delta=0.05)
        self.assertAlmostEqual(summary["calibration_slope"], 1.0, delta=0.1)

    def test_gcs_windows_bands_and_ventilation(self):
        data = _dataset()
        gcs = DYNAMIC_FEATURES.index("GCS")
        vent = DYNAMIC_FEATURES.index("MechVent")
        data.count[0, [2, 9], gcs] = 1
        data.value_last[0, 2, gcs], data.value_last[0, 9, gcs] = 6, 14
        data.count[1, 40, gcs], data.value_last[1, 40, gcs] = 1, 10
        data.count[2, 5, vent], data.value_max[2, 5, vent] = 1, 1
        np.testing.assert_allclose(last_observed(data, "GCS", 0, 12)[:2], [14, np.nan])
        np.testing.assert_allclose(last_observed(data, "GCS", 36, 48)[1], 10)
        np.testing.assert_array_equal(gcs_band(np.array([3, 8, 9, 12, 13, np.nan])), [0, 0, 1, 1, 2, -1])
        np.testing.assert_array_equal(mechanically_ventilated(data), [False, False, True])


if __name__ == "__main__":
    unittest.main()
