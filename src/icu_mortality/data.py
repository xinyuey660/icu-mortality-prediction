"""Load PhysioNet Challenge 2012 records without patient-level leakage.

The loader keeps the raw 48-hour grid and the sufficient hourly statistics used
by every model branch.  It deliberately does not impute or standardize values;
those operations are fitted later on a training split only.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

STATIC_FEATURES = ("Age", "Gender", "Height", "ICUType")

# The six invasive/non-invasive blood-pressure streams are merged into three
# clinical concepts. Weight remains time-varying, matching the public files.
DYNAMIC_FEATURES = (
    "Albumin", "ALP", "ALT", "AST", "Bilirubin", "BUN", "Cholesterol",
    "Creatinine", "BP_Dias", "FiO2", "GCS", "Glucose", "HCO3", "HCT", "HR",
    "K", "Lactate", "BP_MAP", "MechVent", "Mg", "Na", "PaCO2", "PaO2", "pH",
    "Platelets", "RespRate", "SaO2", "BP_Sys", "Temp", "TroponinI",
    "TroponinT", "Urine", "WBC", "Weight",
)

BP_MERGE = {
    "DiasABP": ("BP_Dias", 0),
    "NIDiasABP": ("BP_Dias", 1),
    "MAP": ("BP_MAP", 0),
    "NIMAP": ("BP_MAP", 1),
    "SysABP": ("BP_Sys", 0),
    "NISysABP": ("BP_Sys", 1),
}

# Broad plausibility filters, inherited from the audited course pipeline. They
# are quality-control rules, not clinical reference intervals or diagnoses.
CLINICAL_BOUNDS = {
    "Albumin": (0.6, 6.0), "ALP": (1.0, 4000.0), "ALT": (2.0, 10000.0),
    "AST": (6.0, 20000.0), "Bilirubin": (0.1, 60.0), "BUN": (0.1, 250.0),
    "Cholesterol": (32.0, 600.0), "Creatinine": (0.1, 18.0),
    "BP_Dias": (1.0, 375.0), "FiO2": (0.21, 1.0), "GCS": (3.0, 15.0),
    "Glucose": (30.0, 2000.0), "HCO3": (1.0, 60.0), "HCT": (5.0, 75.0),
    "HR": (1.0, 300.0), "K": (1.0, 12.0), "Lactate": (0.4, 30.0),
    "BP_MAP": (14.0, 330.0), "MechVent": (0.0, 1.0), "Mg": (0.5, 20.0),
    "Na": (100.0, 225.0), "PaCO2": (5.0, 200.0), "PaO2": (32.0, 700.0),
    "pH": (6.1, 7.9), "Platelets": (5.0, 2000.0),
    "RespRate": (0.1, 70.0), "SaO2": (0.1, 100.0), "BP_Sys": (1.0, 300.0),
    "Temp": (25.0, 45.0), "TroponinI": (0.01, 50.0),
    "TroponinT": (0.01, 20.0), "Urine": (0.0, 1000.0),
    "WBC": (0.1, 500.0), "Weight": (1.0, 300.0), "Height": (100.0, 210.0),
    "Age": (15.0, 120.0), "Gender": (0.0, 1.0), "ICUType": (1.0, 4.0),
}


@dataclass(frozen=True)
class ICUDataset:
    """Patient-level arrays on a 48 x feature hourly grid."""

    count: np.ndarray
    value_sum: np.ndarray
    value_min: np.ndarray
    value_max: np.ndarray
    value_first: np.ndarray
    value_last: np.ndarray
    static: np.ndarray
    patient_ids: np.ndarray
    y: np.ndarray

    def __len__(self) -> int:
        return int(self.y.shape[0])

    @property
    def mask(self) -> np.ndarray:
        return (self.count > 0).astype(np.float32)

    def subset(self, indices: Iterable[int]) -> ICUDataset:
        idx = np.asarray(list(indices), dtype=int)
        return ICUDataset(
            count=self.count[idx], value_sum=self.value_sum[idx],
            value_min=self.value_min[idx], value_max=self.value_max[idx],
            value_first=self.value_first[idx], value_last=self.value_last[idx],
            static=self.static[idx], patient_ids=self.patient_ids[idx], y=self.y[idx],
        )


class PhysioNet2012Loader:
    """Read public Set A from a flat or nested directory layout."""

    hours = 48

    def __init__(self, data_dir: str | Path, outcomes_path: str | Path | None = None):
        self.data_dir = Path(data_dir).expanduser().resolve()
        self.outcomes_path = Path(outcomes_path).expanduser().resolve() if outcomes_path else None
        self.feature_to_index = {name: i for i, name in enumerate(DYNAMIC_FEATURES)}

    @staticmethod
    def _parse_time(value: object) -> tuple[int, int] | None:
        try:
            hour_text, minute_text = str(value).split(":", maxsplit=1)
            hour, minute = int(hour_text), int(minute_text)
        except (TypeError, ValueError):
            return None
        if not (0 <= hour < 48 and 0 <= minute < 60):
            return None
        return hour, hour * 60 + minute

    def _discover(self) -> tuple[Path, Path]:
        feature_candidates = (
            self.data_dir / "set-a",
            self.data_dir / "Features_group_7" / "Features_group_10",
            self.data_dir,
        )
        feature_dir = next(
            (p for p in feature_candidates if p.is_dir() and any(p.glob("[0-9]*.txt"))),
            None,
        )
        if feature_dir is None:
            raise FileNotFoundError(
                f"No patient .txt files found below {self.data_dir}. "
                "Expected set-a/ or Features_group_7/Features_group_10/."
            )

        if self.outcomes_path is not None:
            outcome = self.outcomes_path
        else:
            outcome_candidates = (
                self.data_dir / "Outcomes-a.txt",
                self.data_dir / "Outcomes-7.txt",
                feature_dir.parent / "Outcomes-a.txt",
                feature_dir.parent / "Outcomes-7.txt",
            )
            outcome = next((p for p in outcome_candidates if p.is_file()), None)
        if outcome is None or not outcome.is_file():
            raise FileNotFoundError("Outcomes file not found; pass --outcomes-path explicitly.")
        return feature_dir, outcome

    @staticmethod
    def _apply_bounds(name: str, value: float) -> float:
        bounds = CLINICAL_BOUNDS.get(name)
        if bounds is None:
            return value
        return value if bounds[0] <= value <= bounds[1] else np.nan

    def _read_patient(self, path: Path) -> tuple[np.ndarray, ...]:
        frame = pd.read_csv(path)
        required = {"Time", "Parameter", "Value"}
        if not required.issubset(frame.columns):
            raise ValueError(f"{path.name} is missing columns: {sorted(required - set(frame.columns))}")
        frame = frame.loc[:, ["Time", "Parameter", "Value"]].copy()
        frame["Value"] = pd.to_numeric(frame["Value"], errors="coerce")
        parsed = frame["Time"].map(self._parse_time)
        frame = frame.loc[parsed.notna()].copy()
        parsed = parsed.loc[parsed.notna()]
        frame["hour"] = [item[0] for item in parsed]
        frame["minute"] = [item[1] for item in parsed]
        frame["row"] = np.arange(len(frame))

        static = np.full(len(STATIC_FEATURES), np.nan, dtype=np.float32)
        for index, name in enumerate(STATIC_FEATURES):
            candidates = frame.loc[frame["Parameter"].eq(name)].sort_values(["minute", "row"])
            if not candidates.empty:
                value = float(candidates.iloc[0]["Value"])
                static[index] = self._apply_bounds(name, value)

        # Map BP names and retain invasive measurements whenever both sources
        # occur within the same hour.
        dynamic = frame.copy()
        dynamic["feature"] = dynamic["Parameter"]
        dynamic["priority"] = np.zeros(len(dynamic), dtype=np.int64)
        bp_mask = dynamic["Parameter"].isin(BP_MERGE)
        dynamic.loc[bp_mask, "feature"] = dynamic.loc[bp_mask, "Parameter"].map(
            lambda name: BP_MERGE[name][0]
        )
        dynamic.loc[bp_mask, "priority"] = (
            dynamic.loc[bp_mask, "Parameter"]
            .map(lambda name: BP_MERGE[name][1])
            .astype(np.int64)
        )
        dynamic = dynamic.loc[dynamic["feature"].isin(DYNAMIC_FEATURES)].copy()
        if not dynamic.empty:
            best = dynamic.groupby(["hour", "feature"])["priority"].transform("min")
            dynamic = dynamic.loc[dynamic["priority"].eq(best)].copy()
            dynamic["Value"] = [
                self._apply_bounds(name, float(value))
                for name, value in zip(dynamic["feature"], dynamic["Value"])
            ]
            dynamic = dynamic.dropna(subset=["Value"]).sort_values(
                ["hour", "feature", "minute", "row"]
            )

        shape = (self.hours, len(DYNAMIC_FEATURES))
        count = np.zeros(shape, dtype=np.float32)
        value_sum = np.zeros(shape, dtype=np.float32)
        value_min = np.full(shape, np.nan, dtype=np.float32)
        value_max = np.full(shape, np.nan, dtype=np.float32)
        value_first = np.full(shape, np.nan, dtype=np.float32)
        value_last = np.full(shape, np.nan, dtype=np.float32)

        for (hour, feature), group in dynamic.groupby(["hour", "feature"], sort=False):
            feature_index = self.feature_to_index[str(feature)]
            values = group["Value"].to_numpy(dtype=np.float32)
            count[int(hour), feature_index] = len(values)
            value_sum[int(hour), feature_index] = values.sum(dtype=np.float32)
            value_min[int(hour), feature_index] = values.min()
            value_max[int(hour), feature_index] = values.max()
            value_first[int(hour), feature_index] = values[0]
            value_last[int(hour), feature_index] = values[-1]

        return count, value_sum, value_min, value_max, value_first, value_last, static

    def load(self, limit: int | None = None) -> ICUDataset:
        feature_dir, outcome_path = self._discover()
        outcomes = pd.read_csv(outcome_path)
        required = {"RecordID", "In-hospital_death"}
        if not required.issubset(outcomes.columns):
            raise ValueError(f"Outcome file is missing: {sorted(required - set(outcomes.columns))}")
        outcomes["RecordID"] = outcomes["RecordID"].astype(str)
        labels = pd.to_numeric(outcomes["In-hospital_death"], errors="coerce")
        label_by_id = dict(zip(outcomes["RecordID"], labels))

        paths = sorted(feature_dir.glob("[0-9]*.txt"))
        if limit is not None:
            paths = paths[: int(limit)]
        rows = []
        ids = []
        y = []
        for path in paths:
            label = label_by_id.get(path.stem, np.nan)
            if pd.isna(label):
                continue
            rows.append(self._read_patient(path))
            ids.append(path.stem)
            y.append(int(label))
        if not rows:
            raise ValueError("No patient records with valid mortality labels were found.")

        columns = list(zip(*rows))
        return ICUDataset(
            count=np.stack(columns[0]), value_sum=np.stack(columns[1]),
            value_min=np.stack(columns[2]), value_max=np.stack(columns[3]),
            value_first=np.stack(columns[4]), value_last=np.stack(columns[5]),
            static=np.stack(columns[6]), patient_ids=np.asarray(ids, dtype=str),
            y=np.asarray(y, dtype=np.int64),
        )
