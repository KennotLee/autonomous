"""Model inputs for AAR regression: leakage-guarded column selection, calendar features,
and a JSON-serialisable preprocessor shared by training and predict_aar()."""
import json
import re
from pathlib import Path

import pandas as pd

from . import aar_config as cfg

# Derived from target_hour_utc in the airport's local time. The target hour is known when the
# forecast is issued, so these carry no future information. Prefixed to avoid clashing with P1.
CALENDAR = ["p2_local_hour", "p2_local_dow", "p2_local_month", "p2_is_weekend"]


def add_calendar(df: pd.DataFrame, timezones: dict[str, str]) -> pd.DataFrame:
    df = df.copy()
    local_hour = pd.Series(index=df.index, dtype="float64")
    local_dow = pd.Series(index=df.index, dtype="float64")
    local_month = pd.Series(index=df.index, dtype="float64")
    for icao, idx in df.groupby("icao").groups.items():
        local = df.loc[idx, cfg.TIME].dt.tz_convert(timezones[icao])
        local_hour[idx] = local.dt.hour
        local_dow[idx] = local.dt.dayofweek
        local_month[idx] = local.dt.month
    df["p2_local_hour"] = local_hour.astype(int)
    df["p2_local_dow"] = local_dow.astype(int)
    df["p2_local_month"] = local_month.astype(int)
    df["p2_is_weekend"] = (df["p2_local_dow"] >= 5).astype(int)
    return df


def exclusion_reason(col: str) -> str | None:
    """Why a column may not be a model input, or None if it may."""
    if col in cfg.EXCLUDED_COLUMNS:
        return "excluded: realised outcome"
    if col.startswith(cfg.EXCLUDED_PREFIXES):
        return "excluded prefix (oracle_/arr_count)"
    if col in cfg.NON_FEATURE_COLUMNS:
        return "timestamp/identifier"
    if col not in cfg.ALLOWED_OVERRIDES and any(t in col.lower() for t in cfg.SUSPICIOUS_TOKENS):
        return "name suggests realised outcome; add to ALLOWED_OVERRIDES once P1 confirms it is past-only"
    return None


def select_feature_columns(df: pd.DataFrame) -> tuple[list[str], dict[str, str]]:
    features, dropped = [], {}
    for col in df.columns:
        reason = exclusion_reason(col)
        if reason is None and col not in cfg.CATEGORICAL:
            dtype = df[col].dtype
            if not (pd.api.types.is_numeric_dtype(dtype) or pd.api.types.is_bool_dtype(dtype)):
                reason = f"non-numeric dtype {dtype}"
        if reason:
            dropped[col] = reason
        else:
            features.append(col)
    for required in ("icao", "h"):
        if required not in features:
            raise ValueError(f"{required} must be a model input")
    unknown = [c for c in features if provenance(c) is None]
    if unknown:
        raise ValueError(
            f"inputs with no known availability at forecast time: {unknown}. Fail-closed: confirm with P1 "
            "that each is observed at or before t - h (or known in advance), then add it to "
            "aar_config.ALLOWED_OVERRIDES or the allowlists."
        )
    return features, dropped


def provenance(col: str) -> str | None:
    """Why an input is known at forecast time t - h, or None if that is not established."""
    if col in ("icao", "h"):
        return "identifier (airport, horizon)"
    if col in cfg.CALENDAR_COLUMNS or col in CALENDAR:
        return "calendar of the target hour (known in advance)"
    if col == "sched_arr_count" or re.fullmatch(r"p2_sched_[mp]\d", col):
        return "published schedule (known in advance)"
    if col in cfg.WEATHER_OBS_COLUMNS or re.fullmatch(cfg.WEATHER_OBS_PATTERN, col):
        return "hub weather observed at feature_obs_time <= t - h"
    if col.startswith(("p2_hub_", "p2_sched_x_", "p2_sched_m1_x_")):
        return "derived from hub weather observed <= t - h and the schedule"
    if col.startswith("p2_org_"):
        return "origin-airport weather observed <= t - h"
    if col in cfg.ALLOWED_OVERRIDES:
        return "confirmed by P1 as observed <= t - h"
    return None


class AARPreprocessor:
    """Fixes the feature list, column order and categorical levels at fit time."""

    def __init__(self, timezones: dict[str, str]):
        self.timezones = timezones
        self.feature_columns: list[str] = []
        self.dropped: dict[str, str] = {}
        self.categories: dict[str, list[str]] = {}

    def fit(self, df: pd.DataFrame) -> "AARPreprocessor":
        self.feature_columns, self.dropped = select_feature_columns(add_calendar(df, self.timezones))
        self.categories = {c: sorted(df[c].astype(str).unique()) for c in cfg.CATEGORICAL}
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = add_calendar(df, self.timezones)
        missing = [c for c in self.feature_columns if c not in df]
        if missing:
            raise KeyError(f"input is missing model features: {missing}")
        X = df[self.feature_columns].copy()
        for col in self.feature_columns:
            if col in self.categories:
                # Unseen levels become NaN rather than silently mapping to another airport.
                X[col] = pd.Categorical(X[col].astype(str), categories=self.categories[col])
            else:
                X[col] = X[col].astype("float64")
        return X

    def to_dict(self) -> dict:
        return {
            "timezones": self.timezones,
            "feature_columns": self.feature_columns,
            "categories": self.categories,
            "provenance": {c: provenance(c) for c in self.feature_columns},
            "dropped": self.dropped,
        }

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "AARPreprocessor":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        pre = cls(d["timezones"])
        pre.feature_columns, pre.categories, pre.dropped = d["feature_columns"], d["categories"], d["dropped"]
        return pre
