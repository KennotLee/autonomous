"""Load, validate and chronologically split data/processed/hourly_features.parquet."""
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from . import aar_config as cfg


class SchemaError(ValueError):
    """The processed file breaks the team contract; training must not proceed."""


def load_airport_timezones(path: Path = cfg.AIRPORTS_YAML) -> dict[str, str]:
    with open(path, encoding="utf-8") as f:
        airports = yaml.safe_load(f)["airports"]
    return {icao: a["timezone"] for icao, a in airports.items()}


def normalise(df: pd.DataFrame) -> pd.DataFrame:
    """Timestamps -> tz-aware UTC, icao -> str, h -> int. Does not drop or fill anything."""
    df = df.copy()
    for col in (cfg.TIME, cfg.OBS_TIME):
        if col in df:
            df[col] = pd.to_datetime(df[col], utc=True)
    if "icao" in df:
        df["icao"] = df["icao"].astype(str)
    if "h" in df and df["h"].notna().all():
        df["h"] = df["h"].astype(int)
    return df


def load_hourly(path: Path = cfg.HOURLY_FEATURES) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. It is produced by P1's pipeline (scripts/run_pipeline.py). "
            "No model was trained and no metrics were written."
        )
    return normalise(pd.read_parquet(path))


def validate_hourly(df: pd.DataFrame, timezones: dict[str, str]) -> dict:
    """Check keys, timestamps, horizons, missingness and target availability.

    Returns a JSON-serialisable report with "errors" (contract violations that would
    corrupt training or leak) and "warnings" (usable but worth raising with P1).
    """
    errors, warnings = [], []
    report = {"n_rows": int(len(df)), "errors": errors, "warnings": warnings}

    required = cfg.KEYS + [cfg.OBS_TIME, cfg.TARGET]
    missing_cols = [c for c in required if c not in df]
    if missing_cols:
        errors.append(f"missing required columns: {missing_cols}")
        return report

    for col in (cfg.TIME, cfg.OBS_TIME):
        if not isinstance(df[col].dtype, pd.DatetimeTZDtype):
            errors.append(f"{col} is not a UTC timestamp (dtype {df[col].dtype})")
    for col in cfg.KEYS:
        if df[col].isna().any():
            errors.append(f"{int(df[col].isna().sum())} null values in key column {col}")
    if errors:
        return report

    n_dup = int(df.duplicated(cfg.KEYS).sum())
    if n_dup:
        errors.append(f"{n_dup} duplicate (icao, target_hour_utc, h) rows")

    off_hour = int((df[cfg.TIME] != df[cfg.TIME].dt.floor("h")).sum())
    if off_hour:
        errors.append(f"{off_hour} target_hour_utc values not on an hour boundary")

    horizons = sorted(int(x) for x in df["h"].unique())
    report["horizons"] = horizons
    unexpected = sorted(set(horizons) - set(cfg.HORIZONS))
    if unexpected:
        errors.append(f"unexpected horizons {unexpected}; contract is {list(cfg.HORIZONS)}")
    absent = sorted(set(cfg.HORIZONS) - set(horizons))
    if absent:
        warnings.append(f"horizons {absent} have no rows")

    # Core leakage contract: features observed at or before target time minus h.
    n_obs_null = int(df[cfg.OBS_TIME].isna().sum())
    if n_obs_null:
        errors.append(f"{n_obs_null} rows with null feature_obs_time")
    cutoff = df[cfg.TIME] - pd.to_timedelta(df["h"], unit="h")
    n_late = int((df[cfg.OBS_TIME] > cutoff).sum())
    if n_late:
        errors.append(f"{n_late} rows with feature_obs_time later than target_hour_utc - h (leakage)")

    unknown = sorted(set(df["icao"]) - set(timezones))
    if unknown:
        errors.append(f"icao values not in config/airports.yaml: {unknown}")

    y = df[cfg.TARGET]
    n_y_null = int(y.isna().sum())
    report["target_missing"] = n_y_null
    if n_y_null:
        warnings.append(f"{n_y_null} rows with null arr_count; they are dropped, never imputed")
    yv = y.dropna()
    if (yv < 0).any():
        errors.append(f"{int((yv < 0).sum())} negative arr_count values")
    if not np.allclose(yv, np.round(yv)):
        warnings.append("arr_count has non-integer values; treated as a continuous target")
    report["target_summary"] = {k: float(v) for k, v in yv.describe().items()}

    coverage = {}
    for icao, g in df.groupby("icao"):
        hours = g[cfg.TIME].drop_duplicates()
        expected = int((hours.max() - hours.min()) / pd.Timedelta(hours=1)) + 1
        coverage[icao] = {
            "first": hours.min().isoformat(),
            "last": hours.max().isoformat(),
            "n_hours": int(len(hours)),
            "expected_hours": expected,
        }
        if len(hours) < expected:
            warnings.append(
                f"{icao}: {expected - len(hours)} of {expected} hours absent from the panel; "
                "if zero-arrival hours are omitted, persistence lookups for them are NaN"
            )
    report["coverage"] = coverage
    if not (df[cfg.TIME] >= cfg.TEST_START).any():
        warnings.append(f"no rows on or after {cfg.TEST_START.date()}; test split is empty")

    miss = df.isna().mean()
    report["missing_share"] = {c: round(float(v), 4) for c, v in miss[miss > 0].sort_values(ascending=False).items()}
    report["oracle_columns"] = [c for c in df.columns if c.startswith("oracle_")]
    return report


def check(report: dict) -> None:
    for w in report["warnings"]:
        print(f"WARNING: {w}")
    if report["errors"]:
        raise SchemaError("hourly_features.parquet failed validation:\n  " + "\n  ".join(report["errors"]))


def split_chronological(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """train < VAL_START <= val < TEST_START <= test, on target_hour_utc. Never shuffled."""
    t = df[cfg.TIME]
    return {
        "train": df[t < cfg.VAL_START],
        "val": df[(t >= cfg.VAL_START) & (t < cfg.TEST_START)],
        "test": df[t >= cfg.TEST_START],
    }
