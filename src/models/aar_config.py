"""P2 AAR regression settings: paths, split dates, leakage rules, LightGBM parameters.

Everything tunable lives here so the other modules stay free of magic values.
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
HOURLY_FEATURES = ROOT / "data" / "processed" / "hourly_features.parquet"
AIRPORTS_YAML = ROOT / "config" / "airports.yaml"

# P2 owns outputs/predictions/ (README team workflow); metrics live next to the predictions
# so P4's outputs/metrics/ is left untouched.
PRED_DIR = ROOT / "outputs" / "predictions"
MODEL_SUBDIR = "aar_model"
PREDS_FILE = "aar_preds.csv"
METRICS_FILE = "aar_metrics.csv"

# Team contract (src/models/README.md).
TARGET = "arr_count"
TIME = "target_hour_utc"
OBS_TIME = "feature_obs_time"
KEYS = ["icao", TIME, "h"]
HORIZONS = (3, 6, 9, 12)

# Chronological split on target_hour_utc. 2025 is held out for the final test.
VAL_START = pd.Timestamp("2024-07-01", tz="UTC")
TEST_START = pd.Timestamp("2025-01-01", tz="UTC")

# Leakage rules. Columns are dropped from model inputs if they are:
#   * listed explicitly (realised outcomes of the target hour),
#   * prefixed oracle_ (same-hour weather, oracle comparison only),
#   * prefixed arr_count (the target or anything derived from it),
#   * named like a realised outcome (delay/cancel/divert/capacity), unless P1 confirms the
#     column is observed at or before feature_obs_time and it is added to ALLOWED_OVERRIDES.
EXCLUDED_COLUMNS = {"arr_count", "capacity_ratio", "mean_arr_delay", "share_delayed_45", "cancellations"}
EXCLUDED_PREFIXES = ("oracle_", "arr_count")
SUSPICIOUS_TOKENS = ("delay", "cancel", "divert", "capacity")
ALLOWED_OVERRIDES: set[str] = set()

# Columns that are identifiers/timestamps, never features. icao and h ARE features.
NON_FEATURE_COLUMNS = {TIME, OBS_TIME}
CATEGORICAL = ["icao"]

# Name of P1's "arrivals in the last completed hour before feature_obs_time" column, once agreed.
# While None, persistence is derived from past arr_count rows of the panel (see aar_baselines.py).
PERSISTENCE_COLUMN: str | None = None

SEED = 42
LGBM_PARAMS = {
    "objective": "regression",  # L2; "poisson" is a natural next try for count targets
    "metric": ["rmse", "mae"],
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_data_in_leaf": 20,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "seed": SEED,
    "deterministic": True,
    "force_col_wise": True,
    "verbose": -1,
}
NUM_BOOST_ROUND = 2000
EARLY_STOPPING_ROUNDS = 100
