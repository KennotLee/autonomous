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
EXCLUDED_COLUMNS = {"arr_count", "capacity_ratio", "mean_arr_delay", "share_delayed_45", "cancellations", "diversions"}
EXCLUDED_PREFIXES = ("oracle_", "arr_count")
SUSPICIOUS_TOKENS = ("delay", "cancel", "divert", "divers", "capacity")
ALLOWED_OVERRIDES: set[str] = set()

# Fail-closed input allowlist: every model input must belong to one of these "known at forecast
# time" groups (see aar_features.provenance); anything else stops training. P1 weather block <F>
# (CLAUDE.md "Output schemas v1.1"), all observed at feature_obs_time <= t - h:
WEATHER_OBS_COLUMNS = {
    "temp_c", "dwpt_c", "rhum_pct", "prcp_mm", "wdir_deg", "wspd_kt", "wpgt_kt", "pres_hpa", "coco",
    "headwind_kt", "crosswind_kt", "tailwind_kt", "gust_headwind_kt", "gust_crosswind_kt", "gust_tailwind_kt",
}
WEATHER_OBS_PATTERN = r"(wspd_max|wpgt_max|prcp_sum|pres_chg)_\d+h_(kt|mm|hpa)"  # P1 rolling windows
CALENDAR_COLUMNS = {"hour_local", "dow_local", "month_local", "hour_utc"}  # P1 calendar of t

# Columns that are identifiers/timestamps, never features. icao and h ARE features.
NON_FEATURE_COLUMNS = {TIME, OBS_TIME}
CATEGORICAL = ["icao"]

# Name of P1's "arrivals in the last completed hour before feature_obs_time" column, once agreed.
# While None, persistence is derived from past arr_count rows of the panel (see aar_baselines.py).
PERSISTENCE_COLUMN: str | None = None

SEED = 42
LGBM_PARAMS = {
    "objective": "poisson",  # count target; beat L2 slightly on validation (see ablation_aar.py)
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

# Prediction intervals: one LightGBM quantile model per bound, same inputs as the point model.
QUANTILES = {"p10": 0.1, "p90": 0.9}

# aar_preds.csv contract: team schema (CLAUDE.md "B. Prediction CSVs") + p10/p90 (P4 request).
# pred_arr_count is an AAR PROXY: realised arrivals per UTC hour, not runway capacity.
PRED_COL = "pred_arr_count"
ACTUAL_COL = "actual_arr_count"
PREDS_COLUMNS = ["icao", TIME, "h", PRED_COL, "p10", "p90"]
METRICS_TABLE_FILE = "aar_metrics_table.md"
