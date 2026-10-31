"""P2 AAR settings: data sources, paths, split dates, leakage rules, LightGBM parameters.

Everything tunable lives here so the other modules stay free of magic values.

Data (challenge 5, updated problem statement):
* Weather: challenge5_hourly_weather_ATL_DFW.csv (Meteostat hourly / NOAA ISD, stations 72219 ATL and
  72259 DFW, UTC + local hour, runway wind components), described in challenge5_data_guide.xlsx.
* Arrivals: full BTS Reporting Carrier On-Time Performance data, downloaded by aar_bts.py (the data
  guide's route for teams that want hourly AAR; the challenge's flight sample is too sparse for it).
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
AIRPORTS_YAML = ROOT / "config" / "airports.yaml"  # P1 owns; read-only here (iata, timezone, heading)

# Inputs. Raw files are never committed (data/raw/ and data/interim/ are gitignored).
RAW_DIR = ROOT / "data" / "raw" / "challenge5"
WEATHER_CSV = RAW_DIR / "challenge5_hourly_weather_ATL_DFW.csv"
DATA_GUIDE = RAW_DIR / "challenge5_data_guide.xlsx"
BTS_DIR = ROOT / "data" / "interim" / "bts"          # filtered monthly BTS parquet cache
PANEL = ROOT / "data" / "interim" / "p2_hourly_panel.parquet"  # built by aar_dataset.py
BTS_START, BTS_END = "2023-01", "2025-07"            # months covered by the weather file

# Outputs. P2 owns outputs/predictions/ (README team workflow); metrics live next to the predictions
# so P4's outputs/metrics/ is left untouched.
PRED_DIR = ROOT / "outputs" / "predictions"
MODEL_SUBDIR = "aar_model"
PREDS_FILE = "aar_preds.csv"
METRICS_FILE = "aar_metrics.csv"
METRICS_TABLE_FILE = "aar_metrics_table.md"

# Panel schema.
TARGET = "arr_count"
TIME = "target_hour_utc"
OBS_TIME = "feature_obs_time"
KEYS = ["icao", TIME, "h"]
HORIZONS = (3, 6, 9, 12)

# Chronological split on target_hour_utc. 2025 is held out for the final test.
VAL_START = pd.Timestamp("2024-07-01", tz="UTC")
TEST_START = pd.Timestamp("2025-01-01", tz="UTC")

# Weather file columns used (data guide). wind_gust_kmh is 100% missing and runway_heading_deg is a
# per-airport constant, so both are dropped. All are observed at the hour stamped time_utc.
WEATHER_COLUMNS = [
    "temp_c", "dewpoint_c", "rel_humidity_pct", "precip_mm", "wind_dir_deg", "wind_speed_kmh",
    "pressure_hpa", "condition_code", "headwind_kmh", "crosswind_kmh", "tailwind_kmh",
]
# Rolling windows (T - w, T] ending at the observation hour T = t - h.
ROLLING_WINDOWS_H = (3, 6)

# Leakage rules. Columns are dropped from model inputs if they are:
#   * listed explicitly (realised outcomes of the target hour),
#   * prefixed oracle_ (weather at the target hour itself: evaluation/diagnostics only),
#   * prefixed arr_count (the target or anything derived from it),
#   * named like a realised outcome (delay/cancel/divert/capacity), unless confirmed to be known
#     at t - h and added to ALLOWED_OVERRIDES.
EXCLUDED_COLUMNS = {"arr_count", "capacity_ratio", "mean_arr_delay", "share_delayed_45", "cancellations", "diversions"}
EXCLUDED_PREFIXES = ("oracle_", "arr_count")
SUSPICIOUS_TOKENS = ("delay", "cancel", "divert", "divers", "capacity")
ALLOWED_OVERRIDES: set[str] = set()

# Fail-closed input allowlist: every model input must belong to a "known at forecast time" group
# (see aar_features.provenance); anything else stops training. Weather observed at t - h:
WEATHER_OBS_COLUMNS = set(WEATHER_COLUMNS)
WEATHER_OBS_PATTERN = r"(precip_sum|wind_max|crosswind_max|pressure_chg)_\d+h_(mm|kmh|hpa)"

# Columns that are identifiers/timestamps, never features. icao and h ARE features.
NON_FEATURE_COLUMNS = {TIME, OBS_TIME}
CATEGORICAL = ["icao"]

# Name of an external "arrivals in the last completed hour before t - h" column, if ever supplied.
# While None, persistence is derived from past arr_count rows of the panel (see aar_baselines.py).
PERSISTENCE_COLUMN: str | None = None

SEED = 42
LGBM_PARAMS = {
    "objective": "poisson",  # count target
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
NUM_BOOST_ROUND = 3000
EARLY_STOPPING_ROUNDS = 100

# Prediction intervals: one LightGBM quantile model per bound, same inputs as the point model.
QUANTILES = {"p10": 0.1, "p90": 0.9}

# aar_preds.csv contract: team schema (CLAUDE.md "B. Prediction CSVs") + p10/p90.
# pred_arr_count is an AAR PROXY: realised arrivals per UTC hour, not runway capacity.
PRED_COL = "pred_arr_count"
ACTUAL_COL = "actual_arr_count"
PREDS_COLUMNS = ["icao", TIME, "h", PRED_COL, "p10", "p90"]
