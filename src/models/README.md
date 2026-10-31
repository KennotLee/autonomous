P2 Models.

Inputs (team contract, column names fixed; full details in CLAUDE.md):
- data/processed/hourly_features.parquet: icao, target_hour_utc, h, feature_obs_time, [weather + wind component + rolling + calendar features], arr_count
- data/processed/flight_features.parquet: flight_id, origin_icao, dest_icao, sched_time_utc, h, feature_obs_time, [origin_* and dest_* features], delay_min, label_45

h ∈ {3,6,9,12}; all non-oracle features observed at or before target time − h. Columns prefixed `oracle_` are same-hour weather for oracle comparison only — never train production models on them.

Outputs: outputs/predictions/aar_preds.csv and outputs/predictions/delay_preds.csv — schema "TBD, pending P1/team agreement; see CLAUDE.md".

---

## P2 — AAR regression (`arr_count`)

Predicts realised arrivals per UTC hour (AAR proxy) at KATL/KDFW for h ∈ {3, 6, 9, 12}.

### Commands (from repo root)

```
py -3.12 -m venv .venv                                   # Windows; macOS/Linux: python3 -m venv .venv
.venv\Scripts\pip install -r requirements.txt -r src/models/requirements-aar.txt
.venv\Scripts\python -m src.models.train_aar --validate-only   # schema + leakage checks only
.venv\Scripts\python -m src.models.train_aar                   # train, evaluate, save
.venv\Scripts\python -m pytest src/models/tests -q             # tests (synthetic data)
```

Training refuses to run (no artifacts written) if `data/processed/hourly_features.parquet` is
missing or fails validation.

**Until P1's branch is merged into main**, the input lives on `origin/p1`. Copy it into the
gitignored `data/interim/` (run in Git Bash; PowerShell's `>` corrupts binary files) and point at it:

```
git fetch origin
git cat-file blob origin/p1:data/processed/hourly_features.parquet > data/interim/hourly_features.parquet
.venv\Scripts\python -m src.models.train_aar --data data\interim\hourly_features.parquet
.venv\Scripts\python -c "from src.models.predict_aar import predict_aar; print(predict_aar(icao='KATL', h=6, start='2025-03-03', end='2025-03-03 23:00', data_path='data/interim/hourly_features.parquet'))"
```

Re-run training whenever P1 pushes a new parquet, and check any new column against the leakage
rules below (P1 v1.1 added `diversions`, which is now excluded).

### Files

| file | role |
|---|---|
| `aar_config.py` | paths, horizons, split dates, leakage denylist, LightGBM params |
| `aar_data.py` | load, validate, chronological split |
| `aar_features.py` | feature selection with leakage guards, local calendar features, preprocessor |
| `aar_baselines.py` | persistence, historical average |
| `aar_metrics.py` | RMSE / MAE by airport, horizon, airport × horizon |
| `train_aar.py` | CLI entry point |
| `predict_aar.py` | `predict_aar()` for the dashboard |

### Outputs (`outputs/predictions/`)

- `aar_preds.csv`: `split, icao, target_hour_utc, h, y_true, pred_lgbm, pred_persistence, pred_hist_avg` (val + test rows).
- `aar_metrics.csv`: `split, model, icao, h, n, rmse, mae`; `ALL` marks an aggregated level.
- `aar_model/model.txt` (LightGBM), `preprocessor.json` (feature list, column order, categories,
  dropped columns with reasons), `metadata.json` (params, best iteration, split sizes, feature
  importance, validation report). Text/JSON only, no pickles.

| what | path |
|---|---|
| trained model | `outputs/predictions/aar_model/model.txt` |
| model inputs + excluded columns | `outputs/predictions/aar_model/preprocessor.json` |
| settings, feature importance, data checks | `outputs/predictions/aar_model/metadata.json` |
| predictions | `outputs/predictions/aar_preds.csv` |
| scores | `outputs/predictions/aar_metrics.csv` |

### Results (P1 v1.1 data, trained 2026-10-31)

Rows: train 104,980 · val 35,328 · test 40,740. LightGBM best iteration 128 (of 2,000, early stopping).

| split | airport | model | RMSE | MAE |
|---|---|---|---|---|
| test 2025 | KATL | **LightGBM** | **0.391** | **0.226** |
| | | historical avg | 0.503 | 0.355 |
| | | persistence | 0.732 | 0.408 |
| test 2025 | KDFW | **LightGBM** | **0.366** | **0.203** |
| | | historical avg | 0.426 | 0.280 |
| | | persistence | 0.620 | 0.313 |
| val 2024 H2 | KATL | LightGBM | 0.365 | 0.209 |
| val 2024 H2 | KDFW | LightGBM | 0.349 | 0.195 |

- LightGBM beats both baselines at both airports: RMSE about 22% lower than the historical average at
  KATL and 14% lower at KDFW.
- Scores are flat across horizons (test RMSE 0.390–0.391 at KATL, 0.365–0.366 at KDFW for h = 3…12),
  because the dominant input, the schedule, is known in advance regardless of lead time.
- Val and test are close, so no sign of overfitting.
- Overall bias is negligible: mean predicted vs actual arrivals/hour on test (h = 6) is
  0.239 vs 0.249 at KATL and 0.176 vs 0.179 at KDFW.

**Feature importance (share of total gain):** `sched_arr_count` 86.4%, `hour_local` 3.0%,
`hour_utc` 1.8%, `dwpt_c` 1.2%, `pres_hpa` 1.2%, `month_local` 1.1%, `dow_local` 0.9%, `icao` 0.7%.
The forecast is mostly "how many flights are scheduled to land, adjusted by time of day". Weather
adds little; with ~5 flights/day per airport the sample has too few weather-affected hours to learn from.

**Summary for slides:** the model forecasts hourly arrivals 3–12 h ahead with ~20% lower error than a
historical-average baseline (KATL; ~14% at KDFW), with the same skill at every horizon. It is driven
mainly by the published schedule; this sparse sample cannot show a meaningful weather effect.

### How to read the outputs

**The data.** Each row is one airport (`icao`), one UTC hour (`target_hour_utc`) and one horizon
`h` (forecast made `h` hours before the hour starts). `y_true` = planes that actually landed that hour.
On the 2025 test set 81% of hours have 0 arrivals (h = 6 counts: 0 → 8,292 hours, 1 → 1,626,
2 → 246, 3 → 20, 4 → 1). Real KATL lands ~100/hour, so trust the comparisons, not the absolute capacity.

**`aar_preds.csv` columns.**

| column | meaning |
|---|---|
| `split` | `val` = Jul–Dec 2024, `test` = 2025 (never used for training or tuning) |
| `y_true` | actual arrivals |
| `pred_lgbm` | model forecast |
| `pred_persistence` | arrivals in the last full hour before the forecast was made |
| `pred_hist_avg` | usual count for this airport × local hour × weekday (training data) |

Forecasts are **expected values**, not whole numbers: 0.70 means "a plane is likely", roughly a 70%
chance of one arrival. Example (KATL, h = 6):

```
target_hour_utc   y_true  pred_lgbm  pred_persistence  pred_hist_avg
2025-01-05 04:00     1      0.70          0.0              0.13     model expected a plane; one came
2025-01-05 05:00     0      0.02          0.0              0.10     correctly expected none
2025-01-05 08:00     0      0.01          1.0              0.00     persistence carried over a stale arrival
```

Average KATL forecast by actual outcome (test, h = 6): 0 → 0.13, 1 → 0.56, 2 → 1.02, 3 → 1.60,
4 → 2.14. It ranks quiet vs busy hours well but pulls rare busy hours toward the average.

**`aar_metrics.csv`.** MAE = average miss in arrivals/hour (0.23 ≈ off by a quarter of a plane per
hour). RMSE = same idea but big misses weigh more; the main score. Lower is better for both. `n` = rows
scored; `ALL` = that level aggregated.

**`aar_model/`.** `metadata.json → feature_importance_gain` shows what drives forecasts.
`preprocessor.json → dropped_columns` lists every excluded column and why (targets, delays,
cancellations, diversions, all `oracle_*`): the evidence the model does not use future information.
`model.txt` is the model itself, loaded by `predict_aar()`; not meant to be read by hand.

### Dashboard API

```python
from src.models.predict_aar import predict_aar
predict_aar(icao="KATL", h=6, start="2025-03-01", end="2025-03-07")   # reads the parquet
predict_aar(features_df)                                              # or pass rows yourself
# -> DataFrame[icao, target_hour_utc, h, aar_pred]; aar_pred is arrivals/hour, >= 0
```

The model is loaded once and cached. Inputs must contain every column in
`preprocessor.json["feature_columns"]`; a missing column raises `KeyError`.

### Method

- **Split (by `target_hour_utc`, UTC, never shuffled):** train < 2024-07-01 ≤ val < 2025-01-01 ≤ test.
  LightGBM early-stops on val; it is then refit on train + val with the best iteration count, and the
  historical average is refit on train + val too, before scoring 2025. `--no-refit` scores test with
  the train-only fits.
- **Persistence:** arrivals in the last UTC hour that had fully ended by the prediction time `t − h`
  (`[t − h − 1h, t − h)`; P1's `feature_obs_time` is the weather stamp and may lag). Taken from P1's column once `PERSISTENCE_COLUMN` is set;
  until then derived from past `arr_count` rows of the panel. Absent hours give NaN, not 0.
- **Historical average:** mean `arr_count` by (icao, local hour, local day of week), each hour
  counted once across horizons, fit on training rows only. Unseen groups fall back to
  (icao, local hour), then icao.
- **LightGBM:** L2 regression, defaults in `aar_config.LGBM_PARAMS`, `seed=42`, deterministic.
  Inputs: every numeric P1 column that passes the leakage guards (weather, wind components,
  scheduled traffic, rolling features), `icao` (categorical), `h`, and local hour/day of week/month/weekend
  derived from `target_hour_utc`. Predictions are clipped at 0.

### Leakage safeguards

- Validation fails if any row has `feature_obs_time > target_hour_utc − h`, duplicate keys, off-hour
  timestamps, unknown horizons or airports.
- Never inputs: `arr_count`, `capacity_ratio`, `mean_arr_delay`, `share_delayed_45`, `cancellations`, `diversions`,
  any `oracle_*` or `arr_count*` column, and (conservatively) any column whose name contains
  `delay`, `cancel`, `divert`, `divers` or `capacity`. If P1 confirms such a column is observed before
  `feature_obs_time`, add it to `ALLOWED_OVERRIDES`. Every dropped column and the reason are saved
  in `preprocessor.json`.
- Calendar features come only from the target hour itself, which is known when the forecast is issued.
- Historical averages and the preprocessor's categories are fit on training rows only.
- Tests check that scrambling the target and oracle columns does not change predictions.

### Assumptions and open dependencies (for P1)

- Schema: P1's "Output schemas (v1.1)" in CLAUDE.md on branch `p1`. Feature names beyond the keys
  are not hardcoded; any numeric column passing the leakage rules is used.
- `target_hour_utc` labels the hour starting at that time; timestamps are UTC (naive is read as UTC).
- `feature_obs_time` is P1's weather stamp (≤ t − h, up to 2 h older); validation enforces ≤ t − h.
- Rows with null `arr_count` are dropped, never imputed (v1.1 has none).
- **Persistence** is derived from past `arr_count` rows. This is valid because P1's grid covers every
  hour including zero-arrival hours (validation confirms 22,631/22,631 hours per airport). If P1 adds a
  "last observed arrivals" column, set `PERSISTENCE_COLUMN`.
- `lightgbm` and `pytest` need adding to `requirements.txt` (P1 owns it).

### Limitations

- The raw sample is sparse (~5 arrivals/day/airport, `arr_count` mostly 0/1, see CLAUDE.md), so
  `arr_count` is far below real AAR and RMSE/MAE are on that tiny scale. The persistence and
  historical-average baselines are the honest bar to beat.
- Val metrics are slightly optimistic because val also picks the iteration count.
- L2 on counts is a default, not a tuned choice; `objective="poisson"` is the obvious next experiment.
- Weather barely contributes (about 3% of gain combined) because the sample is sparse; this is a data
  limitation, not evidence that weather does not affect real capacity. Gusts are 97–99% missing.
- Busy hours are under-predicted (actual 3–4 → forecast ~1.6–2.1), the usual shrinkage for rare values.
