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
git cat-file blob origin/p1:data/processed/flight_features.parquet > data/interim/flight_features.parquet   # ablation only
.venv\Scripts\python -m src.models.ablation_aar --data data\interim\hourly_features.parquet --flights data\interim\flight_features.parquet
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
| `aar_context.py` | neighbouring-hour schedule (used) and origin-airport weather (tested, not used) |
| `aar_baselines.py` | persistence, historical average |
| `aar_metrics.py` | RMSE / MAE by airport, horizon, airport × horizon |
| `train_aar.py` | CLI entry point |
| `ablation_aar.py` | which inputs help? variants scored on validation only → `aar_ablation.csv` |
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
| input ablation (validation) | `outputs/predictions/aar_ablation.csv` |

### Results (P1 v1.1 data, model v2, trained 2026-10-31)

Rows: train 104,980 · val 35,328 · test 40,740. Poisson LightGBM, best iteration 266 (of 2,000,
early stopping), 38 inputs.

| split | airport | model | RMSE | MAE |
|---|---|---|---|---|
| test 2025 | KATL | **LightGBM v2** | **0.369** | **0.212** |
| | | LightGBM v1 (previous) | 0.391 | 0.226 |
| | | historical avg | 0.503 | 0.355 |
| | | persistence | 0.732 | 0.408 |
| test 2025 | KDFW | **LightGBM v2** | **0.343** | **0.181** |
| | | LightGBM v1 (previous) | 0.366 | 0.203 |
| | | historical avg | 0.426 | 0.280 |
| | | persistence | 0.620 | 0.313 |
| val 2024 H2 | KATL | LightGBM v2 | 0.350 | 0.191 |
| val 2024 H2 | KDFW | LightGBM v2 | 0.323 | 0.165 |

- v2 beats both baselines at both airports: RMSE 27% lower than the historical average at KATL and
  19% lower at KDFW. Against v1, test RMSE fell 5.6% (KATL) and 6.3% (KDFW); KDFW MAE fell 11%.
- Scores are flat across horizons (test RMSE 0.369–0.370 at KATL, 0.342–0.343 at KDFW for h = 3…12),
  because the dominant inputs, the schedule, are known in advance regardless of lead time.
- Val and test are close, so no sign of overfitting. The 2025 test set was scored once, after all
  choices were made on validation.
- Overall bias is negligible: mean predicted vs actual arrivals/hour on test (h = 6) is
  0.248 vs 0.249 at KATL and 0.181 vs 0.179 at KDFW.

**Feature importance (share of total gain):** `sched_arr_count` 63.3%, `p2_sched_p1` (next hour's
schedule) 20.1%, `p2_sched_m1` (previous hour's) 5.0%, `hour_local` 2.1%, `hour_utc` 1.9%,
`p2_sched_m2` 1.1%, `pres_hpa` 0.9%, `dwpt_c` 0.8%, `month_local` 0.8%. All hub weather together: 4.4%.

**Summary for slides:** the model forecasts hourly arrivals 3–12 h ahead with 27% lower error than a
historical-average baseline at KATL (19% at KDFW), with the same skill at every horizon. It is driven
by the published schedule for the target hour and its neighbours. Ablation shows that in this sparse
sample even perfect same-hour weather adds no skill, so weather's small role reflects the data, not the model.

### Does weather help? (ablation, `aar_ablation.csv`)

A reviewer flagged that weather adds almost nothing. `ablation_aar.py` tests that directly: each
variant is trained on train and scored on **validation only**. "Bad-weather hours" are the 3,436
validation hours with actual hub precipitation ≥ 1 mm, coco ≥ 7 (rain or worse), wind ≥ 20 kt or
crosswind ≥ 15 kt.

| variant | val RMSE | val RMSE, bad-weather hours |
|---|---|---|
| historical-average baseline | 0.4616 | 0.4499 |
| A v1 model (hub weather + schedule t, L2) | 0.3570 | 0.3692 |
| B A without any weather | 0.3569 | 0.3673 |
| C A + schedule t−2..t+2 | 0.3374 | 0.3607 |
| D C without weather | 0.3366 | 0.3601 |
| **D2 C with Poisson objective (v2, production)** | **0.3364** | **0.3599** |
| E C + origin-airport weather (from `flight_features.parquet`) | 0.3374 | 0.3614 |
| F E with Poisson objective | 0.3363 | 0.3627 |
| G C with **perfect** same-hour hub weather (`oracle_*`, diagnostic only) | 0.3367 | 0.3586 |

What this shows:
- **The real gain is the schedule of neighbouring hours** (A → C: −5.5%). About 30% of scheduled
  flights land outside their scheduled hour, so the next hour's schedule matters a lot.
- **Weather adds no skill here, even when weather is bad** (B ≈ A, D ≈ C, also on bad-weather hours).
- **Even perfect knowledge of the weather at the target hour gains only ~0.4% on bad-weather hours**
  (G vs D). No model could extract more from weather in this data: the signal is not there. With ~5
  arrivals/day, a storm hour usually has 0–1 scheduled arrivals, so there is little for weather to disrupt.
- **Origin-airport weather was tried and rejected:** the model gave it 22% of the gain but validation
  error did not improve, which is a sign of fitting noise. It also would make `predict_aar()` need a second
  file. The code stays in `aar_context.py` for re-testing on denser data.
- Hub weather is kept in v2: it is part of the brief, costs nothing (D2 vs D), and would matter with
  real traffic volumes.

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

Forecasts are **expected values**, not whole numbers: 0.64 means "a plane is likely", roughly a 64%
chance of one arrival. Example (KATL, h = 6, model v2):

```
target_hour_utc   y_true  pred_lgbm  pred_persistence  pred_hist_avg
2025-01-05 04:00     1      0.64          0.0              0.13     model expected a plane; one came
2025-01-05 05:00     0      0.20          0.0              0.10     some chance of a late arrival spilling over
2025-01-05 08:00     0      0.00          1.0              0.00     persistence carried over a stale arrival
```

Average KATL forecast by actual outcome (test, h = 6): 0 → 0.13, 1 → 0.61, 2 → 1.10, 3 → 1.70,
4 → 2.11. It ranks quiet vs busy hours well but pulls rare busy hours toward the average.

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
- **LightGBM:** Poisson objective (count target), otherwise defaults in `aar_config.LGBM_PARAMS`,
  `seed=42`, deterministic. Inputs: every numeric P1 column that passes the leakage guards (weather,
  wind components, scheduled traffic, rolling features), `icao` (categorical), `h`, local
  hour/day of week/month/weekend derived from `target_hour_utc`, and `p2_sched_m2/m1/p1/p2`:
  `sched_arr_count` 2 and 1 hours before and 1 and 2 hours after the target hour (NaN off the grid).
  Predictions are clipped at 0.
- `predict_aar()` builds the neighbouring-schedule inputs from the rows it is given before filtering;
  when passing your own `features`, include at least 2 hours either side of the hours you want.

### Leakage safeguards

- Validation fails if any row has `feature_obs_time > target_hour_utc − h`, duplicate keys, off-hour
  timestamps, unknown horizons or airports.
- Never inputs: `arr_count`, `capacity_ratio`, `mean_arr_delay`, `share_delayed_45`, `cancellations`, `diversions`,
  any `oracle_*` or `arr_count*` column, and (conservatively) any column whose name contains
  `delay`, `cancel`, `divert`, `divers` or `capacity`. If P1 confirms such a column is observed before
  `feature_obs_time`, add it to `ALLOWED_OVERRIDES`. Every dropped column and the reason are saved
  in `preprocessor.json`.
- Calendar features come only from the target hour itself, which is known when the forecast is issued.
- Neighbouring-hour inputs use only `sched_arr_count` (the published schedule, including flights later
  cancelled), never realised arrivals; a test checks they do not change when `arr_count` is scrambled.
- Origin-weather aggregation (tested, not used) keeps only observations at or before `t − h` of the
  arrival hour; a test checks later observations are dropped.
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
- Hyperparameters are LightGBM defaults; only the objective and inputs were chosen on validation.
- Weather barely contributes (4.4% of gain) and the ablation shows no skill from it, even with perfect
  weather. This is a data limitation, not evidence that weather does not affect real capacity. Gusts
  are 97–99% missing.
- Busy hours are under-predicted (actual 3–4 → forecast ~1.7–2.1), the usual shrinkage for rare values.
