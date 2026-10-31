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
| `aar_context.py` | neighbouring-hour schedule and hub weather exposure (used); origin-airport weather (tested, not used) |
| `aar_baselines.py` | persistence, historical average |
| `aar_metrics.py` | RMSE / MAE by airport, horizon, airport × horizon |
| `train_aar.py` | CLI entry point |
| `ablation_aar.py` | which inputs help? variants scored on validation only → `aar_ablation.csv` |
| `predict_aar.py` | `predict_aar()` for the dashboard |

### Outputs (`outputs/predictions/`)

- `aar_preds.csv`: `split, icao, target_hour_utc, h, y_true, pred_lgbm, pred_persistence, pred_hist_avg, hub_weather_actual` (val + test rows; `hub_weather_actual` = normal/bad/severe from the weather that actually occurred, for analysis only).
- `aar_metrics.csv`: `split, model, icao, h, weather, n, rmse, mae`; `ALL` marks an aggregated level; `weather` rows score normal/bad/severe hours per airport.
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

### Results (P1 v1.1 data, model v3, trained 2026-10-31)

Rows: train 104,980 · val 35,328 · test 40,740. Poisson LightGBM, best iteration 458 (of 2,000,
early stopping), 44 inputs. The 2025 test set is scored once, after every choice was made on validation.

| split | airport | model | RMSE | MAE |
|---|---|---|---|---|
| test 2025 | KATL | **LightGBM v3** | **0.371** | **0.212** |
| | | LightGBM v2 | 0.369 | 0.212 |
| | | LightGBM v1 | 0.391 | 0.226 |
| | | historical avg | 0.503 | 0.355 |
| | | persistence | 0.732 | 0.408 |
| test 2025 | KDFW | **LightGBM v3** | **0.342** | **0.179** |
| | | LightGBM v2 | 0.343 | 0.181 |
| | | LightGBM v1 | 0.366 | 0.203 |
| | | historical avg | 0.426 | 0.280 |
| | | persistence | 0.620 | 0.313 |
| val 2024 H2 | KATL | LightGBM v3 | 0.349 | 0.190 |
| val 2024 H2 | KDFW | LightGBM v3 | 0.323 | 0.164 |

Versions: v1 = hub weather + schedule of hour t (L2). v2 = + schedule t−2..t+2, Poisson.
v3 = v2 + hub weather-exposure features (below).

- v3 beats both baselines at both airports: RMSE 26% lower than the historical average at KATL and
  20% lower at KDFW. v1 → v3 lowered test RMSE 5–7%; v2 → v3 is a wash overall (it targets bad weather).
- Scores are flat across horizons (test RMSE 0.370–0.371 KATL, 0.342–0.343 KDFW for h = 3…12).
- Overall bias is negligible: mean predicted vs actual arrivals/hour on test (h = 6) is
  0.248 vs 0.249 at KATL and 0.181 vs 0.179 at KDFW.

**Test scores by the weather that actually occurred at the hub** (`aar_metrics.csv`, `weather` column):

| hub weather in hour t | rows | v3 RMSE | v3 MAE | v2 RMSE | v2 MAE |
|---|---|---|---|---|---|
| normal | 34,644 | 0.354 | 0.193 | 0.354 | 0.193 |
| bad (rain, ≥ 1 mm/h, wind ≥ 20 kt, crosswind ≥ 15 kt) | 4,960 | 0.369 | 0.207 | 0.367 | 0.207 |
| severe (thunderstorm/hail, snow/ice, ≥ 4 mm/h, wind ≥ 25 kt) | 1,136 | **0.391** | **0.239** | 0.393 | 0.242 |

The model is least accurate in severe weather, where it over-predicts (mean forecast 0.254 vs actual 0.183).

**Feature importance (share of total gain):** `sched_arr_count` 58.9%, `p2_sched_p1` 20.0%,
`p2_sched_m1` 4.8%, `hour_local` 3.1%, `hour_utc` 2.0%, `pres_hpa` 1.1%, `p2_sched_m2` 1.1%,
`dwpt_c` 1.0%, `p2_sched_x_hub_bad` 0.9%. All weather inputs: 7.4% (exposure features 1.5% of that).

**Summary for slides:** the model forecasts hourly arrivals 3–12 h ahead with 26% lower error than a
historical-average baseline at KATL (20% at KDFW), at every horizon. It is driven by the published
schedule. Weather clearly drives delays and cancellations in this data, and severe weather cuts realised
arrivals by about 19%; the model captures part of that drop, limited by how few severe-weather hours the
sparse sample contains.

### Weather: what it does to flights, and what the model can use

**Weather strongly affects delays and cancellations** (P1 `flight_features.parquet`, completed flights
at h = 3, by the weather that actually occurred at the hub; Meteostat condition codes):

| hub weather | flights into hubs | mean arr. delay | delayed > 45 min | cancelled | flights out of hubs | mean delay | delayed > 45 min | cancelled |
|---|---|---|---|---|---|---|---|---|
| 1–4 clear to overcast | 8,129 | 8.7 min | 9.4% | 1.1% | 14,000 | 8.5 min | 9.8% | 0.8% |
| 5–6 fog | 188 | 5.1 | 8.6% | 0.5% | 366 | 6.4 | 10.0% | 0.8% |
| 7–8 light rain / rain | 831 | 16.6 | 13.2% | 3.0% | 1,292 | 24.4 | 19.0% | 3.3% |
| 9 heavy rain | 122 | 17.8 | 13.5% | 7.4% | 200 | 26.3 | 21.6% | 7.5% |
| 10–16 freezing rain / sleet / snow | 22 | 92.5 | 37.5% | **63.6%** | 26 | 143.8 | 50.0% | **69.2%** |
| 17–22 showers | 93 | 42.1 | 23.1% | 2.2% | 177 | 37.8 | 31.4% | 4.0% |
| 23–27 lightning / hail / thunderstorm | 70 | 32.4 | 23.3% | 7.1% | 86 | 83.8 | 50.6% | 10.5% |

Precipitation over 4 mm/h at the hub: mean arrival delay 58 min, 39% delayed > 45 min. Wind over 20 kt:
36 min, 21%. Weather is therefore a key input for **P3's delay model**.

**Why it moves hourly arrival counts much less:** a delayed plane still lands, usually in the same or
next hour (captured by the neighbouring-hour schedule); severe weather is rare (about 1.6% of inbound
flights); and a severe hour here has about 1 scheduled arrival. The effect is still real: in 2025 test
hours with at least one scheduled arrival, when severe weather was **observed at forecast time** (t − h),
realised arrivals averaged 0.624 vs 0.773 otherwise (−19%). v3 lowers its forecast in those hours from
0.802 to 0.736; v2 only to 0.767. So the model now responds to weather in the right direction, but only
part of the way: the training data holds only a few hundred such hours.

**Weather-exposure features (v3, `aar_context.py`).** Instead of leaving the model to discover the
weather × traffic interaction from a handful of storm hours, these state it directly. All use the hub
weather P1 observed at `feature_obs_time` (≤ t − h): `p2_hub_bad`, `p2_hub_severe`, `p2_hub_winter`
(0/1 flags), `p2_sched_x_hub_bad`, `p2_sched_x_hub_severe` (scheduled arrivals in t exposed to it),
`p2_sched_m1_x_hub_bad` (flights due in t − 1 that may be delayed into t).

### Ablation (`aar_ablation.csv`, validation only)

Each variant is trained on train and scored on 2024 H2 validation. Bad-weather hours: 3,436; severe: 364
(defined from the actual hub weather, for scoring only).

| variant | val RMSE | bad-weather hours | severe hours |
|---|---|---|---|
| historical-average baseline | 0.4616 | 0.4499 | 0.4834 |
| A v1 (hub weather + schedule t, L2) | 0.3570 | 0.3692 | 0.4124 |
| B A without any weather | 0.3569 | 0.3673 | 0.3965 |
| C A + schedule t−2..t+2 | 0.3374 | 0.3607 | 0.4012 |
| D C without weather | 0.3366 | 0.3601 | 0.3922 |
| D2 C with Poisson (v2) | 0.3364 | 0.3599 | 0.4069 |
| E C + origin-airport weather maxima | 0.3373 | 0.3604 | 0.4091 |
| F E with Poisson | 0.3362 | 0.3611 | 0.4074 |
| G C with **perfect** same-hour hub weather (`oracle_*`, diagnostic) | 0.3367 | 0.3586 | 0.4000 |
| **H D2 + hub weather exposure (v3, production)** | 0.3365 | **0.3580** | 0.3992 |
| I H + origin-airport weather exposure (all weather) | **0.3361** | 0.3594 | 0.4031 |
| J I without raw hub weather | 0.3362 | 0.3633 | 0.3991 |

What this shows:
- The big gain is the neighbouring-hour schedule (A → C, −5.5%).
- **Raw hub weather alone hurt in severe hours** (D2 0.4069 vs D 0.3922): with so few storm hours the
  model was fitting noise. **The exposure features recover most of that** (H 0.3992) and give the best
  bad-weather score of all variants, slightly better even than perfect same-hour weather (G).
- Origin-airport weather (I) is marginally best overall but worse in bad weather than H, and would make
  `predict_aar()` depend on a second file, so it is not in production. Code kept for denser data.
- Also tried and not adopted: monotone constraints forcing the exposure features to only lower the
  forecast (val RMSE 0.3360 overall, but 0.3601 bad / 0.4012 severe, both worse; only 59 validation hours
  had severe weather observed at forecast time, too few to justify it).
- On 2025 test, v3's severe-weather gain over v2 shrank to about 0.5% RMSE (validation: about 2%). Treat
  it as a small, directionally correct improvement, not a large one.

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

Forecasts are **expected values**, not whole numbers: 0.63 means "a plane is likely", roughly a 63%
chance of one arrival. Example (KATL, h = 6, model v3):

```
target_hour_utc   y_true  pred_lgbm  pred_persistence  pred_hist_avg
2025-01-05 04:00     1      0.63          0.0              0.13     model expected a plane; one came
2025-01-05 05:00     0      0.21          0.0              0.10     some chance of a late arrival spilling over
2025-01-05 08:00     0      0.00          1.0              0.00     persistence carried over a stale arrival
```

Average KATL forecast by actual outcome (test, h = 6): 0 → 0.13, 1 → 0.61, 2 → 1.10, 3 → 1.69,
4 → 2.01. It ranks quiet vs busy hours well but pulls rare busy hours toward the average.

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
  `sched_arr_count` 2 and 1 hours before and 1 and 2 hours after the target hour (NaN off the grid),
  and the hub weather-exposure features (flags and flags × schedule, from weather observed at t − h).
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
- Weather-exposure flags use only the hub weather observed at `feature_obs_time` (≤ t − h), never
  `oracle_*`; a test checks they ignore oracle columns. `hub_weather_actual` (from `oracle_*`) is used
  only to group evaluation scores.
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
- Weather has a large effect on delays and cancellations but a small one on hourly arrival counts in
  this sparse sample; the model captures part of the ~19% severe-weather drop and is least accurate in
  severe hours. More severe-weather hours (full BTS volume, more years) are the real fix. Gusts are
  97–99% missing.
- Busy hours are under-predicted (actual 3–4 → forecast ~1.7–2.0), the usual shrinkage for rare values.
