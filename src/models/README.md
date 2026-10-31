P2 Models.

Inputs (team contract, column names fixed; full details in CLAUDE.md):
- data/processed/hourly_features.parquet: icao, target_hour_utc, h, feature_obs_time, [weather + wind component + rolling + calendar features], arr_count
- data/processed/flight_features.parquet: flight_id, origin_icao, dest_icao, sched_time_utc, h, feature_obs_time, [origin_* and dest_* features], delay_min, label_45

h ∈ {3,6,9,12}; all non-oracle features observed at or before target time − h. Columns prefixed `oracle_` are same-hour weather for oracle comparison only — never train production models on them.

Outputs: outputs/predictions/aar_preds.csv (`icao, target_hour_utc, h, pred_arr_count, p10, p90`, CLAUDE.md "B. Prediction CSVs" + P10/P90) and outputs/predictions/delay_preds.csv (P3).

---

# P2 — Hourly arrivals forecast (AAR proxy)

Forecasts how many planes will land at KATL and KDFW in each UTC hour, 3, 6, 9 and 12 hours ahead,
with a p10–p90 uncertainty band for the dashboard.

> **The target is a proxy.** `arr_count` = realised arrivals (landed, not cancelled or diverted) per UTC
> hour, from a ~0.5% BTS sample (about 5 arrivals per airport per day, 0 in 82% of hours). It stands in
> for the arrival acceptance rate (AAR); it is **not** runway capacity. Compare models on it; do not read
> the absolute numbers as capacity.

## Brief checklist

| P2 requirement | Where |
|---|---|
| Target = actual arrivals per airport per hour, labelled as an AAR proxy | above; `metadata.json → target`; `aar_metrics_table.md` header |
| Baseline: persistence | last **completed** hour before t − h (the hour starting at t − h has not happened yet at forecast time) |
| Baseline: hour-of-day × day-of-week mean | airport × local hour × local weekday, training years only |
| Extra baseline (strongest): schedule × completion ratio | scheduled arrivals × (arrivals / scheduled) per airport, training years only |
| LightGBM, one model with h as a feature | `train_aar.py`; `h` is an input |
| Train on 2023–24, test on 2025 | final models fit on 2023-01-01..2024-12-31 only; 2025 scored once |
| RMSE by horizon × airport, model vs baselines | `aar_metrics_table.md` (readable), `aar_metrics.csv` (all slices) |
| p10/p90 via quantile regression | two LightGBM quantile models (α = 0.1, 0.9) |
| `aar_preds.csv (icao, target_time, h, pred, p10, p90)` | team schema names: `icao, target_hour_utc, h, pred_arr_count, p10, p90` |
| No future data, purely predictive | fail-closed input allowlist + tests; see [No future data](#no-future-data) |

## Quick start (from repo root, Windows)

```
py -3.12 -m venv .venv
.venv\Scripts\pip install -r requirements.txt -r src/models/requirements-aar.txt
.venv\Scripts\python -m src.models.train_aar --validate-only   # schema + leakage checks only
.venv\Scripts\python -m src.models.train_aar                   # train, evaluate, save
.venv\Scripts\python -m pytest src/models/tests -q             # 18 tests (synthetic data)
```

Training refuses to run (no artifacts written) if `data/processed/hourly_features.parquet` is missing or
fails validation.

**Until P1's branch is merged into main**, the input lives on `origin/p1`. Copy it into the gitignored
`data/interim/` (run the `git cat-file` lines in Git Bash; PowerShell's `>` corrupts binary files):

```
git fetch origin
git cat-file blob origin/p1:data/processed/hourly_features.parquet > data/interim/hourly_features.parquet
.venv\Scripts\python -m src.models.train_aar --data data\interim\hourly_features.parquet

# optional: input ablation (needs P1's flight file too)
git cat-file blob origin/p1:data/processed/flight_features.parquet > data/interim/flight_features.parquet
.venv\Scripts\python -m src.models.ablation_aar --data data\interim\hourly_features.parquet --flights data\interim\flight_features.parquet
```

Re-run training whenever P1 pushes a new parquet. A new column that is not on the input allowlist stops
training until someone confirms it is known at forecast time (P1 v1.1 added `diversions`, an outcome; it
is excluded).

## Outputs (`outputs/predictions/`)

| file | contents |
|---|---|
| `aar_preds.csv` | contract columns first: `icao, target_hour_utc, h, pred_arr_count, p10, p90`; then `split` (val = 2024 H2, test = 2025), `actual_arr_count`, `pred_schedule`, `pred_persistence`, `pred_hist_avg`, `eval_hub_weather_actual` |
| `aar_metrics_table.md` | 2025 test RMSE by airport × horizon, model vs the three baselines, plus interval calibration |
| `aar_metrics.csv` | `split, model, icao, h, weather, n, rmse, mae, coverage_p10_p90, share_below_p10, share_above_p90, mean_interval_width`; `ALL` = aggregated level |
| `aar_ablation.csv` | which inputs help (validation only) |
| `aar_model/model.txt`, `model_p10.txt`, `model_p90.txt` | point and quantile LightGBM models |
| `aar_model/preprocessor.json` | input list and order, **provenance of every input**, every excluded column with the reason |
| `aar_model/metadata.json` | parameters, iteration counts, fit date range, split sizes, feature importance, baseline ratios, validation report |

`eval_hub_weather_actual` (normal / bad / severe) is the weather that **actually occurred** in hour t,
from `oracle_*`. It is for slicing evaluation and the failure-case analysis only; it is not known at
forecast time and the model never sees it. The dashboard should not show it as forecast information.

### Dashboard API

```python
from src.models.predict_aar import predict_aar
predict_aar(icao="KATL", h=6, start="2025-03-01", end="2025-03-07")   # reads the parquet
predict_aar(features_df)                                              # or pass rows yourself
# -> DataFrame[icao, target_hour_utc, h, pred_arr_count, p10, p90], arrivals/hour, all >= 0
```

Models are loaded once and cached. Pass `features` covering at least 2 hours either side of the hours you
want (neighbouring-hour schedule inputs; missing ones become NaN). A missing model input column raises
`KeyError`. Realised arrivals, outcome and `oracle_*` columns in `features` are ignored.

Suggested "Low confidence" rule for P4: `p90 − p10 ≥ 1` (flags 23.6% of 2025 test rows, mostly hours with
scheduled flights), or input weather missing, or severe weather observed (see the failure case).

## Results (2025 test, scored once)

Model v3, P1 data v1.1. Rows: train 104,980 · val 35,328 · test 40,740. Point model: Poisson, 458
iterations; p90: 404; p10: 1 (see intervals). 44 inputs.

| airport | LightGBM RMSE | schedule × ratio | hour × weekday mean | persistence | vs best baseline | LightGBM MAE |
|---|---|---|---|---|---|---|
| KATL | **0.371** | 0.431 | 0.503 | 0.732 | −14% | 0.212 |
| KDFW | **0.342** | 0.419 | 0.426 | 0.620 | −18% | 0.179 |

Validation (2024 H2): KATL 0.349, KDFW 0.323. P4's independent evaluation (`origin/p4`,
`outputs/metrics/test_set_summary.md`) reproduces these numbers and the schedule baseline, with 90%
bootstrap CIs excluding zero at every horizon.

- **Beats every baseline at every airport and horizon** (`aar_metrics_table.md`). Quote the gain against
  the schedule baseline (−14% KATL, −18% KDFW): it shows what the model adds beyond reading the schedule.
- **Flat across horizons** (KATL 0.370–0.371, KDFW 0.342–0.343 for h = 3…12): the main inputs, the
  schedule, are known in advance at every lead time.
- **Unbiased overall:** mean forecast vs actual (h = 6) 0.248 vs 0.249 at KATL, 0.181 vs 0.179 at KDFW.
- **Under-predicts rare busy hours:** average KATL forecast by actual 0/1/2/3/4 arrivals =
  0.13/0.61/1.10/1.69/2.01.

**Feature importance (gain share):** `sched_arr_count` 58.9%, next hour's schedule `p2_sched_p1` 20.0%,
previous hour's `p2_sched_m1` 4.8%, local hour 3.1%, UTC hour 2.0%, pressure 1.1%, `p2_sched_m2` 1.1%,
dew point 1.0%, `p2_sched_x_hub_bad` 0.9%. All weather inputs together: 7.4%.

### By actual weather at the hub (2025 test RMSE)

| airport | weather in hour t | rows | LightGBM | schedule × ratio | hour × weekday mean | persistence |
|---|---|---|---|---|---|---|
| KATL | normal | 17,040 | **0.369** | 0.431 | 0.502 | 0.728 |
| KATL | bad | 2,732 | **0.374** | 0.424 | 0.513 | 0.757 |
| KATL | severe | 596 | **0.400** | 0.467 | 0.505 | 0.721 |
| KDFW | normal | 17,604 | **0.339** | 0.413 | 0.430 | 0.627 |
| KDFW | bad | 2,228 | **0.362** | 0.446 | 0.420 | 0.597 |
| KDFW | severe | 540 | 0.380 | 0.492 | **0.312** | 0.469 |

Bad = rain, ≥ 1 mm/h, wind ≥ 20 kt or crosswind ≥ 15 kt. Severe = thunderstorm/hail, snow/ice,
≥ 4 mm/h or wind ≥ 25 kt. Rows count every horizon (4 per hour).

### Failure case: severe weather at KDFW

In severe-weather hours at KDFW, the plain hour × weekday mean (0.312) beats the model (0.380) by 18%.
**Why:** the model leans on the schedule, but storms and ice at DFW cancel or delay scheduled flights, so
realised arrivals fall below schedule; the model only partly lowers its forecast for weather (see below)
because training holds only a few hundred such hours. The hour × weekday mean happens to be low in those
hours. **How a user would notice:** severe weather observed at the hub at forecast time, together with a
forecast close to the schedule. The dashboard can flag that combination ("severe weather observed;
forecast may be too high"). P4's evaluation finds the same failure.

### Prediction intervals (p10–p90)

Two LightGBM quantile models (α = 0.1 and 0.9), same inputs, early-stopped on validation (quantile
loss) and refit on 2023–24. Bounds are widened if needed so p10 ≤ forecast ≤ p90, and clipped at 0.

| 2025 test | coverage (nominal 0.80) | below p10 (nominal ≤ 0.10) | above p90 (nominal ≤ 0.10) | mean width |
|---|---|---|---|---|
| KATL, all hours | 0.929 | 0.000 | 0.071 | 0.54 |
| KDFW, all hours | 0.936 | 0.000 | 0.064 | 0.41 |
| hours with ≥ 1 scheduled arrival | 0.876 | 0.000 | 0.124 | — |
| hours with 1 scheduled | 0.874 | 0.000 | 0.126 | — |
| hours with 3+ scheduled | 0.850 | 0.000 | 0.150 | — |

How to read this honestly:
- **p10 is 0 everywhere, and that is almost always correct.** In 2023–24 the empirical 10th percentile of
  arrivals is 0 whenever 0–2 flights are scheduled (99% of hours); the quantile model learns nothing
  better (it stops after 1 iteration). Only in hours with 3+ scheduled flights (~1%) is the true p10 = 1,
  so the lower bound is too low there (conservative).
- **Coverage above 0.80 is mostly an artefact of whole-number, mostly-zero outcomes:** a band of
  [0, 0.06] contains every zero hour. In hours that have scheduled flights, coverage is 0.876.
- **p90 is slightly too narrow in busy hours:** 12–15% of actuals exceed p90 when flights are scheduled
  (nominal 10%). Treat the upper bound as optimistic in busy hours.
- Width 50%/90%/99% quantiles: 0.06 / 1.03 / 2.1 arrivals.

## Model

- **One LightGBM point model** (Poisson objective, count target; `seed=42`, deterministic; defaults
  otherwise, `aar_config.LGBM_PARAMS`) with `h` as an input, plus the two quantile models. Predictions
  clipped at 0.
- **Split** (by `target_hour_utc`, UTC, never shuffled): train 2023-01..2024-06, validation 2024-07..12,
  test 2025-01..07. Iteration counts and every design choice are picked on validation; final models are
  then refit on all of 2023–24; 2025 is scored once. `--no-refit` scores test with train-only fits.
- **Inputs (44), each with a provenance recorded in `preprocessor.json`:**

| group | inputs | why it is known at forecast time t − h |
|---|---|---|
| identifier (2) | `icao` (categorical), `h` | — |
| calendar (8) | `hour_local`, `dow_local`, `month_local`, `hour_utc`, `p2_local_hour/dow/month`, `p2_is_weekend` | derived from the target hour itself |
| schedule (5) | `sched_arr_count` (hour t), `p2_sched_m2/m1/p1/p2` (t−2..t+2) | published schedule, incl. flights later cancelled |
| hub weather (23) | P1 weather block: temp, dew point, humidity, precipitation, wind, gust, pressure, condition code, rolling 3 h/6 h maxima/sums/changes, head/cross/tailwind | observed at `feature_obs_time` ≤ t − h |
| weather × schedule (6) | `p2_hub_bad/severe/winter`, `p2_sched_x_hub_bad/severe`, `p2_sched_m1_x_hub_bad` | flags from the observed weather above × the schedule |

  About 30% of scheduled flights land outside their scheduled hour, which is why the neighbouring-hour
  schedule matters (it was the single biggest improvement).

- **Baselines.** Persistence: realised arrivals in hour `[t − h − 1 h, t − h)`, the last hour fully over
  at forecast time. Hour × weekday mean: airport × local hour × local weekday, each hour counted once
  across horizons, training rows only; unseen groups fall back to airport × hour, then airport. Schedule ×
  ratio: `sched_arr_count` × training-period arrivals/scheduled per airport (KATL 0.990, KDFW 0.976).

## Weather: what it does, and what the model can use

**Weather strongly drives delays and cancellations** (P1 `flight_features.parquet`, completed flights,
by the weather that actually occurred at the hub; Meteostat condition codes):

| hub weather | flights into hubs | mean arr. delay | delayed > 45 min | cancelled | flights out of hubs | mean delay | delayed > 45 min | cancelled |
|---|---|---|---|---|---|---|---|---|
| 1–4 clear to overcast | 8,129 | 8.7 min | 9.4% | 1.1% | 14,000 | 8.5 min | 9.8% | 0.8% |
| 5–6 fog | 188 | 5.1 | 8.6% | 0.5% | 366 | 6.4 | 10.0% | 0.8% |
| 7–8 light rain / rain | 831 | 16.6 | 13.2% | 3.0% | 1,292 | 24.4 | 19.0% | 3.3% |
| 9 heavy rain | 122 | 17.8 | 13.5% | 7.4% | 200 | 26.3 | 21.6% | 7.5% |
| 10–16 freezing rain / sleet / snow | 22 | 92.5 | 37.5% | **63.6%** | 26 | 143.8 | 50.0% | **69.2%** |
| 17–22 showers | 93 | 42.1 | 23.1% | 2.2% | 177 | 37.8 | 31.4% | 4.0% |
| 23–27 lightning / hail / thunderstorm | 70 | 32.4 | 23.3% | 7.1% | 86 | 83.8 | 50.6% | 10.5% |

Precipitation over 4 mm/h at the hub: mean arrival delay 58 min, 39% delayed > 45 min; wind over 20 kt:
36 min, 21%. Weather is therefore a key input for P3's delay model.

**It moves hourly arrival counts much less:** a delayed plane still lands, usually in the same or the
next hour (captured by the neighbouring-hour schedule); severe weather touches ~1.6% of inbound flights;
and a severe hour here has about one scheduled arrival. The effect is still real: in 2025 hours with
scheduled arrivals, when severe weather was **observed at forecast time**, realised arrivals averaged
0.624 vs 0.773 otherwise (−19%). The v3 model lowers its forecast in those hours from 0.802 to 0.736
(v2 only to 0.767): the right direction, but only part of the way.

**Weather-exposure inputs (v3).** Rather than leave the model to discover the weather × traffic
interaction from a few hundred storm hours, they state it: bad/severe/winter flags from the hub weather
observed at t − h, scheduled arrivals in t exposed to them, and flights due in t − 1 (which bad weather
may delay into t).

### Ablation (`aar_ablation.csv`, validation only)

Each variant trained on train, scored on 2024 H2. Bad-weather hours 3,436; severe 364 (actual weather,
for scoring only).

| variant | val RMSE | bad-weather hours | severe hours |
|---|---|---|---|
| hour × weekday mean baseline | 0.4616 | 0.4499 | 0.4834 |
| A v1 (hub weather + schedule t, L2) | 0.3570 | 0.3692 | 0.4124 |
| B A without any weather | 0.3569 | 0.3673 | 0.3965 |
| C A + schedule t−2..t+2 | 0.3374 | 0.3607 | 0.4012 |
| D C without weather | 0.3366 | 0.3601 | 0.3922 |
| D2 C with Poisson (v2) | 0.3364 | 0.3599 | 0.4069 |
| E C + origin-airport weather maxima | 0.3373 | 0.3604 | 0.4091 |
| F E with Poisson | 0.3362 | 0.3611 | 0.4074 |
| G C with **perfect** same-hour hub weather (`oracle_*`, diagnostic only) | 0.3367 | 0.3586 | 0.4000 |
| **H D2 + weather exposure (v3, production)** | 0.3365 | **0.3580** | 0.3992 |
| I H + origin-airport weather exposure | **0.3361** | 0.3594 | 0.4031 |
| J I without raw hub weather | 0.3362 | 0.3633 | 0.3991 |

- The big gain is the neighbouring-hour schedule (A → C, −5.5%).
- Raw hub weather alone hurt severe hours (D2 vs D): too few storm hours, so the model fit noise. The
  exposure inputs recover most of that (H) and give the best bad-weather score, slightly better even than
  perfect same-hour weather (G).
- Origin-airport weather (I): marginally best overall, worse in bad weather than H, and would make
  `predict_aar()` need a second file, so not in production. Code kept in `aar_context.py`.
- Tried and not adopted: monotone constraints forcing exposure inputs to only lower the forecast (val
  0.3360 overall but 0.3601 bad / 0.4012 severe; only 59 validation hours had severe weather observed
  at forecast time).
- On 2025, v3's severe-weather gain over v2 shrank to ~0.5% (validation ~2%): small and directionally
  right, not large.

## No future data

The model is purely predictive: every input is known at the forecast time t − h, and nothing from 2025
is used for fitting or tuning.

- **Fail-closed input allowlist.** Every input must belong to a known-at-forecast-time group (table in
  [Model](#model)); its reason is saved in `preprocessor.json → provenance`. Any other column stops
  training with an error, so a new P1 column cannot slip in unreviewed.
- **Never inputs:** `arr_count`, `capacity_ratio`, `mean_arr_delay`, `share_delayed_45`, `cancellations`,
  `diversions`, any `oracle_*` or `arr_count*` column, and any column named like an outcome (`delay`,
  `cancel`, `divert`, `divers`, `capacity`). Each exclusion and its reason are saved in `preprocessor.json`.
- **Weather:** validation fails if any row has `feature_obs_time > target_hour_utc − h`. Exposure flags
  use the observed weather only; a test checks they ignore `oracle_*`.
- **Schedule** inputs (t−2..t+2) use only `sched_arr_count`, the published schedule; a test checks they
  do not change when realised arrivals are scrambled.
- **Baselines:** persistence uses only the last hour fully over at t − h (asserted in code); the
  hour × weekday mean and the schedule ratio are fit on training years only.
- **Time:** chronological split; the final models are fit on rows before 2025-01-01 (asserted in code,
  and a test spies on every LightGBM training set). Validation picks iteration counts and design choices;
  2025 is scored once at the end.
- **End-to-end test:** setting the target, every outcome column and every `oracle_*` column to garbage
  leaves the forecast and the p10/p90 band unchanged.
- `eval_hub_weather_actual` in `aar_preds.csv` is the only future-derived column; it is evaluation-only.

## How to read the outputs

**`aar_preds.csv`.** One row per airport × UTC hour × horizon. `pred_arr_count` is the **expected**
number of arrivals, not a whole number: 0.63 means "a plane is likely". `p10`/`p90` are the band.
Example (KATL, h = 6, 2025):

```
target_hour_utc   actual  pred_arr_count  p10   p90   pred_persistence  pred_hist_avg
2025-01-05 04:00     1        0.63         0.0   1.00  0.0               0.13     model expected a plane; one came
2025-01-05 05:00     0        0.21         0.0   0.98  0.0               0.10     some chance of a late arrival spilling over
2025-01-05 08:00     0        0.00         0.0   0.00  1.0               0.00     persistence carried over a stale arrival
```

**`aar_metrics.csv` / `aar_metrics_table.md`.** MAE = average miss in arrivals per hour (0.21 ≈ off by a
fifth of a plane). RMSE = same idea, big misses weigh more; the main score. Lower is better. `n` = rows
scored; `ALL` = aggregated. Interval columns are filled for LightGBM only.

**`aar_model/`.** `preprocessor.json → provenance` says why each input is available at forecast time;
`dropped` lists every excluded column and why. `metadata.json → feature_importance_gain` shows what drives
forecasts. The `.txt` files are the models, loaded by `predict_aar()`.

## Files

| file | role |
|---|---|
| `aar_config.py` | paths, horizons, split dates, input allowlist and denylist, LightGBM and quantile settings, output schema |
| `aar_data.py` | load, validate (keys, timestamps, horizons, `feature_obs_time ≤ t − h`, missingness), chronological split |
| `aar_features.py` | input selection, provenance (fail-closed), local calendar features, preprocessor |
| `aar_context.py` | neighbouring-hour schedule and weather exposure (used); origin-airport weather (tested, not used) |
| `aar_baselines.py` | persistence, hour × weekday mean, schedule × ratio |
| `aar_metrics.py` | RMSE / MAE and interval scores by airport, horizon, weather |
| `train_aar.py` | CLI: train, evaluate, save |
| `ablation_aar.py` | input ablation on validation → `aar_ablation.csv` |
| `predict_aar.py` | `predict_aar()` for the dashboard |
| `tests/test_aar.py` | 18 tests, synthetic data: validation, exclusions, provenance, baselines, intervals, no-future-data |
| `requirements-aar.txt` | `lightgbm`, `pytest` (P1 owns the shared `requirements.txt`) |

## Assumptions and dependencies

- Schema: P1's "Output schemas (v1.1)" in CLAUDE.md (branch `p1`). `target_hour_utc` labels the hour
  starting at that time; timestamps are UTC.
- `feature_obs_time` is P1's weather stamp (≤ t − h, up to 2 h older); the persistence baseline uses
  t − h itself.
- Rows with null `arr_count` would be dropped, never imputed (v1.1 has none). P1's grid covers every
  hour including zero-arrival hours (22,631/22,631 per airport), which the persistence baseline relies on.
- The brief names the columns `target_time` and `pred`; we use the team-contract names `target_hour_utc`
  and `pred_arr_count` (CLAUDE.md, P4's request).
- `lightgbm` and `pytest` need adding to `requirements.txt` (P1).

## Limitations

- **Proxy target on a sparse sample:** ~5 arrivals per airport per day; scores are on that tiny scale and
  say nothing about real runway capacity.
- **Severe weather is the weak spot:** worst accuracy, over-prediction, and at KDFW the simple hour ×
  weekday mean wins (failure case above). More severe-weather hours (full BTS volume, more years) are the
  real fix.
- **Intervals:** p10 is always 0 (correct for 99% of hours, too low for hours with 3+ scheduled flights);
  p90 is slightly too narrow in busy hours (12–15% above vs 10% nominal).
- Busy hours are under-predicted (actual 3–4 → forecast ~1.7–2.0).
- Hyperparameters are LightGBM defaults; only the objective and inputs were chosen on validation, and
  validation scores are slightly optimistic because validation also picks iteration counts.
- Gusts are 97–99% missing.
