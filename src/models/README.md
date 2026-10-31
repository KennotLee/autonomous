P2 Models.

Outputs: outputs/predictions/aar_preds.csv (`icao, target_hour_utc, h, pred_arr_count, p10, p90`, CLAUDE.md "B. Prediction CSVs" + P10/P90) and outputs/predictions/delay_preds.csv (P3).

---

# P2 — Hourly arrivals forecast (AAR proxy)

Forecasts how many planes will land at KATL and KDFW in each UTC hour, 3, 6, 9 and 12 hours ahead,
with a p10–p90 uncertainty band for the dashboard.

> **The target is a proxy.** `arr_count` = realised arrivals (wheels-on, not cancelled or diverted)
> per UTC hour, from the **full** BTS on-time data: about 900 arrivals per airport per day, mean
> 35/hour, peak 103/hour. It stands in for the arrival acceptance rate (AAR) and is **not** a
> declared runway capacity: an hour with low demand has few arrivals even if capacity was high.

## Data (challenge 5)

| source | what | used for |
|---|---|---|
| `challenge5_hourly_weather_ATL_DFW.csv` | Meteostat hourly (NOAA ISD), stations 72219 ATL / 72259 DFW, 2023-01-01..2025-07-31, UTC + local hour, runway head/cross/tailwind (ATL 80°, DFW 180°) | all weather inputs |
| `challenge5_data_guide.xlsx` | column meanings and rules | leakage rule, units, codes |
| BTS Reporting Carrier On-Time Performance, 2023-01..2025-07 | every flight landing at ATL/DFW (1,644,216 flights) | target and schedule |

The data guide makes hourly AAR optional because the challenge's flight sample is too sparse
(~6 ATL arrivals a day) and says teams who want AAR "may pull the full BTS on-time data
automatically". `aar_bts.py` does exactly that. The challenge's flight sample is not used by P2.

Data rules from the guide that P2 follows:
- **No leakage:** weather is joined on airport + hour, then shifted by h ∈ {3, 6, 9, 12}: every weather
  input for target hour t is the observation of hour t − h.
- **Weather row available at time_utc + 1 h, per P1's finding** (a row stamped HH:00 can hold the HH:52
  METAR; `config/airports.yaml` `weather.availability_lag_hours`): the row joined for target hour t is the one
  stamped t − h − 1, so no weather input was observed after t − h.
- **Times:** BTS hhmm times are local; the weather file gives UTC and local hours. Everything is UTC.
- **Generality:** airports come from `config/airports.yaml` (IATA, time zone); `aar_bts.py` works for any
  airport there. The weather file covers ATL and DFW only, so that is what is trained and scored.
- `wind_gust_kmh` is 100% missing in the weather file and is dropped.

Raw files live in `data/raw/challenge5/` and caches in `data/interim/` (both gitignored).

## Brief checklist

| P2 requirement | Where |
|---|---|
| Target = actual arrivals per airport per hour, labelled as an AAR proxy | above; `metadata.json → target`; `aar_metrics_table.md` header |
| Baseline: persistence | arrivals in the last **completed** hour before t − h (the hour starting at t − h has not happened yet at forecast time) |
| Baseline: hour-of-day × day-of-week mean | airport × local hour × local weekday, training years only |
| Extra baseline: schedule × completion ratio | scheduled arrivals × (arrivals / scheduled) per airport, training years only |
| LightGBM, one model with h as a feature | `train_aar.py`; `h` is an input |
| Train on 2023–24, test on 2025 | final models fit on 2023-01-02..2024-12-31 only; 2025 scored once |
| RMSE by horizon × airport, model vs baselines | `aar_metrics_table.md` (readable), `aar_metrics.csv` (all slices) |
| p10/p90 via quantile regression | two LightGBM quantile models (α = 0.1, 0.9) |
| `aar_preds.csv (icao, target_time, h, pred, p10, p90)` | team schema names: `icao, target_hour_utc, h, pred_arr_count, p10, p90` |
| No future data, purely predictive | fail-closed input allowlist + tests; see [No future data](#no-future-data) |

## Quick start (from repo root, Windows)

```
py -3.12 -m venv .venv
.venv\Scripts\pip install -r requirements.txt -r src/models/requirements-aar.txt

# 1. put challenge5_hourly_weather_ATL_DFW.csv (and the data guide) in data\raw\challenge5\
# 2. download BTS arrivals for the airports in config/airports.yaml (~31 months, ~1 GB streamed,
#    ~30 min; resumable, keeps only small filtered parquet files in data\interim\bts\)
.venv\Scripts\python -m src.models.aar_bts --start 2023-01 --end 2025-07
# 3. build the hourly panel -> data\interim\p2_hourly_panel.parquet
.venv\Scripts\python -m src.models.aar_dataset
# 4. validate, train, evaluate, save
.venv\Scripts\python -m src.models.train_aar --validate-only
.venv\Scripts\python -m src.models.train_aar
# optional: which inputs help (validation only)
.venv\Scripts\python -m src.models.ablation_aar
# tests (synthetic data, no downloads needed)
.venv\Scripts\python -m pytest src/models/tests -q
```

Training refuses to run (no artifacts written) if the panel is missing or fails validation.

## Outputs (`outputs/predictions/`)

| file | contents |
|---|---|
| `aar_preds.csv` | contract columns first: `icao, target_hour_utc, h, pred_arr_count, p10, p90`; then `split` (val = 2024 H2, test = 2025), `actual_arr_count`, `pred_schedule`, `pred_persistence`, `pred_hist_avg`, `eval_hub_weather_actual` |
| `aar_metrics_table.md` | 2025 test RMSE by airport × horizon, model vs the three baselines, plus interval calibration |
| `aar_metrics.csv` | `split, model, icao, h, weather, n, rmse, mae, coverage_p10_p90, share_below_p10, share_above_p90, mean_interval_width`; `ALL` = aggregated level |
| `aar_ablation.csv` | which inputs help (validation only) |
| `aar_model/model.txt`, `model_p10.txt`, `model_p90.txt` | point and quantile LightGBM models |
| `aar_model/preprocessor.json` | input list and order, **provenance of every input**, every excluded column with the reason |
| `aar_model/metadata.json` | data sources, parameters, iteration counts, fit date range, split sizes, feature importance, baseline ratios, validation report |

`eval_hub_weather_actual` (normal / bad / severe) is the weather that **actually occurred** in hour t.
It is for slicing evaluation only; it is not known at forecast time and the model never sees it. The
dashboard should not show it as forecast information.

### Dashboard API

```python
from src.models.predict_aar import predict_aar
predict_aar(icao="KATL", h=6, start="2025-03-01", end="2025-03-07")   # reads the panel
predict_aar(features_df)                                              # or pass panel rows yourself
# -> DataFrame[icao, target_hour_utc, h, pred_arr_count, p10, p90], arrivals/hour, all >= 0
```

Models are loaded once and cached. Pass `features` covering at least 2 hours either side of the hours you
want (neighbouring-hour schedule inputs; missing ones become NaN). A missing model input column raises
`KeyError`. Realised arrivals, outcome and `oracle_*` columns in `features` are ignored.

Suggested "Low confidence" rule for P4: `p90 − p10 ≥ 20` arrivals (17.5% of 2025 test rows), or input
weather missing, or severe weather observed at the hub. Show the band as a **~70% band**, not 80% (see
intervals).

## Results (2025 test, scored once)

Rows: train 104,784 · val 35,328 · test 40,704 (4 horizons per hour). Point model: Poisson LightGBM,
1,670 iterations; p10: 278; p90: 292. 39 inputs.

| airport | LightGBM RMSE | schedule × ratio | hour × weekday mean | persistence | vs best baseline | LightGBM MAE |
|---|---|---|---|---|---|---|
| KATL | **6.86** | 8.68 | 9.32 | 34.28 | −21% | see `aar_metrics.csv` |
| KDFW | **8.96** | 14.58 | 10.68 | 34.91 | −16% | see `aar_metrics.csv` |

Units: arrivals per hour (KATL averages 35/hour in 2025). Validation (2024 H2): KATL 6.80, KDFW 8.58.
Before the weather-availability fix (row stamped t − h): KATL 6.82, KDFW 8.89. The weather, interval,
importance and ablation tables below were produced before that fix; the shift moves scores by < 1%.

| h | KATL RMSE | vs best baseline | KDFW RMSE | vs best baseline |
|---|---|---|---|---|
| 3 | 6.76 | −22.1% | 8.78 | −17.8% |
| 6 | 6.92 | −20.3% | 8.93 | −16.4% |
| 9 | 6.87 | −20.9% | 8.92 | −16.5% |
| 12 | 6.91 | −20.4% | 9.20 | −13.9% |

- **Beats every baseline at every airport and horizon.** Accuracy declines gently with lead time,
  mostly at KDFW, as the weather and recent-traffic inputs grow staler.
- **Persistence is a poor baseline here** (RMSE ~34): arrivals follow a strong daily cycle, so "same as
  the last completed hour before t − h" is from a different part of the day.
- **Unbiased overall:** mean forecast vs actual (h = 6) 35.3 vs 35.0 at KATL, 35.1 vs 34.9 at KDFW.

**Feature importance (gain share):** `sched_arr_count` 72.9%, local hour 16.1%, next hour's schedule
4.2%, schedule t+2 1.9%, previous hour's schedule 1.7%. Weather inputs hold 1.5% of the gain, but the
ablation shows removing them makes validation RMSE 3.8% worse: gain share understates their value
because they matter in the minority of hours with weather.

### By actual weather at the hub (2025 test RMSE)

| airport | weather in hour t | rows | LightGBM | schedule × ratio | hour × weekday mean | persistence |
|---|---|---|---|---|---|---|
| KATL | normal | 17,152 | **5.90** | 7.78 | 8.25 | 34.66 |
| KATL | bad | 2,892 | **9.90** | 11.74 | 12.45 | 32.41 |
| KATL | severe | 308 | **14.81** | 17.83 | 22.10 | 29.41 |
| KDFW | normal | 17,728 | **8.05** | 13.91 | 9.86 | 35.24 |
| KDFW | bad | 2,204 | **12.12** | 16.83 | 13.56 | 33.41 |
| KDFW | severe | 420 | **17.97** | 25.36 | 21.36 | 27.93 |

Bad = rain (codes ≥ 7), ≥ 1 mm/h, wind ≥ 37 km/h or crosswind ≥ 28 km/h. Severe = thunderstorm/hail
(≥ 23), freezing rain/sleet/snow, ≥ 4 mm/h or wind ≥ 46 km/h. Rows count every horizon.

The model beats all baselines in every weather class, but error rises 2–3× from normal to severe hours:
storms cut realised arrivals in ways the inputs available 3–12 h ahead only partly anticipate.

### Failure case: severe weather

Severe-weather hours have the largest errors (RMSE 14.8 KATL, 18.0 KDFW vs 5.9 / 8.1 in normal weather).
**Why:** a thunderstorm that builds within the forecast horizon is not in the weather observed at
t − h, and even a forecastable storm's effect (ground stops, holding, diversions) varies a lot from hour
to hour; 2023–24 holds only about 240 severe-weather hours per airport (out of ~17,500). **How a user would notice:**
a wide p10–p90 band, severe weather observed at the hub, or a forecast close to the schedule while
storms are reported. The dashboard can flag these ("severe weather; forecast uncertain").

### Prediction intervals (p10–p90)

Two LightGBM quantile models (α = 0.1 and 0.9), same inputs, early-stopped on validation (quantile loss)
and refit on 2023–24. Bounds are widened if needed so p10 ≤ forecast ≤ p90, and clipped at 0.

| 2025 test | coverage (nominal 0.80) | below p10 (nominal 0.10) | above p90 (nominal 0.10) | mean width |
|---|---|---|---|---|
| KATL | 0.733 | 0.137 | 0.131 | 11.9 |
| KDFW | 0.686 | 0.159 | 0.154 | 15.1 |

**The band is too narrow:** it holds 69–73% of actuals instead of 80%, with misses on both sides
roughly equally. Treat it as a ~70% band. A conformal adjustment on validation residuals would restore
80% coverage; not done yet. Width median 12.9, 90th percentile 23.7 arrivals.

## Model

- **One LightGBM point model** (Poisson objective; `seed=42`, deterministic; defaults otherwise,
  `aar_config.LGBM_PARAMS`) with `h` as an input, plus the two quantile models. Predictions clipped at 0.
- **Split** (by `target_hour_utc`, UTC, never shuffled): train 2023-01-02..2024-06, validation
  2024-07..12, test 2025-01..07. Iteration counts and every design choice are picked on validation;
  final models are refit on all of 2023–24; 2025 is scored once. `--no-refit` scores test with
  train-only fits.
- **Inputs (39), each with a provenance recorded in `preprocessor.json`:**

| group | inputs | why it is known at forecast time t − h |
|---|---|---|
| identifier (2) | `icao` (categorical), `h` | — |
| calendar (4) | `p2_local_hour`, `p2_local_dow`, `p2_local_month`, `p2_is_weekend` | derived from the target hour itself |
| schedule (7) | `sched_arr_count` (hour t), `p2_sched_m2/m1/p1/p2` (t−2..t+2), `p2_ops_sched_last1h/3h` | published schedule, incl. flights later cancelled |
| hub weather (18) | weather file columns (temperature, dew point, humidity, precipitation, wind direction/speed, pressure, condition code, head/cross/tailwind) + rolling: precipitation sum, wind max and crosswind max over 3 h and 6 h, pressure change over 3 h | row stamped t − h − 1 (available by t − h = `feature_obs_time`), windows ending at that stamp |
| weather × schedule (6) | `p2_hub_bad/severe/winter`, `p2_sched_x_hub_bad/severe`, `p2_sched_m1_x_hub_bad` | flags from the t − h weather × the schedule |
| recent arrivals (2) | `p2_ops_arr_last1h`, `p2_ops_arr_last3h` | flights that **landed** in the hours ending at t − h |

- **Baselines.** Persistence: arrivals in hour `[t − h − 1 h, t − h)`, the last hour fully over at
  forecast time. Hour × weekday mean: airport × local hour × local weekday, each hour counted once across
  horizons, training rows only. Schedule × ratio: `sched_arr_count` × training-period arrivals/scheduled
  per airport (KATL 0.989, KDFW 0.979).

## Ablation (`aar_ablation.csv`, validation only)

Each variant trained on train, scored on 2024 H2 validation (35,328 rows; 3,380 bad-weather, 412 severe,
by the actual weather, for scoring only).

| variant | val RMSE | bad-weather hours | severe hours |
|---|---|---|---|
| baseline: schedule × ratio | 11.85 | 14.76 | 17.99 |
| baseline: hour × weekday mean | 10.65 | 12.43 | 15.79 |
| A schedule + calendar only | 8.17 | 12.67 | 16.90 |
| B A + hub weather at t − h | **7.73** | 12.01 | 16.03 |
| C B + weather exposure | 7.81 | 12.02 | 15.90 |
| **D C + recent arrivals (all inputs, production)** | 7.73 | **11.85** | **15.70** |
| E D without any weather | 8.03 | 12.29 | 16.69 |
| F D with L2 objective | 7.73 | 12.02 | 16.20 |
| G D with the weather of hour t itself (`oracle_*`, diagnostic only) | 7.73 | 11.63 | 15.80 |

- **Weather helps:** adding the t − h weather cuts validation RMSE 5.5% (A → B), and removing it from the
  full model costs 3.8% overall, 3.8% in bad weather and 6.2% in severe weather (D vs E).
- **D is production:** best in bad and severe weather among the deployable variants, tied best overall.
- Perfect knowledge of the target hour's weather (G) adds little beyond D overall, a bit in bad weather:
  the t − h observation already carries most of the usable weather signal.
- Poisson beats L2 in bad and severe weather (D vs F).

## No future data

The model is purely predictive: every input is known at the forecast time t − h, and nothing from 2025
is used for fitting or tuning.

- **Fail-closed input allowlist.** Every input must belong to a known-at-forecast-time group (table in
  [Model](#model)); its reason is saved in `preprocessor.json → provenance`. Any other column stops
  training with an error.
- **Never inputs:** `arr_count`, `cancellations`, `diversions`, `capacity_ratio`, `mean_arr_delay`,
  `share_delayed_45`, any `oracle_*` or `arr_count*` column, and any column named like an outcome
  (`delay`, `cancel`, `divert`, `divers`, `capacity`). Each exclusion and its reason are saved.
- **Weather:** the row stamped t − h − 1 (available at stamp + 1 h, so by t − h), rolling windows end at that stamp; validation fails if any row has
  `feature_obs_time > target_hour_utc − h`. A test checks the shift for every h and that `oracle_*` is
  the target hour.
- **Recent arrivals** count only hours that ended by t − h (by landing time, not scheduled time); a test
  checks no later hour is counted.
- **Schedule** inputs use only `sched_arr_count`, the published schedule; a test checks they do not
  change when realised arrivals are scrambled.
- **Baselines:** persistence uses only the last hour fully over at t − h (asserted in code); the
  hour × weekday mean and the schedule ratio are fit on training years only.
- **Time:** chronological split; the final models are fit on rows before 2025-01-01 (asserted in code,
  and a test spies on every LightGBM training set). 2025 is scored once at the end.
- **End-to-end test:** setting the target, every outcome column and every `oracle_*` column to garbage
  leaves the forecast and the p10/p90 band unchanged.

## How to read the outputs

**`aar_preds.csv`.** One row per airport × UTC hour × horizon. `pred_arr_count` is the expected number
of arrivals in that hour; `p10`/`p90` the band. Example (KATL, h = 6, 13 Jan 2025):

```
target_hour_utc   actual  pred_arr_count  p10    p90    pred_schedule  pred_hist_avg
2025-01-13 12:00    50        50.4        41.8   59.7      29.7           68.9
2025-01-13 13:00    67        67.1        58.5   81.9      82.1           80.4
2025-01-13 14:00    58        51.7        45.7   56.8      54.4           55.1   (actual above p90)
```

**`aar_metrics.csv` / `aar_metrics_table.md`.** MAE = average miss in arrivals per hour (4.5 at KATL).
RMSE = same idea, big misses weigh more; the main score. Lower is better. `n` = rows scored; `ALL` =
aggregated. Interval columns are filled for LightGBM only.

## Files

| file | role |
|---|---|
| `aar_config.py` | data sources and paths, horizons, split dates, input allowlist and denylist, LightGBM and quantile settings, output schema |
| `aar_bts.py` | download full BTS on-time data for configured airports; local → UTC arrival and landing times; hourly arrivals |
| `aar_dataset.py` | build the panel: weather joined on airport + hour and shifted by h, rolling windows, recent arrivals, oracle weather for evaluation |
| `aar_data.py` | load, validate (keys, timestamps, horizons, `feature_obs_time ≤ t − h`, missingness), chronological split |
| `aar_features.py` | input selection, provenance (fail-closed), local calendar features, preprocessor |
| `aar_context.py` | neighbouring-hour schedule and weather-exposure inputs |
| `aar_baselines.py` | persistence, hour × weekday mean, schedule × ratio |
| `aar_metrics.py` | RMSE / MAE and interval scores by airport, horizon, weather |
| `train_aar.py` | CLI: train, evaluate, save |
| `ablation_aar.py` | input ablation on validation → `aar_ablation.csv` |
| `predict_aar.py` | `predict_aar()` for the dashboard |
| `tests/test_aar.py` | 19 tests, synthetic data: BTS time conversion, weather shift, recent-arrival window, validation, exclusions, provenance, baselines, intervals, no-future-data |
| `requirements-aar.txt` | `lightgbm`, `pytest` (P1 owns the shared `requirements.txt`) |

## Assumptions

- `target_hour_utc` labels the hour starting at that time. A flight counts in the hour its wheels touch
  down; `sched_arr_count` counts every scheduled arrival, including flights later cancelled.
- The repeated hour when clocks go back is read as standard time (one hour a year); the skipped spring
  hour is shifted forward.
- The panel starts 2023-01-02 local (so red-eyes that departed before 2023 are not missing) and ends at
  the last weather hour, 2025-07-31 23:00 UTC.
- A weather observation missing at exactly t − h stays missing (NaN, ~0.01–0.15% of rows); it is not
  filled from a neighbouring hour.

## Limitations

- **Proxy target:** realised arrivals mix capacity with demand; a quiet hour is not a low-capacity hour.
- **Severe weather is the weak spot** (2–3× normal error), though the model still beats every baseline.
- **Intervals are too narrow** (69–73% coverage vs 80% nominal).
- **Weather is ATL and DFW only:** the BTS pipeline works for any configured airport, but weather for
  other airports would need another source.
- Hyperparameters are LightGBM defaults; only inputs and objective were chosen on validation, and
  validation scores are slightly optimistic because validation also picks iteration counts.
