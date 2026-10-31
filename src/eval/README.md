P4 Evaluation. Reads outputs/predictions/*.csv + data/processed/*.parquet, writes outputs/metrics/. Prediction schemas:
- outputs/predictions/aar_preds.csv — columns: icao (str, e.g. KATL), target_hour_utc (ISO-8601 UTC timestamp, hour start), h (int hours ahead, one of 3,6,9,12), pred_arr_count (float, predicted arrivals in that UTC hour)
- outputs/predictions/delay_preds.csv — **current (organiser update, P1 v2)**: event_id (matches data/processed/event_features.parquet), h (3,6,9,12), risk_prob (0–1, P(delay > 45 min)), top_drivers (text). Older flight_id/p_delay45 version: see v1 sections.

# P4: baselines, model-vs-baseline evaluation, leakage review

All numbers below are on the **2025 test set** (2025-01-01 UTC onward). Every baseline is fit on data
before 2025 only, with no random splits. The full tables are in
[`outputs/metrics/test_set_summary.md`](../../outputs/metrics/test_set_summary.md), generated from the
CSVs, never typed by hand.

## Run (from repo root; Windows `.venv/Scripts/python`, macOS/Linux `.venv/bin/python`)

```
python src/eval/baselines.py all          # baselines + CSV-weather alignment check
# fetch teammates' predictions from their branches (Git Bash; PowerShell's > re-encodes files)
git show origin/p2:outputs/predictions/aar_preds.csv   > data/interim/p2_aar_preds.csv
git show origin/p3:outputs/predictions/delay_preds.csv > data/interim/p3_delay_preds.csv
python src/eval/evaluate_aar.py   --preds data/interim/p2_aar_preds.csv --model-col pred_arr_count   # band: p10, p90
python src/eval/evaluate_delay.py --preds data/interim/p3_delay_preds.csv --name p3_gbm
python src/eval/leakage_checks.py
python src/eval/cross_airport.py         # AAR train-on-one-hub, test-on-other
python src/eval/summarize.py              # -> outputs/metrics/test_set_summary.md
```

Needs P1's `data/processed/*.parquet` (branch `p1`), `config/airports.yaml`, the raw CSV in
`data/raw/` (for the alignment and leakage checks), and `scikit-learn` alongside `requirements.txt`.

## Files

| file | what it does | output in `outputs/metrics/` |
|---|---|---|
| `baselines.py` | delay baselines: global rate, smoothed rate per origin × local dep hour × month, and dest × arr hour × month. Arrival baselines: persistence of the last *completed* hour before t − h, hour-of-day / hour-of-week means, scheduled arrivals × train completion ratio. P10/P90 from train residuals. Also checks the raw CSV's weather alignment. | `delay_baseline_*.csv`, `delay_share_by_hub.csv`, `aar_baseline_*.csv`, `weather_alignment_check.csv` |
| `evaluate_aar.py` | P2 predictions vs baselines on identical test hours, per icao × h × slice, with a 90% day-block bootstrap CI and P10–P90 band calibration (model vs schedule-baseline band) | `aar_model_vs_baseline.csv` |
| `evaluate_delay.py` | delay model vs baselines on identical test flights, per h × slice, with a 90% bootstrap CI | `delay_model_vs_baseline_<name>.csv` |
| `leakage_checks.py` | measures how much of a scheduled-hour "hub state" feature is unknowable at issue time | `leakage_hub_state.csv` |
| `cross_airport.py` | AAR model trained on one hub, scored on the other (no `icao` input) vs within-airport | `aar_cross_airport.csv` |
| `summarize.py` | builds the test-set summary tables | `test_set_summary.md` |
| — | problem statement, before/after workflow, sourced cost figures | `problem_framing.md` |

## Findings v2: event-level delay > 45 min (organiser update, P1 `event_features.parquet`)
Events are departures from ATL/DFW (DEP_DELAY > 45) and arrivals at ATL/DFW (ARR_DELAY > 45).
- Train: FL_DATE 2023–2024, 19,480 events, 10.0% delayed. Test: FL_DATE 2025, 5,704 events, 13.2% delayed.
- 428 of 25,612 events (1.7%) were dropped: 350 cancelled, 78 diverted.
- Weather comes from the organisers' hourly file at t − h.

```
python src/eval/baselines_events.py [--events <parquet>]     # B0-B3 -> event_baseline_metrics.csv, outputs/predictions/event_baseline_preds.csv
python src/eval/evaluate_events.py --preds <csv> --name <tag>  # model vs best baseline -> event_model_vs_baseline_<tag>.csv, event_calibration_<tag>.csv
```

**Baselines, h = 3, all test events** (top 15% = 856 flags):

| baseline | PR-AUC | Brier | precision top 15% | recall top 15% |
|---|---|---|---|---|
| B0 constant train rate | 0.132 | 0.116 | 0.123 | 0.139 |
| B1 airport × direction × local hour | 0.189 | 0.113 | 0.217 | 0.247 |
| B2 B1 + month | 0.198 | 0.113 | 0.230 | 0.262 |
| **B3 B2 × adverse-weather multiplier at t − h** | **0.215** | **0.112** | **0.243** | **0.276** |

- **B3's rule, fit on train:**
  - Adverse weather means a fog/rain/thunderstorm code, crosswind ≥ 35 km/h or tailwind ≥ 15 km/h.
  - The multiplier at h = 3 is ×1.45 when adverse and ×0.88 when not.
  - 25.6% of test events have adverse weather at t − 3h.
- **B3 vs B2:** +0.017 PR-AUC at h = 3 (90% CI [+0.007, +0.028]). The gain fades to +0.004 at h = 12, where the CI includes 0.
- **Top-15% threshold:** set across both airports, so flags concentrate at DFW, which has the higher historical rate.

**P3 v1 (`b27ba59`) vs the best baseline (B3), all test events.** The 90% CI resamples whole days.

| h | model | B3 | change | 90% CI of PR-AUC difference |
|---|---|---|---|---|
| 3 | 0.220 | 0.215 | +2% | [−0.015, +0.022] |
| 6 | 0.222 | 0.208 | +7% | [−0.005, +0.036] |
| 9 | 0.221 | 0.209 | +6% | [−0.007, +0.029] |
| 12 | 0.219 | 0.202 | +9% | [+0.001, +0.042] |

- **DFW:** +5% / +11% / +14% / +17% at h = 3 / 6 / 9 / 12. The gain is significant at h = 9 and 12.
- **ATL:** +1% / +2% / −4% / −1%. There is no gain at any h.
- **The model is flat across h,** so it gets nothing from fresher weather.
- **Calibration:** both the model and B3 under-predict the lower bins by 2–4 points, because 2025 has a higher delay rate (13.2% vs 10.0% in train).

The v1 sections below use the old flight table and Meteostat weather. They are kept only for history.

## Findings v1 (superseded; flight table + Meteostat, test set 2025)

### 1. Hourly arrivals (AAR proxy): P2 LightGBM v3 (`6ac3dcf`) beats every baseline
| airport | model RMSE | schedule baseline | hour-of-week mean | persistence | vs strongest baseline (90% CI of RMSE diff) |
|---|---|---|---|---|---|
| KATL | **0.370–0.371** | 0.431 | 0.503 | 0.723–0.737 | −13.9% to −14.2% ([−0.071, −0.051]) |
| KDFW | **0.342–0.343** | 0.419 | 0.426 | 0.615–0.626 | −18.2% to −18.3% ([−0.086, −0.066]) |

The ranges cover h = 3, 6, 9 and 12, over 5,092 KATL and 5,093 KDFW hours. The v3 point forecasts are essentially the same as v2's. v3's new hub weather-exposure inputs are leak-safe: P1 weather at t − h times the published schedule, with a fail-closed input allowlist.

- **P10–P90 band:**
  - Coverage is 87–89% in hours with scheduled arrivals (nominal 80%) and 93–94% over all hours.
  - 0% of outcomes fall below P10 (P10 is 0, which is correct for 0–2 scheduled flights). 12–13% exceed P90 in busy hours, so the top of the band is slightly narrow.
  - My schedule-baseline band (train residual quantiles) is nearly zero-width (0.01–0.02) and covers only 44–49% of busy hours. Use P2's band in the dashboard, not mine.

- **Quote the gain against the schedule baseline,** not the historical average. The model is driven by the published schedule (target hour plus neighbours, about 90% of gain). The schedule baseline shows how much the model adds beyond simply reading the schedule.
- **Skill is flat across h.** P2's validation ablation shows weather adds no skill on this sample. Even perfect same-hour weather gains only about 0.4% on bad-weather hours. Present this honestly: *the sample (about 5 arrivals per day) is too sparse for weather to show; this is not evidence that weather doesn't matter.*
- **Snow/storm hours are a failure case.**
  - KDFW (62 hours): the plain hour-of-week mean (RMSE 0.270) beats the model (0.334–0.366) by 24–35%.
  - KATL (102 hours): the model's edge over the schedule baseline is not significant at h = 3, 6 and 9 (the CI reaches 0.000 to +0.012).
- **Caveat:** `arr_count` is 0 in about 82% of hours (a 0.5% BTS sample). This RMSE is not runway-capacity accuracy.

### 1b. Cross-airport generalisation (`cross_airport.py` → `aar_cross_airport.csv`)
This uses the P2 v3 recipe without the `icao` input. Each model is trained on one hub and scored on the other hub's 2025 hours.

| test airport | within (trained here) | cross (trained on other hub) | schedule baseline | cross vs schedule |
|---|---|---|---|---|
| KATL | 0.372 | 0.381 | 0.428 | −11.1% |
| KDFW | 0.343 | 0.353 | 0.422 | −16.2% |

- A model that never saw the target airport loses only about 0.01 RMSE (2.4–3.1%) and still beats that airport's schedule baseline. The pipeline transfers because:
  - wind components come from the configured runway heading;
  - the main driver, the schedule, exists at every airport.
- Severe-weather hours: the cross model is *no worse* than the within model. Mean prediction is 0.20 vs an actual 0.08 at KDFW (62 hours), so both over-predict.

### 2. Delay > 45 min: P3 `gbm` (leak fixed in `b8618cc`) beats the best baseline at every h
| h | model PR-AUC | best baseline | global rate | gain (90% CI of PR-AUC diff) | Brier model / baseline |
|---|---|---|---|---|---|
| 3 | **0.333** | 0.199 | 0.136 | +68% ([+0.108, +0.158]) | 0.106 / 0.116 |
| 6 | **0.312** | 0.199 | 0.136 | +57% ([+0.089, +0.135]) | 0.108 / 0.116 |
| 9 | **0.279** | 0.199 | 0.136 | +40% ([+0.058, +0.104]) | 0.110 / 0.116 |
| 12 | **0.272** | 0.199 | 0.136 | +37% ([+0.051, +0.098]) | 0.111 / 0.116 |

The test set has 5,641 hub flights, 767 of them late. Cancelled and diverted flights are excluded, per P1's `label_45`.

- **Leak fixed and re-verified.**
  - The old `hub_*` ops-state features bucketed each flight by its *scheduled* hour. 60–86% of what they counted was unknown at issue time (`leakage_hub_state.csv`; this measures the old features).
  - P3 now buckets by *actual* arrival/departure hour (`pipeline/panel.py` `*_act`). They are joined at `_ops_hour = floor(t − h) − 1 h` (`src/models/delay/features.py:82`), so every counted flight had landed or left by issue time.
  - The leaky version scored 0.343 at h = 3, so the leak was worth only about 0.01 PR-AUC. Recent hub delays are a genuine early signal, not an artefact.
- **Gain shrinks with h** (+68% → +37%) because the hub state goes stale. Even at 12 h the CI excludes 0.
- **By slice (h = 3):**
  - KATL arrivals 0.246 vs 0.121; KATL departures 0.398 vs 0.235.
  - KDFW arrivals 0.277 vs 0.188; KDFW departures 0.363 vs 0.212.
  - KDFW arrivals at h = 12: the CI touches 0 ([−0.001, +0.092]).
- **Failure case: snow/storm at the hub end** (94 flights).
  - The model does not beat the baseline at any h (−3% to −17%; every CI includes 0).
  - At KATL (69 flights, 23 late) the model scores 0.45–0.51 vs a baseline of 0.64 at every h.
  - The thunderstorm-hour signal is too rare in train for the model to learn. Say so on the failure slide.

### 3. Leakage review status
| item | status |
|---|---|
| P1 features: `feature_obs_time ≤ t − h` | ✅ passes on 100% of rows in both parquets |
| Raw CSV weather | ❌ misaligned by 4–6 h (naive local-as-UTC key matches 95–100% of rows, correct UTC key 5–6%; `weather_alignment_check.csv`). Not used by anyone. |
| P2 AAR v3 | ✅ no outcome or `oracle_*` inputs; persistence uses only completed hours; neighbouring-hour schedule is known in advance; hub/origin weather-exposure inputs use obs ≤ t − h; fail-closed input allowlist; ablation run on validation only |
| P3 delay `gbm` | ✅ since `b8618cc`: ops state bucketed by actual time, last hour closes at issue time. `hub_sched_*` = published schedule. (❌ before: scheduled-hour bucketing.) |

### Open requests
- ~~P2: add P10/P90 and use `pred_arr_count`~~ — done in `6ac3dcf`.
- ~~P3: fix the leaky `hub_*` features~~ — done in `b8618cc`, re-scored above.
- **P1:** add `scikit-learn` to `requirements.txt`. `lightgbm` and `pytest` were added in `1fd3662`.
