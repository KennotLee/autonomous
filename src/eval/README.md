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
| — | 8-slide outline, 3-min script with [PLACEHOLDER] model numbers, dashboard QA checklist | `pitch.md` |

## Findings v2: event-level delay > 45 min (organiser update, P1 `event_features.parquet`)
Events are departures from ATL/DFW (DEP_DELAY > 45) and arrivals at ATL/DFW (ARR_DELAY > 45).
- Train: FL_DATE 2023–2024, 19,480 events, 10.0% delayed. Test: FL_DATE 2025, 5,704 events, 13.2% delayed.
- 428 of 25,612 events (1.7%) were dropped: 350 cancelled, 78 diverted.
- Weather comes from the organisers' hourly file at t − h.

```
python src/eval/baselines_events.py [--events <parquet>]     # B0-B3 -> event_baseline_metrics.csv, outputs/predictions/event_baseline_preds.csv
python src/eval/evaluate_events.py --preds <csv> --name <tag>  # model vs best baseline -> event_model_vs_baseline_<tag>.csv, event_calibration_<tag>.csv
python src/eval/failure_cases.py [--preds <csv> --name <tag>]  # missed storm + false alarm -> failure_cases_<tag>.md (default B3)
python src/eval/bts_arrivals.py download && python src/eval/bts_arrivals.py hourly   # full BTS -> independent hourly arrivals
python src/eval/evaluate_aar_bts.py --preds <p2 aar_preds.csv>  # -> aar_bts_model_vs_baseline.csv
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

### Failure cases (`failure_cases.py` → `failure_cases_<name>.md`, h = 3)
B3 and P3 v1 fail on **the same two days**.

**Missed storm: DFW, 2025-04-23**
- 11 events, 9 of them late by more than 45 minutes (82%). B3 expected 1.6 late events and P3 1.9.
- The first five late events, between 08:10 and 13:39, had *overcast, no adverse weather* at t − 3h.
- A thunderstorm shows up at the scheduled time only at 12:00, so convection formed less than 3 h ahead.
- BTS cause minutes on the late events: late aircraft 574, weather 243, NAS 139, carrier 51. That points to delay propagating in from upstream airports and earlier rotations.
- **Fix:**
  - TAF or convective-outlook features;
  - inbound-aircraft and origin-airport delay;
  - hub state (recent *actual* delays). The hub was already running late by 09:00, which would have raised the afternoon risk.
- **How a controller notices:** the dashboard shows low risk while the live delay feed is red. So the dashboard should show "observed delays in the last 2 h" next to the forecast.

**False alarm: DFW, 2025-05-30**
- 22 events, 0 late. B3 flagged 9 in the top 15%; P3 flagged 4.
- The weather at t − 3h was adverse on 73% of events. It was light rain plus a **17–30 km/h tailwind on the assumed runway heading 180°**.
- A wind from the north is a tailwind on 18 but a headwind on 36. DFW simply operates in north flow, so the fixed-heading rule misfires.
- Evidence across all events: "tailwind only" triggers have a delay rate of 0.135 in train and 0.145 in test. That is close to normal weather (0.087 / 0.117) and far below fog/rain codes (0.153 / 0.202) and thunderstorms (0.348 / 0.300).
- In 2025 test, "tailwind only" accounts for 703 of the 1,458 events that the adverse rule fires on (48%).
- **Fix:** take the minimum tailwind over both runway ends (the airport picks its runway direction), and separate light rain from convection.

### Arrivals per hour (optional AAR): independent check of P2 `19e65a6` (full BTS, leak-fixed)
`bts_arrivals.py` downloads 31 BTS months (1.64M ATL/DFW arrivals). It rebuilds the hourly landings from BTS's own WheelsOn clock time; P2 uses scheduled arrival + ArrDelay − TaxiIn. `evaluate_aar_bts.py` scores P2's predictions and writes `aar_bts_model_vs_baseline.csv`.

P2's `19e65a6` (on branch `p2-fix3`) makes weather available at stamp + 1 h, fixing the leak flagged below. That raised RMSE at h = 3 from 6.70 to 6.76 (KATL) and from 8.68 to 8.78 (KDFW).
- **P2's target is correct:** it equals P4's on 99.4% of 2025 hours and is within ±1 on 100%. Mean is 34.98 arrivals per hour in both.
- **Stronger baseline:** `sched_spread` spreads the published schedule over the 2023–24 distribution of landing-hour offsets for each airport × local hour, including the not-landed share. RMSE: KATL 7.01 and KDFW 9.44, against P2's best baselines of 8.68 and 10.68.

| icao | h | P2 model | sched_spread | P2's best baseline | vs sched_spread (90% CI of RMSE diff) |
|---|---|---|---|---|---|
| KATL | 3 | 6.76 | 7.01 | 8.68 | −3.5% [−0.49, −0.03] |
| KATL | 12 | 6.91 | 7.01 | 8.68 | −1.4% [−0.31, +0.06] |
| KDFW | 3 | 8.78 | 9.44 | 10.68 | −7.0% [−1.04, −0.32] |
| KDFW | 12 | 9.20 | 9.44 | 10.68 | −2.5% [−0.51, +0.02] |

- **The honest headline is −1% to −7% vs reading the schedule well.** The gain is significant at KATL only at h = 3, and at KDFW at h = 3–9.
- **The value is in bad weather.** These slices use P2's oracle label for the target hour, for slicing only:
  - in hours with **bad** weather the model is 4–13% better, with every CI below 0;
  - in hours with **severe** weather it is 1–22% better (the KDFW h = 12 CI includes 0);
  - in normal-weather hours at KATL it is slightly *worse* (+1% to +3%).
- **The P10–P90 band is too narrow.** Coverage is 69–75% overall against a nominal 80%, and 42–56% in severe hours.

### Leakage review v2 (P1 `b064886`, P3 v1 `b27ba59`): no leak found
| file:line | check | result |
|---|---|---|
| p1 `src/data/events.py:63-64` | flights-CSV weather `*_ORIGIN/*_DEST` | kept only as `oracle_*`; P3 excludes them with an assert (`train.py:54-59`) ✅ |
| p1 `src/data/events.py:49-50,77-79` | DEP_DELAY / ARR_DELAY / CANCELLED / DIVERTED | used only to build `delay_min`, `label_45`, `cancelled`, `diverted` (targets) ✅ |
| p1 `src/data/build_events.py:37` + `hourly_weather.py:50` | weather join shifted ≥ h before *scheduled* time | cutoff = t − h, and a row counts as available at time_utc + 1 h. In the data, `feature_obs_time ≤ t − h` on 100% of rows with weather (the 2 failing rows are the first events of 2023-01-01 at h = 12, which have no earlier weather, so NaN) ✅ |
| p1 `hourly_weather.py:31,50` | UTC/local, DST | Against Meteostat (P4 cache), organiser row T equals Meteostat T + 1 h (the METAR at about T:52) on 66% of hours. Any other offset from −7 to +2 h matches ≤ 1.3%, with similar rates in DST and standard time. So the file is true UTC, and the +1 h availability lag is *needed* and sufficient. `events.py:73` turns DST-ambiguous local times into NaT; 0 events dropped ✅ |
| p1 `hourly_weather.py:40-46` | rolling windows including the target hour | lags are shift(k) on a complete grid; `prcp_sum_6h` covers (T − 6 h, T] at row T, which is available at T + 1 h ≤ t − h ✅ |
| p1 `events.py:80-82` | calendar | hour/dow/month from the *scheduled* local time ✅ |
| p3 `train.py:37,49-50,78-83` | target encodings / tuning on test | no target encoding; early stopping and threshold on 2024-H2 validation. The test split is on UTC, so it is a superset of FL_DATE 2025 and no test event reaches training ✅ |
| p1 `hourly_weather.py:36` | minor | duplicate `time_utc` rows silently keep the first; log the count |
| p1 `events.py:78` | minor | diverted *departures* keep a label (DEP_DELAY is known); P4 drops them (78 events) and P3 trains on them. This is a consistency issue, not a leak |

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
