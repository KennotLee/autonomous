P4 Evaluation. Reads outputs/predictions/*.csv + data/processed/*.parquet, writes outputs/metrics/. Prediction schemas:
- outputs/predictions/aar_preds.csv — columns: icao (str, e.g. KATL), target_hour_utc (ISO-8601 UTC timestamp, hour start), h (int hours ahead, one of 3,6,9,12), pred_arr_count (float, predicted arrivals in that UTC hour)
- outputs/predictions/delay_preds.csv — columns: flight_id (str, matches data/processed/flight_features.parquet), sched_time_utc (ISO-8601 UTC, scheduled departure), h (int, one of 3,6,9,12), p_delay45 (float 0–1, probability arrival delay > 45 min)

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
python src/eval/evaluate_aar.py   --preds data/interim/p2_aar_preds.csv            # P2 column pred_lgbm
python src/eval/evaluate_delay.py --preds data/interim/p3_delay_preds.csv --name p3_gbm
python src/eval/leakage_checks.py
python src/eval/summarize.py              # -> outputs/metrics/test_set_summary.md
```

Needs P1's `data/processed/*.parquet` (branch `p1`), `config/airports.yaml`, the raw CSV in
`data/raw/` (for the alignment and leakage checks), and `scikit-learn` alongside `requirements.txt`.

## Files

| file | what it does | output in `outputs/metrics/` |
|---|---|---|
| `baselines.py` | delay baselines: global rate, smoothed rate per origin × local dep hour × month, and dest × arr hour × month. Arrival baselines: persistence of the last *completed* hour before t − h, hour-of-day / hour-of-week means, scheduled arrivals × train completion ratio. P10/P90 from train residuals. Also checks the raw CSV's weather alignment. | `delay_baseline_*.csv`, `delay_share_by_hub.csv`, `aar_baseline_*.csv`, `weather_alignment_check.csv` |
| `evaluate_aar.py` | P2 predictions vs baselines on identical test hours, per icao × h × slice, with a 90% day-block bootstrap CI | `aar_model_vs_baseline.csv` |
| `evaluate_delay.py` | delay model vs baselines on identical test flights, per h × slice, with a 90% bootstrap CI | `delay_model_vs_baseline_<name>.csv` |
| `leakage_checks.py` | measures how much of a scheduled-hour "hub state" feature is unknowable at issue time | `leakage_hub_state.csv` |
| `summarize.py` | builds the test-set summary tables | `test_set_summary.md` |
| — | problem statement, before/after workflow, sourced cost figures | `problem_framing.md` |

## Findings (test set 2025)

### 1. Hourly arrivals (AAR proxy): P2 LightGBM v2 beats every baseline
| airport | model RMSE | schedule baseline | hour-of-week mean | persistence | vs strongest baseline (90% CI of RMSE diff) |
|---|---|---|---|---|---|
| KATL | **0.369–0.370** | 0.431 | 0.503 | 0.723–0.737 | −14.2% to −14.4% ([−0.072, −0.052]) |
| KDFW | **0.342–0.343** | 0.419 | 0.426 | 0.615–0.626 | −18.1% to −18.2% ([−0.086, −0.066]) |

The ranges cover h = 3, 6, 9 and 12, over 5,092 KATL and 5,093 KDFW hours.

- **Quote the gain against the schedule baseline,** not the historical average. The model is driven by the published schedule (target hour plus neighbours, about 90% of gain). The schedule baseline shows how much the model adds beyond simply reading the schedule.
- **Skill is flat across h.** P2's validation ablation shows weather adds no skill on this sample. Even perfect same-hour weather gains only about 0.4% on bad-weather hours. Present this honestly: *the sample (about 5 arrivals per day) is too sparse for weather to show; this is not evidence that weather doesn't matter.*
- **Snow/storm hours are a failure case.**
  - KDFW (62 hours): the plain hour-of-week mean (RMSE 0.270) beats the model (0.344–0.359) by 27–33%.
  - KATL (102 hours): the model's edge over the schedule baseline is not significant at h = 3, 6 and 9 (the CI reaches +0.003 to +0.007).
- **Caveat:** `arr_count` is 0 in about 82% of hours (a 0.5% BTS sample). This RMSE is not runway-capacity accuracy.

### 2. Delay > 45 min: P3 `gbm` beats the baselines overall, but its headline is inflated by leakage
| h | model PR-AUC | best baseline | global rate | gain (90% CI of PR-AUC diff) | Brier model / baseline |
|---|---|---|---|---|---|
| 3 | **0.343** | 0.199 | 0.136 | +72% ([+0.116, +0.167]) | 0.105 / 0.116 |
| 6 | **0.314** | 0.199 | 0.136 | +58% ([+0.091, +0.138]) | 0.108 / 0.116 |
| 9 | **0.281** | 0.199 | 0.136 | +41% ([+0.060, +0.105]) | 0.110 / 0.116 |
| 12 | **0.272** | 0.199 | 0.136 | +37% ([+0.050, +0.096]) | 0.110 / 0.116 |

The test set has 5,641 hub flights, 767 of them late. Cancelled and diverted flights are excluded, per P1's `label_45`.

- **⚠ Leak in P3's `hub_*` ops-state features.**
  - `pipeline/panel.py` assigns each arrival's and departure's final delay, its >45 flag and cancellations to its *scheduled* hour.
  - The window does end at or before t − h, but it still contains flights that hadn't landed or departed at t − h.
  - Measured on hub flights (`leakage_hub_state.csv`): **60.0%** of the >45-min arrivals counted in `hub_sev_rate_3h`, **85.8%** of the arrival-delay minutes in `hub_arr_delay_mean`, and **84.5%** of the departure-delay minutes in `hub_dep_delay_mean` (the #1 feature) are unknown at issue time.
  - The gain shrinking with horizon (+72% at 3 h down to +37% at 12 h) is consistent with this.
  - **Fix:** bucket by *actual* time and count only flights completed by t − h, or drop those features. Until then, the headline must be P3's leakage-free `gbm_p1_only`. P3 reports PR-AUC 0.277 for it (0.290 at h = 3, 0.265 at h = 12); we could not re-score it here because its predictions are not in the CSV.
- **Snow/storm at the hub end** (94 flights): the model does not beat the baseline at any h (−1% to −14%; every CI includes 0).

### 3. Leakage review status
| item | status |
|---|---|
| P1 features: `feature_obs_time ≤ t − h` | ✅ passes on 100% of rows in both parquets |
| Raw CSV weather | ❌ misaligned by 4–6 h (naive local-as-UTC key matches 95–100% of rows, correct UTC key 5–6%; `weather_alignment_check.csv`). Not used by anyone. |
| P2 AAR v2 | ✅ no outcome or `oracle_*` inputs; persistence uses only completed hours; neighbouring-hour schedule is known in advance; ablation run on validation only |
| P3 delay `gbm` | ❌ `hub_*` ops-state leak (above). `hub_sched_*` (schedule) is fine. |

### Open requests
- **P2:** add P10/P90 to `aar_preds.csv`, and rename `pred_lgbm` → `pred_arr_count` (CLAUDE.md schema) so P3's dashboard can show uncertainty.
- **P3:** fix or drop the leaky `hub_*` features, and add `gbm_p1_only` predictions to `delay_preds.csv`.
- **P1:** add `scikit-learn`, `lightgbm` and `pytest` to `requirements.txt`.
