# 🔍 Review: P3 delay-risk model (branch `p3`)

- **Reviewed:** `origin/p3` @ `9616f56` (2026-10-31, ~12:55 SGT) by P1
- **Scope:** `src/models/delay/` (features, train, failure) and the parts of `pipeline/` it imports
  (`panel.py`, `features.py`). Code was read, and the leak was measured on P1's `flight_features.parquet`.
  The model was **not** retrained here (that needs the full BTS download, 31 monthly files).
- **Verdict:** well-built model, but the **headline `gbm` score includes future information**.
  `gbm_p1_only` (PR-AUC 0.277 vs base rate 0.136) is clean and can be used now.

## 🚨 Must fix: leakage in the hub-state features

**Where:** `pipeline/panel.py` → `build_panel()` aggregates `n_sev`, `arr_delay_sum` and `dep_delay_sum` by each
flight's **scheduled** hour (`h_sched`). `pipeline/features.py` → `ops_state()` turns them into
`sev_rate_3h`, `arr_delay_mean`, `dep_delay_mean`. `src/models/delay/features.py` → `add_hub_features()`
then reads hour `floor(t − h) − 1` as `hub_sev_rate_3h`, `hub_arr_delay_mean`, `hub_dep_delay_mean`.

**Why it leaks:** the bucket closes by t − h, but a flight *scheduled* in that hour that ends up 3 h late has
not landed or departed by t − h, so its delay is not known at issue time.

**How big (measured on P1 hub arrivals, 2023-01..2025-07):**
- 16.8% of arrivals had **not** arrived when their scheduled-hour bucket closed.
- Among arrivals more than 45 min late, **97.7%** had not arrived at bucket close, and those flights carry **86%** of all positive delay minutes.
- ⇒ `hub_sev_rate_3h`, `hub_arr_delay_mean` and `hub_dep_delay_mean` are mostly built from future information.

**Supporting evidence in P3's own metrics (`outputs/metrics/delay_metrics.md`):**
- `hub_dep_delay_mean` is the #1 feature (gain share 0.083 at h = 6).
- The gain from hub features shrinks with horizon, which is the pattern this leak produces:

| h | gbm_p1_only | gbm (with hub features) | gain |
|---|---|---|---|
| 3 | 0.290 | 0.343 | +0.053 |
| 6 | 0.283 | 0.314 | +0.031 |
| 9 | 0.276 | 0.281 | +0.005 |
| 12 | 0.265 | 0.272 | +0.007 |

**Fix (a few lines):** in `build_panel()`, aggregate delay outcomes by the **actual** hour (`h_act`, already
computed): arrival delays and the >45 min share by actual arrival hour, departure delays by actual departure
hour. A delay is known once the flight has landed or left. Then retrain.

**Until fixed:** write `delay_preds.csv` from `gbm_p1_only` (P1 features only, covered by P1's leakage test).

**Not affected (clean):** `hub_cap_ratio_3h` (uses actual arrival hours) and `hub_sched_arr` / `hub_sched_dep`
(the schedule is known in advance).

## ✅ What's good

- One LightGBM per horizon; split by date: train before 2024-07, validation 2024-07..12, test 2025-01..07.
  The threshold is chosen on validation only.
- Outcome columns and `oracle_*` are excluded; builds on P1's leakage-tested features.
- Strong baselines (base rate, weather rule, shrunk climatology) and a P1-only ablation.
- Good extras: oracle comparison, calibration and PR plots, threshold trade-off table, SHAP top drivers per
  flight, auto-selected failure case.
- `delay_preds.csv` follows the contract (`flight_id, sched_time_utc, h, p_delay45`) plus extra columns.

## ⚠️ Smaller issues

1. **`delay_preds.csv` includes train/validation (in-sample) rows**, marked in `split`. P4 must score only
   `split == "test"`.
2. **Merge conflicts with `p1`:** both branches added their own `tests/test_leakage.py`. `p3` also has a
   root-level `run_pipeline.py` and an older parallel `pipeline/` folder, so the team could end up with two
   pipelines. Agree which files are canonical before merging to `main`.
3. **Possible version mismatch:** `pipeline/download.py` appears to use meteostat's newer API (`ms.hourly`),
   while the shared `requirements.txt` pins `meteostat==1.7.6`. Check before merging.

## 🔬 Suggested ablations (by priority)

| # | Ablation | What it answers |
|---|---|---|
| 1 | **Retrain with hub features grouped by actual hour** | The real value of hub state. **Must-do before the demo.** |
| 2 | **Remove all weather** from `gbm_p1_only` | How much the honest model gains from weather |
| 3 | **Oracle gap** (already: 0.303 → 0.346) | The ceiling for better weather forecasts; repeat it against the fixed model |
| 4 | **Remove one group at a time:** schedule congestion / hub state / origin weather / destination weather / carrier and calendar | Which signals matter; origin vs destination weather |
| 5 | **One model for all h with `h` as an input, vs one per h** | Each per-h model has only ~11k training flights; pooling gives 4× the rows |
| 6 | **Seeds and bootstrap by day** | Whether gaps like 0.277 vs 0.303 are real or noise |
| 7 | **Calibration** (isotonic, fitted on validation) | Better probabilities and Brier score for the dashboard |
| 8 | **Cost-based threshold** instead of max F1 | An operating point that fits how alerts are used |
| 9 | **Count cancelled/diverted flights as positives** | Label policy; P3's panel already treats them as "severe" |
| 10 | **Previous leg of the same aircraft** (tail number), only legs completed by t − h | Late aircraft are the biggest real cause of delays (needs full BTS and careful timing) |

## ▶️ Next step for P3

Apply the `h_act` fix and retrain (~15 min) before the 14:30 predictions checkpoint; meanwhile ship
`gbm_p1_only` predictions.
