# P3 · Delay risk model

P(`label_45`) = P(delay > 45 min) per ATL/DFW event (departure judged on `DEP_DELAY`, arrival on `ARR_DELAY`), for horizons h ∈ {3, 6, 9, 12}, from P1's `data/processed/event_features.parquet` (branch `p1`). If the file isn't present locally, `train.py` reads it from `origin/p1` via git.

```
python -m src.models.delay.train      # model + outputs/predictions/delay_preds.csv
python -m src.models.delay.evaluate   # outputs/metrics/*
```

## Features
Every column except keys (`event_id`, `flight_key`, `sched_time_local`, `sched_time_utc`, `feature_obs_time`, `other_feature_obs_time`), targets (`delay_min`, `label_45`, `cancelled`, `diverted`) and `oracle_*` columns. An assert enforces this.

- Categorical: `event_type`, `airport`, `other_airport`, `carrier`.
- `h` is a feature (one model for all horizons).

## Split and training
- Split by `sched_time_utc`: train 2023-01..2024-06, validation 2024-07..12 (early stopping and threshold), test 2025 (data ends Jul 2025).
- Rows with `label_45` NaN (cancelled, diverted arrivals) are not trained or evaluated. They still get a row in `delay_preds.csv`.
- No class reweighting, which keeps probabilities calibrated for the Brier score and reliability curve. Imbalance is handled by the decision threshold, set to max F1 on validation.

## Outputs
- `outputs/predictions/delay_preds.csv`: columns `event_id`, `h`, `risk_prob`, `top_drivers`, for every 2025 row of `event_features` (join coverage asserted 100%).
  - `top_drivers`: up to 3 features pushing risk up, from LightGBM `pred_contrib`, formatted `feature=value` and joined by `; `.
- `src/models/delay/artifacts/`: `gbm.txt` (model) and `meta.json` (threshold and feature list).
- `outputs/metrics/`: `metrics.csv`, `threshold.md`, `calibration.png`, `oracle_gap.csv`, `generalisation.csv`, `failure_cases.md`.

## Baselines
- **Base rate:** the training positive rate.
- **Weather rule:** `ts_flag` OR crosswind >= c OR tailwind >= t, with c and t chosen on train only (max F1).
