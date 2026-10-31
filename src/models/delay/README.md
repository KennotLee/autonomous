# P3 · Delay risk model

Per-flight P(`label_45`) = P(arrival delay > 45 min), using P1's label and features, h ∈ {3, 6, 9, 12}.

```
# 1. P1's features (branch p1)
git show origin/p1:data/processed/flight_features.parquet > data/processed/flight_features.parquet
# 2. Optional, for the hub congestion/state features: full BTS months + hub weather
python -m pipeline.download
# 3. Train, evaluate, write outputs (~1 min)
python -m src.models.delay.train
```

## Inputs
- **P1 `flight_features.parquet`:** one row per flight × h. The forecast time t = scheduled departure (UTC), and all weather is as of t − h. `label_45` is blank for cancelled/diverted flights; those are not scored but still get a prediction. `oracle_*` columns are used only by the oracle model.
- **P3 additions:**
  - **Thunderstorm / heavy-precipitation flags** at each end, derived from P1's `coco`.
  - **Hub schedule congestion:** scheduled arrivals and departures at ATL/DFW in the hour the flight uses the hub. It comes from the full BTS schedule, which is known in advance; the team CSV is a 0.5% sample and can't measure congestion.
  - **Hub state** in the last complete hour before t − h: arrival capacity ratio, severe-delay share, cancellation rate, mean arrival and departure delay.
- Without the BTS files, the script skips the P3 additions and trains on P1 features only.

## Model and evaluation
- **Model:** one LightGBM per horizon.
- **Split** (by scheduled departure): train < 2024-07, validation 2024-07..12 (early stopping and threshold choice), test 2025-01..07.
- **Baselines:** base rate, a weather rule (thunderstorm/heavy-rain code, gust > 25 kt or wind > 20 kt at either end), and climatology by hub × direction × arrival hour.
- **Ablation:** `gbm_p1_only` uses P1's features only.
- **Threshold:** maximum F1 on validation. The trade-off table in the metrics shows other operating points.

## Outputs
- `outputs/predictions/delay_preds.csv`: contract columns `flight_id, sched_time_utc, h, p_delay45`, plus `high_risk`, `top_drivers` (the three TreeSHAP log-odds contributions that push risk up), `label_45` and `split`. Rows from train/validation are in-sample.
- `outputs/metrics/delay_metrics.md`, `delay_calibration.png`, `delay_pr_curve.png`.
- `outputs/metrics/delay_failure_case.md` + `.png`: worst missed day, auto-selected. The narrative comes from `failure_notes.md`.
