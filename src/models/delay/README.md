# P3 · Delay risk model

Per-flight P(arrival delay > 45 min) for ATL/DFW flights, issued h ∈ {3, 6, 9, 12} h before scheduled arrival.

```
python -m pipeline.download --skip-bts                 # hub weather (meteostat>=2)
python -m pipeline.download                            # BTS months -> data/raw/bts/*.parquet
python -m src.models.delay.weather                     # weather for every airport in the network (~10 min)
python -m src.models.delay.train                       # ~4 min: train, evaluate, write outputs
```

Put the team CSV at `data/raw/Flight_Data_With_Weather_Final.csv`. Its flights become `delay_preds.csv`.

## Data
- **Flights:** full BTS on-time files for every ATL/DFW flight, 2023-01..2025-07. The team CSV is a ~0.5% sample, too thin to train on or to measure hub congestion.
- **Weather:** Meteostat hourly at **both ends** (210 of 213 airports; ACY, AVL and SGU have no station, so their weather features are blank). The CSV's own weather columns are not used: they are joined at the flight's own hour, misaligned by the UTC offset (see CLAUDE.md).
- **Missing fields:** dew point and visibility are missing at the hubs, and gusts are missing ~97% of hours.

## Leakage rule
Forecasts are issued at t0 = scheduled arrival (UTC) − h. Weather and hub state come from the hour that ends at or before t0 (`feature_obs_time` = floor(t0) − 1 h). The schedule, carrier, distance and calendar are known in advance. `now_*` columns hold weather at the flight's own hour and are used only by the oracle model.

## Features
- Origin and destination weather: precipitation, wind, crosswind/tailwind (hubs only, against the configured runway heading), thunderstorm/snow/fog codes, pressure change, 3/6/24 h windows.
- Hub scheduled arrivals/departures in the flight's hour.
- Hub state over the 3 h before issue: arrival capacity ratio, severe-delay share, cancellations, mean arrival and departure delay.
- Carrier, hour, weekday, month, distance, block time.

## Split and threshold
- **Train:** 2023-01..2024-06.
- **Validation:** 2024-07..12, used for early stopping and to choose the threshold.
- **Test:** 2025-01..07.
- Seeded random sample of 300k / 100k / 200k flights.
- **Threshold:** maximum F1 on validation, which weights a missed delay and a false alert equally.

## Outputs
- `outputs/predictions/delay_preds.csv`: `flight_id, h, risk_prob, high_risk, top_drivers` plus context columns. `flight_id` = `date_carrier_flightnum_origin_dest`. `top_drivers` lists the three TreeSHAP contributions that raise risk, in log-odds. `split` marks train-period rows as in-sample.
- `outputs/metrics/delay_metrics.md`, `delay_calibration.png`, `delay_pr_curve.png`.
- `outputs/metrics/delay_failure_case.md` + `.png`: worst missed day, auto-selected. Its narrative comes from `failure_notes.md`.

Code reuses `pipeline/` (panel building and weather feature functions).
