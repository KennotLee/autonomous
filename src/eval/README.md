P4 Evaluation. Reads outputs/predictions/*.csv + data/processed/*.parquet, writes outputs/metrics/. Prediction schemas:
- outputs/predictions/aar_preds.csv — columns: icao (str, e.g. KATL), target_hour_utc (ISO-8601 UTC timestamp, hour start), h (int hours ahead, one of 3,6,9,12), pred_arr_count (float, predicted arrivals in that UTC hour)
- outputs/predictions/delay_preds.csv — columns: flight_id (str, matches data/processed/flight_features.parquet), sched_time_utc (ISO-8601 UTC, scheduled departure), h (int, one of 3,6,9,12), p_delay45 (float 0–1, probability arrival delay > 45 min)
