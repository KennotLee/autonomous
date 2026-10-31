P2 Models.

Inputs (team contract, column names fixed; full details in CLAUDE.md):
- data/processed/hourly_features.parquet: icao, target_hour_utc, h, feature_obs_time, [weather + wind component + rolling + calendar features], arr_count
- data/processed/flight_features.parquet: flight_id, origin_icao, dest_icao, sched_time_utc, h, feature_obs_time, [origin_* and dest_* features], delay_min, label_45

h ∈ {3,6,9,12}; all non-oracle features observed at or before target time − h. Columns prefixed `oracle_` are same-hour weather for oracle comparison only — never train production models on them.

Outputs: outputs/predictions/aar_preds.csv and outputs/predictions/delay_preds.csv — schema "TBD, pending P1/team agreement; see CLAUDE.md".
