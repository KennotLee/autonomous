P2 Models.

Inputs (team contract, column names fixed; full details in CLAUDE.md):
- data/processed/hourly_features.parquet: icao, target_hour_utc, h, feature_obs_time, [weather + wind component + rolling + calendar features], arr_count
- data/processed/flight_features.parquet: flight_id, origin_icao, dest_icao, sched_time_utc, h, feature_obs_time, [origin_* and dest_* features], delay_min, label_45

h ∈ {3,6,9,12}; all non-oracle features observed at or before target time − h. Columns prefixed `oracle_` are same-hour weather for oracle comparison only — never train production models on them.

Outputs: outputs/predictions/aar_preds.csv and outputs/predictions/delay_preds.csv — schema "TBD, pending P1/team agreement; see CLAUDE.md".

---

## P2 — AAR regression (`arr_count`)

Predicts realised arrivals per UTC hour (AAR proxy) at KATL/KDFW for h ∈ {3, 6, 9, 12}.

### Commands (from repo root)

```
py -3.12 -m venv .venv                                   # Windows; macOS/Linux: python3 -m venv .venv
.venv\Scripts\pip install -r requirements.txt -r src/models/requirements-aar.txt
.venv\Scripts\python -m src.models.train_aar --validate-only   # schema + leakage checks only
.venv\Scripts\python -m src.models.train_aar                   # train, evaluate, save
.venv\Scripts\python -m pytest src/models/tests -q             # tests (synthetic data)
```

Training refuses to run (no artifacts written) if `data/processed/hourly_features.parquet` is
missing or fails validation.

### Files

| file | role |
|---|---|
| `aar_config.py` | paths, horizons, split dates, leakage denylist, LightGBM params |
| `aar_data.py` | load, validate, chronological split |
| `aar_features.py` | feature selection with leakage guards, local calendar features, preprocessor |
| `aar_baselines.py` | persistence, historical average |
| `aar_metrics.py` | RMSE / MAE by airport, horizon, airport × horizon |
| `train_aar.py` | CLI entry point |
| `predict_aar.py` | `predict_aar()` for the dashboard |

### Outputs (`outputs/predictions/`)

- `aar_preds.csv`: `split, icao, target_hour_utc, h, y_true, pred_lgbm, pred_persistence, pred_hist_avg` (val + test rows).
- `aar_metrics.csv`: `split, model, icao, h, n, rmse, mae`; `ALL` marks an aggregated level.
- `aar_model/model.txt` (LightGBM), `preprocessor.json` (feature list, column order, categories,
  dropped columns with reasons), `metadata.json` (params, best iteration, split sizes, feature
  importance, validation report). Text/JSON only, no pickles.

### Dashboard API

```python
from src.models.predict_aar import predict_aar
predict_aar(icao="KATL", h=6, start="2025-03-01", end="2025-03-07")   # reads the parquet
predict_aar(features_df)                                              # or pass rows yourself
# -> DataFrame[icao, target_hour_utc, h, aar_pred]; aar_pred is arrivals/hour, >= 0
```

The model is loaded once and cached. Inputs must contain every column in
`preprocessor.json["feature_columns"]`; a missing column raises `KeyError`.

### Method

- **Split (by `target_hour_utc`, UTC, never shuffled):** train < 2024-07-01 ≤ val < 2025-01-01 ≤ test.
  LightGBM early-stops on val; it is then refit on train + val with the best iteration count, and the
  historical average is refit on train + val too, before scoring 2025. `--no-refit` scores test with
  the train-only fits.
- **Persistence:** arrivals in the last UTC hour that had fully ended by the prediction time `t − h`
  (`[t − h − 1h, t − h)`; P1's `feature_obs_time` is the weather stamp and may lag). Taken from P1's column once `PERSISTENCE_COLUMN` is set;
  until then derived from past `arr_count` rows of the panel. Absent hours give NaN, not 0.
- **Historical average:** mean `arr_count` by (icao, local hour, local day of week), each hour
  counted once across horizons, fit on training rows only. Unseen groups fall back to
  (icao, local hour), then icao.
- **LightGBM:** L2 regression, defaults in `aar_config.LGBM_PARAMS`, `seed=42`, deterministic.
  Inputs: every numeric P1 column that passes the leakage guards (weather, wind components,
  scheduled traffic, rolling features), `icao` (categorical), `h`, and local hour/day of week/month/weekend
  derived from `target_hour_utc`. Predictions are clipped at 0.

### Leakage safeguards

- Validation fails if any row has `feature_obs_time > target_hour_utc − h`, duplicate keys, off-hour
  timestamps, unknown horizons or airports.
- Never inputs: `arr_count`, `capacity_ratio`, `mean_arr_delay`, `share_delayed_45`, `cancellations`, `diversions`,
  any `oracle_*` or `arr_count*` column, and (conservatively) any column whose name contains
  `delay`, `cancel`, `divert`, `divers` or `capacity`. If P1 confirms such a column is observed before
  `feature_obs_time`, add it to `ALLOWED_OVERRIDES`. Every dropped column and the reason are saved
  in `preprocessor.json`.
- Calendar features come only from the target hour itself, which is known when the forecast is issued.
- Historical averages and the preprocessor's categories are fit on training rows only.
- Tests check that scrambling the target and oracle columns does not change predictions.

### Assumptions and open dependencies (for P1)

- Schema is the README contract; P1's processed-schema section in CLAUDE.md is still pending.
  Feature names beyond the keys are not assumed.
- `target_hour_utc` labels the hour starting at that time; timestamps are UTC (naive is read as UTC).
- Rows with null `arr_count` are dropped, never imputed.
- **Persistence column:** we need P1's "arrivals in the last completed hour before
  `feature_obs_time`" (name TBD → `PERSISTENCE_COLUMN`). The derived fallback is correct only if the
  panel lists every hour, including zero-arrival hours; validation warns when hours are missing.
- `lightgbm` and `pytest` need adding to `requirements.txt` (P1 owns it).

### Limitations

- The raw sample is sparse (~5 arrivals/day/airport, `arr_count` mostly 0/1, see CLAUDE.md), so
  `arr_count` is far below real AAR and RMSE/MAE are on that tiny scale. The persistence and
  historical-average baselines are the honest bar to beat.
- Val metrics are slightly optimistic because val also picks the iteration count.
- L2 on counts is a default, not a tuned choice; `objective="poisson"` is the obvious next experiment.
- Weather quality depends on P1 fixing the misaligned upstream weather join (CLAUDE.md).
