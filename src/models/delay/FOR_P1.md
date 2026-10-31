# P3 → P1: ready to merge `p3` (handover, 2026-10-31)

`p3` is ready for you to merge (p1 → p3 → main). The model trains on your `event_features.parquet` only.

## 1. When merging
- **Keep `p3`'s side of `src/models/delay/`.** `p3` deletes the old `features.py`, `failure.py`, `failure_notes.md`
  and `airports.py` (they read `flight_features.parquet` and import `pipeline/`); `origin/p1` still has the old copies.
- **`config/airports.yaml`:** keep yours. `p3`'s copy is older and has no `weather` section; until the merge,
  P3's code reads yours from `origin/p1` (`src/models/delay/train.py: config()`).
- **Data needed to run:** `data/processed/event_features.parquet` (yours) and the organiser hourly file at
  `weather.hourly_file` (`data/raw/challenge5_hourly_weather_ATL_DFW.csv`, used by `evaluate.py` for the true oracle).
- **Packages:** `lightgbm`, `scikit-learn`, `matplotlib`, `pyyaml`, `tabulate` (already added to `requirements.txt` on main).

## 2. Please update CLAUDE.md
- **Prediction contract:** replace `flight_id, sched_time_utc, h, p_delay45` with
  `event_id, airport, event_type, sched_time_utc, h, risk_prob, high_risk, top_drivers`
  (details: `outputs/predictions/SCHEMA.md`).
- **`oracle_*` timing (finding):** your `oracle_*` (flights-CSV weather) matches the organiser hourly file at the
  event's true UTC hour in ~5% of rows, but at the *local hour read as UTC* in ~44%: it is the misaligned upstream
  join, about 4–6 h before the event, not same-time weather. P3's oracle gap therefore uses the hourly file at
  floor(sched_time_utc) instead (`outputs/metrics/notes.md`). Your lagged weather matches the hourly file 100%
  on all 9 columns (checked).

## 3. Proposal: `pipeline/` is unused (fix 9, your call as integrator)
Nothing in `src/` imports it on `p3`. It is only used by the old AAR/BTS run, which the organiser update dropped:

| file | used by |
|---|---|
| `pipeline/` (`config.py`, `download.py`, `evaluate.py`, `features.py`, `panel.py`) | `run_pipeline.py`, `tests/test_pipeline_leakage.py` |
| `run_pipeline.py` | nothing (writes `reports/metrics.md`) |
| `tests/test_pipeline_leakage.py` | tests `pipeline/` only |
| `reports/metrics.md` | output of `run_pipeline.py` (old AAR / hourly severe-delay metrics) |

**Proposal:** delete all four in the merge (I can't: they're outside P3's folders). If you'd rather keep them, add
a line at the top of `run_pipeline.py` / `pipeline/__init__.py` saying "unused since the organiser update".

Then please update:
- **SOURCES.md #2** (BTS full months via `pipeline/download.py`): remove, or mark "no longer used".
- **ASSUMPTIONS.md #21** (P3 hub state from full BTS): remove; the model has no hub-state features now.
- **ASSUMPTIONS.md #22** (P3 rule baseline and climatology): replace with "weather rule = `ts_flag` OR crosswind
  ≥ c OR tailwind ≥ t, with c, t = max F1 on train (currently 7.5 and 10.0 km/h); no climatology baseline".
- **ASSUMPTIONS.md #23** (`pipeline/config.py` hardcodes airports): remove with `pipeline/`. P3's code reads
  airports from `config/airports.yaml`.

## 4. Where the results are
`outputs/metrics/notes.md` (demo notes first), `metrics.csv`, `metrics_ci.csv`, `threshold.md`, `failure_cases.md`,
`calibration.png`, `oracle_gap.csv`, `generalisation.csv`, `cancellations.csv`.
