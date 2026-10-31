# delay_preds.csv: schema (P3, proposed final, 2026-10-31)

**P1:** please replace the old contract in CLAUDE.md (`flight_id, sched_time_utc, h, p_delay45`) with this.
**P4:** please read these columns for the dashboard.
Reply on the `p3` branch or in chat if anything needs to change.

One row per (event_id, h) for every 2025 row of P1's `event_features.parquet` (join coverage with P1 asserted 100%).

| column | type | meaning |
|---|---|---|
| event_id | str | P1's `event_id` (`flight_key + "_" + event_type`), join key with `event_features` |
| airport | str | event hub, ICAO (KATL / KDFW) |
| event_type | str | `dep` (judged on DEP_DELAY) or `arr` (judged on ARR_DELAY) |
| sched_time_utc | datetime, UTC | scheduled time of the event at the hub (t) |
| h | int | forecast horizon in hours, 3 / 6 / 9 / 12 (forecast issued at t − h) |
| risk_prob | float 0..1 | predicted P(delay > 45 min) |
| high_risk | 0/1 | `risk_prob >= threshold`; threshold chosen on validation only (see `outputs/metrics/threshold.md`) |
| top_drivers | str | up to 3 features pushing risk up, `feature=value` joined by `; ` (LightGBM pred_contrib) |

Notes
- Events with no label (cancelled, diverted arrivals) are included: the forecast is made before the outcome is known.
  Score metrics only on rows where P1's `label_45` is not NaN.
- Rates, not counts, when comparing airports (organiser rule).
