# Notes on the metrics

- **Oracle gap uses the true event-hour weather.** `pr_auc_true_oracle` = same model with all lagged event-airport weather replaced by the organiser hourly file at floor(sched_time_utc) (the file P1's weather features come from; checked 100% equal on all 9 weather columns). P1's `oracle_*` (`pr_auc_p1_oracle`) is not event-time weather: it matches the event hour in ~5% of rows and the *local hour read as UTC* in ~44% (the misaligned upstream join in CLAUDE.md), i.e. ~4-6 h before the event.
- **Why the gap is small:** event-hour weather matters a lot per flight (thunderstorm coded: 37.7% severe vs 10.6%; >= 2.5 mm/h rain: 28.5% vs 10.4%) but it is rare (61 thunderstorm and ~1.4% heavy-rain events out of 25,231), so it barely moves an all-events PR-AUC. Test PR-AUC: no event-airport weather 0.207, lagged weather 0.219, event-hour weather 0.221.
- **Weather rule** cut-offs were chosen by max F1 on train; with a weak signal that lands on loose cut-offs (crosswind >= 7.5, tailwind >= 10.0 km/h) and flags about half of all events.
- **Generalisation** models drop `airport` (constant in training).
