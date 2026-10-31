# Notes for the demo

- **2025 is a worse year.** Severe-delay rate: train 10.1%, validation 9.6%, test 13.2%; cancellations 1.1% -> 1.3% -> 2.1%. The model under-predicts in 2025: events scored 0-0.1 (mean 0.067) were severely delayed 9.5% of the time; 0.1-0.2 (mean 0.137) 16.1% (see calibration.png). Read probabilities as slightly low.
- **Weather explains only a small part of severe delays here.** Model vs no event-airport weather: +0.012 PR-AUC (95% CI -0.004 to +0.029). Bad weather does matter when it happens (see the wx_* slices in metrics.csv), but it is rare. The worst misses (failure_cases.md) are calm-weather flights (carrier OO); their cause is not in our data (possibly late inbound aircraft or airline operations, which we cannot observe).
- **Arrivals are weaker than departures** (PR-AUC 0.177 vs 0.241): the hourly file only covers ATL/DFW, so the weather at an arrival's origin is unknown.
- **Cancellations are not scored** (organiser: label NaN), but they get a forecast. Mean risk_prob in 2025: cancelled 0.135 (121 events) vs flown 0.105; cancellations look like high-risk events (cancellations.csv).
- **Alerts:** about 4 of 5 alerts are false at the chosen thresholds (failure_cases.md, alert workload).

# Notes on the metrics

- **Oracle gap uses the true event-hour weather.** `pr_auc_true_oracle` = same model with all lagged event-airport weather replaced by the organiser hourly file at floor(sched_time_utc) (the file P1's weather features come from; checked 100% equal on all 9 weather columns). P1's `oracle_*` (`pr_auc_p1_oracle`) is not event-time weather: it matches the event hour in ~5% of rows and the *local hour read as UTC* in ~44% (the misaligned upstream join in CLAUDE.md), i.e. ~4-6 h before the event.
- **Why the gap is small:** event-hour weather matters a lot per flight (thunderstorm coded: 37.7% severe vs 10.6%; >= 2.5 mm/h rain: 28.5% vs 10.4%) but it is rare (61 thunderstorm and ~1.4% heavy-rain events out of 25,231), so it barely moves an all-events PR-AUC. Test PR-AUC: no event-airport weather 0.207, lagged weather 0.219, event-hour weather 0.221.
- **Weather rule** cut-offs were chosen by max F1 on train; with a weak signal that lands on loose cut-offs (crosswind >= 7.5, tailwind >= 10.0 km/h) and flags about half of all events.
- **Generalisation** models drop `airport` (constant in training).
