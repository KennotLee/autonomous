# Notes on the metrics

- **Oracle gap is not a leakage ceiling.** P1's `oracle_*` = flights-CSV weather. Checked against the organiser hourly file: it matches the true event hour (UTC) in ~5% of rows but the *local hour read as UTC* in ~44% (the misaligned upstream join in CLAUDE.md), i.e. it is weather from ~4-6 h before the event. So `oracle_gap.csv` compares lagged weather with other lagged weather; the small gap does not show what true same-time weather would fake.
- **Weather rule** cut-offs were chosen by max F1 on train; with a weak signal that lands on loose cut-offs (crosswind >= 7.5, tailwind >= 10.0 km/h) and flags about half of all events.
- **Generalisation** models drop `airport` (constant in training).
