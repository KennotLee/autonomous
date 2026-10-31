# Decision thresholds (per airport)

Max F1 on each airport's validation slice (2024-07..12, never test): a missed severe delay and a false alert weigh the same, judged within each airport.

| airport | threshold | val precision | val recall | val alert rate |
|---|---|---|---|---|
| KATL | 0.098 | 0.195 | 0.456 | 0.198 |
| KDFW | 0.123 | 0.219 | 0.454 | 0.226 |

Previous single threshold (`gbm_global_thr` in metrics.csv): 0.119 (val precision 0.213, recall 0.403). Airports not in the table fall back to it.

Weather rule baseline: ts_flag OR crosswind >= 7.5 km/h OR tailwind >= 10.0 km/h (cut-offs = max F1 on train 2023-01..2024-06, F1 0.190).
