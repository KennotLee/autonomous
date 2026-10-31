# Decision threshold: 0.119

Max F1 on the validation slice (2024-07..12, never test): a missed severe delay and a false alert weigh the same. Validation precision 0.213, recall 0.403.

Weather rule baseline: ts_flag OR crosswind >= 7.5 km/h OR tailwind >= 10.0 km/h (cut-offs = max F1 on train 2023-01..2024-06, F1 0.190).
