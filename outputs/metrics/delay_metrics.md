# P3 delay risk: P(arrival delay > 45 min) = P1 `label_45`

Data: P1 `flight_features.parquet` (team CSV flights touching ATL/DFW). Test = scheduled departure 2025-01..07: 5,641 flights x 4 horizons, base rate 0.136. Cancelled/diverted (label NaN) excluded from scoring. One model per h. Threshold 0.148 = max F1 on validation (2024 H2; precision 0.30, recall 0.42). Rule baseline is binary (scored at 0.5).

`gbm_p1_only` = P1 features only; `gbm` adds hub schedule congestion + hub state (full BTS, as of t - h); `gbm_oracle` adds P1 same-hour `oracle_*` weather.

## Overall (all horizons)

|             |   pr_auc |   brier |   precision |   recall |   alert_rate |
|:------------|---------:|--------:|------------:|---------:|-------------:|
| base_rate   |    0.136 |   0.119 |     nan     |  nan     |      nan     |
| rule        |    0.148 |   0.202 |       0.205 |    0.168 |        0.111 |
| climatology |    0.189 |   0.116 |       0.204 |    0.286 |        0.19  |
| gbm_p1_only |    0.277 |   0.11  |       0.273 |    0.453 |        0.226 |
| gbm         |    0.299 |   0.109 |       0.279 |    0.478 |        0.233 |
| gbm_oracle  |    0.338 |   0.106 |       0.296 |    0.513 |        0.235 |

## PR-AUC by horizon

|    |   rule |   climatology |   gbm_p1_only |   gbm |   gbm_oracle |   base_rate |
|---:|-------:|--------------:|--------------:|------:|-------------:|------------:|
|  3 |  0.161 |         0.189 |         0.29  | 0.333 |        0.361 |       0.136 |
|  6 |  0.148 |         0.189 |         0.283 | 0.312 |        0.348 |       0.136 |
|  9 |  0.143 |         0.189 |         0.276 | 0.279 |        0.327 |       0.136 |
| 12 |  0.141 |         0.189 |         0.265 | 0.272 |        0.32  |       0.136 |

## By hub (gbm)

|      |   pr_auc |   brier |   precision |   recall |   alert_rate |   base_rate |   climatology_pr_auc |
|:-----|---------:|--------:|------------:|---------:|-------------:|------------:|---------------------:|
| KATL |    0.291 |   0.098 |       0.292 |    0.411 |        0.17  |       0.121 |                0.176 |
| KDFW |    0.309 |   0.121 |       0.271 |    0.537 |        0.303 |       0.153 |                0.198 |

## Threshold trade-off (gbm, test)

|       |   pr_auc |   brier |   precision |   recall |   alert_rate |
|------:|---------:|--------:|------------:|---------:|-------------:|
| 0.1   |    0.299 |   0.109 |       0.229 |    0.659 |        0.391 |
| 0.148 |    0.299 |   0.109 |       0.279 |    0.478 |        0.233 |
| 0.15  |    0.299 |   0.109 |       0.282 |    0.474 |        0.229 |
| 0.2   |    0.299 |   0.109 |       0.33  |    0.341 |        0.141 |
| 0.25  |    0.299 |   0.109 |       0.371 |    0.232 |        0.085 |
| 0.3   |    0.299 |   0.109 |       0.42  |    0.15  |        0.049 |
| 0.4   |    0.299 |   0.109 |       0.53  |    0.057 |        0.015 |
| 0.5   |    0.299 |   0.109 |       0.689 |    0.017 |        0.003 |

## Top features (gain share)

|                     |   gain_share (h=6) |
|:--------------------|-------------------:|
| carrier             |              0.08  |
| dep_hour_local      |              0.08  |
| origin_dwpt_c       |              0.056 |
| month_local         |              0.051 |
| origin_pres_hpa     |              0.048 |
| hub_cap_ratio_3h    |              0.043 |
| hub_arr_delay_mean  |              0.042 |
| origin_temp_c       |              0.041 |
| origin_coco         |              0.041 |
| dest_dwpt_c         |              0.04  |
| hub_sev_rate_3h     |              0.032 |
| arr_hour_local      |              0.03  |
| hub_dep_delay_mean  |              0.028 |
| origin_headwind_kt  |              0.026 |
| origin_crosswind_kt |              0.024 |
