# P3 delay risk: P(arrival delay > 45 min) = P1 `label_45`

Data: P1 `flight_features.parquet` (team CSV flights touching ATL/DFW). Test = scheduled departure 2025-01..07: 5,641 flights x 4 horizons, base rate 0.136. Cancelled/diverted (label NaN) excluded from scoring. One model per h. Threshold 0.150 = max F1 on validation (2024 H2; precision 0.30, recall 0.42). Rule baseline is binary (scored at 0.5).

`gbm_p1_only` = P1 features only; `gbm` adds hub schedule congestion + hub state (full BTS, as of t - h); `gbm_oracle` adds P1 same-hour `oracle_*` weather.

## Overall (all horizons)

|             |   pr_auc |   brier |   precision |   recall |   alert_rate |
|:------------|---------:|--------:|------------:|---------:|-------------:|
| base_rate   |    0.136 |   0.119 |     nan     |  nan     |      nan     |
| rule        |    0.148 |   0.202 |       0.205 |    0.168 |        0.111 |
| climatology |    0.189 |   0.116 |       0.207 |    0.284 |        0.186 |
| gbm_p1_only |    0.277 |   0.11  |       0.275 |    0.446 |        0.22  |
| gbm         |    0.303 |   0.108 |       0.288 |    0.483 |        0.229 |
| gbm_oracle  |    0.346 |   0.105 |       0.299 |    0.514 |        0.233 |

## PR-AUC by horizon

|    |   rule |   climatology |   gbm_p1_only |   gbm |   gbm_oracle |   base_rate |
|---:|-------:|--------------:|--------------:|------:|-------------:|------------:|
|  3 |  0.161 |         0.189 |         0.29  | 0.343 |        0.367 |       0.136 |
|  6 |  0.148 |         0.189 |         0.283 | 0.314 |        0.361 |       0.136 |
|  9 |  0.143 |         0.189 |         0.276 | 0.281 |        0.336 |       0.136 |
| 12 |  0.141 |         0.189 |         0.265 | 0.272 |        0.324 |       0.136 |

## By hub (gbm)

|      |   pr_auc |   brier |   precision |   recall |   alert_rate |   base_rate |   climatology_pr_auc |
|:-----|---------:|--------:|------------:|---------:|-------------:|------------:|---------------------:|
| KATL |    0.29  |   0.098 |       0.301 |    0.41  |        0.164 |       0.121 |                0.176 |
| KDFW |    0.319 |   0.12  |       0.279 |    0.549 |        0.301 |       0.153 |                0.198 |

## Threshold trade-off (gbm, test)

|      |   pr_auc |   brier |   precision |   recall |   alert_rate |
|-----:|---------:|--------:|------------:|---------:|-------------:|
| 0.1  |    0.303 |   0.108 |       0.231 |    0.663 |        0.391 |
| 0.15 |    0.303 |   0.108 |       0.287 |    0.483 |        0.229 |
| 0.2  |    0.303 |   0.108 |       0.332 |    0.354 |        0.145 |
| 0.25 |    0.303 |   0.108 |       0.376 |    0.248 |        0.09  |
| 0.3  |    0.303 |   0.108 |       0.418 |    0.162 |        0.053 |
| 0.4  |    0.303 |   0.108 |       0.494 |    0.058 |        0.016 |
| 0.5  |    0.303 |   0.108 |       0.658 |    0.024 |        0.005 |

## Top features (gain share)

|                    |   gain_share (h=6) |
|:-------------------|-------------------:|
| hub_dep_delay_mean |              0.083 |
| carrier            |              0.076 |
| dep_hour_local     |              0.067 |
| origin_pres_hpa    |              0.055 |
| origin_dwpt_c      |              0.048 |
| month_local        |              0.046 |
| origin_temp_c      |              0.046 |
| hub_sev_rate_3h    |              0.045 |
| origin_coco        |              0.043 |
| dest_dwpt_c        |              0.037 |
| arr_hour_local     |              0.032 |
| hub_arr_delay_mean |              0.032 |
| hub_cap_ratio_3h   |              0.029 |
| dest_coco          |              0.023 |
| origin_headwind_kt |              0.021 |
