# P3 delay risk: P(arrival delay > 45 min)

Test = sampled BTS ATL/DFW flights with scheduled arrival 2025-01..07 (204,119 flights x 4 horizons). Cancelled/diverted flights excluded (no arrival delay). Decision threshold 0.137 = max F1 on validation (2024 H2; precision 0.27, recall 0.42). Rule baseline is binary (scored at 0.5).

## Overall (all horizons)

|             |   pr_auc |   brier |   precision |   recall |   alert_rate |
|:------------|---------:|--------:|------------:|---------:|-------------:|
| base_rate   |    0.131 |   0.115 |     nan     |  nan     |      nan     |
| rule        |    0.147 |   0.188 |       0.222 |    0.176 |        0.104 |
| climatology |    0.203 |   0.111 |       0.21  |    0.46  |        0.287 |
| gbm         |    0.321 |   0.103 |       0.273 |    0.537 |        0.257 |
| gbm_oracle  |    0.359 |   0.1   |       0.292 |    0.578 |        0.259 |

## PR-AUC by horizon

|    |   climatology_pr_auc |   rule_pr_auc |   gbm_pr_auc |   gbm_oracle_pr_auc |   base_rate |
|---:|---------------------:|--------------:|-------------:|--------------------:|------------:|
|  3 |                0.203 |         0.168 |        0.381 |               0.4   |       0.131 |
|  6 |                0.203 |         0.15  |        0.33  |               0.367 |       0.131 |
|  9 |                0.203 |         0.14  |        0.292 |               0.34  |       0.131 |
| 12 |                0.203 |         0.135 |        0.27  |               0.324 |       0.131 |

## By hub (gbm)

|     |   pr_auc |   brier |   precision |   recall |   alert_rate |   base_rate |   clim_pr_auc |
|:----|---------:|--------:|------------:|---------:|-------------:|------------:|--------------:|
| ATL |    0.309 |   0.093 |       0.295 |    0.452 |        0.178 |       0.116 |         0.194 |
| DFW |    0.334 |   0.113 |       0.261 |    0.605 |        0.338 |       0.146 |         0.213 |

## Team CSV flights in the test period

|             |   pr_auc |   brier |   precision |   recall |   alert_rate |
|:------------|---------:|--------:|------------:|---------:|-------------:|
| climatology |    0.213 |   0.114 |       0.221 |    0.467 |        0.288 |
| gbm         |    0.326 |   0.107 |       0.276 |    0.537 |        0.265 |

## Threshold trade-off (gbm, test)

|       |   pr_auc |   brier |   precision |   recall |   alert_rate |
|------:|---------:|--------:|------------:|---------:|-------------:|
| 0.15  |    0.321 |   0.103 |       0.287 |    0.496 |        0.226 |
| 0.2   |    0.321 |   0.103 |       0.336 |    0.372 |        0.145 |
| 0.25  |    0.321 |   0.103 |       0.385 |    0.275 |        0.093 |
| 0.3   |    0.321 |   0.103 |       0.433 |    0.198 |        0.06  |
| 0.4   |    0.321 |   0.103 |       0.523 |    0.103 |        0.026 |
| 0.5   |    0.321 |   0.103 |       0.612 |    0.055 |        0.012 |
| 0.137 |    0.321 |   0.103 |       0.273 |    0.536 |        0.256 |

## Top features (gain share)

|                    |   gain_share |
|:-------------------|-------------:|
| hour_local         |        0.13  |
| OP_UNIQUE_CARRIER  |        0.127 |
| hub_dep_delay_mean |        0.11  |
| month              |        0.069 |
| hub_sev_rate_3h    |        0.061 |
| hub_cap_ratio_3h   |        0.05  |
| origin_temp        |        0.046 |
| origin_pres        |        0.035 |
| hub_arr_delay_mean |        0.029 |
| hub_cancel_rate_3h |        0.027 |
| dest_pres          |        0.025 |
| origin_cldc        |        0.023 |
| dest_temp          |        0.023 |
| dow                |        0.022 |
| h                  |        0.02  |
