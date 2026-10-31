# Failure cases: 3 worst high-confidence misses (test 2025)

False negatives with the lowest predicted risk (per-airport thresholds, see threshold.md); one row per event, ties broken by larger delay. Weather 'seen' = event-airport row available at feature_obs_time; 'oracle' = P1's oracle_* (flights-CSV weather; its join puts it ~4-6 h before the event, see notes.md). Never a feature.

## 2025-03-17_OO4927_MAF_DFW_arr (h = 3)
- scheduled 2025-03-17 07:19:00 local (2025-03-17 12:19:00+00:00 UTC), arr at KDFW, carrier OO; delay 187 min; predicted risk 0.033
- feature_obs_time 2025-03-17 09:00:00+00:00: temp=11.1, prcp=0, wspd=11.2, pres=1.02e+03, coco=1, crosswind=3.8, tailwind=0, ts_flag=0
- oracle: oracle_temp=11.1, oracle_prcp=0, oracle_wspd=9.4, oracle_pres=1.02e+03, oracle_coco=1
- likely reason: weather was benign in both rows (coco 1/1, no precipitation, wind <= 11 km/h): nothing in our data explains the 187-min delay, so its cause is outside the weather features

## 2025-02-28_OO6272_DFW_MGM_dep (h = 6)
- scheduled 2025-02-28 14:50:00 local (2025-02-28 20:50:00+00:00 UTC), dep at KDFW, carrier OO; delay 109 min; predicted risk 0.033
- feature_obs_time 2025-02-28 14:00:00+00:00: temp=6.7, prcp=0, wspd=11.2, pres=1.02e+03, coco=1, crosswind=7.2, tailwind=0, ts_flag=0
- oracle: oracle_temp=8.3, oracle_prcp=0, oracle_wspd=14.8, oracle_pres=1.02e+03, oracle_coco=2
- likely reason: weather was benign in both rows (coco 1/2, no precipitation, wind <= 15 km/h): nothing in our data explains the 109-min delay, so its cause is outside the weather features

## 2025-07-10_OO6381_MAF_DFW_arr (h = 6)
- scheduled 2025-07-10 06:28:00 local (2025-07-10 11:28:00+00:00 UTC), arr at KDFW, carrier OO; delay 98 min; predicted risk 0.034
- feature_obs_time 2025-07-10 05:00:00+00:00: temp=24.4, prcp=0, wspd=11.2, pres=1.01e+03, coco=2, crosswind=1.9, tailwind=0, ts_flag=0
- oracle: oracle_temp=25, oracle_prcp=0, oracle_wspd=9.4, oracle_pres=1.01e+03, oracle_coco=2
- likely reason: weather was benign in both rows (coco 2/2, no precipitation, wind <= 11 km/h): nothing in our data explains the 98-min delay, so its cause is outside the weather features

# False alarms: 3 highest-risk events that were not severely delayed

Flagged (risk >= the airport's threshold) but delay <= 45 min; one row per event.

## 2025-05-19_F93424_DFW_LGA_dep (h = 3)
- scheduled 2025-05-19 17:23:00 local (2025-05-19 22:23:00+00:00 UTC), dep at KDFW, carrier F9; delay 6 min; predicted risk 0.469 (threshold 0.123)
- feature_obs_time 2025-05-19 19:00:00+00:00: temp=28.3, prcp=0.1, wspd=37, pres=1e+03, coco=8, crosswind=12.7, tailwind=0, ts_flag=0
- oracle: oracle_temp=26.7, oracle_prcp=0, oracle_wspd=24.1, oracle_pres=1e+03, oracle_coco=8
- top drivers: carrier=F9; coco=8; hour=17
- likely reason: adverse weather was visible at feature_obs_time (condition code 8 (rain or worse)) and pushed the risk up (top drivers: carrier=F9; coco=8; hour=17), but the event was not severely delayed (delay +6 min)

## 2025-04-23_AA1092_DFW_ELP_dep (h = 3)
- scheduled 2025-04-23 20:40:00 local (2025-04-24 01:40:00+00:00 UTC), dep at KDFW, carrier AA; delay -2 min; predicted risk 0.441 (threshold 0.123)
- feature_obs_time 2025-04-23 22:00:00+00:00: temp=26.1, prcp=0, wspd=16.6, pres=1.01e+03, coco=18, crosswind=14.4, tailwind=0, ts_flag=0
- oracle: oracle_temp=26.1, oracle_prcp=0, oracle_wspd=20.5, oracle_pres=1.01e+03, oracle_coco=7
- top drivers: hour=20; coco=18; carrier=AA
- likely reason: adverse weather was visible at feature_obs_time (condition code 18 (rain or worse)) and pushed the risk up (top drivers: hour=20; coco=18; carrier=AA), but the event was not severely delayed (delay -2 min)

## 2025-06-04_AA2708_DFW_LGA_dep (h = 12)
- scheduled 2025-06-04 19:01:00 local (2025-06-05 00:01:00+00:00 UTC), dep at KDFW, carrier AA; delay 2 min; predicted risk 0.439 (threshold 0.123)
- feature_obs_time 2025-06-04 12:00:00+00:00: temp=19, prcp=0, wspd=24, pres=1.01e+03, coco=8, crosswind=12, tailwind=20.8, ts_flag=0
- oracle: oracle_temp=23.3, oracle_prcp=0, oracle_wspd=18.4, oracle_pres=1.01e+03, oracle_coco=4
- top drivers: hour=19; carrier=AA; coco=8
- likely reason: adverse weather was visible at feature_obs_time (condition code 8 (rain or worse)) and pushed the risk up (top drivers: hour=19; carrier=AA; coco=8), but the event was not severely delayed (delay +2 min)

# Alert workload (test 2025, per-airport thresholds)

| airport | events x h | flagged | alert rate | precision | false alerts per true alert |
|---|---|---|---|---|---|
| all | 22,888 | 7,735 | 33.8% | 0.199 | 4.0 |
| KATL | 12,140 | 3,755 | 30.9% | 0.194 | 4.2 |
| KDFW | 10,748 | 3,980 | 37.0% | 0.204 | 3.9 |

Precision 0.199: about 4 of every 5 alerts are false. The flight CSV is a ~0.5% sample, so counts are not real daily volumes; use the rates.
