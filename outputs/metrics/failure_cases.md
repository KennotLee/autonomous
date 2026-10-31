# Failure cases: 3 worst high-confidence misses (test 2025)

False negatives with the lowest predicted risk (threshold 0.119); one row per event, ties broken by larger delay. Weather 'seen' = event-airport row available at feature_obs_time; 'oracle' = P1's oracle_* (flights-CSV weather; its join puts it ~4-6 h before the event, see notes.md). Never a feature.

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
