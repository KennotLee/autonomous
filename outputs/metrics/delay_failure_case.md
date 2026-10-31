# Failure case: KATL on 2025-05-11 (h = 6 h)

Auto-selected: the test hub-day with the most > 45 min delays scored below the alert threshold (0.148) at h = 6.

![timeline](delay_failure_case.png)

## Flights that day

| flight_id                 | hub_time_local   |   delay_min |   oracle_h6 |   hub_coco_at_issue |   hub_coco_at_flight_hour | top_drivers_h6                                                                                     |   risk_h3 |   risk_h6 |   risk_h9 |   risk_h12 |
|:--------------------------|:-----------------|------------:|------------:|--------------------:|--------------------------:|:---------------------------------------------------------------------------------------------------|----------:|----------:|----------:|-----------:|
| 2025-05-11_DL987_ATL_BOS  | 07:00            |         -18 |        0.04 |                   5 |                         7 | hub_dep_delay_mean=158 (+0.14); origin_wspd_max_6h_kt=18.1 (+0.07); hub_arr_delay_mean=110 (+0.07) |      0.04 |      0.04 |      0.04 |       0.05 |
| 2025-05-11_DL1376_PBI_ATL | 07:50            |         -19 |        0.04 |                   7 |                         7 | origin_dwpt_c=22.1 (+0.23); dest_coco=7 (+0.19); hub_arr_delay_mean=31.4 (+0.07)                   |      0.03 |      0.04 |      0.07 |       0.08 |
| 2025-05-11_NK2404_ATL_SJU | 08:00            |          26 |        0.09 |                   7 |                         5 | dest_dwpt_c=24.4 (+0.25); carrier=NK (+0.24); dest_coco=7 (+0.22)                                  |      0.08 |      0.13 |      0.09 |       0.11 |
| 2025-05-11_DL550_LGA_ATL  | 08:20            |           6 |        0.04 |                   7 |                         5 | dest_coco=7 (+0.17); hub_arr_delay_mean=31.4 (+0.06); month_local=5 (+0.06)                        |      0.03 |      0.03 |      0.04 |       0.05 |
| 2025-05-11_DL3108_ATL_DAY | 08:55            |         -20 |        0.03 |                   7 |                         5 | origin_coco=7 (+0.20); hub_dep_delay_mean=148 (+0.10); hub_arr_delay_mean=202 (+0.09)              |      0.04 |      0.04 |      0.03 |       0.06 |
| 2025-05-11_AA3128_ATL_CLT | 09:08            |          57 |        0.08 |                   4 |                         5 | carrier=AA (+0.34); dest_coco=8 (+0.25); dest_rhum_pct=81 (+0.06)                                  |      0.06 |      0.06 |      0.05 |       0.07 |
| 2025-05-11_UA593_ATL_EWR  | 09:51            |          -3 |        0.06 |                   4 |                         5 | carrier=UA (+0.29); origin_wdir_deg=80 (+0.04); origin_rhum_pct=93 (+0.04)                         |      0.07 |      0.04 |      0.06 |       0.08 |
| 2025-05-11_WN175_ATL_DCA  | 11:00            |          -9 |        0.04 |                   7 |                         5 | origin_coco=7 (+0.24); month_local=5 (+0.07); dow_local=6 (+0.07)                                  |      0.04 |      0.05 |      0.06 |       0.05 |
| 2025-05-11_DL1631_FLL_ATL | 11:49            |          24 |        0.05 |                   4 |                         5 | origin_dwpt_c=21.7 (+0.25); month_local=5 (+0.08); dest_coco=4 (+0.08)                             |      0.05 |      0.06 |      0.08 |       0.06 |
| 2025-05-11_DL431_DEN_ATL  | 11:55            |           1 |        0.03 |                   4 |                         5 | origin_wspd_max_6h_kt=18.1 (+0.10); dest_rhum_pct=93 (+0.07); dest_coco=4 (+0.07)                  |      0.04 |      0.04 |      0.06 |       0.06 |
| 2025-05-11_DL838_ATL_PBI  | 13:59            |         147 |        0.08 |                   7 |                         7 | dest_dwpt_c=20.6 (+0.27); origin_coco=7 (+0.26); month_local=5 (+0.09)                             |      0.08 |      0.08 |      0.17 |       0.11 |
| 2025-05-11_DL2908_MCI_ATL | 15:13            |          25 |        0.05 |                   7 |                         7 | dest_coco=7 (+0.15); dest_rhum_pct=90 (+0.09); month_local=5 (+0.08)                               |      0.04 |      0.04 |      0.07 |       0.06 |
| 2025-05-11_DL2542_MSY_ATL | 15:17            |          15 |        0.05 |                   7 |                         7 | dest_coco=7 (+0.16); month_local=5 (+0.10); dest_rhum_pct=90 (+0.06)                               |      0.05 |      0.05 |      0.09 |       0.07 |
| 2025-05-11_DL2329_ATL_ORF | 15:18            |         -15 |        0.05 |                   5 |                         7 | hub_cap_ratio_3h=0.819 (+0.12); dep_hour_local=15 (+0.07); origin_coco=5 (+0.07)                   |      0.04 |      0.03 |      0.04 |       0.05 |
| 2025-05-11_DL1121_SAT_ATL | 15:30            |          12 |        0.05 |                   7 |                         7 | dest_coco=7 (+0.16); month_local=5 (+0.09); dest_rhum_pct=90 (+0.07)                               |      0.05 |      0.04 |      0.06 |       0.07 |
| 2025-05-11_DL2168_ATL_EYW | 15:39            |          30 |        0.12 |                   5 |                         7 | dest_dwpt_c=24.9 (+0.50); dep_hour_local=15 (+0.11); hub_cap_ratio_3h=0.819 (+0.10)                |      0.12 |      0.1  |      0.15 |       0.12 |
| 2025-05-11_DL3083_ATL_VPS | 16:03            |          17 |        0.16 |                   5 |                         8 | dest_dwpt_c=20.9 (+0.29); dep_hour_local=16 (+0.27); hub_arr_delay_mean=21.2 (+0.19)               |      0.25 |      0.11 |      0.11 |       0.1  |
| 2025-05-11_DL1348_ATL_RIC | 16:57            |          17 |        0.08 |                   5 |                         8 | dep_hour_local=16 (+0.23); hub_arr_delay_mean=21.2 (+0.19); origin_coco=5 (+0.09)                  |      0.16 |      0.06 |      0.05 |       0.08 |
| 2025-05-11_DL2622_CVG_ATL | 17:31            |          14 |        0.06 |                   5 |                         8 | hub_cap_ratio_3h=0.819 (+0.16); month_local=5 (+0.08); dest_rhum_pct=93 (+0.08)                    |      0.05 |      0.05 |      0.05 |       0.06 |
| 2025-05-11_DL2136_ATL_SJC | 17:59            |          83 |        0.17 |                   5 |                         8 | dep_hour_local=17 (+0.24); hub_arr_delay_mean=14.5 (+0.18); arr_hour_local=20 (+0.11)              |      0.31 |      0.08 |      0.08 |       0.08 |
| 2025-05-11_DL1257_ATL_BWI | 18:54            |          -3 |        0.29 |                   5 |                         8 | dep_hour_local=18 (+0.23); hub_arr_delay_mean=13.9 (+0.17); arr_hour_local=20 (+0.09)              |      0.3  |      0.07 |      0.05 |       0.06 |
| 2025-05-11_DL2893_ATL_SAT | 18:58            |          63 |        0.32 |                   5 |                         8 | dep_hour_local=18 (+0.24); hub_arr_delay_mean=13.9 (+0.18); origin_coco=5 (+0.10)                  |      0.39 |      0.08 |      0.06 |       0.08 |
| 2025-05-11_B6819_JFK_ATL  | 19:39            |           8 |        0.25 |                   5 |                         7 | carrier=B6 (+0.40); dep_hour_local=17 (+0.32); hub_arr_delay_mean=14.5 (+0.18)                     |      0.27 |      0.16 |      0.19 |       0.13 |
| 2025-05-11_DL903_ATL_SEA  | 21:25            |         -25 |        0.29 |                   7 |                         7 | dep_hour_local=21 (+0.31); origin_coco=7 (+0.29); hub_arr_delay_mean=49.6 (+0.25)                  |      0.13 |      0.17 |      0.09 |       0.13 |
| 2025-05-11_DL2340_EWR_ATL | 21:44            |         226 |        0.17 |                   7 |                         7 | dep_hour_local=19 (+0.26); hub_arr_delay_mean=14.8 (+0.19); dest_coco=7 (+0.19)                    |      0.11 |      0.13 |      0.07 |       0.07 |
| 2025-05-11_DL3044_ATL_BHM | 21:55            |          28 |        0.27 |                   7 |                         7 | dep_hour_local=21 (+0.33); origin_coco=7 (+0.28); hub_arr_delay_mean=49.6 (+0.25)                  |      0.21 |      0.2  |      0.12 |       0.13 |
| 2025-05-11_DL2144_ATL_MSN | 21:59            |          52 |        0.18 |                   7 |                         7 | origin_coco=7 (+0.31); dep_hour_local=21 (+0.27); hub_arr_delay_mean=49.6 (+0.23)                  |      0.11 |      0.09 |      0.06 |       0.07 |
| 2025-05-11_DL3121_ATL_BNA | 22:49            |          42 |        0.15 |                   8 |                         7 | origin_coco=8 (+0.38); dep_hour_local=22 (+0.28); hub_arr_delay_mean=60.1 (+0.23)                  |      0.21 |      0.11 |      0.11 |       0.06 |
| 2025-05-11_DL1100_ATL_CHA | 22:50            |          81 |        0.16 |                   8 |                         7 | origin_coco=8 (+0.42); dep_hour_local=22 (+0.29); hub_arr_delay_mean=60.1 (+0.24)                  |      0.17 |      0.11 |      0.11 |       0.08 |

## Worst-missed hub-days

|                                            |   flights |   severe |   missed |   mean_risk |   mean_oracle |
|:-------------------------------------------|----------:|---------:|---------:|------------:|--------------:|
| ('KATL', Timestamp('2025-05-11 00:00:00')) |        29 |        7 |        7 |       0.078 |         0.119 |
| ('KATL', Timestamp('2025-07-13 00:00:00')) |        18 |        7 |        6 |       0.098 |         0.1   |
| ('KDFW', Timestamp('2025-05-02 00:00:00')) |        14 |        6 |        5 |       0.128 |         0.182 |
| ('KDFW', Timestamp('2025-06-23 00:00:00')) |        21 |        6 |        5 |       0.129 |         0.117 |
| ('KATL', Timestamp('2025-01-21 00:00:00')) |        12 |        5 |        5 |       0.096 |         0.136 |

## What happened

ATL, Sunday 11 May 2025. Weather at the hub (Meteostat) was poor all day but never dramatic: fog
or light rain (coco 5/7) from 06:00, overcast (cloud cover 8/8), east-northeast wind of 10–14 kt
from 070–090°, then heavy rain at 18:00–19:00 (9.1 and 7.4 mm/h). No thunderstorm was coded.

- 7 of the 29 sampled ATL flights that day were more than 45 min late (24%). The 6-hour model
  scored **all 7 below the 0.148 alert threshold** (highest 0.125), with a mean of 8% over the day.
- The worst delays came in the evening (DL2340 EWR→ATL +226 min, DL838 ATL→PBI +147 min,
  DL2136 ATL→SJC +83 min, DL1100 ATL→CHA +81 min).
- The **3-hour forecast caught part of it**: for the ATL departures between 16:00 and 19:00 it rose to 0.16–0.39,
  above the threshold, and it flagged 3 of the 7 severe flights (DL2136 0.31, DL2893 0.39,
  DL1100 0.17). Its top driver was the hub's arrival capacity ratio, down to 0.75–0.81 over the
  previous 3 h, followed by mean departure delay (13–31 min) and the share of landed arrivals
  more than 45 min late (rising from 6% to 32%).

## Why the model missed it

1. **The cause isn't in our weather fields.** Wind was a headwind on the 080° runway, so crosswind
   and tailwind looked benign. Rain without thunder is common at ATL and usually harmless, so the
   model learned to ignore it. What most likely cut capacity was **low ceilings and visibility**
   (fog, overcast all day). Visibility is 100% missing in Meteostat at the hubs and ceiling isn't
   provided at all, so the model can't see instrument-condition days. *(Inferred from the codes;
   not checked against FAA ATCSCC advisories.)*
2. **The breakdown built up slowly.** Six hours before the evening flights, the hub was only
   mildly late (departure delay ~13–18 min, capacity ratio 0.81–1.04). The trouble shows up in
   hub state about 3 hours ahead, not 6.
3. **The data is thin.** The team CSV has ~6 flights per hub per day, so a single day gives few
   examples, and climatology cells are small.

## How a user would notice

- **Short horizons react first.** The 3-hour board flagged the 16:00–19:00 ATL departures while the
  6-hour board still showed 6–11%. The dashboard should lead with the newest, shortest-horizon
  forecast and highlight jumps.
- **Drivers change character.** When the top drivers switch from calendar and weather terms
  (`dep_hour_local`, `month_local`, `dest_dwpt_c`) to `hub_cap_ratio_3h` and `hub_dep_delay_mean`,
  the airport is already degrading and older long-horizon forecasts are stale.
- **Low probability means "no signal", not "safe".** On a fog/overcast day the coordinator still
  reads the TAF ceilings and visibility, which this model does not have. Adding TAF or METAR
  ceiling/visibility is the most direct fix.
