## What happened

ATL, Sunday 11 May 2025. Weather at the hub (Meteostat) was poor all day but never dramatic: fog
or light rain (coco 5/7) from 06:00, overcast (cloud cover 8/8), east-northeast wind of 10–14 kt
from 070–090°, then heavy rain at 18:00–19:00 (9.1 and 7.4 mm/h). No thunderstorm was coded.

- 7 of the 29 sampled ATL flights that day were more than 45 min late (24%). The 6-hour model
  scored **all 7 below the 0.15 alert threshold**, with a mean of 8% over the day.
- The worst delays came in the evening (DL2340 EWR→ATL +226 min, DL838 ATL→PBI +147 min,
  DL2136 ATL→SJC +83 min, DL1100 ATL→CHA +81 min).
- The **3-hour forecast caught part of it**: for the 17:00–19:00 departures it rose to 0.24–0.30,
  above the threshold. Its top drivers were the hub's own state: mean departure delay 51–61 min
  and arrival capacity ratio 0.75–0.80 over the previous 3 h.

## Why the model missed it

1. **The cause isn't in our weather fields.** Wind was a headwind on the 080° runway, so crosswind
   and tailwind looked benign. Rain without thunder is common at ATL and usually harmless, so the
   model learned to ignore it. What most likely cut capacity was **low ceilings and visibility**
   (fog, overcast all day). Visibility is 100% missing in Meteostat at the hubs and ceiling isn't
   provided at all, so the model can't see instrument-condition days. *(Inferred from the codes;
   not checked against FAA ATCSCC advisories.)*
2. **The breakdown built up slowly.** Six hours before the evening flights, the hub was only
   mildly late (departure delay ~13–16 min). The trouble shows up in hub state about 3 hours
   ahead, not 6.
3. **The data is thin.** The team CSV has ~6 flights per hub per day, so a single day gives few
   examples, and climatology cells are small.

## How a user would notice

- **Short horizons react first.** The 3-hour board flagged the 17:00–19:00 flights while the
  6-hour board still showed 7–8%. The dashboard should lead with the newest, shortest-horizon
  forecast and highlight jumps.
- **Drivers change character.** When the top drivers switch from calendar terms
  (`dep_hour_local`, `month_local`) to `hub_dep_delay_mean` and `hub_cap_ratio_3h`, the airport is
  already degrading and older long-horizon forecasts are stale.
- **Low probability means "no signal", not "safe".** On a fog/overcast day the coordinator still
  reads the TAF ceilings and visibility, which this model does not have. Adding TAF or METAR
  ceiling/visibility is the most direct fix.
