## What happened

DFW, Wednesday 11 June 2025. Meteostat codes a **thunderstorm at DFW 07:00–08:00 local** (coco 25),
followed by rain until ~13:00. Overnight (01:00–04:00) there was light rain but no thunder, and
pressure was flat (1014–1015 hPa), so no pressure-tendency or convective signal existed earlier.

- Arrivals scheduled **07:00–10:00**: 63–88% were > 45 min late. The 6-hour model said **4–6%**,
  well below the 0.14 alert threshold. **Every horizon (3, 6, 9, 12 h) missed these hours**: the
  issue times (01:00–07:00) all fall before the first thunder observation.
- Arrivals **11:00–13:00**: still missed at h = 6 (issued 05:00–07:00), but the **3-hour forecast
  caught them** (risk 0.18 → 0.39, above threshold), because by 08:00 the storm and the first
  departure delays at DFW were in its inputs.
- From **14:00** even the 6-hour forecast flags every hour (0.25 → 0.66), driven by the hub's own
  state (mean departure delay 60–97 min, 3-hour severe share 0.78–0.90), not by weather.

## Why the model missed it

1. **Pop-up convection is not in the past observations.** The model only sees weather up to the
   issue time. A storm that forms within the forecast window is invisible; this is the leakage
   rule working as intended, not a bug.
2. **Even same-hour weather doesn't explain it.** The oracle (weather at the arrival hour) only
   reached 0.18 at 08:00. The damage comes from the backlog a 1–2 h storm creates at a hub
   (holding, ground stop, misplaced aircraft and crews), which lasts 6+ hours after the rain stops.
   Hourly station weather (one point, 1-hour resolution, gusts and visibility missing) can't
   describe that.
3. The fix is a **forecast input** (TAF thunder probability, convective outlook/CCFP), not more
   past observations. We did not have those archives within the hackathon time.

## How a user would notice

- **Refresh at short horizons.** The 3-hour board flipped to high risk at 08:00, about 3 hours
  before the worst arrivals and 6 hours before the 6-hour board caught up. The dashboard should
  show the newest, shortest-horizon forecast first and mark a jump in it.
- **Watch the drivers.** The top drivers switch from calendar/carrier terms to `hub_dep_delay_mean`
  and `hub_sev_rate_3h`. A board full of "hub state" drivers is a sign the airport is already in
  trouble and earlier long-horizon forecasts are stale.
- **Low probability is not "safe".** On a summer morning with moisture around, the model's 5%
  means "no signal yet", not "no storm". The coordinator still checks the TAF and the convective
  outlook; the model does not replace them.
