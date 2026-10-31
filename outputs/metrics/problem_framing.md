# Problem framing (P4, Task 2)

## Problem statement

1. At Atlanta (ATL) and Dallas/Fort Worth (DFW), a traffic management coordinator (TMC) in the tower or TRACON sets the hourly Airport Acceptance Rate (AAR) from experience and weather forecasts. The FAA Command Center (ATCSCC) then turns any hour where demand exceeds that rate into a Ground Delay Program (GDP), or, once the weather has already hit, a Ground Stop [S1, S2].
2. FAA training material calls predicting capacity from weather forecasts "very subjective", and says ground stops "are issued at the time of the constraint and are rarely, if ever, issued for a start time in the future" [S2]. The decision that matters is made by judgment, and the fallback is purely reactive.
3. Weather is the main cause of the problem:
   - NOAA's Aviation Weather Center, citing the FAA, attributes "as much as 67%" of national airspace delay to weather, at roughly USD 23 billion a year [S3].
   - In our 2023–25 BTS sample, 73.9% of cancelled ATL/DFW flights carry the weather cancellation code [D1].
4. Decisions made late push delay into the air:
   - A narrowbody burns about 35 kg of fuel per minute in arrival management (holding and sequencing), against about 12 kg/min taxiing and none parked at the gate [S4].
   - US airlines' average cost is $98.41 per block minute [S5].
5. We give the TMC and the airline operations duty manager a forecast made 3–12 hours ahead:
   - hourly arrivals with an uncertainty band;
   - each flight's risk of arriving more than 45 minutes late;
   - the rule and timestamped evidence behind every recommendation.
   
   They can then set rates and programs earlier. **The human decides; the model advises.**

## Before vs after

**Before (today)**
- The TMC sets the AAR by experience. The starting point is a baseline rate from the FAA's performance metrics, adjusted for runways in use, weather and navigation-aid limits [S1]. The rate is agreed by consensus among the tower, TRACON, Center and ATCSCC [S1].
- A GDP is modelled in the Flight Schedule Monitor (FSM) software once forecast demand clearly exceeds the AAR. Flights get departure slot times (EDCTs), and the rate is revised as the weather unfolds [S1, S2].
- If storms arrive earlier or stronger than forecast, the fallback is a Ground Stop issued at the moment of the constraint [S2]. Flights already in the air absorb the delay as airborne holding, the most expensive kind [S4, S6].
- The airline duty manager learns which flights are hit only when EDCTs or the Ground Stop arrive. That leaves little time to swap aircraft, re-crew or cancel early.

**After (with our tool)**
- 3–12 hours ahead, the dashboard shows forecast hourly arrivals at ATL/DFW with a P10–P90 band, alongside scheduled demand. Runway crosswind and tailwind are computed against the configured runway heading.
- Every hour where the forecast falls below scheduled demand is flagged, showing the rule that fired and the timestamped weather it used. That gives the TMC evidence for an earlier, smoother GDP rate inside the 5–6 hour decision window the FAA already plans around [S2].
- Flights are ranked by their risk of arriving more than 45 minutes late, with the top contributing factors shown. The airline duty manager can then protect connections, swap aircraft or re-crew before the EDCTs arrive.
- **The human decides.** The TMC accepts, edits or rejects each flag, and the decision is logged with the forecast version. After the event, forecast is compared with actual so trust is earned, not assumed.

## Cost and context figures

Figures marked "verified" were read in the source text itself. "Unverified" means the source page couldn't be read here (faa.gov and bts.gov refuse automated access); check those in a browser before putting them on a slide.

| ID | Figure | Source | Status |
|---|---|---|---|
| S1 | "The first step is for an Airport Acceptance Rate (AAR) to be set… By rule, the tower/terminal TMC considers these metrics, all other aspects, to set and adjust the actual AAR." The EWR example shows 47 arrivals expected against an AAR of 32. A GDP is used when other initiatives "are not enough to support the TM mission and limit airborne holding." | FAA TFM Learning, *Appendix A: The Use of a Ground Delay Program in the U.S. NAS*, https://tfmlearning.faa.gov/tfm-training/atfm-basics/files/pdf/L4-Appendix%20A_Ground%20Delay%20Program.pdf | Verified (PDF text) |
| S2 | "Ground stops are issued at the time of the constraint and are rarely, if ever, issued for a start time in the future." "Determining what the actual (or available) capacity will be based on weather forecasts can be very subjective." "The Critical Decision Window (CDW) for implementation is five to six hours prior to the forecasted convective weather activity." | FAA TFM Learning, *NE AFP Strategy of Use Plan* (final draft v2), https://tfmlearning.faa.gov/assets/media/tfm-training/NE_AFP_Strategy_of_Use_Plan_final_draft_v2.pdf | Verified (PDF text). Written for northeast airspace flow programs, not ATL/DFW specifically. |
| S3 | "As much as 67% of the delay in the NAS can be attributed to weather, resulting in US economic losses of roughly 23,000,000,000 USD annually." The same slide's pie chart shows weather at 63%. | NOAA/NWS Aviation Weather Center, *The Value of Your Aviation Weather Program* (J. Stroozas), citing FAA Administrator's Handbook (Dec 2020), https://www.weather.gov/media/aviation/afp/Aviation_Weather_Program_Value.pdf | Verified (PDF text). The 67% text and 63% pie disagree, so quote "as much as 67%" with the citation. |
| S3b | Weather = 74.26% of system-impacting delays over 15 minutes (OPSNET, June 2017 – May 2023) | FAA, *FAQ: Weather Delay*, https://www.faa.gov/NextGen/programs/weather/faq | **Unverified.** Seen only in a search-result summary; the page refused automated access. |
| S4 | Average fuel burn, narrow-body (B738, A320, A319, A20N, A321, A21N, B38M, BCS3, B734): taxi 11.7 kg/min, en route 40.1 kg/min, arrival management 35.2 kg/min. Based on EUROCONTROL BADA and PRISME data. No separate "holding" row; arrival management is the closest phase. | EUROCONTROL Standard Inputs for CBA, ch. 6 "Rate of fuel burn", Table 6.1, https://standard-inputs.netlify.app/chapters/rate_of_fuel_burn.html (redirected from ansperformance.eu) | Verified (page text) |
| S4b | Taxi-out fuel flow: A320 24.3 lb/min (flight-data-recorder model, 160 A320s, 76% of the ICAO rate); A320 21.3 lb/min (A4A data, 63% of ICAO); B737 22.0 lb/min (A4A data, 76% of ICAO). That's about 11.0, 9.7 and 10.0 kg/min (our conversion, 1 lb = 0.4536 kg). | ASCENT Project 46 (MIT / MIT Lincoln Lab, FAA-funded), Oct 2019, https://ascent.aero/documents/2019/10/october-2019-ascent-046.pdf/ | Verified (PDF text). This corrects a search summary that had mixed up the A320 figures. |
| S4c | Airbus A320-214 holding fuel flow at FL100 is plotted on an axis spanning 1,400–2,800 kg/h against weight (40–70 t) and technique; clean configuration at green-dot speed gives the lowest flow. | Airbus, *Getting to grips with fuel economy*, Issue 4 (2004), §5.5, https://ansperformance.eu/library/airbus-fuel-economy.pdf | Partly verified. The curve values sit in a chart we couldn't read; only the axis range and conclusion are confirmed. Don't quote a single number from it. |
| S5 | "In 2025, the average cost of aircraft block (taxi plus airborne) time for U.S. passenger airlines was $98.41 per minute": crew $37.01, fuel $29.34, maintenance $18.35, ownership $9.76, other $3.95. Passenger time is valued at $47/hour (FAA-recommended value). | Airlines for America, *U.S. Passenger Carrier Delay Costs*, https://www.airlines.org/dataset/u-s-passenger-carrier-delay-costs/ | Verified (page text, dated Jul 23, 2026) |
| S6 | Strategic cost of delay per minute (2022 prices): at gate €17.78, taxi €46.69, en-route (cruise extension) €82.95. Tactical cost with knock-on (reactionary) delay, all delay lengths: gate €166, taxi €182, en-route €212, arrival management €206. | EUROCONTROL Standard Inputs, ch. 16 "Cost of delay", Tables 16.1–16.2, from University of Westminster (2015) v4.1, https://standard-inputs.netlify.app/chapters/cost_of_delay.html | Verified (page text) |
| D1 | In the provided CSV (2023-01 to 2025-07): of cancelled flights touching ATL/DFW, 73.9% are code B (weather), 22.3% A (carrier) and 3.8% C (NAS). The BTS "extreme weather" cause is 7.3% of delay-cause minutes and NAS is 17.9%. | Our computation on `Flight_Data_With_Weather_Final.csv` | Computed. BTS "extreme weather" understates weather: weather-related NAS delay is counted under NAS. BTS explains this on its causes page, but that page blocked automated access. |

### Derived figure (our arithmetic on S4 and S6; label it as such on slides)
For each aircraft-minute of delay moved from airborne arrival management to the gate:
- **about 35 kg of fuel saved** (35.2 vs about 0 kg/min with engines off) [S4];
- **about €65 saved** (€82.95 − €17.78, strategic, 2022 prices; en-route is the closest airborne category) [S6].

### Couldn't verify (not used above)
- Official ATL/DFW arrival rates in good vs bad weather (FAA Airport Capacity Profiles): faa.gov refused access.
- FAA Order JO 7210.3 chapter 17 wording on GDPs and Ground Stops: faa.gov refused access.
- The BTS method for adding NAS-weather delays to get weather's full share: bts.gov refused access.
