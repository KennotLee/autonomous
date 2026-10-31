# Problem framing (P4, Task 5; updated for the organiser's main target: per-flight delay > 45 min)

## User
- **Primary user:** the airline operations-control **duty manager** for the ATL or DFW hub. They decide hours ahead which departures and arrivals to protect, swap, re-crew or pre-cancel.
- **Secondary user:** the FAA **traffic management coordinator (TMC)** at the tower or TRACON. They set the arrival rate and request Ground Delay Programs (GDPs) from the FAA Command Center (ATCSCC) [S1].

## Problem statement (5 sentences)
1. At ATL and DFW, the airline duty manager has to decide hours ahead which flights to protect, but the usual trigger is an FAA Ground Delay Program or Ground Stop issued once weather is already constraining the airport [S1, S2].
2. FAA training material itself calls capacity prediction from weather forecasts "very subjective" and says ground stops "are issued at the time of the constraint and are rarely, if ever, issued for a start time in the future" [S2].
3. Weather accounts for "as much as 67%" of US airspace delay, roughly USD 23 billion a year [S3]. In our 2025 test data, 13.2% of ATL/DFW departures and arrivals were more than 45 minutes late [D2].
4. Reacting late is expensive: a block minute costs a US airline USD 98.41 [S5], and delay absorbed in airborne arrival management burns about 35 kg of fuel per minute against about 0 at the gate [S4].
5. We give the duty manager each flight's probability of a delay over 45 minutes, 3–12 hours ahead, using only weather observed at least h hours before the scheduled time; every flag shows its drivers and the weather timestamp, and **the human decides**.

## Before vs after

**Before (today)**
- Disruption shows up as FAA programs. The TMC sets the arrival rate from experience and the published baseline. ATCSCC issues a GDP, with departure slot times (EDCTs), once demand clearly exceeds that rate. If storms arrive early, it issues a Ground Stop at the moment of the constraint [S1, S2].
- The airline duty manager mostly finds out which flights are hit when the EDCTs or the Ground Stop arrive. By then aircraft swaps, re-crewing and early cancellations are expensive or impossible.
- The only forward-looking tool is judgment: "storms are forecast, so expect trouble this evening." It isn't tied to specific flights and isn't checked afterwards [S2].
- Late reaction pushes delay into the most expensive places: taxiing and airborne holding cost about €47–83 per minute, against about €18 at the gate (strategic, 2022 prices) [S6].

**After (with our tool)**
- 3, 6, 9 or 12 hours ahead, every departure and arrival at ATL/DFW gets a probability of being more than 45 minutes late. The duty manager works from a ranked list instead of a hunch.
- Each flag shows why: historical rate for that airport, direction, hour and month; the weather observation used (with its UTC timestamp); and the top model drivers. The duty manager can check the evidence in seconds.
- **The human decides.** The duty manager accepts, overrides or ignores each flag, and the decision is logged with the forecast version. Nothing acts on its own.
- After the day, forecast vs actual (including misses like DFW 2025-04-23) is visible, so trust is earned, not assumed. The same pipeline runs for any configured airport code.

## Our data figures
| ID | Figure | Source |
|---|---|---|
| D2 | 2025 test set: 5,704 ATL/DFW events (departures + arrivals), 753 (13.2%) late by more than 45 minutes. Train 2023–24: 19,480 events, 10.0% late. | `baselines_events.py` on P1 `event_features.parquet` |
| D3 | Best transparent baseline (history × adverse-weather rule at t − 3h): PR-AUC 0.215 against 0.132 for a constant guess. | `event_baseline_metrics.csv` |

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
