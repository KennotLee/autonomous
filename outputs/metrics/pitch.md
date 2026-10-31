# Pitch: 8 slides, 3-minute script, dashboard QA checklist (P4, Task 6)

Baseline numbers are final (2025 test set, `event_baseline_metrics.csv`). Model numbers are `[PLACEHOLDER]`.
Fill them from `event_model_vs_baseline_<name>.csv` after P3's final run. The current P3 v1 values are in brackets after each placeholder.
Sources [S#] are in `problem_framing.md`.

## Slide outline

| # | slide | on screen | evidence |
|---|---|---|---|
| 1 | **Problem & user** | Airline duty manager at the ATL/DFW hub, 14:00, evening bank ahead. "Which of my flights will be more than 45 min late tonight?" | user definition, `problem_framing.md` |
| 2 | **Today & cost** | Ground Stops "issued at the time of the constraint" [S2]; weather = up to 67% of airspace delay, about $23B/yr [S3]; $98.41 per block minute [S5]; 35 kg fuel/min in arrival management vs about 0 at the gate [S4]; 13.2% of 2025 ATL/DFW flights more than 45 min late | S2–S5, D2 |
| 3 | **Data & no-leakage design** | Timeline: weather obs ≤ t − h (and +1 h because a report lands up to 52 min after its stamp) → scheduled time t. The flights file's own weather is crossed out. Train 2023–24, test 2025-01..07; no random split | leakage review v2 (README) |
| 4 | **Demo** | DFW, h = 3: ranked flights, risk, drivers, weather timestamp, accept/override buttons. Switch to ATL and h = 12 | P3 dashboard |
| 5 | **Results vs baselines** | PR-AUC bar chart: constant 0.132, history 0.189, history + month 0.198, weather rule 0.215, **model [PLACEHOLDER] (0.220)**. Second row by h = 3/6/9/12 and ATL/DFW, with 90% CIs | `event_model_vs_baseline_*.csv` |
| 6 | **Failure case** | DFW 2025-04-23: 9 of 11 late, model expected about 2; overcast at t − 3h, thunderstorm at 12:00; cause mostly late-arriving aircraft. Plus false alarm 2025-05-30: north wind counted as tailwind on runway 18 | `failure_cases_*.md` |
| 7 | **Limitations & next steps** | 0.5% flight sample; observations only (no TAF); single runway heading; no inbound-aircraft delay. Next: TAF/convective forecasts, inbound tail delay, both runway directions, live hub-delay panel | README v2 |
| 8 | **Impact** | Hours of warning, not minutes, for the person who can still swap an aircraft cheaply. "The model ranks; the human decides." | |

## Script (about 420 words, 3:00)

**[0:00–0:20] Slide 1: Problem & user.**
Meet our user: the airline duty manager at Atlanta or Dallas–Fort Worth. Every afternoon they decide which flights to protect, swap or cancel before the evening rush. Today they usually learn a flight is in trouble when the FAA issues a ground stop, and by then every fix is expensive.

**[0:20–0:45] Slide 2: Today & cost.**
The FAA's own training material says ground stops are issued at the time of the constraint, rarely in advance. Weather causes up to two-thirds of US airspace delay, about 23 billion dollars a year. A block minute costs a US airline about 98 dollars, and a minute of airborne holding burns around 35 kilograms of fuel. In 2025, thirteen percent of ATL and DFW flights in our data were more than 45 minutes late.

**[0:45–1:10] Slide 3: Data & no leakage.**
For every departure and arrival at the two hubs, we predict the probability of a delay over 45 minutes, three to twelve hours ahead. Leakage is ruled out by construction. We only use hourly weather observed at least that many hours before the scheduled time, plus one more hour, because a report can land 52 minutes after its timestamp. We never use the weather attached to the flights file. We train on 2023 and 2024 and test on 2025.

**[1:10–1:40] Slide 4: Demo.**
Here is Dallas, three hours ahead. Every flight shows its risk, the reasons, and the exact weather observation it used, with its timestamp. The duty manager sorts by risk, reads why, and decides: accept, override, or ignore. Switch to Atlanta, or twelve hours ahead, and the same view updates.

**[1:40–2:05] Slide 5: Results.**
On 5,704 unseen 2025 flights, a constant guess scores a PR-AUC of 0.13. History by airport, direction and hour reaches 0.19. The rule a controller eyeballs today, raising risk when weather looks bad three hours out, reaches 0.215. Our model scores [PLACEHOLDER] at three hours and [PLACEHOLDER] at twelve, a [PLACEHOLDER] percent gain over that rule, with day-level confidence intervals.

**[2:05–2:30] Slide 6: Failure case.**
Here is where we fail. On April 23rd at Dallas, nine of eleven flights were badly late, and we expected two. Three hours earlier the sky was only overcast. The thunderstorm formed later, and most of the delay was late aircraft arriving from elsewhere. On May 30th we raised a false alarm: a north wind counted as a tailwind on runway 18, but Dallas simply turned around.

**[2:30–2:45] Slide 7: Limitations & next steps.**
This is a half-percent sample of flights, built on observations, not forecasts. Next: weather forecasts, inbound-aircraft delay, and both runway directions.

**[2:45–3:00] Slide 8: Impact.**
The result is hours of warning instead of minutes, for the person who can still swap an aircraft cheaply. The model ranks; the human decides.

Placeholder values from P3 v1 (`b27ba59`), in script order:
- PR-AUC at h = 3: 0.220
- PR-AUC at h = 12: 0.219
- gain over the weather rule: +2% at h = 3 (CI includes 0) and +9% at h = 12 (CI [+0.001, +0.042])

If the final model still ties the rule at h = 3, say: "it matches the rule at three hours and beats it by X% at twelve, where the rule's weather signal has gone stale".

## Dashboard QA checklist (click through as the DFW duty manager)

1. **Both airports and every horizon.**
   - ATL and DFW each load, and h = 3, 6, 9 and 12 each change the risks.
   - An unconfigured airport code shows a clear message, not a blank page or a crash.
2. **Weather timestamp on every flag.** Pick one flight and check its weather obs time is ≤ scheduled time − h − 1 h, in UTC, with local time shown too.
3. **Every recommendation shows its rule.**
   - The rule behind the flag: historical rate cell, adverse-weather trigger (code / crosswind / tailwind) and the top model drivers.
   - No driver is an actual-time, delay, taxi, cancelled/diverted or flights-file weather field.
4. **Uncertainty is visible.**
   - Risk appears as a probability, with a calibration hint, for example "of flights shown at about 20%, 23% were late in 2025".
   - No bare yes/no.
5. **The human decides.**
   - Accept, override and dismiss controls exist, and nothing auto-acts.
   - Each decision is logged with user, time and forecast version.
6. **The "high risk" threshold is explained** (top 15%, or a probability cut). It is set per airport, so DFW's higher base rate doesn't absorb every flag.
7. **Model vs rule side by side.** The weather-rule (B3) risk appears next to the model risk, so the user sees when the model disagrees and why.
8. **Times and DST.**
   - Hub local times are labelled with their time zone.
   - Flights on 2025-03-09 (DST start) show sensible local and UTC times.
9. **Failure days replay honestly.** Load DFW 2025-04-23 (missed storm) and 2025-05-30 (false alarm). The dashboard shows what the model believed then, and an "observed delays in the last 2 h" panel shows the miss.
10. **Data-as-of and exclusions.**
    - The header shows the data-as-of time and the model version.
    - Cancelled and diverted flights are marked as excluded, not scored.
    - The 0.5% sample caveat is visible.
