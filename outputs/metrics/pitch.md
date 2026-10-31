# Pitch: 8 slides, 3-minute script, dashboard QA checklist (P4)

All numbers are measured on the 2025 test set (5,704 ATL/DFW departures and arrivals, 13.2% late by more than 45 minutes) using P3's final predictions (`34eeb18`).
- Sources: `event_baseline_metrics.csv`, `event_model_vs_baseline_p3_final.csv` and the dashboard's "How good is it?" tab.
- Cited figures [S#] are in `problem_framing.md`.
- Rubric mapping: problem realism (slides 1–2), technical soundness (3–4), honest evaluation (5–6), trust & traceability (4), impact (8), time discipline (under 3:00).

## Slide outline

| # | slide | on screen | rubric |
|---|---|---|---|
| 1 | **Problem & user** | The FAA **traffic management coordinator (TMC)** at ATL/DFW, with the airline duty manager next to them. 14:00: "How many flights can we take tonight, and which are in trouble?" | problem realism |
| 2 | **Today & cost** | Ground Stops "issued at the time of the constraint" [S2]; weather is up to 67% of airspace delay, about $23B/yr [S3]; $98.41 per block minute [S5]; 35 kg fuel/min in arrival management vs about 0 at the gate [S4]; 13.2% of 2025 ATL/DFW flights more than 45 min late | problem realism |
| 3 | **Data & no-leakage design** | Timeline: weather obs ≤ t − h, plus 1 h because a report lands up to 52 min after its stamp. Flights-file weather crossed out. Head/cross/tailwind on runway 080°/180°. Train 2023–24, test 2025. Any airport code; train-on-ATL-test-on-DFW keeps most skill | technical soundness |
| 4 | **Demo** | DFW, h = 3: recommendations with the rule that fired and the weather timestamp; flight table with risk, "similar flights were late X% (CI)", cautions, and the ⛔ no-forecast decline; decision logged. Then ATL and h = 12 | trust & traceability |
| 5 | **Results vs baselines** | PR-AUC: constant 0.132, history 0.189, + month 0.198, weather rule 0.215, **model 0.220**. By h: model 0.219–0.222, flat; rule 0.215 → 0.202. Tie at h = 3, +9% at h = 12 (CI > 0). No gain at ATL. Bonus AAR (arrivals/hour, full BTS): RMSE 6.76 ATL / 8.78 DFW at h = 3, 3.5% / 7.0% below the best schedule baseline; 4–13% below in bad-weather hours | honest evaluation |
| 6 | **Failure case** | DFW 2025-04-23: 9 of 11 late, model expected about 2; overcast at t − 3h, thunderstorm at 12:00, cause mostly late aircraft. False alarm 2025-05-30: north wind counted as a tailwind on runway 18. How the user knows: "no elevated windows" while live delays rise; the tailwind rule is named | honest evaluation |
| 7 | **Limitations & next steps** | Most misses are calm-weather delays; 2025 ran hotter than training (13.2% vs 10.0%), so risks read low; AI use: no LLM at runtime, built with Claude Code; credit Meteostat (CC BY 4.0). 0.5% flight sample; observations, not forecasts; one runway heading; no inbound-aircraft delay; alerts on about 1 in 3 flights is too many. Next: TAF/convective forecasts, inbound tail delay, both runway ends, alert budget per shift | honest evaluation |
| 8 | **Impact** | Measured: 49–53% of severe delays flagged 3–12 h ahead, about 1 in 5 alerts right. Today that warning comes at 0 h. The weather rule alone gets about the same at the same alert count, so the value is the early, traceable workflow; the model adds at 12 h | impact |

## Say out loud (checklist)
- **Assumptions:**
  - runway headings 080° (ATL) and 180° (DFW), as given by the organisers;
  - a 1 h weather buffer, because a stamp can precede its METAR by about 52 min (all models use it);
  - cancelled and diverted flights are not scored;
  - rates = severe events ÷ scored events.
- **AI use:** no LLM at runtime. Risk comes from LightGBM, recommendations from explicit rules, and reasons from LightGBM feature contributions. We built it with AI coding assistance (Claude Code); explanatory text was AI-drafted and reviewed by the team.
- **Where it fails:**
  - **Calm-weather delays:** late aircraft and airline operations, which our weather features can't see.
  - **2025 had more delays than training** (13.2% vs 10.0%), so risks read low.
  - **How the board shows it:** a "tie with the rule" trust line, the calibration chart, and the two failure days.
- **Any airport:** demo presets are the evaluated ATL/DFW days; hubs, runway headings and defaults come from `config/airports.yaml`.
- **Credits:** weather data from Meteostat (CC BY 4.0, NOAA ISD). Flights from the US DOT BTS On-Time Performance data; we have **not verified** the BTS licence (it's US government data).

## Bonus: arrivals per hour (AAR proxy), for slide 5 and Q&A
These numbers come from an independent P4 check of P2's leak-fixed model (`19e65a6`, weather available at stamp + 1 h).
- **Data:** the full BTS data for 2023-01 to 2025-07, 1.64M ATL/DFW arrivals. Test = 2025, 5,088 hours per airport.
- **Target check:** P4 rebuilt the hourly target from BTS WheelsOn, and it matches P2's on 99.4% of hours.
- **Baseline:** the best schedule baseline (`sched_spread`) spreads the published schedule over the 2023–24 landing-hour pattern. It has RMSE 7.01 at ATL and 9.44 at DFW. P2's own best baselines are weaker (8.68 / 10.68).
- Source: `outputs/metrics/aar_bts_model_vs_baseline.csv`.

| airport | h | model RMSE | vs best schedule baseline (90% CI of RMSE diff) |
|---|---|---|---|
| ATL | 3 | 6.76 | −3.5% [−0.49, −0.03] |
| ATL | 6–12 | 6.87–6.92 | −1.3% to −2.0% (CI includes 0) |
| DFW | 3 | 8.78 | −7.0% [−1.04, −0.32] |
| DFW | 6–9 | 8.92–8.93 | −5.4% to −5.5% (significant) |
| DFW | 12 | 9.20 | −2.5% [−0.51, +0.02] (CI includes 0) |

- **Where it helps:**
  - **bad-weather hours:** 4–13% lower error, significant at every h;
  - **severe-weather hours:** 1–22% lower (the DFW h = 12 CI includes 0);
  - **normal-weather hours at ATL:** slightly worse, +1% to +3%.
- **Uncertainty band:** the P10–P90 band covers 69–75% of hours (nominal 80%) and 42–56% in severe weather, so it is too narrow when it matters most.
- **What to say:** "Arrivals per hour: 1–7% better than reading the schedule well, and 4–13% better in bad-weather hours. Don't quote −21%: that was against weaker baselines."

## Script (about 420 words, 3:00)

**[0:00–0:20] Slide 1: Problem & user.**
Meet our user: the traffic management coordinator at Atlanta or Dallas–Fort Worth, with the airline duty manager beside them. Every afternoon they decide how many flights the airport can take tonight. Today they cut rates when the storm arrives, often with a ground stop, and by then every fix is expensive.

**[0:20–0:40] Slide 2: Today & cost.**
FAA training says ground stops are issued at the time of the constraint. Weather causes up to two-thirds of US airspace delay. A block minute costs an airline about 98 dollars, and airborne holding burns around 35 kilograms of fuel a minute. In 2025, thirteen percent of ATL and DFW flights in our data were more than 45 minutes late.

**[0:40–1:05] Slide 3: Data & no leakage.**
For every departure and arrival at both hubs, we predict the chance of a delay over 45 minutes, three to twelve hours ahead. We only use hourly weather observed at least that early, plus one more hour, because reports land up to 52 minutes after their timestamp. Our assumptions: runway headings of 80 and 180 degrees for crosswind and tailwind, and cancelled flights are not scored. Train is 2023 and 2024; test is 2025.

**[1:05–1:35] Slide 4: Demo.**
Here is Dallas, three hours ahead. The board names the time windows at risk, the rule that fired and the weather it used, with its timestamp. Each flight shows its risk, how often similar flights were really late, and a caution when evidence is thin. With no weather in time, it declines to forecast. The coordinator decides, and the decision is logged.

**[1:35–2:00] Slide 5: Results.**
On 5,704 unseen flights, guessing scores a PR-AUC of 0.13. History reaches 0.19, and a simple weather rule 0.215. Our model scores 0.22 at every horizon. That ties the rule at three hours and beats it by nine percent at twelve, where the rule's weather has gone stale. At Atlanta we gain nothing.

**[2:00–2:25] Slide 6: Failure case.**
Here is where we fail. On April 23rd at Dallas, nine of eleven flights were badly late, and we expected two. Three hours earlier it was only overcast; the storm formed later, and most of the delay was late aircraft from elsewhere. On May 30th, a north wind counted as a tailwind on runway 18, but Dallas simply turned around.

**[2:25–2:40] Slide 7: Limitations.**
Most misses happen in calm weather, and 2025 ran hotter than our training years, so risks read low; the board shows both. Next: weather forecasts, inbound-aircraft delay, both runway directions. No AI runs at decision time; we built it with Claude Code. Weather data: Meteostat.

**[2:40–3:00] Slide 8: Impact.**
Measured on 2025, half of all severe delays are flagged three to twelve hours ahead, with one alert in five correct. Today that warning comes at zero hours. The model ranks; the human decides.

## Dashboard QA checklist (click through as the DFW TMC, then as the duty manager)

1. **Both airports and every horizon.**
   - ATL and DFW each load, and h = 3, 6, 9 and 12 each change the risks.
   - Typing ORD gives a clear "not configured" message.
2. **Weather timestamp on every row.** Pick one flight and check its obs time is ≤ scheduled time − h, in UTC plus local. Head/cross/tailwind are shown and the runway heading is in the sidebar.
3. **Recommendations show their rule.**
   - Each elevated window lists the rule that fired and the weather context.
   - The TMC view suggests a rate review or an early GDP; the duty manager view lists the flights to check.
4. **Uncertainty is visible.**
   - "Flights scored like this were late X% (90% CI)".
   - Cautions for rare conditions and for model/rule disagreement.
   - The ⛔ no-forecast decline when there's no weather ≤ t − h.
5. **The human decides.**
   - Logging is refused without a name.
   - The log records role, risk, rule risk, threshold, weather obs time and model version.
6. **The threshold is explained:** the per-airport default comes from 2024 H2 validation, and the slider shows how alerts change.
7. **Historical trends:**
   - monthly late *rates* for both hubs, with the train/test line;
   - late rate by hour, adverse vs normal;
   - late rate by weather trigger;
   - weekly predicted vs actual for 2025.
8. **Times and DST:** local times on 2025-03-09 (DST start) look sensible.
9. **Failure days replay honestly:** in replay mode, DFW 2025-04-23 shows the misses and 2025-05-30 shows the false alarms.
10. **Evaluation tab is honest:**
    - "not significant: treat as a tie" at h = 3;
    - precision and recall at the threshold vs the rule;
    - cross-airport table;
    - the impact box separates "measured" from "reasoned".
