"""P4 dashboard: weather delay-risk board for ATL/DFW (user: ATC traffic management coordinator; airline ops alongside).

    .venv/bin/streamlit run app/app.py        (Windows: .venv\\Scripts\\streamlit run app/app.py)

Centre of the page: P3's model risk P(delay > 45 min) per flight, with the evidence to check it (plain-language
drivers, the weather observation and its timestamp, the range seen for similar flights) and a logged human decision.
Reads committed team outputs only:
  data/processed/event_features.parquet        P1 events + weather observed at t - h (outcomes only in replay)
  outputs/predictions/delay_preds.csv          P3 model risk per (event_id, h)
  outputs/predictions/event_baseline_preds.csv P4 weather-rule baseline (B3), used on the performance tab
  outputs/metrics/event_*.csv, src/models/delay/artifacts/meta.json, outputs/predictions/aar_preds.csv (P2)
Writes app/decision_log.csv. No LLM at runtime (see AI_USE).
"""
import json
import math
import re
import subprocess
from datetime import date, datetime, timezone
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
import yaml

ROOT = Path(__file__).resolve().parents[1]
PRED, MET = ROOT / "outputs/predictions", ROOT / "outputs/metrics"
LOG = ROOT / "app/decision_log.csv"
HORIZONS = [3, 6, 9, 12]
TEST_START = "2025-01-01"  # FL_DATE split: train 2023-24, test 2025
ADVERSE_CODES, CROSS_KMH, TAIL_KMH = {5, 6, 7, 8, 9, 25, 26, 27}, 35, 15  # organiser "adverse weather" rule
RARE = {"thunderstorm": {23, 24, 25, 26, 27}, "snow/ice": {10, 11, 12, 13, 14, 15, 16, 19, 20, 21, 22}}
ELEVATED_X, ELEVATED_ALERTS = 1.5, 2  # a 3-h window is elevated if mean risk >= 1.5x normal or >= 2 alerts
COCO = {1: "clear", 2: "fair", 3: "cloudy", 4: "overcast", 5: "fog", 6: "freezing fog", 7: "light rain", 8: "rain",
        9: "heavy rain", 10: "freezing rain", 11: "heavy freezing rain", 12: "sleet", 13: "heavy sleet",
        14: "light snow", 15: "snow", 16: "heavy snow", 17: "rain shower", 18: "heavy rain shower",
        19: "sleet shower", 20: "heavy sleet shower", 21: "snow shower", 22: "heavy snow shower", 23: "lightning",
        24: "hail", 25: "thunderstorm", 26: "heavy thunderstorm", 27: "storm"}
MONTHS = "Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split()
DAYS = "Mon Tue Wed Thu Fri Sat Sun".split()
# demo presets = the evaluated test days (src/eval/failure_cases.py); shown only for airports present in the data
DEMO_DAYS = {"✅ Good day (hand-picked):": ("KATL", date(2025, 4, 6)),
             "❌ Missed storm:": ("KDFW", date(2025, 4, 23)),
             "❌ False alarm:": ("KDFW", date(2025, 5, 30))}
DECISIONS = ["—", "Act", "Watch", "Override: no action"]
AI_USE = ("No LLM at runtime: risk = LightGBM, recommendations = explicit rules, reasons = LightGBM feature "
          "contributions. Built with AI coding assistance (Claude Code); explanatory text was AI-drafted and "
          "reviewed by the team.")
SRC = "Data: organiser files (BTS on-time sample; Meteostat / NOAA ISD weather, CC BY 4.0)."
SRC_BTS = "Data: US DOT BTS On-Time Performance (full, fetched by script); Meteostat / NOAA ISD weather, CC BY 4.0."
RATE = "rate = severe events (> 45 min late) ÷ scored events (cancelled and diverted are not scored)"

st.set_page_config(page_title="Weather delay-risk board", page_icon="✈️", layout="wide")


# ------------------------------------------------------------------ data
@st.cache_data
def load_config():
    with open(ROOT / "config/airports.yaml") as f:
        return yaml.safe_load(f)


@st.cache_data
def load_events():
    cols = ["event_id", "event_type", "airport", "other_airport", "sched_time_local", "sched_time_utc", "h",
            "feature_obs_time", "coco", "crosswind", "tailwind", "headwind", "carrier", "delay_min", "label_45",
            "cancelled", "diverted"]
    ev = pd.read_parquet(ROOT / "data/processed/event_features.parquet", columns=cols)
    ev["test"] = ev["event_id"].str[:10] >= TEST_START
    ev["scored"] = ev["label_45"].notna() & (ev["cancelled"] == 0) & (ev["diverted"] == 0)
    ev["adverse"] = ev["coco"].isin(ADVERSE_CODES) | (ev["crosswind"].abs() >= CROSS_KMH) | (ev["tailwind"] >= TAIL_KMH)
    ev["rare"] = ""
    for name, codes in RARE.items():
        ev.loc[ev["coco"].isin(codes), "rare"] = name
    return ev


@st.cache_data
def load_preds():
    p = pd.read_csv(PRED / "delay_preds.csv", usecols=["event_id", "h", "risk_prob", "top_drivers"])
    b = pd.read_csv(PRED / "event_baseline_preds.csv")
    b = b[b["model"] == "B3_B2_x_adverse_wx"][["event_id", "h", "risk_prob"]].rename(columns={"risk_prob": "rule_risk"})
    return p.merge(b, on=["event_id", "h"], how="left")


@st.cache_data
def load_meta():
    meta = json.loads((ROOT / "src/models/delay/artifacts/meta.json").read_text())
    try:
        sha = subprocess.run(["git", "log", "-1", "--format=%h", "--", "outputs/predictions/delay_preds.csv"],
                             cwd=ROOT, capture_output=True, text=True).stdout.strip()
    except OSError:
        sha = ""
    return meta, sha or "unknown"


@st.cache_data
def load_csv(name):
    path = MET / name
    return pd.read_csv(path) if path.exists() else None


@st.cache_data
def train_stats():
    ev = load_events()
    tr = ev[(ev["h"] == 3) & ~ev["test"] & ev["scored"]]
    return tr.groupby("airport")["label_45"].mean().to_dict(), tr[tr["rare"] != ""].groupby("rare").size().to_dict()


@st.cache_data
def operating_point(thr_by_apt, default_thr):
    """2025 test: alerts at the per-airport threshold vs the weather rule given the same number of alerts."""
    ev, preds = load_events(), load_preds()
    t = ev[ev["test"] & ev["scored"]].merge(preds, on=["event_id", "h"])
    rows = []
    for (apt, h), g in t.groupby(["airport", "h"]):
        y = g["label_45"].to_numpy() == 1
        f = g["risk_prob"].to_numpy() >= thr_by_apt.get(apt, default_thr)
        fr = np.zeros(len(g), bool)
        fr[np.argsort(-g["rule_risk"].to_numpy(), kind="stable")[:f.sum()]] = True
        rows.append({"airport": apt, "h": h, "late flights": int(y.sum()), "alerts": int(f.sum()),
                     "model precision": (f & y).sum() / max(f.sum(), 1), "model recall": (f & y).sum() / max(y.sum(), 1),
                     "rule recall (same alerts)": (fr & y).sum() / max(y.sum(), 1)})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ helpers
def wilson(p, n, z=1.645):
    d = 1 + z * z / n
    c, w = (p + z * z / (2 * n)) / d, z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(c - w, 0), c + w


def similar(risk, cal):
    """Observed late rate of 2025 flights the model scored in the same risk band, with a 90% Wilson range."""
    if cal is None or pd.isna(risk):
        return ""
    row = cal[cal["pred_max"] >= risk].head(1)
    r = (cal.tail(1) if row.empty else row).iloc[0]
    lo, hi = wilson(r.observed_rate, int(r.n))
    return f"{r.observed_rate:.0%} ({lo:.0%}–{hi:.0%})"


def plain_driver(item):
    """'pres_lag3h=1.01e+03' -> 'pressure 1010 hPa (3 h earlier)'. Unknown features pass through unchanged."""
    if "=" not in item:
        return item
    name, val = item.split("=", 1)
    where = " at other end" if name.startswith("other_") and name != "other_airport" else ""
    name = name if name == "other_airport" else name.removeprefix("other_")
    lag = re.search(r"_lag(\d+)h$", name)
    when = f" ({lag.group(1)} h earlier)" if lag else ""
    name = re.sub(r"_lag\d+h$", "", name)
    text = {"carrier": lambda: f"carrier {val}", "other_airport": lambda: f"route to/from {val}",
            "airport": lambda: f"hub {val}", "event_type": lambda: {"dep": "departure", "arr": "arrival"}.get(val, val)}
    num = {"h": lambda x: f"{x:.0f} h ahead", "hour": lambda x: f"{int(x):02d}:00 slot",
           "month": lambda x: MONTHS[int(x) - 1], "dow": lambda x: DAYS[int(x)],
           "coco": lambda x: COCO.get(int(x), f"weather code {int(x)}"), "temp": lambda x: f"{x:.0f} °C",
           "prcp": lambda x: f"rain {x:.1f} mm", "prcp_sum_6h": lambda x: f"rain {x:.1f} mm in 6 h",
           "wspd": lambda x: f"wind {x:.0f} km/h", "wdir": lambda x: f"wind from {x:.0f}°",
           "pres": lambda x: f"pressure {x:.0f} hPa", "pres_tend_3h": lambda x: f"pressure {x:+.1f} hPa in 3 h",
           "headwind": lambda x: f"headwind {x:.0f} km/h", "crosswind": lambda x: f"crosswind {abs(x):.0f} km/h",
           "tailwind": lambda x: f"tailwind {x:.0f} km/h",
           "ts_flag": lambda x: "thunderstorm" if x == 1 else "no thunderstorm",
           "snow_flag": lambda x: "snow" if x == 1 else "no snow"}
    try:
        out = text[name]() if name in text else num[name](float(val)) if name in num else item
    except (ValueError, IndexError):
        out = item
    return out + where + when


def board(icao, day, h, tz):
    ev, preds = load_events(), load_preds()
    d = ev[(ev["airport"] == icao) & (ev["h"] == h) & (ev["sched_time_local"].dt.date == day)]
    d = d.merge(preds, on=["event_id", "h"], how="left").sort_values("sched_time_local").reset_index(drop=True)
    obs_local = d["feature_obs_time"].dt.tz_convert(tz)
    d["weather"] = [
        f"{'⚠ ' if a else ''}{COCO.get(int(c), '?') if pd.notna(c) else 'n/a'}, cross {abs(x):.0f} / tail {t:.0f} km/h"
        f" · obs {o:%H:%M}Z ({ol:%H:%M} local)" if pd.notna(o) else "—"
        for a, c, x, t, o, ol in zip(d["adverse"], d["coco"], d["crosswind"].fillna(0), d["tailwind"].fillna(0),
                                     d["feature_obs_time"], obs_local)]
    d["why"] = [" · ".join(plain_driver(s.strip()) for s in str(v).split(";")) if pd.notna(v) else ""
                for v in d["top_drivers"]]
    return d


# ------------------------------------------------------------------ sidebar
cfg, ev, preds = load_config(), load_events(), load_preds()
meta, model_version = load_meta()
airports = cfg["airports"]
thr_by_apt = {k: v["threshold"] for k, v in meta.get("thresholds_by_airport", {}).items()}
base_rates, rare_counts = train_stats()
res = load_csv("event_model_vs_baseline_p3_final.csv")
cal_all = load_csv("event_calibration_p3_final.csv")

hubs = [k for k in airports if k in set(ev["airport"])]  # configured airports that have events
demo_days = {f"{lab} {airports[a]['iata']} {d:%d %b}": (a, d) for lab, (a, d) in DEMO_DAYS.items() if a in hubs}
first = next(iter(demo_days.values()), (hubs[0], date(2025, 5, 2)))
HEADINGS = ", ".join(f"{airports[k]['iata']} {airports[k]['runway_heading_deg']:03d}°" for k in hubs)
for key, val in [("icao", first[0]), ("day", first[1])]:
    st.session_state.setdefault(key, val)


def jump(icao, day):
    st.session_state.update(icao=icao, day=day)


def pick_code():
    c = st.session_state["code"].strip().upper()
    match = c if c in airports else next((k for k, v in airports.items() if v["iata"] == c), None)
    if not c:
        msg = None
    elif match is None:
        msg = f"{c} is not configured: add it to config/airports.yaml and run scripts/run_pipeline.py --airports {c}."
    elif match not in set(ev["airport"]):
        msg = f"{match} is configured but has no events yet: run the pipeline and model for it."
    else:
        msg = None
        st.session_state["icao"] = match
    st.session_state["code_msg"] = msg


with st.sidebar:
    st.markdown("### ✈️ Weather delay-risk board")
    icao = st.selectbox("Hub", hubs, key="icao",
                        format_func=lambda k: f"{airports.get(k, {}).get('iata', k)} ({k})")
    h = st.radio("Warn how far ahead?", HORIZONS, index=0, horizontal=True, format_func=lambda x: f"{x} h")
    test_days = sorted(ev.loc[ev["event_id"].isin(preds["event_id"]), "sched_time_local"].dt.date.unique())
    day = st.date_input("Day (2025)", key="day", min_value=test_days[0], max_value=test_days[-1])
    reveal = st.toggle("Replay: show what actually happened", value=False)
    user = st.text_input("Your name (to log decisions)", value="")
    st.markdown("**Demo days**")
    for label, (a, dd) in demo_days.items():
        st.button(label, on_click=jump, args=(a, dd), width="stretch")
    with st.expander("Settings"):
        st.text_input("Other airport code (IATA/ICAO)", key="code", on_change=pick_code, placeholder="e.g. ORD")
        if st.session_state.get("code_msg"):
            st.error(st.session_state["code_msg"])
        default_thr = thr_by_apt.get(icao, meta["threshold"])
        if st.session_state.get("thr_for") != icao:  # new hub -> its validated default threshold
            st.session_state.update(thr=float(round(default_thr, 3)), thr_for=icao)
        thr = st.slider("Alert at risk ≥", 0.02, 0.50, step=0.005, key="thr",
                        help="Default: tuned per airport on 2024-H2 validation data (never on test).")
        st.caption(f"Runway heading {airports[icao]['runway_heading_deg']}° (organiser assumption).")
    st.caption(f"Model: P3 LightGBM ({model_version}). {SRC} No LLM at runtime; built with Claude Code assistance.")

tz = airports[icao]["timezone"]
iata = airports[icao]["iata"]
base_rate = base_rates.get(icao, 0.10)
cal = None if cal_all is None else cal_all[(cal_all["h"] == h) & (cal_all["predictor"] == "model")].sort_values("bin")

st.markdown(f"## {iata} · {day:%a %d %b %Y} · risk of a > 45 min delay, forecast {h} h ahead")
st.caption("For the ATC traffic management coordinator (and airline ops). Today, rates are cut when the weather "
           "arrives; this board ranks flights hours earlier from weather already observed. **It advises; you decide.**")

tabs = st.tabs(["🛫 Risk board", "📈 Performance & failures", "🛬 Arrivals/hour (bonus)", "🔒 How it works"])

# ------------------------------------------------------------------ 1 risk board
with tabs[0]:
    if res is not None:
        m = res[(res["h"] == h) & (res["slice"] == "ALL")].iloc[0]
        tie = m["pr_auc_diff_ci90_lo"] <= 0
        st.info(f"**How much to trust this at {h} h:** PR-AUC {m['pr_auc_model']:.3f} vs {m['pr_auc_base']:.3f} for a "
                f"simple weather rule and 0.13 for guessing (2025 test). "
                + ("That is a **tie with the rule**: use the risk to rank flights, not as a certainty."
                   if tie else f"That is **{m['pr_auc_impr_pct']:.0f}% better** than the rule (significant)."))
    d = board(icao, day, h, tz)
    if d.empty:
        st.warning("No flights for this hub and day in the flight sample (~0.5% of traffic). Pick another day.")
    else:
        d["alert"] = d["risk_prob"] >= thr
        c = st.columns(4 if reveal else 3)
        c[0].metric("Flights (sample)", len(d))
        c[1].metric("Alerts", int(d["alert"].sum()), help=f"risk ≥ {thr:.2f}")
        c[2].metric("Expected late rate", f"{d['risk_prob'].mean():.0%}", delta=f"normal {base_rate:.0%}", delta_color="off",
                    help=f"Mean model risk. Normal = 2023–24 {RATE}.")
        if reveal:
            lab = d[d["scored"]]
            tp = int((lab["alert"] & (lab["label_45"] == 1)).sum())
            c[3].metric("Actual late rate", f"{lab['label_45'].mean():.0%}",
                        help=f"{RATE}. {tp} of {int(lab['label_45'].sum())} late flights had an alert.")

        # recommendations: explicit rule on 3-hour windows, one line each
        d["win"] = (d["sched_time_local"].dt.hour // 3) * 3
        for w, g in d[d["risk_prob"].notna()].groupby("win"):
            n_alert, mean_risk = int(g["alert"].sum()), g["risk_prob"].mean()
            if mean_risk < ELEVATED_X * base_rate and n_alert < ELEVATED_ALERTS:
                continue
            wx = "adverse weather at t−h" if g["adverse"].any() else "no adverse weather: risk from history"
            obs = g["feature_obs_time"].dropna()
            sig = ((f"signal: weather observed {obs.min():%H:%M}Z"
                    + (f"–{obs.max():%H:%M}Z" if obs.max() != obs.min() else "")) if len(obs) else "signal: none")
            st.warning(f"**{w:02d}:00–{w + 3:02d}:00 elevated** — {n_alert} alerts, mean risk {mean_risk:.0%} vs normal "
                       f"{base_rate:.0%}; {wx} ({sig}). *Consider* reviewing the rate for this window and planning a GDP "
                       f"early rather than a ground stop at onset; airline: consider checking connections, aircraft "
                       f"and crew for alerted flights. Rule: mean risk ≥ {ELEVATED_X}× normal or ≥ {ELEVATED_ALERTS} "
                       f"alerts in a 3-hour window.")

        order = d.sort_values("risk_prob", ascending=False, na_position="last").reset_index(drop=True)
        caution = []
        for r in order.itertuples():
            if pd.isna(r.risk_prob):
                caution.append("⛔ no forecast: no weather observed ≤ t−h")
            elif r.rare:
                caution.append(f"rare: {r.rare} (only {rare_counts.get(r.rare, 0)} in training)")
            else:
                caution.append("")
        view = pd.DataFrame({
            "event_id": order["event_id"],
            "decision": "—",
            "time": order["sched_time_local"].dt.strftime("%H:%M"),
            "flight": np.where(order["event_type"] == "dep", "🛫 ", "🛬 ") + order["carrier"] + " "
                      + np.where(order["event_type"] == "dep", "→ ", "← ") + order["other_airport"],
            "risk": order["risk_prob"],
            "alert": np.where(order["risk_prob"] >= thr, "🔴", ""),
            "similar flights late": [similar(r, cal) for r in order["risk_prob"]],
            "why (model)": order["why"],
            "weather used (obs ≤ t−h)": order["weather"],
            "caution": caution,
            "note": ""})
        if reveal:  # actual outcome next to the risk, plus whether the alert was right
            mins = order["delay_min"].round().astype("Int64").astype(str)
            late, alerted = order["label_45"] == 1, order["risk_prob"] >= thr
            outcome = np.select([order["cancelled"] == 1, order["diverted"] == 1, late, order["label_45"] == 0],
                                ["cancelled", "diverted", "LATE " + mins + " min", "on time (" + mins + " min)"], "—")
            verdict = np.select([late & alerted, late & ~alerted, (order["label_45"] == 0) & alerted],
                                ["✅ caught", "❌ missed", "⚠ false alarm"], "")
            view.insert(view.columns.get_loc("alert") + 1, "actual", outcome)
            view.insert(view.columns.get_loc("actual") + 1, "alert right?", verdict)
        if not any(caution):
            view = view.drop(columns="caution")
        cols = list(view.columns)
        if reveal:
            colour = {"LATE": "background-color: #f8d7da; color: #842029; font-weight: 600",
                      "on time": "background-color: #d1e7dd; color: #0f5132",
                      "cancelled": "background-color: #e2e3e5; color: #41464b",
                      "diverted": "background-color: #e2e3e5; color: #41464b"}
            view = view.style.map(lambda v: next((css for k, css in colour.items() if str(v).startswith(k)), ""),
                                  subset=["actual"])
        b1, b2 = st.columns([1, 4])
        log_clicked = b1.button("✅ Log my decisions", type="primary")
        b2.caption("Pick **Act / Watch / Override** in the first column (add a note if you like), then log. "
                   "Nothing happens until you decide.")
        log_msg = st.container()
        edited = st.data_editor(
            view, hide_index=True, width="stretch", key=f"ed_{icao}_{day}_{h}",
            disabled=[c for c in cols if c not in ("decision", "note")],
            column_config={
                "event_id": None,
                "risk": st.column_config.ProgressColumn("risk", min_value=0, max_value=1, format="percent"),
                "similar flights late": st.column_config.TextColumn(
                    help="Of 2025 flights the model scored in this risk band, the share that was really late "
                         "(90% range). This is the uncertainty on the risk."),
                "why (model)": st.column_config.TextColumn(help="Top factors pushing this flight's risk up."),
                "weather used (obs ≤ t−h)": st.column_config.TextColumn(
                    help="The exact hourly observation the forecast used, and when it was observed. ⚠ = adverse."),
                "decision": st.column_config.SelectboxColumn(options=DECISIONS, required=True),
            })
        if log_clicked:  # handled after the editor so it sees this run's edits; messages show above the table
            chosen = edited[edited["decision"] != "—"]
            if not user.strip():
                log_msg.warning("Enter your name in the sidebar first: every decision is attributed.")
            elif chosen.empty:
                log_msg.warning("Pick a decision for at least one flight.")
            else:
                rows = chosen.merge(order[["event_id", "risk_prob", "feature_obs_time"]], on="event_id").assign(
                    logged_at_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"), user=user.strip(),
                    airport=icao, h=h, threshold=thr, model_version=model_version)
                rows[["logged_at_utc", "user", "airport", "h", "event_id", "risk_prob", "threshold", "feature_obs_time",
                      "decision", "note", "model_version"]].to_csv(LOG, mode="a", header=not LOG.exists(), index=False)
                log_msg.success(f"Logged {len(rows)} decision(s).")
        st.caption(f"{SRC} Risk: P3 LightGBM ({model_version}); flights shown are a ~0.5% sample of real traffic.")
        with st.expander("Decision log"):
            if LOG.exists():
                st.dataframe(pd.read_csv(LOG).iloc[::-1], hide_index=True, width="stretch")
            else:
                st.caption("No decisions logged yet.")

# ------------------------------------------------------------------ 2 performance & failures
with tabs[1]:
    if res is None:
        st.warning("Run src/eval/baselines_events.py and evaluate_events.py first.")
    else:
        st.markdown(f"**Tested on 5,704 unseen 2025 flights (13.2% late), trained on 2023–24. Horizon: {h} h.**")
        basem = load_csv("event_baseline_metrics.csv")
        b = basem[(basem["slice"] == "ALL") & (basem["h"] == h)][["model", "pr_auc"]]
        b["model"] = b["model"].map({"B0_const": "Guess (base rate)", "B1_apt_dir_hour": "History: airport × hour",
                                     "B2_apt_dir_hour_month": "History + month", "B3_B2_x_adverse_wx": "Simple weather rule"})
        m = res[(res["h"] == h) & (res["slice"] == "ALL")].iloc[0]
        b = pd.concat([b, pd.DataFrame({"model": ["Our model"], "pr_auc": [m["pr_auc_model"]]})])
        bar = alt.Chart(b).mark_bar().encode(
            x=alt.X("pr_auc:Q", title="PR-AUC (higher is better)", scale=alt.Scale(domain=[0, 0.3])),
            y=alt.Y("model:N", sort=None, title=None),
            color=alt.condition(alt.datum.model == "Our model", alt.value("#d62728"), alt.value("#9e9e9e")))
        c1, c2 = st.columns([3, 2])
        c1.altair_chart((bar + bar.mark_text(align="left", dx=3).encode(text=alt.Text("pr_auc:Q", format=".3f")))
                        .properties(height=180), width="stretch")
        c1.caption(f"2025 test, {h} h ahead. {SRC}")
        tie = m["pr_auc_diff_ci90_lo"] <= 0
        c2.markdown(f"**vs simple weather rule:** {m['pr_auc_impr_pct']:+.0f}%  \n90% CI of difference "
                    f"[{m['pr_auc_diff_ci90_lo']:+.3f}, {m['pr_auc_diff_ci90_hi']:+.3f}]  \n"
                    + ("➡ **not significant: a tie**" if tie else "➡ **significant**"))
        per_apt = res[(res["h"] == h) & res["slice"].isin([airports[k]["iata"] for k in hubs])]
        c2.caption(", ".join(f"{r.slice} {r.pr_auc_impr_pct:+.0f}%" for r in per_apt.itertuples()) + " vs the rule. "
                   "Across h the model stays at 0.22 while the rule fades (0.215 → 0.202).")

        op = operating_point(thr_by_apt, meta["threshold"])
        st.markdown("**At the alert threshold** (precision = share of alerts that were right; recall = share of late "
                    "flights alerted):")
        st.dataframe(op[op["h"] == h].drop(columns="h").style.format(
            {"model precision": "{:.0%}", "model recall": "{:.0%}", "rule recall (same alerts)": "{:.0%}"}),
            hide_index=True, width="stretch")
        st.caption(f"Precision = right alerts ÷ alerts; recall = alerted late flights ÷ late flights. {SRC}")

        if cal_all is not None:
            c = cal_all[(cal_all["h"] == h) & (cal_all["predictor"] == "model")]
            pts = alt.Chart(c).mark_line(point=True, color="#d62728").encode(
                x=alt.X("mean_pred:Q", title="predicted risk"), y=alt.Y("observed_rate:Q", title="actually late"),
                tooltip=["n", alt.Tooltip("mean_pred:Q", format=".2f"), alt.Tooltip("observed_rate:Q", format=".2f")])
            diag = alt.Chart(pd.DataFrame({"x": [0, 0.3], "y": [0, 0.3]})).mark_line(strokeDash=[4, 4], color="gray").encode(x="x", y="y")
            hist = ev[(ev["h"] == 3) & ev["scored"]].copy()
            hist["month"] = hist["sched_time_local"].dt.to_period("M").dt.to_timestamp()
            mm = hist.groupby(["month", "airport"])["label_45"].mean().rename("late rate").reset_index()
            c1, c2 = st.columns(2)
            c1.markdown("**Is the risk honest?** Predicted vs actual (dashed = perfect)")
            c1.altair_chart((diag + pts).properties(height=220), width="stretch")
            c1.caption(f"Slightly low: 2025 had more delays (13%) than the training years (10%). {SRC}")
            c2.markdown("**History:** monthly late rate (rates, so hubs compare fairly)")
            c2.altair_chart(alt.Chart(mm).mark_line(point=True).encode(
                x=alt.X("month:T", title=None), y=alt.Y("late rate:Q", axis=alt.Axis(format="%")), color="airport:N")
                .properties(height=220), width="stretch")
            c2.caption(f"Monthly {RATE}. Train 2023–24, test 2025. {SRC}")

    st.markdown("### Where it fails, and how you would know")
    f1, f2 = st.columns(2)
    with f1:
        st.markdown("**❌ Missed storm: DFW 23 Apr 2025.** 9 of 11 flights late; the model expected about 2. "
                    "Three hours before, it was only overcast; the thunderstorm formed later, and most delay was late "
                    "aircraft from other airports.  \n*Fix:* forecasts (TAF), inbound-aircraft delay.  \n"
                    "*How you'd know:* the board stays calm while morning flights are already late, so always "
                    "cross-check live delays.")
        st.button("Open this day", key="f1", on_click=jump, args=("KDFW", date(2025, 4, 23)))
    with f2:
        st.markdown("**❌ False alarm: DFW 30 May 2025.** 0 of 22 late, many alerts. A north wind counted as a "
                    "*tailwind* on the assumed runway 180°, but DFW simply turned around to land north.  \n"
                    "*Fix:* use both runway directions.  \n*How you'd know:* the weather column shows the "
                    "tailwind behind the alerts, so you can discount it.")
        st.button("Open this day", key="f2", on_click=jump, args=("KDFW", date(2025, 5, 30)))

# ------------------------------------------------------------------ 3 arrivals (bonus)
with tabs[2]:
    aar_path = PRED / "aar_preds.csv"
    if not aar_path.exists():
        st.info("No aar_preds.csv.")
    else:
        a = pd.read_csv(aar_path, parse_dates=["target_hour_utc"])
        a = a[(a["icao"] == icao) & (a["h"] == h)]
        a["local"] = a["target_hour_utc"].dt.tz_convert(tz).dt.tz_localize(None)
        a = a[a["local"].dt.date == day]
        st.markdown(f"**Arrivals per hour at {iata}, forecast {h} h ahead** (P2 model on full BTS data)")
        if a.empty:
            st.info("No arrival forecasts for this day.")
        else:
            base = alt.Chart(a).encode(x=alt.X("local:T", title="local time"))
            layers = (base.mark_area(opacity=0.25).encode(y=alt.Y("p10:Q", title="arrivals per hour"), y2="p90:Q")
                      + base.mark_line(point=True, color="#d62728").encode(y="pred_arr_count:Q")
                      + base.mark_line(strokeDash=[4, 4], color="gray").encode(y="pred_schedule:Q"))
            if reveal:
                layers += base.mark_point(color="black", filled=True).encode(y="actual_arr_count:Q")
            st.altair_chart(layers.properties(height=280), width="stretch")
            st.caption("Red = forecast, shaded = P10–P90 range, dashed = schedule"
                       + (", dots = actual" if reveal else "") + ". Measured on 2025 (P2 leak-fixed model): 1–7% lower error than "
                       "reading the schedule well, 4–13% lower in bad-weather hours; the range is too narrow (covers "
                       "69–75%, should be 80%). " + SRC_BTS)

# ------------------------------------------------------------------ 4 how it works
with tabs[3]:
    sub = ev.dropna(subset=["feature_obs_time"])
    margin = (sub["sched_time_utc"] - pd.to_timedelta(sub["h"], unit="h") - sub["feature_obs_time"]) / pd.Timedelta(hours=1)
    st.metric("Rows where the weather used was observed ≥ h hours before the flight", f"{(margin >= 0).mean():.2%}",
              help=f"Checked live on {len(sub):,} rows.")
    st.markdown(f"""
- **Risk** = LightGBM trained on 2023–24 flights (departures: DEP_DELAY > 45, arrivals: ARR_DELAY > 45), tested on 2025.
- **Inputs** = hourly weather at the hub observed **at least h hours earlier** (+1 h because each report lands up to
  52 min after its timestamp), wind split into head/cross/tailwind on the runway ({HEADINGS}), and the
  schedule (hour, day, month, carrier, route). The flights file's own weather and anything known only after
  departure (actual times, delays, cancellations) are **never** used.
- **Alerts and recommendations** = explicit thresholds and rules, shown on screen. **Ranges** = statistics.
- **AI use:** {AI_USE}
- **Any airport:** add it to `config/airports.yaml`; trained on one hub and tested on the other, the model keeps
  most of its skill.
""")
    for doc in ["SOURCES.md", "ASSUMPTIONS.md"]:
        path = ROOT / doc
        with st.expander(f"{doc[:-3].title()} (read from {doc})"):
            st.markdown(path.read_text(encoding="utf-8") if path.exists() else f"{doc} not found in this checkout.")
    with st.expander("Rebuild everything (scripts, no manual steps)"):
        st.code("python scripts/run_pipeline.py --airports ATL DFW\npython tests/test_leakage.py\n"
                "python -m src.models.delay.train\npython src/eval/baselines_events.py\n"
                "python src/eval/evaluate_events.py --preds outputs/predictions/delay_preds.csv --name p3_final\n"
                "python -m src.models.aar_bts && python -m src.models.aar_dataset && python -m src.models.train_aar\n"
                "streamlit run app/app.py", language="bash")
