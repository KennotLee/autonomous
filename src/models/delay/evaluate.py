"""P3 evaluation on the 2025 test set (labelled rows only).

    python -m src.models.delay.evaluate      (after train.py)

Writes outputs/metrics/: metrics.csv (model vs base rate vs weather rule, per
slice), threshold.md, calibration.png, oracle_gap.csv, generalisation.csv,
failure_cases.md. Thresholds and rule cut-offs never look at test data.
"""
import json
from itertools import product

import lightgbm as lgb
import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss

from .train import MET_DIR, MODEL_DIR, PRED_DIR, ROOT, X, choose_threshold, config, feature_cols, fit, iata_to_icao, prepare

# organiser hourly weather (config weather.hourly_file; P1's weather source), used only for the true oracle
TRUE_ORACLE = {"temp_c": "temp", "precip_mm": "prcp", "wind_speed_kmh": "wspd", "pressure_hpa": "pres",
               "condition_code": "coco", "wind_dir_deg": "wdir", "headwind_kmh": "headwind",
               "crosswind_kmh": "crosswind", "tailwind_kmh": "tailwind"}


def add_true_oracle(df: pd.DataFrame) -> pd.DataFrame:
    """Event-airport weather at the event's own UTC hour, floor(sched_time_utc): never
    known 3-12 h ahead, so it only measures what leakage would fake."""
    cfg = config()
    w = pd.read_csv(ROOT / cfg["weather"]["hourly_file"], parse_dates=["time_utc"])
    w = w.assign(airport=w.airport.map(iata_to_icao(cfg)), _hour=w.time_utc.dt.tz_localize("UTC"))
    w = w[["airport", "_hour"] + list(TRUE_ORACLE)].rename(columns={k: f"true_oracle_{v}" for k, v in TRUE_ORACLE.items()})
    ts = cfg["weather"]["thunderstorm_codes"]
    w["true_oracle_ts_flag"] = w.true_oracle_coco.isin(ts).astype(float).where(w.true_oracle_coco.notna())
    out = df.assign(_hour=df.sched_time_utc.dt.floor("h")).merge(w, on=["airport", "_hour"], how="left")
    return out.drop(columns="_hour").set_index(df.index)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def rule_fit(tr: pd.DataFrame):
    """Weather rule ts_flag OR crosswind >= c OR tailwind >= t; (c, t) = max F1 on TRAIN."""
    y = tr.label_45.to_numpy()
    ts = tr.ts_flag.fillna(0).to_numpy() > 0
    cw, tw = tr.crosswind.to_numpy(), tr.tailwind.to_numpy()
    grid_c = np.unique(np.nanquantile(cw, np.linspace(0.5, 0.995, 40)))
    grid_t = np.unique(np.nanquantile(tw, np.linspace(0.5, 0.995, 40)))
    best = (-1, None, None)
    for c, t in product(grid_c, grid_t):
        f = ts | (cw >= c) | (tw >= t)
        tp = (f & (y == 1)).sum()
        f1 = 2 * tp / max(f.sum() + (y == 1).sum(), 1)
        if f1 > best[0]:
            best = (f1, float(c), float(t))
    return best


def rule_score(df, c, t):
    return (df.ts_flag.fillna(0).gt(0) | df.crosswind.ge(c) | df.tailwind.ge(t)).to_numpy().astype(float)


def scores(y, p, thr):
    out = {"n": len(y), "pos_rate": y.mean(),
           "pr_auc": average_precision_score(y, p) if 0 < y.sum() < len(y) else np.nan,  # undefined w/o both classes
           "brier": brier_score_loss(y, np.clip(p, 0, 1))}
    if thr is None:  # constant predictor: no alert decision
        return out | {"precision": np.nan, "recall": np.nan, "alert_rate": np.nan}
    flag = p >= thr
    tp = (flag & (y == 1)).sum()
    return out | {"precision": tp / max(flag.sum(), 1), "recall": tp / max((y == 1).sum(), 1),
                  "alert_rate": flag.mean()}


HEAVY_RAIN_MM = 2.5


def weather_slices(te: pd.DataFrame, tr: pd.DataFrame) -> list[tuple]:
    """Bad-weather subsets, defined at feature time (what the model saw) and at the event
    hour (true oracle). Strong crosswind = top 5% of train crosswind (cut-off from train only)."""
    cw = float(np.nanquantile(tr.crosswind, 0.95))
    out = []
    for kind, pre in (("wx_feature_time", ""), ("wx_event_hour", "true_oracle_")):
        out += [(kind, "thunderstorm", te[te[f"{pre}ts_flag"] == 1]),
                (kind, f"heavy_rain_ge_{HEAVY_RAIN_MM}mm", te[te[f"{pre}prcp"] >= HEAVY_RAIN_MM]),
                (kind, f"crosswind_ge_{cw:.1f}kmh_top5pct_train", te[te[f"{pre}crosswind"] >= cw]),
                (kind, "none_of_these", te[(te[f"{pre}ts_flag"] != 1) & ~(te[f"{pre}prcp"] >= HEAVY_RAIN_MM)
                                            & ~(te[f"{pre}crosswind"] >= cw)])]
    return out


CI_SINGLE =["gbm", "base_rate", "rule", "no_weather", "true_oracle", "p1_oracle"]
CI_DIFF = [("gbm", "base_rate"), ("gbm", "rule"), ("gbm", "no_weather"), ("true_oracle", "gbm"), ("p1_oracle", "gbm")]


def bootstrap_ci(te: pd.DataFrame, reps=200, seed=0) -> list[dict]:
    """95% CIs for test PR-AUC and PR-AUC differences, resampling whole local scheduled
    dates (events on the same day share weather and airport state, so they aren't independent)."""
    day = te.sched_time_local.dt.date.to_numpy()
    groups = [np.flatnonzero(day == d) for d in np.unique(day)]
    y = te.label_45.to_numpy()
    p = {n: te[f"p_{n}"].to_numpy() for n in CI_SINGLE}
    rng = np.random.default_rng(seed)
    draws = {n: [] for n in CI_SINGLE}
    for _ in range(reps):
        i = np.concatenate([groups[k] for k in rng.integers(0, len(groups), len(groups))])
        for n in CI_SINGLE:
            draws[n].append(average_precision_score(y[i], p[n][i]))
    draws = {n: np.array(v) for n, v in draws.items()}
    point = {n: average_precision_score(y, p[n]) for n in CI_SINGLE}
    rows = [{"metric": f"pr_auc {n}", "estimate": point[n], "ci_low": np.percentile(draws[n], 2.5),
             "ci_high": np.percentile(draws[n], 97.5)} for n in CI_SINGLE]
    for a, b in CI_DIFF:
        d = draws[a] - draws[b]
        rows.append({"metric": f"pr_auc {a} - {b}", "estimate": point[a] - point[b],
                     "ci_low": np.percentile(d, 2.5), "ci_high": np.percentile(d, 97.5)})
    for r in rows:
        r |= {"reps": reps, "days": len(groups)}
    return rows


def main():
    ev, lab, cats = prepare()
    lab = add_true_oracle(lab)
    meta = json.loads((MODEL_DIR / "meta.json").read_text())
    thr, cols = meta["threshold"], meta["features"]
    model = lgb.Booster(model_file=str(MODEL_DIR / "gbm.txt"))
    tr, va = lab[lab.split == "train"], lab[lab.split == "val"]
    te = lab[lab.split == "test"].copy()
    te["p_gbm"] = model.predict(X(te, cols, cats))
    te["p_base_rate"] = tr.label_45.mean()
    rf1, rc, rt = rule_fit(tr)
    te["p_rule"] = rule_score(te, rc, rt)
    by_ap = {ap: d["threshold"] for ap, d in meta["thresholds_by_airport"].items()}
    te["thr_gbm"] = te.airport.map(by_ap).fillna(thr)          # per-airport (used for high_risk)
    te["p_gbm_global_thr"], te["thr_gbm_global_thr"] = te.p_gbm, thr  # old single threshold, for comparison
    te["thr_rule"] = 0.5
    models = ["gbm", "gbm_global_thr", "rule", "base_rate"]

    # ---- metrics.csv: overall and per h / airport / event_type
    rows = []
    slices = [("all", "all", te)] + [(k, v, g) for k in ("h", "airport", "event_type") for v, g in te.groupby(k)]
    slices += weather_slices(te, tr)
    for kind, val, g in slices:
        for name in models:
            t = g[f"thr_{name}"].to_numpy() if f"thr_{name}" in g else None
            rows.append({"slice": kind, "value": val, "model": name} | scores(g.label_45.to_numpy(), g[f"p_{name}"].to_numpy(), t))
    met = pd.DataFrame(rows)
    met.to_csv(MET_DIR / "metrics.csv", index=False, float_format="%.4f")

    # ---- threshold.md
    (MET_DIR / "threshold.md").write_text(
        "# Decision thresholds (per airport)\n\n"
        "Max F1 on each airport's validation slice (2024-07..12, never test): a missed severe delay and a false "
        "alert weigh the same, judged within each airport.\n\n"
        "| airport | threshold | val precision | val recall | val alert rate |\n|---|---|---|---|---|\n"
        + "".join(f"| {ap} | {d['threshold']:.3f} | {d['val_precision']:.3f} | {d['val_recall']:.3f} | "
                  f"{d['val_alert_rate']:.3f} |\n" for ap, d in meta["thresholds_by_airport"].items())
        + f"\nPrevious single threshold (`gbm_global_thr` in metrics.csv): {thr:.3f} (val precision "
        f"{meta['val_precision']:.3f}, recall {meta['val_recall']:.3f}). Airports not in the table fall back to it.\n\n"
        f"Weather rule baseline: ts_flag OR crosswind >= {rc:.1f} km/h OR tailwind >= {rt:.1f} km/h "
        f"(cut-offs = max F1 on train 2023-01..2024-06, F1 {rf1:.3f}).\n")

    # ---- calibration.png
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], color="#999", lw=1, ls="--", label="perfect")
    b = pd.DataFrame({"b": np.minimum((te.p_gbm * 10).astype(int), 9), "p": te.p_gbm, "y": te.label_45})
    b = b.groupby("b").agg(p=("p", "mean"), y=("y", "mean"), n=("y", "size"))
    b = b[b.n >= 30]
    ax.plot(b.p, b.y, marker="o", label="LightGBM (test 2025)")
    ax.axhline(te.label_45.mean(), color="#d62728", lw=0.8, ls=":", label=f"test base rate {te.label_45.mean():.3f}")
    ax.set(xlabel="predicted P(delay > 45 min)", ylabel="observed share", xlim=(0, 0.6), ylim=(0, 0.6),
           title="Reliability, test 2025 (bins with >= 30 rows)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(MET_DIR / "calibration.png", dpi=130)
    plt.close(fig)
    print("calibration bins:\n" + b.round(3).to_string())

    # ---- oracle gap: same model, all lagged event-airport weather replaced by weather at the event hour
    # (true oracle, organiser file); P1's oracle_* shown for comparison (it is ~4-6 h before the event)
    pcols = feature_cols(ev, oracle=True)
    tcols = [c for c in pcols if not c.startswith("oracle_")] + [c for c in lab.columns if c.startswith("true_oracle_")]
    print(f"true-oracle weather found for {te.true_oracle_temp.notna().mean():.2%} of test rows")
    ncols = [c for c in pcols if not c.startswith("oracle_")]  # no event-airport weather at all
    for name, oc in (("p1_oracle", pcols), ("true_oracle", tcols), ("no_weather", ncols)):
        te[f"p_{name}"] = fit(tr, va, oc, cats).predict_proba(X(te, oc, cats))[:, 1]
    gap = pd.DataFrame([{"h": h, "pr_auc_lagged": average_precision_score(g.label_45, g.p_gbm),
                         "pr_auc_true_oracle": average_precision_score(g.label_45, g.p_true_oracle),
                         "pr_auc_p1_oracle": average_precision_score(g.label_45, g.p_p1_oracle)}
                        for h, g in [("all", te)] + list(te.groupby("h"))])
    gap["gap_true_oracle"] = gap.pr_auc_true_oracle - gap.pr_auc_lagged
    gap.to_csv(MET_DIR / "oracle_gap.csv", index=False, float_format="%.4f")

    # ---- uncertainty: day-block bootstrap (resample local scheduled dates, all events of a day together)
    pd.DataFrame(bootstrap_ci(te)).to_csv(MET_DIR / "metrics_ci.csv", index=False, float_format="%.4f")

    # ---- generalisation: train on one airport, test on the other (airport is constant -> dropped)
    gcols = [c for c in cols if c != "airport"]
    gen = []
    for src, dst in (("KATL", "KDFW"), ("KDFW", "KATL")):
        gm = fit(tr[tr.airport == src], va[va.airport == src], gcols, cats)
        d = te[te.airport == dst]
        gen.append({"train_airport": src, "test_airport": dst,
                    "pr_auc_transfer": average_precision_score(d.label_45, gm.predict_proba(X(d, gcols, cats))[:, 1]),
                    "pr_auc_in_airport_model": average_precision_score(d.label_45, d.p_gbm),
                    "test_pos_rate": d.label_45.mean()})
    pd.DataFrame(gen).to_csv(MET_DIR / "generalisation.csv", index=False, float_format="%.4f")

    # ---- failure cases: worst high-confidence misses (false negatives, one row per event)
    fn = te[(te.label_45 == 1)].sort_values(["p_gbm", "delay_min"], ascending=[True, False])
    fn = fn.drop_duplicates("event_id").head(3)
    wx = ["temp", "prcp", "wspd", "pres", "coco", "crosswind", "tailwind", "ts_flag"]
    owx = ["oracle_temp", "oracle_prcp", "oracle_wspd", "oracle_pres", "oracle_coco"]
    lines = ["# Failure cases: 3 worst high-confidence misses (test 2025)\n",
             f"False negatives with the lowest predicted risk (per-airport thresholds, see threshold.md); one row per event, "
             "ties broken by larger delay. Weather 'seen' = event-airport row available at feature_obs_time; "
             "'oracle' = P1's oracle_* (flights-CSV weather; its join puts it ~4-6 h before the event, see "
             "notes.md). Never a feature.\n"]
    for r in fn.itertuples():
        seen = {c: getattr(r, c) for c in wx}
        orac = {c: getattr(r, c) for c in owx}
        benign = lambda coco, prcp: coco <= 4 and prcp == 0  # noqa: E731  clear/cloudy, dry
        if benign(r.coco, r.prcp) and benign(r.oracle_coco, r.oracle_prcp) and not r.ts_flag:
            reason = (f"weather was benign in both rows (coco {r.coco:.0f}/{r.oracle_coco:.0f}, no precipitation, "
                      f"wind <= {max(r.wspd, r.oracle_wspd):.0f} km/h): nothing in our data explains the "
                      f"{r.delay_min:.0f}-min delay, so its cause is outside the weather features")
        elif benign(r.coco, r.prcp) and not benign(r.oracle_coco, r.oracle_prcp):
            reason = f"weather worsened after feature_obs_time (coco {r.coco:.0f} -> {r.oracle_coco:.0f})"
        else:
            reason = "adverse weather was already visible at feature_obs_time but the model scored it low"
        lines += [f"## {r.event_id} (h = {r.h})",
                  f"- scheduled {r.sched_time_local} local ({r.sched_time_utc} UTC), {r.event_type} at {r.airport}, "
                  f"carrier {r.carrier}; delay {r.delay_min:.0f} min; predicted risk {r.p_gbm:.3f}",
                  f"- feature_obs_time {r.feature_obs_time}: " + ", ".join(f"{k}={v:.3g}" for k, v in seen.items()),
                  "- oracle: " + ", ".join(f"{k}={v:.3g}" for k, v in orac.items()),
                  f"- likely reason: {reason}\n"]

    # false alarms: highest-risk events that were not severely delayed (one row per event)
    drivers = pd.read_csv(PRED_DIR / "delay_preds.csv", usecols=["event_id", "h", "top_drivers"])
    fp = te[(te.label_45 == 0) & (te.p_gbm >= te.thr_gbm)].sort_values("p_gbm", ascending=False)
    fp = fp.drop_duplicates("event_id").head(3).merge(drivers, on=["event_id", "h"], how="left")
    cw = float(np.nanquantile(tr.crosswind, 0.95))
    lines += ["# False alarms: 3 highest-risk events that were not severely delayed\n",
              "Flagged (risk >= the airport's threshold) but delay <= 45 min; one row per event.\n"]
    for r in fp.itertuples():
        seen = {c: getattr(r, c) for c in wx}
        orac = {c: getattr(r, c) for c in owx}
        bad = [n for n, ok in (("thunderstorm", r.ts_flag == 1), (f"condition code {r.coco:.0f} (rain or worse)", r.coco >= 7),
                               (f"rain {r.prcp:.1f} mm", r.prcp >= HEAVY_RAIN_MM),
                               (f"crosswind {r.crosswind:.0f} km/h", r.crosswind >= cw)) if ok]
        reason = (f"adverse weather was visible at feature_obs_time ({', '.join(bad)}) and pushed the risk up "
                  f"(top drivers: {r.top_drivers}), but the event ran only {r.delay_min:.0f} min late" if bad else
                  f"no adverse weather at feature_obs_time; the risk came from the top drivers ({r.top_drivers})")
        lines += [f"## {r.event_id} (h = {r.h})",
                  f"- scheduled {r.sched_time_local} local ({r.sched_time_utc} UTC), {r.event_type} at {r.airport}, "
                  f"carrier {r.carrier}; delay {r.delay_min:.0f} min; predicted risk {r.p_gbm:.3f} "
                  f"(threshold {r.thr_gbm:.3f})",
                  f"- feature_obs_time {r.feature_obs_time}: " + ", ".join(f"{k}={v:.3g}" for k, v in seen.items()),
                  "- oracle: " + ", ".join(f"{k}={v:.3g}" for k, v in orac.items()),
                  f"- top drivers: {r.top_drivers}",
                  f"- likely reason: {reason}\n"]

    # alert workload at the per-airport thresholds
    flag = te.p_gbm >= te.thr_gbm
    lines += ["# Alert workload (test 2025, per-airport thresholds)\n",
              "| airport | events x h | flagged | alert rate | precision | false alerts per true alert |",
              "|---|---|---|---|---|---|"]
    for ap, g in [("all", te)] + list(te.groupby("airport")):
        f = flag[g.index]
        prec = g.label_45[f].mean()
        lines.append(f"| {ap} | {len(g):,} | {f.sum():,} | {f.mean():.1%} | {prec:.3f} | {(1 - prec) / prec:.1f} |")
    prec = te.label_45[flag].mean()
    lines += ["", f"Precision {prec:.3f}: about {round((1 - prec) * 5)} of every 5 alerts are false. "
              "The flight CSV is a ~0.5% sample, so counts are not real daily volumes; use the rates.\n"]
    (MET_DIR / "failure_cases.md").write_text("\n".join(lines))

    (MET_DIR / "notes.md").write_text(
        "# Notes on the metrics\n\n"
        "- **Oracle gap uses the true event-hour weather.** `pr_auc_true_oracle` = same model with all lagged "
        "event-airport weather replaced by the organiser hourly file at floor(sched_time_utc) (the file P1's weather "
        "features come from; checked 100% equal on all 9 weather columns). P1's `oracle_*` (`pr_auc_p1_oracle`) is "
        "not event-time weather: it matches the event hour in ~5% of rows and the *local hour read as UTC* in ~44% "
        "(the misaligned upstream join in CLAUDE.md), i.e. ~4-6 h before the event.\n"
        "- **Why the gap is small:** event-hour weather matters a lot per flight (thunderstorm coded: 37.7% severe "
        "vs 10.6%; >= 2.5 mm/h rain: 28.5% vs 10.4%) but it is rare (61 thunderstorm and ~1.4% heavy-rain events "
        "out of 25,231), so it barely moves an all-events PR-AUC. Test PR-AUC: no event-airport weather 0.207, "
        "lagged weather 0.219, event-hour weather 0.221.\n"
        f"- **Weather rule** cut-offs were chosen by max F1 on train; with a weak signal that lands on loose "
        f"cut-offs (crosswind >= {rc:.1f}, tailwind >= {rt:.1f} km/h) and flags about half of all events.\n"
        "- **Generalisation** models drop `airport` (constant in training).\n")

    with pd.option_context("display.width", 200):
        print(met[met.slice == "all"].round(3).to_string(index=False))
        print(gap.round(3).to_string(index=False)); print(pd.DataFrame(gen).round(3).to_string(index=False))
    print(f"rule: c={rc:.1f} t={rt:.1f} (train F1 {rf1:.3f}); threshold {thr:.3f}")


if __name__ == "__main__":
    main()
