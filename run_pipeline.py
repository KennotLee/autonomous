"""End-to-end: flights + weather -> hourly panel -> leakage-safe features -> baselines/model metrics.

    python -m pipeline.download                # once: fetch BTS + Meteostat
    python run_pipeline.py --airports KATL KDFW
    python run_pipeline.py --flights path/to/Flight_Data_With_Weather_Final.csv   # sample file
"""
import argparse

import pandas as pd

from pipeline.config import INTERIM, PROCESSED, REPORTS, get_airport
from pipeline.download import download_weather
from pipeline.evaluate import (aar_predictions, calibration_table, rmse, sev_metrics,
                               sev_predictions, split)
from pipeline.features import build_dataset
from pipeline.panel import add_local_times, build_panel, load_flights

AAR_MODELS = ["schedule", "persistence", "climatology", "gbm", "gbm_oracle"]
SEV_MODELS = ["climatology", "persistence", "gbm", "gbm_oracle"]


def build(airports, flights_src, wx_start, wx_end):
    flights = add_local_times(load_flights(flights_src))
    panels, sets = [], []
    for code in airports:
        a = get_airport(code)
        panel = build_panel(flights, a)
        wx = download_weather(a.icao, wx_start, wx_end)
        panels.append(panel)
        sets.append(build_dataset(panel, wx, a))
        print(f"{a.icao}: {len(panel):,} hours, {panel.sched_arr.sum():,} scheduled arrivals, "
              f"mean {panel.sched_arr.mean():.1f}/h")
    for d in (INTERIM, PROCESSED, REPORTS):
        d.mkdir(parents=True, exist_ok=True)
    panel = pd.concat(panels)
    panel.to_parquet(INTERIM / "panel.parquet")
    ds = pd.concat(sets, ignore_index=True)
    ds.to_parquet(PROCESSED / "dataset.parquet")
    return panel, ds


def aar_tables(pred: pd.DataFrame) -> dict[str, pd.DataFrame]:
    def table(df, by):
        return df.groupby(by).apply(
            lambda g: pd.Series({m: rmse(g.y_act_arr, g[m]) for m in AAR_MODELS} | {"n_hours": len(g)}),
            include_groups=False)

    busy_cut = pred.groupby("airport").sched_arr_T.transform(lambda s: s.quantile(0.75))
    pred = pred.assign(
        subset="all hours",
        ratio=pred.y_act_arr / pred.sched_arr_T.where(pred.sched_arr_T > 0))
    subsets = pd.concat([
        pred,
        pred[pred.sched_arr_T >= busy_cut].assign(subset="busy hours (top 25% demand)"),
        # capacity actually reduced: where the forecast matters most
        pred[(pred.sched_arr_T >= busy_cut) & (pred.ratio < 0.85)].assign(subset="busy & ratio<0.85"),
    ])
    return {
        "aar_by_airport": table(subsets, ["subset", "airport"]),
        "aar_by_horizon": table(pred, ["airport", "h"]),
    }


def sev_tables(pred: pd.DataFrame) -> dict[str, pd.DataFrame]:
    by_ap = pd.DataFrame({ap: sev_metrics(g, SEV_MODELS) for ap, g in pred.groupby("airport")}).T
    by_h = pd.DataFrame({h: sev_metrics(g, SEV_MODELS) for h, g in pred.groupby("h")}).T
    return {"sev_by_airport": by_ap, "sev_by_horizon": by_h,
            "sev_calibration_gbm": calibration_table(pred, "gbm")}


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--airports", nargs="+", default=["KATL", "KDFW"])
    p.add_argument("--flights", default="bts", help="'bts' or a CSV path")
    p.add_argument("--wx-start", default="2023-01")
    p.add_argument("--wx-end", default="2025-12")
    args = p.parse_args(argv)

    panel, ds = build(args.airports, args.flights, args.wx_start, args.wx_end)
    train, test = split(ds)
    print(f"train {train.local_time.min():%Y-%m-%d}..{train.local_time.max():%Y-%m-%d} ({len(train):,} rows), "
          f"test {test.local_time.min():%Y-%m-%d}..{test.local_time.max():%Y-%m-%d} ({len(test):,} rows)")

    aar = aar_predictions(train, test)
    sev = sev_predictions(train, test)
    aar.to_parquet(REPORTS / "aar_predictions.parquet")
    sev.to_parquet(REPORTS / "sev_predictions.parquet")

    tables = aar_tables(aar) | sev_tables(sev)
    with open(REPORTS / "metrics.md", "w") as f:
        f.write("# Metrics (test = 2025, train = 2023-24)\n\n")
        for name, t in tables.items():
            f.write(f"## {name}\n\n{t.round(4).to_markdown()}\n\n")
            print(f"\n== {name}\n{t.round(3).to_string()}")
    print(f"\nwrote {REPORTS / 'metrics.md'}")


if __name__ == "__main__":
    main()
