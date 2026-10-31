"""Delay-model inputs: P1's flight_features.parquet plus P3 hub features.

P1 contract (CLAUDE.md, branch p1): one row per flight x h, t = sched_time_utc
(scheduled departure), weather at both ends as of t - h, label_45 = ARR_DELAY > 45
(NaN if cancelled/diverted), oracle_* = same-hour weather (oracle only).

P3 adds, per flight and h, for the hub end (ATL/DFW) of the flight:
  hub_sched_arr / hub_sched_dep  scheduled movements at the hub in the hour the
                                 flight uses it (schedule: known in advance)
  hub_<ops>                      hub state in the last full hour before t - h:
                                 capacity ratio, severe-delay share, cancellations,
                                 mean arrival/departure delay (3 h pooled)
  {origin,dest}_ts, _heavy_precip  flags derived from P1's coco
The hub features need the full BTS months (python -m pipeline.download); the
team CSV is a 0.5% sample, so its own counts can't measure congestion.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.features import ops_state
from pipeline.panel import add_local_times, build_panel, load_flights

from . import airports

ROOT = Path(__file__).resolve().parents[3]
P1_FILE = ROOT / "data" / "processed" / "flight_features.parquet"
BTS_DIR = ROOT / "data" / "raw" / "bts"
HORIZONS = [3, 6, 9, 12]
OPS_FEATS = ["cap_ratio_3h", "sev_rate_3h", "cancel_rate_3h", "arr_delay_mean", "dep_delay_mean"]
COCO_TS = [23, 24, 25, 26, 27]
COCO_HEAVY = [9, 11, 13, 16, 18, 20, 22, 26]

# P1 columns that are targets/outcomes/ids, never features
TARGETS = ["delay_min", "label_45", "cancelled", "diverted"]
IDS = ["flight_id", "origin_icao", "dest_icao", "sched_time_utc", "sched_arr_utc", "feature_obs_time",
       "origin_feature_obs_time", "dest_feature_obs_time"]


def load_p1() -> pd.DataFrame:
    fl = pd.read_parquet(P1_FILE)
    hubs = airports.hubs()
    fl["hub"] = np.where(fl.dest_icao.isin(hubs), fl.dest_icao, fl.origin_icao)
    fl["arr_to_hub"] = fl.dest_icao.isin(hubs).astype(int)
    for end in ("origin", "dest"):
        fl[f"{end}_ts"] = fl[f"{end}_coco"].isin(COCO_TS).astype(float).where(fl[f"{end}_coco"].notna())
        fl[f"{end}_heavy_precip"] = fl[f"{end}_coco"].isin(COCO_HEAVY).astype(float).where(fl[f"{end}_coco"].notna())
    return fl


def _hub_panels() -> pd.DataFrame:
    flights = add_local_times(load_flights("bts"))
    frames = []
    for icao, a in airports.hubs().items():
        panel = build_panel(flights, a)
        idx = pd.date_range(panel.index.min(), panel.index.max(), freq="h", name="hour_utc")
        p = panel.reindex(idx)
        o = ops_state(p)[OPS_FEATS].astype("float32")
        o["sched_arr"], o["sched_dep"] = p.sched_arr, p.sched_dep
        o["hub"] = icao
        frames.append(o.reset_index())
    return pd.concat(frames, ignore_index=True)


def add_hub_features(fl: pd.DataFrame) -> pd.DataFrame:
    if not any(BTS_DIR.glob("*.parquet")):
        print("no BTS files: skipping hub congestion/state features")
        return fl
    hub = _hub_panels()
    # congestion at the hub hour the flight uses (arrival hour if inbound, else departure hour)
    fl["_hub_hour"] = np.where(fl.arr_to_hub == 1, fl.sched_arr_utc.dt.floor("h"), fl.sched_time_utc.dt.floor("h"))
    fl["_hub_hour"] = pd.to_datetime(fl._hub_hour, utc=True)
    sched = hub[["hub", "hour_utc", "sched_arr", "sched_dep"]].rename(
        columns={"hour_utc": "_hub_hour", "sched_arr": "hub_sched_arr", "sched_dep": "hub_sched_dep"})
    fl = fl.merge(sched, on=["hub", "_hub_hour"], how="left")
    # live hub state: hour bucket [k, k+1) with k + 1 <= t - h, so it is complete at issue time
    fl["_ops_hour"] = (fl.sched_time_utc - pd.to_timedelta(fl.h, unit="h")).dt.floor("h") - pd.Timedelta(hours=1)
    ops = hub[["hub", "hour_utc"] + OPS_FEATS].rename(
        columns={"hour_utc": "_ops_hour"} | {c: f"hub_{c}" for c in OPS_FEATS})
    fl = fl.merge(ops, on=["hub", "_ops_hour"], how="left")
    return fl.drop(columns=["_hub_hour", "_ops_hour"])


def feature_cols(df: pd.DataFrame, oracle=False, p1_only=False) -> list[str]:
    cols = [c for c in df.columns if c not in TARGETS + IDS + ["split"]]
    if not oracle:
        cols = [c for c in cols if not c.startswith("oracle_")]
    if p1_only:
        cols = [c for c in cols if not c.startswith("hub_") or c == "hub"]
    return cols
