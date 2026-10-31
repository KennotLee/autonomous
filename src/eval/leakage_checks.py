"""P4 leakage review: how much of a "hub state over the last hours" feature is unknowable at issue time
when flights are bucketed by SCHEDULED hour (as in P3's pipeline/panel.py build_panel + ops_state).

A window [I - W, I) of scheduled times is read at issue time I = t - h. A flight scheduled at s with delay d
has finished at s + d; if s + d > I its final delay / >45 flag is not yet known. Assuming I - s ~ U(0, W]:
    P(unknown | d) = clip(d, 0, W) / W
Uses the hub flights of the raw CSV (ATL/DFW from config) as the delay distribution.

Usage: python src/eval/leakage_checks.py   ->  outputs/metrics/leakage_hub_state.csv
"""
import numpy as np
import pandas as pd

from baselines import OUT, ROOT, load_airports


def unknown_share(d, window_min):
    return np.clip(d, 0, window_min) / window_min


def main():
    iatas = [c["iata"] for c in load_airports().values()]
    df = pd.read_csv(ROOT / "data/raw/Flight_Data_With_Weather_Final.csv",
                     usecols=["ORIGIN", "DEST", "ARR_DELAY", "DEP_DELAY", "CANCELLED", "DIVERTED"])
    arr = df[df.DEST.isin(iatas) & (df.CANCELLED == 0) & (df.DIVERTED == 0)]
    dep = df[df.ORIGIN.isin(iatas) & (df.CANCELLED == 0)]
    sev = arr.ARR_DELAY[arr.ARR_DELAY > 45]
    arr_pos = arr.ARR_DELAY.clip(lower=0)
    dep_pos = dep.DEP_DELAY.dropna().clip(lower=0)
    rows = [
        {"feature": "hub_sev_rate_3h", "window_min": 180, "measure": "share of counted >45 min arrivals not yet landed",
         "n": len(sev), "unknown_share": unknown_share(sev, 180).mean()},
        {"feature": "hub_arr_delay_mean", "window_min": 60, "measure": "share of positive arrival-delay minutes not yet landed",
         "n": len(arr_pos), "unknown_share": (arr_pos * unknown_share(arr_pos, 60)).sum() / arr_pos.sum()},
        {"feature": "hub_dep_delay_mean", "window_min": 60, "measure": "share of positive departure-delay minutes not yet departed",
         "n": len(dep_pos), "unknown_share": (dep_pos * unknown_share(dep_pos, 60)).sum() / dep_pos.sum()},
    ]
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "leakage_hub_state.csv", index=False)
    print(res.to_string(index=False, float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()
