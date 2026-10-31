"""Single entry point: build data/processed/event_features.parquet (+ a 5-row sample CSV).

Usage:
  .venv/bin/python scripts/run_pipeline.py --airports ATL DFW [--start 2023-01-01 --end 2025-07-31]
      [--raw data/raw/Flight_Data_With_Weather_Final.csv] [--weather data/raw/challenge5_hourly_weather_ATL_DFW.csv]
Airports may be given as IATA or ICAO; they must be configured in config/airports.yaml and present in
the hourly weather file (otherwise their weather features are NaN).
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from src.data.build_events import build_event_features  # noqa: E402
from src.data.config import INTERIM, PROCESSED, RAW_CSV, REPO, hub_airports, load_config  # noqa: E402
from src.data.events import build_events  # noqa: E402
from src.data.hourly_weather import load_hourly  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--airports", "--icao", nargs="+", default=None, help="IATA or ICAO codes (default: all configured)")
    ap.add_argument("--start", default=None, help="first local flight date to keep (YYYY-MM-DD)")
    ap.add_argument("--end", default=None, help="last local flight date to keep (YYYY-MM-DD)")
    ap.add_argument("--raw", default=str(RAW_CSV))
    ap.add_argument("--weather", default=None)
    ap.add_argument("--out", default=str(PROCESSED / "event_features.parquet"))
    args = ap.parse_args()

    cfg = load_config()
    hubs = hub_airports(cfg, args.airports)
    wx_path = args.weather or str(REPO / cfg["weather"]["hourly_file"])
    INTERIM.mkdir(parents=True, exist_ok=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    ev, stats = build_events(args.raw, hubs, cfg)
    if args.start:
        ev = ev[ev.sched_time_local >= pd.Timestamp(args.start)]
    if args.end:
        ev = ev[ev.sched_time_local < pd.Timestamp(args.end) + pd.Timedelta(days=1)]
    wx = load_hourly(wx_path, cfg)
    wx.to_parquet(INTERIM / "hourly_weather_features.parquet", index=False)  # used by tests/test_leakage.py
    feats = build_event_features(ev, wx, cfg["airports"], cfg)
    feats.to_parquet(out, index=False)
    feats.head(5).to_csv(out.with_name("event_features_sample5.csv"), index=False)

    meta = {"stats": stats, "hubs": list(hubs), "rows": len(feats), "cols": feats.shape[1],
            "weather_file": wx_path, "weather_coverage": round(float(feats.temp.notna().mean()), 4)}
    (out.parent / "build_meta.json").write_text(json.dumps(meta, indent=1, default=str))
    print(json.dumps(meta, default=str))


if __name__ == "__main__":
    main()
