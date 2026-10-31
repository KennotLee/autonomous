"""Single entry point: build data/processed/{hourly,flight}_features.parquet.

Usage:
  .venv/bin/python scripts/run_pipeline.py --icao KATL KDFW --start 2023-01-01 --end 2025-07-31 \
      [--weather-scope hubs|all]

--weather-scope hubs : fetch weather only for the hub ICAOs; other-airport features are NaN (fast).
--weather-scope all  : also fetch every other airport seen on hub flights (slow first time, cached).
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from src.data.build import build_flight, build_hourly, weather_table  # noqa: E402
from src.data.config import INTERIM, PROCESSED, RAW_CSV, hub_airports, load_config  # noqa: E402
from src.data.flights import load_hub_flights  # noqa: E402
from src.data.weather import fetch_all, resolve_airports  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--icao", nargs="+", default=None, help="hub ICAOs from config/airports.yaml")
    ap.add_argument("--start", default=None, help="first FL_DATE to keep (YYYY-MM-DD)")
    ap.add_argument("--end", default=None, help="last FL_DATE to keep (YYYY-MM-DD)")
    ap.add_argument("--weather-scope", choices=["hubs", "all"], default="hubs")
    ap.add_argument("--raw", default=str(RAW_CSV))
    args = ap.parse_args()

    cfg = load_config()
    hubs = hub_airports(cfg, args.icao)
    hub_iatas = [v["iata"] for v in hubs.values()]
    INTERIM.mkdir(parents=True, exist_ok=True)
    PROCESSED.mkdir(parents=True, exist_ok=True)

    od = pd.read_csv(args.raw, usecols=["ORIGIN", "DEST"])
    od = od[od.ORIGIN.isin(hub_iatas) | od.DEST.isin(hub_iatas)]
    airports = resolve_airports(set(od.ORIGIN) | set(od.DEST), cfg)
    airports.loc[airports.icao.isin(hubs), "runway_heading_deg"] = [
        float(hubs[i]["runway_heading_deg"]) for i in airports.loc[airports.icao.isin(hubs), "icao"]]
    airports.loc[~airports.icao.isin(hubs), "runway_heading_deg"] = float("nan")
    airports.to_csv(INTERIM / "airports_resolved.csv", index=False)

    flights, val = load_hub_flights(args.raw, hub_iatas, airports)
    if args.start:
        flights = flights[flights.fl_date >= args.start]
    if args.end:
        flights = flights[flights.fl_date <= args.end]
    print("validation:", json.dumps(val))

    m = cfg["meteostat"]
    to_fetch = airports if args.weather_scope == "all" else airports[airports.icao.isin(hubs)]
    raw = fetch_all(to_fetch, m["weather_start"], m["weather_end"])
    print(f"weather: {sum(v is not None for v in raw.values())}/{len(airports)} airports have data")
    wx = weather_table(airports, raw, cfg)
    wx.to_parquet(INTERIM / "weather_features.parquet", index=False)  # used by tests/test_leakage.py

    hourly = build_hourly(flights, hubs, wx, cfg)
    hourly.to_parquet(PROCESSED / "hourly_features.parquet", index=False)
    flight = build_flight(flights, wx, cfg)
    flight.to_parquet(PROCESSED / "flight_features.parquet", index=False)
    meta = {"validation": val, "weather_scope": args.weather_scope, "hubs": list(hubs),
            "hourly_shape": hourly.shape, "flight_shape": flight.shape}
    (PROCESSED / "build_meta.json").write_text(json.dumps(meta, indent=1, default=str))
    print("hourly", hourly.shape, "flight", flight.shape)


if __name__ == "__main__":
    main()
