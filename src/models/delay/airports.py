"""Airport metadata for both ends of every flight.

Hubs (with runway headings) come from config/airports.yaml (P1). Other
airports only need a timezone and a weather station, taken from the
`airportsdata` package, so any airport in BTS works without editing code.
"""
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]

# airportsdata gaps found in the 2023-25 ATL/DFW network
_MANUAL = {"PBI": ("KPBI", 26.6832, -80.0956, "America/New_York")}


@dataclass(frozen=True)
class Airport:
    icao: str
    iata: str
    tz: str
    lat: float
    lon: float
    runway_heading: float | None = None   # only for configured hubs


@cache
def hubs() -> dict[str, Airport]:
    """Configured hubs keyed by IATA code."""
    cfg = yaml.safe_load((ROOT / "config" / "airports.yaml").read_text())["airports"]
    out = {}
    for icao, c in cfg.items():
        a = lookup(c["iata"])
        out[c["iata"]] = Airport(icao, c["iata"], c["timezone"], a.lat, a.lon,
                                 float(c["runway_heading_deg"]))
    return out


@cache
def _db():
    import airportsdata
    return airportsdata.load("IATA")


def lookup(iata: str) -> Airport:
    if iata in _MANUAL:
        icao, lat, lon, tz = _MANUAL[iata]
        return Airport(icao, iata, tz, lat, lon)
    r = _db()[iata]
    return Airport(r["icao"], iata, r["tz"], r["lat"], r["lon"])


def get(iata: str) -> Airport:
    return hubs().get(iata) or lookup(iata)
