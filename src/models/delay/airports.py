"""Hub airports from config/airports.yaml (P1 owns the file)."""
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class Airport:
    icao: str
    iata: str
    tz: str
    runway_heading: float


@cache
def hubs() -> dict[str, Airport]:
    """Configured hubs keyed by ICAO."""
    cfg = yaml.safe_load((ROOT / "config" / "airports.yaml").read_text())["airports"]
    return {icao: Airport(icao, c["iata"], c["timezone"], float(c["runway_heading_deg"]))
            for icao, c in cfg.items()}
