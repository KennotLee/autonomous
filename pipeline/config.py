"""Airport registry and project-wide constants.

Every airport-specific number lives here, so the pipeline runs for any airport
once it has an entry. Add one with `register_airport()` or by editing AIRPORTS.
"""
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = DATA / "raw"
INTERIM = DATA / "interim"
PROCESSED = DATA / "processed"
REPORTS = ROOT / "reports"

KMH_PER_KT = 1.852
SEVERE_DELAY_MIN = 45
HORIZONS = list(range(3, 13))          # forecast lead times in hours
TRAIN_YEARS = (2023, 2024)
TEST_YEARS = (2025,)


@dataclass(frozen=True)
class Airport:
    icao: str
    iata: str              # BTS uses IATA codes
    tz: str
    lat: float
    lon: float
    runway_heading: float  # assumed primary arrival heading, degrees
    wx_station: str | None = None  # Meteostat station id; None = nearest to lat/lon


AIRPORTS: dict[str, Airport] = {
    "KATL": Airport("KATL", "ATL", "America/New_York", 33.6407, -84.4277, 80.0, "72219"),
    "KDFW": Airport("KDFW", "DFW", "America/Chicago", 32.8998, -97.0403, 180.0, "72259"),
    # Extra airports (headings are approximate); used for generalisation checks.
    "KDEN": Airport("KDEN", "DEN", "America/Denver", 39.8561, -104.6737, 170.0, "72565"),
    "KORD": Airport("KORD", "ORD", "America/Chicago", 41.9742, -87.9073, 270.0, "72530"),
}


def register_airport(a: Airport) -> None:
    AIRPORTS[a.icao] = a


def get_airport(code: str) -> Airport:
    """Look up an airport by ICAO (KATL) or IATA (ATL) code."""
    code = code.upper()
    if code in AIRPORTS:
        return AIRPORTS[code]
    for a in AIRPORTS.values():
        if a.iata == code:
            return a
    raise KeyError(f"{code} not in registry; add it with register_airport()")
