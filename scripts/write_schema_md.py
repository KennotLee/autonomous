"""Render data/schema_profile.csv + a hand-written meanings map into a markdown table (stdout).
Usage: python3 -I scripts/write_schema_md.py data/schema_profile.csv > /tmp/schema.md
Meanings for BTS columns follow the BTS "Reporting Carrier On-Time Performance" field definitions.
Weather units are INFERRED (column names + value ranges match Meteostat hourly) and NOT confirmed.
"""
import sys
import pandas as pd

M = {
 "YEAR": ("", "Year of flight date"), "QUARTER": ("", "Quarter 1-4"), "MONTH": ("", "Month 1-12"),
 "DAY_OF_MONTH": ("", "Day of month"), "DAY_OF_WEEK": ("", "1=Mon..7=Sun (BTS)"),
 "FL_DATE": ("local date", "Flight date (scheduled departure, local)"),
 "OP_UNIQUE_CARRIER": ("", "Operating carrier code"), "OP_CARRIER_AIRLINE_ID": ("", "DOT carrier id"),
 "OP_CARRIER": ("", "Carrier code"), "TAIL_NUM": ("", "Aircraft tail number"),
 "OP_CARRIER_FL_NUM": ("", "Flight number"),
 "ORIGIN_AIRPORT_ID": ("", "DOT airport id"), "ORIGIN_AIRPORT_SEQ_ID": ("", "DOT airport seq id"),
 "ORIGIN_CITY_MARKET_ID": ("", "DOT city market id"), "ORIGIN": ("IATA", "Origin airport (IATA, e.g. ATL)"),
 "ORIGIN_CITY_NAME": ("", "Origin city"), "ORIGIN_STATE_ABR": ("", "Origin state"),
 "ORIGIN_STATE_FIPS": ("", "State FIPS"), "ORIGIN_STATE_NM": ("", "State name"), "ORIGIN_WAC": ("", "World area code"),
 "DEST_AIRPORT_ID": ("", "DOT airport id"), "DEST_AIRPORT_SEQ_ID": ("", "DOT airport seq id"),
 "DEST_CITY_MARKET_ID": ("", "DOT city market id"), "DEST": ("IATA", "Destination airport (IATA)"),
 "DEST_CITY_NAME": ("", "Dest city"), "DEST_STATE_ABR": ("", "Dest state"), "DEST_STATE_FIPS": ("", "State FIPS"),
 "DEST_STATE_NM": ("", "State name"), "DEST_WAC": ("", "World area code"),
 "CRS_DEP_TIME": ("hhmm local", "Scheduled departure time"), "DEP_TIME": ("hhmm local", "Actual gate departure"),
 "DEP_DELAY": ("min", "Dep delay, signed (early<0)"), "DEP_DELAY_NEW": ("min", "Dep delay, early clipped to 0"),
 "DEP_DEL15": ("0/1", "Dep delay >=15"), "DEP_DELAY_GROUP": ("15-min bins", "Dep delay group"),
 "DEP_TIME_BLK": ("hhmm-hhmm local", "Scheduled dep hour block"), "TAXI_OUT": ("min", "Taxi-out time"),
 "WHEELS_OFF": ("hhmm local", "Wheels-off (runway takeoff)"), "WHEELS_ON": ("hhmm local", "Wheels-on (runway touchdown)"),
 "TAXI_IN": ("min", "Taxi-in time"), "CRS_ARR_TIME": ("hhmm local", "Scheduled arrival time"),
 "ARR_TIME": ("hhmm local", "Actual gate arrival"), "ARR_DELAY": ("min", "Arr delay, signed"),
 "ARR_DELAY_NEW": ("min", "Arr delay, early clipped to 0"), "ARR_DEL15": ("0/1", "Arr delay >=15"),
 "ARR_DELAY_GROUP": ("15-min bins", "Arr delay group"), "ARR_TIME_BLK": ("hhmm-hhmm local", "Scheduled arr hour block"),
 "CANCELLED": ("0/1", "Cancelled"), "CANCELLATION_CODE": ("A-D", "A carrier, B weather, C NAS, D security"),
 "DIVERTED": ("0/1", "Diverted"), "CRS_ELAPSED_TIME": ("min", "Scheduled elapsed"),
 "ACTUAL_ELAPSED_TIME": ("min", "Actual elapsed"), "AIR_TIME": ("min", "Airborne time"),
 "FLIGHTS": ("count", "Always 1"), "DISTANCE": ("miles", "Route distance"), "DISTANCE_GROUP": ("250-mi bins", "Distance group"),
 "CARRIER_DELAY": ("min", "Cause attribution (only when ARR_DELAY>=15)"),
 "WEATHER_DELAY": ("min", "Cause attribution (post-hoc; LEAKY)"), "NAS_DELAY": ("min", "Cause attribution (post-hoc; LEAKY)"),
 "SECURITY_DELAY": ("min", "Cause attribution (post-hoc; LEAKY)"), "LATE_AIRCRAFT_DELAY": ("min", "Cause attribution (post-hoc; LEAKY)"),
 "FIRST_DEP_TIME": ("hhmm local", "First gate dep (gate return cases)"), "TOTAL_ADD_GTIME": ("min", "Added ground time (gate return)"),
 "LONGEST_ADD_GTIME": ("min", "Longest added ground time"), "DIV_AIRPORT_LANDINGS": ("count", "Diversion landings (9 = BTS code)"),
 "DIV_REACHED_DEST": ("0/1", "Diverted flight reached dest"), "DIV_ACTUAL_ELAPSED_TIME": ("min", "Elapsed, diverted"),
 "DIV_ARR_DELAY": ("min", "Arr delay, diverted"), "DIV_DISTANCE": ("miles", "Diversion distance"),
}
WX = {
 "temp": ("°C? (inferred)", "Air temperature; range -35..48"),
 "prcp": ("mm? (inferred)", "Precipitation (1-h total if Meteostat hourly)"),
 "wspd": ("km/h? (inferred)", "Sustained wind speed; 11% zeros (calm or filled?)"),
 "wpgt": ("km/h? (inferred)", "Peak gust; 97.8% zeros -> almost surely NaN filled with 0"),
 "pres": ("hPa (inferred)", "Sea-level pressure? station? UNKNOWN which"),
 "snow": ("mm? (inferred)", "Snow depth; 100% zeros -> no information"),
 "coco": ("code", "Weather condition code (Meteostat 1-27?); 0.8% zeros = likely filled NaN"),
}
p = pd.read_csv(sys.argv[1])
print("| column | dtype | units | % missing | meaning |")
print("|---|---|---|---|---|")
div_done = False
for r in p.itertuples():
    c = r.col
    if c.startswith(("DIV2", "DIV3", "DIV4", "DIV5")):
        if not div_done:
            print("| DIV2_* .. DIV5_* (32 cols) | mixed | | 100.0 | All-null; drop |")
            div_done = True
        continue
    if c.startswith("DIV1_"):
        u, m = ("", "1st diversion airport fields")
    elif c.split("_")[0] in WX:
        k, side = c.split("_", 1)
        u, m = WX[k]
        m = f"{m} @ {side.lower()} airport, joined at (date, sched local hr)*"
    else:
        u, m = M.get(c, ("", "UNKNOWN"))
    print(f"| {c} | {r.dtype} | {u} | {r.pct_missing} | {m} |")
