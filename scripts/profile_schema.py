"""Profile the raw CSV: dtype, % missing, n unique, min/max, examples per column.
Writes data/schema_profile.csv. Usage: python3 -I scripts/profile_schema.py <csv> <out>
"""
import sys
import pandas as pd

src, out = sys.argv[1], sys.argv[2]
df = pd.read_csv(src, low_memory=False)
rows = []
for c in df.columns:
    s = df[c]
    nn = s.dropna()
    num = pd.api.types.is_numeric_dtype(s)
    rows.append({
        "col": c,
        "dtype": str(s.dtype),
        "pct_missing": round(100 * s.isna().mean(), 2),
        "n_unique": nn.nunique(),
        "min": nn.min() if num and len(nn) else "",
        "max": nn.max() if num and len(nn) else "",
        "examples": "|".join(map(str, nn.unique()[:4])),
    })
pd.DataFrame(rows).to_csv(out, index=False)
print("shape", df.shape)
