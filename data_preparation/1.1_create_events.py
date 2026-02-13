"""
Thesis Data Pipeline - Step 1.1
================================
Parse the SPI index events file (additions & removals) exported from
LSEG Workspace and save a clean CSV.

Input :  events2.xlsx  - raw export with metadata rows above the header
Output:  events2.csv   - cleaned table (Status, Issuer, Code, Date)
"""

import pandas as pd

# The raw export contains metadata rows before the actual header.
# Locate the header by searching for a row that contains "Status".
raw = pd.read_excel("events2.xlsx", header=None)
header_row = raw.index[
    raw.apply(lambda r: r.astype(str).str.contains("Status").any(), axis=1)
][0]

df = pd.read_excel("events2.xlsx", header=header_row)
df = df[["Status", "Issuer", "Code", "Date"]]
df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
df = df.dropna(subset=["Status", "Date"])

df.to_csv("events2.csv", index=False, date_format="%Y-%m-%d")
print(f"Wrote events2.csv  ({len(df)} rows)")