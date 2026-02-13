"""
F3 Evaluation - Data Preparation
====================================================
Reads a multi-sheet Excel workbook with two-level headers (RIC, FIELD),
cleans and combines all sheets, then writes one wide CSV per field
(Date x RIC) for downstream use in the F3 evaluation.

Inputs
------
- final_pull_F3.xlsx : Excel workbook; each sheet has a (RIC, FIELD)
                       multi-header over daily observations.
                       A 'mapping_full' sheet, if present, is ignored.

Outputs
-------
- spi_OFFBK_VOL.csv
- spi_TRNOVR_UNS.csv
- spi_BID.csv
- spi_ASK.csv
- spi_VWAP.csv
"""

import pandas as pd

# ============================================================
# Configuration
# ============================================================
EXCEL_FILE = "final_pull_F3.xlsx"
MAPPING_SHEET = "mapping_full"
FIELDS = ["OFFBK_VOL", "TRNOVR_UNS", "BID", "ASK", "VWAP"]

# Strings that Refinitiv / Excel may produce instead of numeric values
ERROR_STRINGS = {
    "The universe is not found.": pd.NA,
    "#N/A": pd.NA,
    "N/A": pd.NA,
    "#VALUE!": pd.NA,
}

# ============================================================
# 1  Load all data sheets
# ============================================================
sheets = pd.read_excel(EXCEL_FILE, sheet_name=None, header=[0, 1])
sheets.pop(MAPPING_SHEET, None)

# ============================================================
# 2  Clean each sheet and collect into a list
# ============================================================
dfs = []
for sheet_name, df in sheets.items():
    if df.empty:
        continue

    # First column is the timestamp; promote it to index
    df = df.set_index(df.columns[0])
    df.index.name = "Date"
    df.index = pd.to_datetime(df.index, errors="coerce")
    df = df[df.index.notna()].sort_index()

    # Normalise the two-level column headers
    df.columns = pd.MultiIndex.from_tuples(
        [(str(ric).strip(), str(field).strip()) for ric, field in df.columns],
        names=["RIC", "FIELD"],
    )

    # Drop columns with blank RIC or FIELD (artefacts of ragged Excel ranges)
    valid = (
        (df.columns.get_level_values("RIC") != "")
        & (df.columns.get_level_values("FIELD") != "")
    )
    df = df.loc[:, valid]

    df = df.replace(ERROR_STRINGS)
    df = df.apply(pd.to_numeric, errors="coerce")

    dfs.append(df)

# ============================================================
# 3  Combine sheets and resolve duplicates
# ============================================================
# Sheets may cover different RICs or date ranges; horizontal concat
# brings them together.  If the same (RIC, FIELD) appears in more
# than one sheet, keep the first non-NaN value.
combined = pd.concat(dfs, axis=1)

if combined.columns.duplicated().any():
    combined = combined.groupby(level=["RIC", "FIELD"], axis=1).first()

# ============================================================
# 4  Write one CSV per field (Date × RIC)
# ============================================================
available_fields = combined.columns.get_level_values("FIELD")

for field in FIELDS:
    if field not in available_fields:
        print(f"Skipping {field}: not found in workbook.")
        continue

    wide = combined.xs(field, level="FIELD", axis=1)

    if wide.columns.duplicated().any():
        wide = wide.groupby(level=0, axis=1).first()

    out_path = f"spi_{field}.csv"
    wide.to_csv(out_path)
    print(f"Wrote {out_path}  {wide.shape}")