"""
Thesis Data Pipeline - Step 4
==============================
Convert the multi-sheet Excel file of daily SPI constituent prices into
a single, clean CSV with one column per RIC_FULL and a datetime index.

Input
-----
- spi_initial_pull.xlsx : one sheet per batch of RIC_FULL price series,
                          plus a 'mapping_full' sheet (RIC pairings, not
                          price data).

Output
------
- spi_combined_clean_fullric.csv : combined daily prices, Date x RIC_FULL
"""

import pandas as pd

# ============================================================
# 1  LOAD ALL PRICE SHEETS FROM EXCEL
# ============================================================
EXCEL_FILE = "spi_initial_pull.xlsx"
MAPPING_SHEET = "mapping_full"

OUT_FILE = "spi_combined_clean_fullric.csv"

sheets = pd.read_excel(EXCEL_FILE, sheet_name=None)
sheets.pop(MAPPING_SHEET, None)  # discard the RIC-mapping sheet

# ============================================================
# 2  CLEAN EACH SHEET AND COLLECT
# ============================================================
dfs = []
for sheet_name, df in sheets.items():
    df = (
        df.set_index(df.columns[0])                          # first column is the date
          .replace("The universe is not found.", pd.NA)       # Refinitiv error string → NaN
          .apply(pd.to_numeric, errors="coerce")
    )
    dfs.append(df)

# ============================================================
# 3  MERGE SHEETS AND FINALISE INDEX
# ============================================================
combined = pd.concat(dfs, axis=1)
combined.index = pd.to_datetime(combined.index, errors="coerce")
combined.index.name = "Date"
combined = combined[combined.index.notna()].sort_index()

# If the same RIC_FULL appeared in multiple sheets, keep the first occurrence
if combined.columns.duplicated().any():
    combined = combined.groupby(level=0, axis=1).first()

# ============================================================
# 4  SAVE
# ============================================================
combined.to_csv(OUT_FILE)
print(f"Wrote {OUT_FILE} with {combined.shape}")