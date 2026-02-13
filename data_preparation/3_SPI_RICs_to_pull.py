"""
Thesis Data Pipeline - Step 3
==============================
Create the pull universe for daily SPI price downloads.  Reads the
RIC mapping file and writes an Excel workbook with batched sheets of
RIC_FULL identifiers (for pasting into the data provider) plus a
full RIC_FULL ↔ RIC_BASE mapping sheet for later reconciliation.

Inputs
------
- spi_ric_pull_universe.csv  : RIC_FULL and RIC_BASE columns

Outputs
-------
- spi_price_pull_universe_batched.xlsx : one mapping sheet + batched
  pull sheets (50 RICs per sheet)
"""

import math
import pandas as pd

# ============================================================
# Configuration
# ============================================================
BATCH_SIZE = 50
INPUT_PATH = "spi_ric_pull_universe.csv"
OUTPUT_PATH = "spi_price_pull_universe_batched.xlsx"

# ============================================================
# 1  Load and build clean RIC mapping
# ============================================================
pull = pd.read_csv(INPUT_PATH)

mapping = (
    pull[["RIC_FULL", "RIC_BASE"]]
    .dropna()
    .drop_duplicates()
    .sort_values(["RIC_BASE", "RIC_FULL"])
    .reset_index(drop=True)
)

universe = mapping["RIC_FULL"].unique().tolist()
n = len(universe)
num_sheets = math.ceil(n / BATCH_SIZE)

# ============================================================
# 2  Write batched Excel workbook
# ============================================================
# First sheet holds the full mapping for later reconciliation.
# Subsequent sheets each contain one batch of RIC_FULLs sized for
# convenient pasting into the data provider's query interface.

with pd.ExcelWriter(OUTPUT_PATH, engine="openpyxl") as writer:
    mapping.to_excel(writer, sheet_name="mapping_full", index=False)

    for i in range(num_sheets):
        start = i * BATCH_SIZE
        end = min(start + BATCH_SIZE, n)
        batch = pd.DataFrame({"RIC_FULL": universe[start:end]})
        batch.to_excel(writer, sheet_name=f"batch_{i + 1:02d}", index=False)

print(f"Wrote {OUTPUT_PATH} with {num_sheets} batch sheets (+ mapping_full). "
      f"Total RICs: {n}")