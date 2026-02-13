"""
Thesis Data Pipeline - Step 1.2
================================
Convert the current SPI membership list exported from LSEG Workspace
from Excel to CSV.

Input :  current2.xlsx - current SPI constituents
Output:  current2.csv
"""

import pandas as pd

df = pd.read_excel("current2.xlsx")
df.to_csv("current2.csv", index=False)
print(f"Wrote current2.csv  ({len(df)} rows)")