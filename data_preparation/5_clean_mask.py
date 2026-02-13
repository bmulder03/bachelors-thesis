"""
Thesis Data Pipeline - Step 5
==============================
Filter daily SPI stock prices to index-membership windows, select the
best RIC per company, apply quality filters, and compute excess returns.

Inputs
------
- spi_combined_clean_fullric.csv   : daily prices, one column per RIC_FULL
- spi_membership_panel_by_month.csv: monthly SPI membership at RIC_BASE level
- spi_ric_pull_universe.csv        : RIC_FULL → RIC_BASE mapping
- company_key_merges.csv (optional): manual company-key merge overrides
- snb_rates.csv                    : SNB monthly risk-free rates

Outputs
-------
- returns_daily.parquet            : masked & filled daily returns
- mask_membership_daily.parquet    : boolean SPI-membership mask
- prices_daily.parquet             : forward-filled daily prices
- Y_excess_daily_target.parquet    : daily excess returns (training target)
- best_ric_full_per_company_postmask.csv : audit trail for RIC selection
"""

import re
import pandas as pd
import numpy as np

# ============================================================
# 1  LOAD DAILY PRICES
# ============================================================
prices_raw = pd.read_csv("spi_combined_clean_fullric.csv", parse_dates=["Date"])
prices_raw = (
    prices_raw
    .rename(columns={"Date": "date"})
    .sort_values("date")
    .set_index("date")
    .apply(pd.to_numeric, errors="coerce")
)

print("Initial number of stocks:", prices_raw.shape[1])
prices_raw = prices_raw.dropna(axis=1, how="all")
print("Stocks left after removing all-NaN columns:", prices_raw.shape[1], "\n")

# ============================================================
# 2  DAILY RETURNS (raw, unfiltered)
# ============================================================
# shift(-1): return on day t = price change from t to t+1,
# so the target is the *next-day* return available at prediction time.
Y = prices_raw.pct_change(fill_method=None).shift(-1)

# ============================================================
# 3  LOAD SPI MEMBERSHIP PANEL & RIC MAPPING
# ============================================================
panel = pd.read_csv("spi_membership_panel_by_month.csv", parse_dates=["date"])
panel["month"] = panel["date"].dt.to_period("M")

pull = (
    pd.read_csv("spi_ric_pull_universe.csv", usecols=["RIC_FULL", "RIC_BASE"])
    .dropna()
)
pull["RIC_FULL"] = pull["RIC_FULL"].astype(str).str.strip()
pull["RIC_BASE"] = pull["RIC_BASE"].astype(str).str.strip().str.upper()

full_to_base = dict(zip(pull["RIC_FULL"], pull["RIC_BASE"]))

# Map each price column (RIC_FULL) to its RIC_BASE identifier.
# Drop columns that have no mapping — they can't be matched to membership.
col_base = pd.Series({c: full_to_base.get(c) for c in prices_raw.columns})
mapped_cols = col_base.dropna().index.tolist()

prices_raw = prices_raw[mapped_cols]
Y = Y[mapped_cols]
col_base = col_base.loc[mapped_cols]

# ============================================================
# 4  BUILD DAILY SPI-MEMBERSHIP MASK
# ============================================================
# Membership is recorded monthly at the RIC_BASE level.  We broadcast it
# to every trading day in that month and to every RIC_FULL that maps to
# the same RIC_BASE.

stocks = sorted(prices_raw.columns)
bases = sorted(col_base.unique())
daily_months = prices_raw.index.to_period("M")

member_pivot = (
    panel[panel["RIC_BASE"].isin(bases)]
    .assign(val=True)
    .pivot_table(
        index="month", columns="RIC_BASE", values="val",
        aggfunc="max", fill_value=False,
    )
    .astype(bool)
)

mask = pd.DataFrame(False, index=prices_raw.index, columns=stocks)
for base in bases:
    full_cols = col_base.index[col_base == base]
    if base in member_pivot.columns:
        allowed = member_pivot[base].reindex(daily_months, fill_value=False).to_numpy()
        mask.loc[:, full_cols] = allowed[:, None]

# Apply mask: NaN wherever a stock is outside the SPI that month
Y_masked = Y.where(mask)
prices_masked = prices_raw.where(mask)

# ============================================================
# 5  SELECT BEST RIC_FULL PER COMPANY
# ============================================================
# Multiple RIC_FULLs can map to the same underlying company (e.g. line
# changes after corporate actions).  We normalise company names, then
# keep the RIC_FULL with the most non-NaN masked return observations.


def norm_name(x: str) -> str:
    """Strip common legal suffixes and non-alpha characters to create
    a rough company key for deduplication."""
    s = str(x).upper()
    s = re.sub(r"\b(AG|SA|LTD|LIMITED|INC|CORP|CORPORATION|PLC|NV|SE|GMBH)\b", "", s)
    s = re.sub(r"\b(HOLDING|HOLDINGS|GROUP)\b", "", s)
    s = re.sub(r"[^A-Z]", "", s)
    return s


# Map RIC_FULL → normalised company key via RIC_BASE → Name
base_to_name = (
    panel.dropna(subset=["RIC_BASE", "Name"])
    .drop_duplicates(["RIC_BASE"])
    .set_index("RIC_BASE")["Name"]
)
full_to_company_key = col_base.map(base_to_name).map(norm_name)

# Drop columns whose RIC_BASE has no name in the panel
mapped_full = full_to_company_key[full_to_company_key.notna()].index
prices_raw          = prices_raw[mapped_full]
Y                   = Y[mapped_full]
mask                = mask[mapped_full]
Y_masked            = Y_masked[mapped_full]
col_base            = col_base.loc[mapped_full]
full_to_company_key = full_to_company_key.loc[mapped_full]

# Optional: apply manual company-key overrides (e.g. mergers / renamings)
try:
    merges = pd.read_csv("company_key_merges.csv")
    merge_map = dict(zip(
        merges["from_key"].astype(str),
        merges["to_key"].astype(str),
    ))
    company_key_final = full_to_company_key.replace(merge_map)
    print(f"Applied {len(merge_map)} manual company-key merge rules.")
except FileNotFoundError:
    company_key_final = full_to_company_key
    print("No company_key_merges.csv found — skipping manual merge step.")

# Pick the RIC_FULL with the most valid masked return observations per company
scores = Y_masked.notna().sum(axis=0)
best_full = scores.groupby(company_key_final).idxmax()
keep_full = list(best_full.values)

prices_raw    = prices_raw[keep_full]
Y             = Y[keep_full]
mask          = mask[keep_full]
Y_masked      = Y_masked[keep_full]
prices_masked = prices_raw.where(mask)

# Save mapping for audit trail
pd.DataFrame({
    "company_key":   best_full.index,
    "best_RIC_FULL": best_full.values,
}).to_csv("best_ric_full_per_company_postmask.csv", index=False)

print(f"After post-mask RIC selection: {len(keep_full)} columns, "
      f"{len(best_full)} unique companies.")

# ============================================================
# 6  FORWARD-FILL MISSING PRICES WITHIN MEMBERSHIP WINDOWS
# ============================================================
# Gaps inside a membership window (e.g. suspended trading days) are
# forward-filled so that return calculations don't produce spurious NaNs.
# Gaps *outside* the membership window remain NaN.

prices_filled = prices_masked.copy()
for col in prices_filled.columns:
    in_spi = mask[col]
    prices_filled.loc[in_spi, col] = prices_filled.loc[in_spi, col].ffill()

Y_filled = prices_filled.pct_change(fill_method=None).shift(-1)
Y_masked_filled = Y_filled.where(mask)

# Diagnostics: forward-fill impact on zero-return share
zero_before = (Y_masked == 0).sum() / Y_masked.notna().sum() * 100
zero_after  = (Y_masked_filled == 0).sum() / Y_masked_filled.notna().sum() * 100
gaps_before = (mask & prices_masked.isna()).sum().sum()
gaps_after  = (mask & prices_filled.isna()).sum().sum()

print(f"\nForward-fill impact on zero returns:")
print(f"  Before: {zero_before.mean():.1f}%  →  After: {zero_after.mean():.1f}%  "
      f"(+{zero_after.mean() - zero_before.mean():.1f} pp)")
print(f"  Price gaps filled: {gaps_before} → {gaps_after}")

# ============================================================
# 7  QUALITY FILTERS: MINIMUM OBSERVATIONS & LIQUIDITY
# ============================================================
MIN_OBS = 3000          # minimum trading days with a valid return
MAX_ZERO_PCT = 20       # maximum share of zero-return days (%)

valid_counts = Y_masked_filled.notna().sum(axis=0)
keep_cols = valid_counts[valid_counts >= MIN_OBS].index.tolist()
print(f"\nAfter MIN_OBS filter (≥{MIN_OBS}): {len(keep_cols)} stocks")

zero_pct = (Y_masked_filled[keep_cols] == 0).sum() / Y_masked_filled[keep_cols].notna().sum() * 100
keep_cols = zero_pct[zero_pct <= MAX_ZERO_PCT].index.tolist()
print(f"After liquidity filter (≤{MAX_ZERO_PCT}% zeros): {len(keep_cols)} stocks")

Y_final      = Y_masked_filled[keep_cols]
mask_final   = mask[keep_cols]
prices_final = prices_filled[keep_cols]

print(f"\nFinal universe: {len(keep_cols)} stocks, "
      f"{Y_final.index.min().date()} → {Y_final.index.max().date()}")
print(f"Final avg zero-return share: "
      f"{((Y_final == 0).sum() / Y_final.notna().sum() * 100).mean():.1f}%")

# ============================================================
# 8  SAVE INTERMEDIATE OUTPUTS
# ============================================================
Y_final.to_parquet("returns_daily.parquet")
mask_final.to_parquet("mask_membership_daily.parquet")
prices_final.to_parquet("prices_daily.parquet")

print("\nSample of membership mask (first 40 days × 8 stocks):")
print(mask_final.iloc[:40, :8])

# ============================================================
# 9  RISK-FREE RATE & EXCESS RETURNS (training target)
# ============================================================
# Source: Swiss National Bank (SNB) data portal.
# Priority hierarchy: SARON → overnight call money (1TGT) → 3-month
# Confederation bill yield (EG3M).

rf_raw = pd.read_csv("snb_rates.csv", sep=";", quotechar='"')
rf_raw["Value"] = pd.to_numeric(rf_raw["Value"], errors="coerce")
rf_raw = rf_raw[rf_raw["D0"].isin(["SARON", "1TGT", "EG3M"])].copy()

rf_monthly = (
    rf_raw
    .pivot_table(index="Date", columns="D0", values="Value", aggfunc="mean")
    .sort_index()
)

rf_monthly_pct = (
    rf_monthly.get("SARON")
    .combine_first(rf_monthly.get("1TGT"))
    .combine_first(rf_monthly.get("EG3M"))
)
rf_monthly_pct.index = pd.to_datetime(rf_monthly_pct.index + "-01")

# Broadcast monthly annualised rate → daily simple return
rf_daily = (
    rf_monthly_pct
    .reindex(Y_final.index, method="ffill") / 100.0 / 252.0
)
rf_daily.name = "rf_daily"

# Excess return = raw return − risk-free rate, only where the stock is in the SPI
Y_excess = (
    Y_final
    .where(mask_final)
    .sub(rf_daily, axis=0)
    .where(mask_final)
)

Y_excess.to_parquet("Y_excess_daily_target.parquet")
print("\nExcess returns (first 40 rows):")
print(Y_excess.head(40))