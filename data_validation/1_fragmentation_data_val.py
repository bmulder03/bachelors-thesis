"""
Data Validation 1 - Membership Fragmentation Analysis
======================================================
Checks how contiguous each stock's SPI membership is.  Stocks that
drop in and out of the index create short fragments; fragments shorter
than the model's look-back window (LOOKBACK days) are unusable for
sequence-based models.  This script quantifies that loss.

Inputs
------
- mask_membership_daily.parquet : boolean daily SPI-membership mask
- prices_daily.parquet          : forward-filled daily prices (loaded
                                  here only to confirm alignment)
"""

import pandas as pd

# ============================================================
# 1  LOAD DATA
# ============================================================
mask_final   = pd.read_parquet("mask_membership_daily.parquet")
prices_final = pd.read_parquet("prices_daily.parquet")

print(f"Loaded data for {len(mask_final.columns)} stocks")
print(f"Date range: {mask_final.index.min().date()} to "
      f"{mask_final.index.max().date()}\n")

# ============================================================
# 2  HELPER: IDENTIFY CONTIGUOUS MEMBERSHIP PERIODS
# ============================================================

def get_contiguous_periods(mask_series: pd.Series):
    """Return a list of (start, end, length) tuples for every
    contiguous run of True values in *mask_series*."""
    periods = []
    in_period = False
    start_idx = None

    for idx, (date, is_member) in enumerate(mask_series.items()):
        if is_member and not in_period:
            start_idx = idx
            in_period = True
        elif not is_member and in_period:
            periods.append((
                mask_series.index[start_idx],
                mask_series.index[idx - 1],
                idx - start_idx,
            ))
            in_period = False

    # Close a period that extends to the last trading day
    if in_period:
        periods.append((
            mask_series.index[start_idx],
            mask_series.index[-1],
            len(mask_series) - start_idx,
        ))

    return periods

# ============================================================
# 3  FRAGMENTATION ANALYSIS
# ============================================================
# A period is "usable" only if it contains at least LOOKBACK consecutive
# membership days — the minimum needed to form one input sequence.
LOOKBACK = 30

fragmentation_stats = []
for stock in mask_final.columns:
    periods = get_contiguous_periods(mask_final[stock])
    usable_periods = [p for p in periods if p[2] >= LOOKBACK]
    total_usable_days = sum(p[2] for p in usable_periods)
    total_member_days = int(mask_final[stock].sum())

    fragmentation_stats.append({
        "stock":              stock,
        "num_periods":        len(periods),
        "num_usable_periods": len(usable_periods),
        "total_days":         total_member_days,
        "usable_days":        total_usable_days,
        "lost_days":          total_member_days - total_usable_days,
    })

frag_df = pd.DataFrame(fragmentation_stats)

# ============================================================
# 4  PRINT SUMMARY
# ============================================================
print("=" * 80)
print("FRAGMENTATION ANALYSIS")
print("=" * 80)

pct_lost = 100 * frag_df["lost_days"].sum() / frag_df["total_days"].sum()

print(f"\nSummary across {len(mask_final.columns)} stocks:")
print(f"  Avg periods per stock:        {frag_df['num_periods'].mean():.1f}")
print(f"  Avg usable periods per stock: {frag_df['num_usable_periods'].mean():.1f}")
print(f"  Avg days lost to short fragments: "
      f"{frag_df['lost_days'].mean():.0f} ({pct_lost:.1f}% of all member-days)")

single = (frag_df["num_periods"] == 1).sum()
few    = frag_df["num_periods"].between(2, 3).sum()
many   = (frag_df["num_periods"] >= 4).sum()

print(f"\nStocks by number of membership periods:")
print(f"  Single period (no gaps): {single}")
print(f"  2–3 periods:             {few}")
print(f"  4+ periods:              {many}")

print(f"\nTop 10 most fragmented stocks:")
print(frag_df.nlargest(10, "num_periods")[
    ["stock", "num_periods", "usable_days", "lost_days"]
].to_string(index=False))

print(f"\nTop 10 stocks by days lost:")
print(frag_df.nlargest(10, "lost_days")[
    ["stock", "num_periods", "lost_days", "total_days"]
].to_string(index=False))

# ============================================================
# 5  DETAILED BREAKDOWN OF MOST FRAGMENTED STOCK
# ============================================================
worst_stock = frag_df.nlargest(1, "num_periods").iloc[0]["stock"]

print(f"\n{'=' * 80}")
print(f"DETAILED BREAKDOWN: {worst_stock}")
print(f"{'=' * 80}")

for i, (start, end, length) in enumerate(
    get_contiguous_periods(mask_final[worst_stock]), start=1
):
    usable = "✓" if length >= LOOKBACK else "✗"
    print(f"  Period {i:>2}: {start.date()} → {end.date()}  "
          f"{length:>4d} days  {usable}")