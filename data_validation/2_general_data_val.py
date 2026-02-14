"""
Thesis Data Pipeline - Step 6 (Validation)
===========================================
Integrity and sanity checks for the frozen SPI daily dataset produced
by the preprocessing pipeline.

Checks performed: shape alignment, leakage guards, in-mask missingness,
return distribution sanity, per-stock usable lookback windows,
survivorship-bias diagnostics, and non-positive price detection.

Inputs
------
- Y_excess_daily_target.parquet  : daily excess returns
- prices_daily.parquet           : forward-filled daily prices
- mask_membership_daily.parquet  : boolean SPI-membership mask

Outputs
-------
- Console diagnostics only (no files written)
"""

import sys

import numpy as np
import pandas as pd

# ============================================================
# Configuration
# ============================================================
Y_PATH = "Y_excess_daily_target.parquet"
P_PATH = "prices_daily.parquet"
M_PATH = "mask_membership_daily.parquet"

LOOKBACK_DAYS = 30       # for usable-window sanity check
MAX_ABS_RET_WARN = 1.0   # 100 % daily return → warning
MAX_ABS_RET_HARD = 5.0   # 500 % daily return → hard error (likely data bug)


# ============================================================
# Helpers
# ============================================================
def pct(x):
    """Format a fraction as a readable percentage string."""
    return f"{100 * x:.2f}%"


# ============================================================
# Main validation routine
# ============================================================
def main():

    # ============================================================
    # 1  Load datasets
    # ============================================================
    print("Loading datasets...")
    try:
        Y = pd.read_parquet(Y_PATH)
        P = pd.read_parquet(P_PATH)
        M = pd.read_parquet(M_PATH)
    except Exception as e:
        print(f"ERROR: failed to read required files: {e}")
        sys.exit(1)

    for df, name in [(Y, "Y"), (P, "Prices"), (M, "Mask")]:
        if not isinstance(df.index, pd.DatetimeIndex):
            try:
                df.index = pd.to_datetime(df.index)
            except Exception:
                print(f"ERROR: {name} index cannot be converted to DatetimeIndex.")
                sys.exit(1)

    # ============================================================
    # 2  Shape alignment
    # ============================================================
    print("\n" + "=" * 80)
    print("BASIC SHAPE / ALIGNMENT")
    print("=" * 80)

    print(f"Y shape:      {Y.shape}")
    print(f"Prices shape: {P.shape}")
    print(f"Mask shape:   {M.shape}")

    common_idx = Y.index.intersection(P.index).intersection(M.index)
    common_cols = Y.columns.intersection(P.columns).intersection(M.columns)

    if len(common_idx) == 0 or len(common_cols) == 0:
        print("ERROR: no overlap in index/columns across Y / Prices / Mask.")
        sys.exit(1)

    if len(common_idx) != len(Y.index) or len(common_cols) != len(Y.columns):
        print("WARNING: datasets not perfectly aligned; auto-aligning to intersection.")

    Y = Y.loc[common_idx, common_cols].sort_index()
    P = P.loc[common_idx, common_cols].sort_index()
    M = M.loc[common_idx, common_cols].sort_index()

    print(f"\nAligned shapes: Y={Y.shape}, Prices={P.shape}, Mask={M.shape}")
    print(f"Date range: {Y.index.min().date()} → {Y.index.max().date()}")
    print(f"Num stocks: {Y.shape[1]}")

    if M.dtypes.nunique() != 1 or M.dtypes.iloc[0] != bool:
        print("WARNING: Mask is not boolean dtype everywhere; coercing.")
        M = M.astype(bool)

    # ============================================================
    # 3  Hard invariants (leakage guards)
    # ============================================================
    # Returns and prices must be NaN wherever the mask is False.
    # Any violation means the masking step in the pipeline has a bug.
    print("\n" + "=" * 80)
    print("HARD INVARIANTS (LEAKAGE GUARDS)")
    print("=" * 80)

    leak_Y = int((~M & Y.notna()).sum().sum())
    leak_P = int((~M & P.notna()).sum().sum())

    print(f"Non-NaN excess returns outside mask: {leak_Y:,}")
    print(f"Non-NaN prices outside mask:         {leak_P:,}")

    if leak_Y or leak_P:
        print("ERROR: data found outside membership mask — indicates leakage.")
        sys.exit(1)
    print("OK: masking invariants hold.")

    # ============================================================
    # 4  Missingness inside the mask
    # ============================================================
    # After forward-filling, remaining gaps are typically at the very
    # start of a membership window where no prior price exists.
    print("\n" + "=" * 80)
    print("MISSINGNESS INSIDE MASK")
    print("=" * 80)

    total_in_mask = int(M.sum().sum())
    gaps_price = int((M & P.isna()).sum().sum())
    gaps_ret = int((M & Y.isna()).sum().sum())

    print(f"Total in-mask cells:            {total_in_mask:,}")
    print(f"In-mask missing prices:         {gaps_price:,} "
          f"({pct(gaps_price / total_in_mask) if total_in_mask else 'n/a'})")
    print(f"In-mask missing excess returns: {gaps_ret:,} "
          f"({pct(gaps_ret / total_in_mask) if total_in_mask else 'n/a'})")

    # ============================================================
    # 5  Return distribution sanity
    # ============================================================
    print("\n" + "=" * 80)
    print("RETURN DISTRIBUTION SANITY")
    print("=" * 80)

    Y_stack = Y.stack(dropna=True)
    if len(Y_stack) == 0:
        print("ERROR: no non-NaN excess returns found.")
        sys.exit(1)

    print(Y_stack.describe().to_string())

    quantiles = [0.0001, 0.001, 0.01, 0.5, 0.99, 0.999, 0.9999]
    print("\nQuantiles of daily excess returns:")
    for q, v in Y_stack.quantile(quantiles).items():
        print(f"  q={q:>7}: {v: .6f}")

    max_abs = float(np.nanmax(np.abs(Y_stack.values)))
    print(f"\nMax |daily excess return|: {max_abs:.6f}")

    if max_abs > MAX_ABS_RET_HARD:
        print(f"ERROR: extreme return exceeds hard threshold ({MAX_ABS_RET_HARD}). "
              f"Likely bad prices or unadjusted splits.")
        sys.exit(1)
    elif max_abs > MAX_ABS_RET_WARN:
        print(f"WARNING: some returns exceed {MAX_ABS_RET_WARN:.0%}. "
              f"Verify whether legitimate corporate actions explain them.")

    # ============================================================
    # 6  Per-stock sample size & usable lookback windows
    # ============================================================
    # A "usable" training example at day t requires both a valid return
    # at t and L consecutive valid returns in the preceding window
    # (needed to construct lagged-return features).
    print("\n" + "=" * 80)
    print(f"PER-STOCK SAMPLE SIZE & USABLE LOOKBACK WINDOWS (L={LOOKBACK_DAYS})")
    print("=" * 80)

    obs = Y.notna().sum(axis=0)
    print("Valid excess return observations per stock:")
    print(obs.describe().to_string())

    L = LOOKBACK_DAYS
    notna_int = Y.notna().astype(np.int16)
    rolling_sum = notna_int.rolling(L, min_periods=L).sum()
    usable = ((rolling_sum == L) & Y.notna()).sum(axis=0)

    print(f"\nUsable examples per stock (conservative, L={L}):")
    print(usable.describe().to_string())

    num_zero = int((usable == 0).sum())
    print(f"\nStocks with zero usable examples: {num_zero}")
    if num_zero > 0:
        print("First examples:", usable[usable == 0].index.tolist()[:20])
        print("WARNING: consider excluding these stocks or reducing L.")

    # ============================================================
    # 7  Survivorship-bias check
    # ============================================================
    # If every stock's membership runs to the final date the dataset
    # suffers from pure survivorship bias.  A healthy universe should
    # show diverse end-dates (delistings, index removals).
    print("\n" + "=" * 80)
    print("SURVIVORSHIP-BIAS CHECK (MEMBERSHIP END-DATES)")
    print("=" * 80)

    last_in = M.apply(
        lambda s: s.index[s.to_numpy().nonzero()[0][-1]] if s.any() else pd.NaT,
        axis=0,
    )
    end_at_last = int((last_in == M.index.max()).sum())
    print(f"Stocks whose membership extends to final date: "
          f"{end_at_last} / {M.shape[1]}")
    print("\nMost common membership end-dates (top 10):")
    print(last_in.value_counts().head(10).to_string())

    # ============================================================
    # 8  Non-positive price check
    # ============================================================
    print("\n" + "=" * 80)
    print("NON-POSITIVE PRICE CHECK")
    print("=" * 80)

    nonpos = int((M & (P <= 0)).sum().sum())
    print(f"Non-positive prices inside mask: {nonpos:,}")
    if nonpos > 0:
        print("WARNING: non-positive prices detected — inspect those cells.")

    # ============================================================
    # Summary
    # ============================================================
    print("\n" + "=" * 80)
    print("VALIDATION COMPLETE")
    print("=" * 80)
    print("If no ERROR appeared above, the frozen dataset is consistent "
          "and ready for modelling.")


if __name__ == "__main__":
    main()