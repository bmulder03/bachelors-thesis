"""
Co-movement Diagnostics
========================================================
Build a *structural* SPI panel (membership-masked, one series per company,
no ML quality filters), compute monthly returns, then produce rolling
co-movement diagnostics: average pairwise correlation and first principal
component (PC1) variance share.

Inputs
------
- spi_combined_clean_fullric.csv      : daily prices (Date + RIC_FULL columns)
- spi_membership_panel_by_month.csv   : monthly membership (date, RIC_BASE, Name)
- spi_ric_pull_universe.csv           : RIC_FULL → RIC_BASE mapping
- company_key_merges.csv (optional)   : manual company-key merge overrides

Outputs
-------
- prices_daily_structural.parquet             : masked daily prices
- mask_membership_daily_structural.parquet     : boolean membership mask
- best_ric_full_per_company_structural.csv    : audit trail for RIC selection
- returns_monthly_structural.parquet          : end-of-month simple returns
- spi_comovement_rolling_stats.csv            : rolling correlation & PC1 share
- spi_comovement_rolling.png / .pdf           : time-series plot
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# Configuration
# ============================================================
PRICES_CSV = "spi_combined_clean_fullric.csv"
MEMBERSHIP_CSV = "spi_membership_panel_by_month.csv"
MAPPING_CSV = "spi_ric_pull_universe.csv"
COMPANY_MERGES_CSV = "company_key_merges.csv"

WINDOW_MONTHS = 36   # rolling window for correlation / PCA
MIN_OVERLAP = 18     # minimum months an asset needs inside the window


# ============================================================
# Helpers
# ============================================================
def norm_name(x: str) -> str:
    """Strip common legal suffixes and non-alpha characters to create
    a rough company key for deduplication."""
    s = str(x).upper()
    s = re.sub(r"\b(AG|SA|LTD|LIMITED|INC|CORP|CORPORATION|PLC|NV|SE|GMBH)\b", "", s)
    s = re.sub(r"\b(HOLDING|HOLDINGS|GROUP)\b", "", s)
    s = re.sub(r"[^A-Z]", "", s)
    return s


@dataclass
class StructuralPanel:
    """Container for the three artefacts produced by build_structural_panel."""
    prices_daily: pd.DataFrame   # daily prices, masked by membership
    mask_daily: pd.DataFrame     # boolean SPI-membership mask
    best_ric_map: pd.DataFrame   # company_key → best RIC_FULL (audit trail)


# ============================================================
# 1  Build structural daily panel
# ============================================================
# This mirrors the preprocessing pipeline (Step 5) but intentionally
# omits the MIN_OBS and liquidity filters so that the full index
# composition is preserved for structural / descriptive analysis.

def build_structural_panel() -> StructuralPanel:

    # ---- Load daily prices (RIC_FULL columns) ----
    prices_raw = (
        pd.read_csv(PRICES_CSV, parse_dates=["Date"])
        .rename(columns={"Date": "date"})
        .sort_values("date")
        .set_index("date")
        .apply(pd.to_numeric, errors="coerce")
        .dropna(axis=1, how="all")
    )

    # ---- Load monthly membership (RIC_BASE level) ----
    panel = pd.read_csv(MEMBERSHIP_CSV, parse_dates=["date"])
    panel["month"] = panel["date"].dt.to_period("M")

    # ---- Load RIC_FULL → RIC_BASE mapping ----
    pull = pd.read_csv(MAPPING_CSV, usecols=["RIC_FULL", "RIC_BASE"]).dropna()
    pull["RIC_FULL"] = pull["RIC_FULL"].astype(str).str.strip()
    pull["RIC_BASE"] = pull["RIC_BASE"].astype(str).str.strip().str.upper()

    full_to_base = dict(zip(pull["RIC_FULL"], pull["RIC_BASE"]))

    # Map each price column to its RIC_BASE; drop unmapped columns
    col_base = pd.Series({c: full_to_base.get(c) for c in prices_raw.columns})
    mapped_cols = col_base.dropna().index.tolist()
    if not mapped_cols:
        raise RuntimeError("No RIC_FULL columns could be mapped to RIC_BASE.")

    prices_raw = prices_raw[mapped_cols]
    col_base = col_base.loc[mapped_cols]

    # ---- Build daily membership mask from monthly membership ----
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

    prices_masked = prices_raw.where(mask)

    # ---- De-duplicate: pick best RIC_FULL per company ----
    base_to_name = (
        panel.dropna(subset=["RIC_BASE", "Name"])
        .drop_duplicates(["RIC_BASE"])
        .set_index("RIC_BASE")["Name"]
    )
    full_to_company_key = col_base.map(base_to_name).map(norm_name)

    mapped_full = full_to_company_key[full_to_company_key.notna()].index
    prices_raw          = prices_raw[mapped_full]
    prices_masked       = prices_masked[mapped_full]
    mask                = mask[mapped_full]
    col_base            = col_base.loc[mapped_full]
    full_to_company_key = full_to_company_key.loc[mapped_full]

    # Optional manual merges (e.g. post-merger renamings)
    try:
        merges = pd.read_csv(COMPANY_MERGES_CSV)
        merge_map = dict(zip(
            merges["from_key"].astype(str),
            merges["to_key"].astype(str),
        ))
        company_key_final = full_to_company_key.replace(merge_map)
        print(f"Applied {len(merge_map)} manual company-key merge rules.")
    except FileNotFoundError:
        company_key_final = full_to_company_key
        print("No company_key_merges.csv found — skipping manual merge step.")

    # Keep the RIC_FULL with the most valid masked price observations per company
    scores = prices_masked.notna().sum(axis=0)
    best_full = scores.groupby(company_key_final).idxmax()
    keep_full = list(best_full.values)

    prices_daily = prices_raw[keep_full].where(mask[keep_full])
    mask_daily = mask[keep_full]

    best_ric_map = pd.DataFrame({
        "company_key":   best_full.index,
        "best_RIC_FULL": best_full.values,
    })

    print(f"Structural panel: {prices_daily.shape[1]} companies, "
          f"{prices_daily.index.min().date()} → {prices_daily.index.max().date()}")

    return StructuralPanel(
        prices_daily=prices_daily,
        mask_daily=mask_daily,
        best_ric_map=best_ric_map,
    )


# ============================================================
# 2  Monthly returns
# ============================================================
def compute_monthly_returns(prices_daily: pd.DataFrame) -> pd.DataFrame:
    """Simple monthly returns from end-of-month masked prices."""
    prices_eom = prices_daily.resample("ME").last()
    return prices_eom.pct_change(fill_method=None)


# ============================================================
# 3  Rolling co-movement diagnostics
# ============================================================
def rolling_comovement(
    rets_m: pd.DataFrame,
    window_months: int = WINDOW_MONTHS,
    min_overlap_months: int = MIN_OVERLAP,
) -> pd.DataFrame:
    """For each month t, compute the correlation matrix over the trailing
    window and extract two summary statistics: the average off-diagonal
    pairwise correlation and the variance share of the first principal
    component (from an eigen-decomposition of the correlation matrix).

    Assets with fewer than *min_overlap_months* valid observations inside
    the window are excluded for that month.
    """
    if window_months <= 3:
        raise ValueError("window_months should be reasonably large (e.g. 36 or 60).")

    rows: list[tuple] = []
    idx = rets_m.index

    for t in range(window_months, len(idx)):
        w = rets_m.iloc[t - window_months : t]

        # Drop assets with too few observations in this window
        valid_assets = w.notna().sum(axis=0) >= min_overlap_months
        w = w.loc[:, valid_assets]
        n_assets = w.shape[1]

        if n_assets < 3:
            rows.append((idx[t], np.nan, np.nan, n_assets))
            continue

        # Average off-diagonal pairwise correlation
        corr = w.corr(min_periods=min_overlap_months)
        cvals = corr.to_numpy()
        off_diag = cvals[np.triu_indices_from(cvals, k=1)]
        avg_corr = float(np.nanmean(off_diag))

        # PC1 variance share from eigenvalues of the correlation matrix
        corr_filled = corr.fillna(0.0)
        np.fill_diagonal(corr_filled.values, 1.0)
        eigvals = np.maximum(np.linalg.eigvalsh(corr_filled.to_numpy()), 0.0)
        pc1_share = float(eigvals[-1] / eigvals.sum()) if eigvals.sum() > 0 else np.nan

        rows.append((idx[t], avg_corr, pc1_share, n_assets))

    return (
        pd.DataFrame(rows, columns=["date", "avg_corr_offdiag", "pc1_share", "n_assets_used"])
        .set_index("date")
    )


# ============================================================
# 4  Plotting
# ============================================================
def plot_comovement(
    stats: pd.DataFrame,
    out_png: str = "spi_comovement_rolling.png",
    out_pdf: str = "spi_comovement_rolling.pdf",
) -> None:
    """Dual-axis time-series plot of average correlation and PC1 share."""
    fig, ax1 = plt.subplots(figsize=(11, 5.5))

    ax1.plot(stats.index, stats["avg_corr_offdiag"],
             linewidth=1.5, label="Average pairwise correlation")
    ax1.set_ylabel("Average pairwise correlation")
    ax1.set_xlabel("Date")
    ax1.grid(True, which="major", alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(stats.index, stats["pc1_share"],
             linewidth=1.5, linestyle="--", label="PC1 variance share")
    ax2.set_ylabel("PC1 share (correlation PCA)")

    # Combined legend from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2,
               loc="upper left", frameon=False)

    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    fig.savefig(out_pdf)
    plt.close(fig)

    print(f"Saved plot: {out_png}, {out_pdf}")


# ============================================================
# 5  Main
# ============================================================
def main() -> int:

    # ---- Build & save structural panel ----
    sp = build_structural_panel()
    sp.prices_daily.to_parquet("prices_daily_structural.parquet")
    sp.mask_daily.to_parquet("mask_membership_daily_structural.parquet")
    sp.best_ric_map.to_csv("best_ric_full_per_company_structural.csv", index=False)

    # ---- Monthly returns ----
    rets_m = compute_monthly_returns(sp.prices_daily)
    rets_m.to_parquet("returns_monthly_structural.parquet")

    # ---- Rolling co-movement statistics ----
    stats = rolling_comovement(rets_m)
    stats.to_csv("spi_comovement_rolling_stats.csv")

    # ---- Plot ----
    plot_comovement(stats)

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())