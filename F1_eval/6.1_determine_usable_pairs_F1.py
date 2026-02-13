"""
Thesis Data Pipeline - Step 6 (Usable Pairs & F1 Dataset)
==========================================================
Determine which (date, stock) pairs yield valid supervised-learning
examples under the F1 feature specification (L lagged excess returns),
build the flat training dataset, and produce summary statistics for
the thesis.

Inputs
------
- Y_excess_daily_target.parquet  : daily excess returns
- mask_membership_daily.parquet  : boolean SPI-membership mask

Outputs
-------
- usable_F1_L30.parquet                : boolean mask of usable (date, stock) pairs
- dataset_F1_L30.npz                   : compressed arrays (X, y, asset_id, date)
- dataset_F1_L30_id_to_stock.csv       : integer id → RIC_FULL mapping
- F1_summary_statistics.csv            : key descriptive statistics for Table 1
- usable_examples_per_year.pdf / .png  : bar chart for thesis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# Configuration
# ============================================================
Y_PATH = "Y_excess_daily_target.parquet"
M_PATH = "mask_membership_daily.parquet"

L = 30                # lookback window length (trading days)
MIN_TRAIN_OBS = 1000  # minimum pre-cutoff samples per stock
TRAIN_CUTOFF = pd.Timestamp("2010-01-01")

# ============================================================
# 1  Load data
# ============================================================
Y = pd.read_parquet(Y_PATH)
M = pd.read_parquet(M_PATH)

# ============================================================
# 2  Compute usable (date, stock) pairs
# ============================================================

def usable_pairs_F1(Y: pd.DataFrame, mask: pd.DataFrame, L: int) -> pd.DataFrame:
    """
    A pair (t, i) is usable iff:
      1. The target return y_{i,t} exists.
      2. Stock i is in the SPI at time t.
      3. Stock i was continuously in the SPI over [t-L, …, t-1].
      4. All L lagged returns used as features exist.

    This guarantees strict temporal causality: features use only
    information available up to t-1.
    """
    Y_ok = Y.notna()
    M_ok = mask.astype(bool)

    # rolling(L).min() enforces "all True" in the window; shift(1)
    # moves the window back one day so it covers [t-L, …, t-1].
    M_lags_ok = (
        M_ok.rolling(L, min_periods=L).min()
        .shift(1).fillna(False).astype(bool)
    )
    Y_lags_ok = (
        Y_ok.rolling(L, min_periods=L).min()
        .shift(1).fillna(False).astype(bool)
    )

    return M_ok & Y_ok & M_lags_ok & Y_lags_ok


usable = usable_pairs_F1(Y, M, L)
usable.to_parquet("usable_F1_L30.parquet")

# ============================================================
# 3  Feasible date range & summary
# ============================================================
usable_per_date = usable.sum(axis=1)
first_date = usable_per_date[usable_per_date > 0].index.min()
last_date = usable_per_date[usable_per_date > 0].index.max()

print(f"Feasible date range: {first_date.date()} → {last_date.date()}")
print(f"Max usable obs per day: {usable_per_date.max()}")
print(f"Total usable (date, stock) pairs: {usable.sum().sum():,}")
print(f"\nUsable samples per year:")
print(usable_per_date.resample("YE").sum())

# ============================================================
# 4  Publication-quality plot: usable examples per year
# ============================================================
usable_per_year = usable_per_date.resample("YE").sum()

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.linewidth": 0.8,
    "grid.linewidth": 0.5,
    "grid.alpha": 0.7,
})

fig, ax = plt.subplots(figsize=(10, 4.5))

years = usable_per_year.index.year
values = usable_per_year.values

bars = ax.bar(years, values, width=0.8, color="#2E86AB",
              edgecolor="white", linewidth=0.5, alpha=0.9)

# Fade partial first and last years
bars[0].set_alpha(0.6)
bars[-1].set_alpha(0.6)

ax.yaxis.grid(True, linestyle="--", alpha=0.7, zorder=0)
ax.set_axisbelow(True)
ax.set_xlabel("Year")
ax.set_ylabel("Usable examples")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x):,}"))
ax.set_xticks(years)
ax.set_xticklabels(years, rotation=60)

plt.tight_layout()
plt.savefig("usable_examples_per_year.pdf", dpi=300, bbox_inches="tight")
plt.savefig("usable_examples_per_year.png", dpi=300, bbox_inches="tight")
plt.close()

print("\nSaved: usable_examples_per_year.pdf / .png")

# ============================================================
# 5  Build flat F1 dataset
# ============================================================

def build_F1_dataset(Y: pd.DataFrame, usable: pd.DataFrame, L: int):
    """
    Assemble the supervised dataset: for every usable (t, i) pair,
    extract the L-day lagged return vector as features and the
    day-t excess return as the target.

    Returns
    -------
    X            : (n_samples, L)        feature matrix
    y            : (n_samples,)          target vector
    asset_id     : (n_samples,)          integer stock identifier
    sample_dates : (n_samples,)          datetime of each example
    id_to_stock  : dict[int, str]        id → RIC_FULL mapping
    """
    stocks = list(Y.columns)
    stock_to_id = {s: i for i, s in enumerate(stocks)}

    Y_arr = Y.to_numpy(dtype=np.float32)
    U_arr = usable.to_numpy(dtype=bool)
    dates = Y.index.to_numpy(dtype="datetime64[ns]")

    T, N = Y_arr.shape
    X_list, y_list, a_list, d_list = [], [], [], []

    for t in range(L, T):
        idx = np.where(U_arr[t, :])[0]
        if len(idx) == 0:
            continue

        X_list.append(Y_arr[t - L : t, :][:, idx].T)   # (k, L)
        y_list.append(Y_arr[t, idx])                     # (k,)
        a_list.append(idx.astype(np.int32))
        d_list.append(np.full(len(idx), dates[t], dtype="datetime64[ns]"))

    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    asset_id = np.concatenate(a_list, axis=0)
    sample_dates = np.concatenate(d_list, axis=0)

    id_to_stock = {i: s for s, i in stock_to_id.items()}
    return X, y, asset_id, sample_dates, id_to_stock


X, y, asset_id, sample_dates, id_to_stock = build_F1_dataset(Y, usable, L)

print(f"\nF1 dataset shapes: X={X.shape}, y={y.shape}, "
      f"asset_id={asset_id.shape}, dates={sample_dates.shape}")
print(f"Unique assets in samples: {len(np.unique(asset_id))}")

np.savez_compressed(
    "dataset_F1_L30.npz",
    X=X.astype(np.float32),
    y=y.astype(np.float32),
    asset_id=asset_id.astype(np.int32),
    date=sample_dates,
)
pd.Series(id_to_stock).to_csv(
    "dataset_F1_L30_id_to_stock.csv", header=["RIC_FULL"]
)

print("Saved: dataset_F1_L30.npz / dataset_F1_L30_id_to_stock.csv")

# ============================================================
# 6  Sanity checks
# ============================================================
print(f"\nNaNs in X: {np.isnan(X).sum()}  |  NaNs in y: {np.isnan(y).sum()}")

j = np.random.randint(0, len(y))
print(f"Random sample — asset: {id_to_stock[int(asset_id[j])]}, "
      f"date: {sample_dates[j]}, y: {y[j]:.6f}, last lag: {X[j, -1]:.6f}")

# ============================================================
# 7  Pre-cutoff training depth per stock
# ============================================================
pre_cutoff = usable.loc[usable.index < TRAIN_CUTOFF]
counts_pre = pre_cutoff.sum(axis=0).astype(int)
n_ok = int((counts_pre >= MIN_TRAIN_OBS).sum())

print(f"\nStocks with ≥{MIN_TRAIN_OBS} usable samples before "
      f"{TRAIN_CUTOFF.date()}: {n_ok} / {len(counts_pre)}")
print(counts_pre.describe())

fail = counts_pre[counts_pre < MIN_TRAIN_OBS].sort_values()
if len(fail) > 0:
    print(f"\nStocks below {MIN_TRAIN_OBS} samples (showing up to 30):")
    print(fail.head(30))
else:
    print(f"\nAll stocks meet the {MIN_TRAIN_OBS}-sample threshold.")

# ============================================================
# 8  Table 1 statistics (for thesis)
# ============================================================
print("\n" + "=" * 60)
print("TABLE 1: F1 DATASET SUMMARY STATISTICS")
print("=" * 60)

print(f"\nSample period: {first_date.date()} → {last_date.date()}")
print(f"Frequency: daily")
print(f"Number of assets: {len(Y.columns)}")
print(f"Panel type: unbalanced")
print(f"Total usable observations: {usable.sum().sum():,}")

y_series = pd.Series(y)
print(f"\n--- Excess return statistics (target variable) ---")
print(f"Mean:     {y.mean():.6f}")
print(f"Std dev:  {y.std():.6f}")
print(f"Min:      {y.min():.6f}")
print(f"Max:      {y.max():.6f}")
print(f"Median:   {np.median(y):.6f}")
print(f"25th pct: {np.percentile(y, 25):.6f}")
print(f"75th pct: {np.percentile(y, 75):.6f}")
print(f"Skewness: {y_series.skew():.6f}")
print(f"Kurtosis: {y_series.kurtosis():.6f}")

# Exclude partial boundary years for annual density statistics
full_years = usable_per_year[
    (usable_per_year.index.year > first_date.year)
    & (usable_per_year.index.year < last_date.year)
]
mature = full_years[full_years.index.year >= 2000]

print(f"\n--- Sample density over time (full years only) ---")
print(f"Mean:   {full_years.mean():.0f}")
print(f"Median: {full_years.median():.0f}")
print(f"Min:    {full_years.min():.0f}")
print(f"Max:    {full_years.max():.0f}")
print(f"Mature period (2000+) average: {mature.mean():.0f}")

if last_date.year in usable_per_year.index.year:
    partial = usable_per_year[usable_per_year.index.year == last_date.year].values[0]
    print(f"{last_date.year} (partial year): {partial:.0f}")

stats = {
    "sample_start": first_date.date(),
    "sample_end": last_date.date(),
    "n_assets": len(Y.columns),
    "total_obs": int(usable.sum().sum()),
    "mean_excess_return": float(y.mean()),
    "std_excess_return": float(y.std()),
    "min_excess_return": float(y.min()),
    "max_excess_return": float(y.max()),
    "skewness": float(y_series.skew()),
    "kurtosis": float(y_series.kurtosis()),
    "avg_obs_per_year_full": float(full_years.mean()),
    "avg_obs_per_year_mature": float(mature.mean()),
}
pd.Series(stats).to_csv("F1_summary_statistics.csv")

print("\nSaved: F1_summary_statistics.csv")
print("=" * 60)