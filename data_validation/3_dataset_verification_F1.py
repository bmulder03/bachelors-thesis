"""
Data Validation - Step 3
========================
Spot-check the compiled training dataset (dataset_F1_L30.npz) against
the source parquet files to verify that feature windows, targets, and
membership masks are internally consistent.

Inputs
------
- dataset_F1_L30.npz                : compiled training arrays (X, y, asset_id, date)
- dataset_F1_L30_id_to_stock.csv    : asset_id → RIC_FULL mapping
- Y_excess_daily_target.parquet     : daily excess returns (ground truth)
- mask_membership_daily.parquet     : boolean SPI-membership mask

Checks performed
----------------
1. Random spot-checks (x5): feature window & target match source data.
2. Membership mask is True across the entire feature window and at the
   target date.
3. One fully expanded sample is printed for visual inspection.
"""

import numpy as np
import pandas as pd

# ============================================================
# SETUP
# ============================================================
L = 30  # lookback window length (number of lagged return days)

# Stock-ID mapping
id_to_stock = (
    pd.read_csv("dataset_F1_L30_id_to_stock.csv", index_col=0)["RIC_FULL"]
    .to_dict()
)
stock_cols = list(id_to_stock.values())

# Source data
Y = pd.read_parquet("Y_excess_daily_target.parquet")[stock_cols]
M = pd.read_parquet("mask_membership_daily.parquet")[stock_cols]

# Compiled dataset
data = np.load("dataset_F1_L30.npz", allow_pickle=True)
X        = data["X"]
y        = data["y"]
asset_id = data["asset_id"]
date     = data["date"]          # datetime64[ns]

# Position lookup: date → integer row index in Y
idx = Y.index
pos = pd.Series(np.arange(len(idx)), index=idx)

# ============================================================
# 1  RANDOM SPOT-CHECKS (×5)
# ============================================================
# For five random samples, reconstruct the feature window and target
# from the source parquet and compare to the compiled arrays.

for _ in range(5):
    j = np.random.randint(0, len(y))
    s = id_to_stock[int(asset_id[j])]
    t = pd.Timestamp(date[j])
    t_pos = int(pos[t])

    # Feature window: L returns ending at t-1 → indices [t-L, …, t-1]
    w_idx = idx[t_pos - L : t_pos]

    x_true = Y.loc[w_idx, s].to_numpy(dtype=np.float32)
    y_true = np.float32(Y.loc[t, s])

    print(f"Sample {j}  |  Stock: {s}  |  Date t: {t.date()}")
    print(f"  max|X − X_true| = {float(np.max(np.abs(X[j] - x_true)))}")
    print(f"  |y − y_true|    = {float(abs(y[j] - y_true))}")
    print(f"  X last lag: {float(X[j, -1])}   Y[t-1]: {float(Y.loc[idx[t_pos - 1], s])}")
    print(f"  Mask all True on window: {bool(M.loc[w_idx, s].all())}   "
          f"Mask at t: {bool(M.loc[t, s])}")
    print("---")

# ============================================================
# 2  SINGLE EXTRA MASK CHECK (inclusive window variant)
# ============================================================
# Verify membership using the inclusive window [t-(L-1), …, t] to
# confirm the mask holds under both slicing conventions.

j = np.random.randint(0, len(y))
s = id_to_stock[int(asset_id[j])]
t = pd.Timestamp(date[j])
t_pos = pos[t]

w_idx_inclusive = idx[t_pos - (L - 1) : t_pos + 1]

print(f"\nInclusive-window mask check (sample {j}, {s}, {t.date()}):")
print(f"  Mask all True on window: {bool(M.loc[w_idx_inclusive, s].all())}")
print(f"  Mask True at target date: {bool(M.loc[t, s])}")

# ============================================================
# 3  PRINT ONE FULLY EXPANDED TRAINING SAMPLE
# ============================================================
# Render a single sample as a readable table so the feature layout
# can be visually verified against the source returns.

j = np.random.randint(0, len(y))
s = id_to_stock[int(asset_id[j])]
t = pd.Timestamp(date[j])
t_pos = int(pos[t])

w_idx = idx[t_pos - L : t_pos]

sample_df = pd.DataFrame({
    "excess_return (feature)": X[j],
    "lag": np.arange(-L, 0),
}, index=w_idx)
sample_df.index.name = "date"

print("\n================ SINGLE TRAINING SAMPLE ================\n")
print(f"Sample index      : {j}")
print(f"Stock             : {s}")
print(f"Prediction date t : {t.date()}")
print(f"Target y_t        : {y[j]: .6f}")
print("\nFeature window (lagged excess returns):")
print(sample_df)
print(f"\nSanity checks:")
print(f"  Last lag (t-1) feature  : {X[j, -1]}")
print(f"  Y[t-1] from panel       : {float(Y.loc[idx[t_pos - 1], s])}")
print(f"  Target Y[t]             : {float(Y.loc[t, s])}")
print(f"  Membership True (window): {bool(M.loc[w_idx, s].all())}")
print(f"  Membership True (t)     : {bool(M.loc[t, s])}")
print("\n========================================================\n")