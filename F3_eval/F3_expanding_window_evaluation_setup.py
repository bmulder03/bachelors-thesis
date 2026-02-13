"""
F3 Evaluation - Expanding-Window Evaluation Setup
===================================================================
Build expanding-window train / validation / test splits for the F3
feature block (L=30), compute per-window asset eligibility, and
derive a fixed model-capacity plan from the reference window.

Window scheme: train ≤ (Y-2), val = (Y-1), test = Y.

Inputs
------
- dataset_F3_L30.npz              : stacked examples (X, y, asset_id, date)
- dataset_F3_L30_id_to_stock.csv  : asset-ID → RIC mapping (optional)

Outputs
-------
- evaluation_config_F3_L30.pkl    : full config dict (windows, capacity plan, mapping)
- evaluation_windows_summary_F3_L30.csv : one-row-per-window summary table
"""

import os
import pickle

import numpy as np
import pandas as pd

# ============================================================
# Configuration
# ============================================================
FEATURE_BLOCK = "F3"
L = 30

NPZ_PATH = f"dataset_{FEATURE_BLOCK}_L{L}.npz"
ID_TO_STOCK_CSV = f"dataset_{FEATURE_BLOCK}_L{L}_id_to_stock.csv"

OUT_PKL = f"evaluation_config_{FEATURE_BLOCK}_L{L}.pkl"
OUT_SUMMARY_CSV = f"evaluation_windows_summary_{FEATURE_BLOCK}_L{L}.csv"

FIRST_TEST_YEAR = 2011
LAST_TEST_YEAR = 2025
LAST_TEST_DATE = "2025-08-30"               # partial final year

MIN_TRAIN_PER_ASSET = 1000                   # eligibility threshold for I_Y

# Model capacity budget (fixed ex-ante and held constant across all experiments)
JOINT_PARAMS_TOTAL = 10_000_000
TRUNK_FRACTION = 0.80
HEAD_FRACTION = 0.20

# ============================================================
# 1  Load data
# ============================================================
print(f"=== Expanding-window evaluation setup ({FEATURE_BLOCK}, L={L}) ===\n")
print(f"Loading: {NPZ_PATH}")

data = np.load(NPZ_PATH, allow_pickle=True)
X = data["X"]
y = data["y"]
asset_id = data["asset_id"].astype(np.int32)
sample_dates = pd.DatetimeIndex(pd.to_datetime(data["date"]))

n_examples = len(X)
n_assets = int(asset_id.max()) + 1

print(f"Total examples: {n_examples:,}")
print(f"Total assets:   {n_assets}")
print(f"Date range:     {sample_dates.min().date()} → {sample_dates.max().date()}\n")

id_to_stock = None
if os.path.exists(ID_TO_STOCK_CSV):
    try:
        mapping = pd.read_csv(ID_TO_STOCK_CSV, index_col=0)
        id_to_stock = {int(k): str(v) for k, v in mapping.iloc[:, 0].items()}
        print(f"Loaded stock mapping: {len(id_to_stock)} stocks\n")
    except Exception as e:
        print(f"WARNING: could not load stock mapping: {e}\n")

# ============================================================
# 2  Build expanding windows
# ============================================================
print("Building expanding windows...")
print(f"Window scheme: train ≤ (Y−2), val = (Y−1), test = Y")
print(f"Test years: {FIRST_TEST_YEAR} → {LAST_TEST_YEAR}\n")

last_test_ts = pd.Timestamp(LAST_TEST_DATE) if LAST_TEST_DATE else None

windows = []
summary_rows = []

for test_year in range(FIRST_TEST_YEAR, LAST_TEST_YEAR + 1):

    train_start = sample_dates.min()
    train_end = pd.Timestamp(f"{test_year - 2}-12-31")
    val_start = pd.Timestamp(f"{test_year - 1}-01-01")
    val_end = pd.Timestamp(f"{test_year - 1}-12-31")
    test_start = pd.Timestamp(f"{test_year}-01-01")
    test_end = pd.Timestamp(f"{test_year}-12-31")

    if test_year == LAST_TEST_YEAR and last_test_ts:
        test_end = last_test_ts

    if train_end < train_start or test_start > sample_dates.max():
        continue

    # Boolean masks → integer indices
    idx_train = np.where((sample_dates >= train_start) & (sample_dates <= train_end))[0]
    idx_val = np.where((sample_dates >= val_start) & (sample_dates <= val_end))[0]
    idx_test = np.where((sample_dates >= test_start) & (sample_dates <= test_end))[0]

    # Per-asset observation counts in each split
    train_counts = np.bincount(asset_id[idx_train], minlength=n_assets)
    val_counts = np.bincount(asset_id[idx_val], minlength=n_assets)
    test_counts = np.bincount(asset_id[idx_test], minlength=n_assets)

    # Eligibility vector I_Y: enough training data *and* non-trivial
    # presence in validation and test so that evaluation is meaningful.
    I_Y = (
        (train_counts >= MIN_TRAIN_PER_ASSET)
        & (val_counts > 10)
        & (test_counts > 10)
    )
    n_eligible = int(I_Y.sum())

    # Restrict each split to eligible assets only
    idx_train_elig = idx_train[I_Y[asset_id[idx_train]]]
    idx_val_elig = idx_val[I_Y[asset_id[idx_val]]]
    idx_test_elig = idx_test[I_Y[asset_id[idx_test]]]

    windows.append({
        "test_year": test_year,
        "train_start": train_start,
        "train_end": train_end,
        "val_start": val_start,
        "val_end": val_end,
        "test_start": test_start,
        "test_end": test_end,
        "idx_train": idx_train,
        "idx_val": idx_val,
        "idx_test": idx_test,
        "I_Y": I_Y,
        "train_asset_counts": train_counts,
        "idx_train_elig": idx_train_elig,
        "idx_val_elig": idx_val_elig,
        "idx_test_elig": idx_test_elig,
    })

    summary_rows.append({
        "test_year": test_year,
        "train_period": f"{train_start.date()} to {train_end.date()}",
        "val_period": f"{val_start.date()} to {val_end.date()}",
        "test_period": f"{test_start.date()} to {test_end.date()}",
        "n_eligible_assets": n_eligible,
        "n_train_raw": len(idx_train),
        "n_train_elig": len(idx_train_elig),
        "n_val_elig": len(idx_val_elig),
        "n_test_elig": len(idx_test_elig),
    })

summary_df = pd.DataFrame(summary_rows)

# ============================================================
# 3  Reference window & capacity plan
# ============================================================
# The reference window (largest eligible training set) pins the
# parameter budget.  That budget is then held constant across all
# windows so that model complexity is comparable.

print("Computing capacity plan from reference window...\n")

ref_idx = int(summary_df["n_train_elig"].values.argmax())
ref_window = windows[ref_idx]
ref_row = summary_df.iloc[ref_idx]

n_train_ref_raw = int(ref_row["n_train_raw"])
n_train_ref_elig = int(ref_row["n_train_elig"])
n_eligible_ref = int(ref_row["n_eligible_assets"])

# The peak eligible-asset count may occur in a different window
# (not all assets appear in every test block simultaneously).
max_elig_idx = int(summary_df["n_eligible_assets"].values.argmax())
n_eligible_max = int(summary_df.iloc[max_elig_idx]["n_eligible_assets"])
max_elig_year = windows[max_elig_idx]["test_year"]

print(f"Reference window: test year {ref_window['test_year']} (max n_train_elig)")
print(f"  Training period:          {ref_window['train_start'].date()} → {ref_window['train_end'].date()}")
print(f"  Training samples (raw):   {n_train_ref_raw:,}")
print(f"  Training samples (elig):  {n_train_ref_elig:,}")
print(f"  Peak eligible assets:     {n_eligible_max} "
      f"(at test year {max_elig_year})\n")

# Joint model budget
params_trunk = int(JOINT_PARAMS_TOTAL * TRUNK_FRACTION)
params_heads_total = int(JOINT_PARAMS_TOTAL * HEAD_FRACTION)
params_per_head = params_heads_total // n_assets

# Separate-model budget (equal share of the joint total per asset)
params_sep_total = JOINT_PARAMS_TOTAL // n_assets
params_sep_trunk = int(params_sep_total * TRUNK_FRACTION)
params_sep_head = int(params_sep_total * HEAD_FRACTION)

# Implied parameter-to-sample ratios at the reference window
P_over_n_raw = JOINT_PARAMS_TOTAL / n_train_ref_raw
P_over_n_elig = JOINT_PARAMS_TOTAL / n_train_ref_elig

capacity_plan = {
    "reference_window": {
        "test_year": int(ref_window["test_year"]),
        "train_period": f"{ref_window['train_start'].date()} to {ref_window['train_end'].date()}",
        "n_train_raw": n_train_ref_raw,
        "n_train_eligible": n_train_ref_elig,
        "n_eligible_assets": n_eligible_ref,
        "selected_by": "max_n_train_elig",
    },
    "P_over_n_at_reference_raw": float(P_over_n_raw),
    "P_over_n_at_reference_elig": float(P_over_n_elig),
    "joint_model": {
        "params_total": JOINT_PARAMS_TOTAL,
        "params_trunk": params_trunk,
        "params_heads_total": params_heads_total,
        "params_per_head": params_per_head,
        "trunk_fraction": TRUNK_FRACTION,
        "head_fraction": HEAD_FRACTION,
    },
    "separate_models": {
        "n_models": n_assets,
        "params_per_model": params_sep_total,
        "params_trunk": params_sep_trunk,
        "params_head": params_sep_head,
        "total_params_all_models": params_sep_total * n_assets,
    },
}

print("Capacity plan (fixed across all windows):")
print(f"  Joint model:    {JOINT_PARAMS_TOTAL / 1e6:.1f}M params "
      f"(trunk {params_trunk / 1e6:.1f}M, "
      f"heads {params_heads_total / 1e6:.1f}M, "
      f"{params_per_head:,} per head)")
print(f"  Separate models: {params_sep_total / 1e3:.1f}k params each "
      f"(trunk {params_sep_trunk / 1e3:.1f}k, head {params_sep_head / 1e3:.1f}k)")
print(f"  Reference P/n (raw):  {P_over_n_raw:.3f}")
print(f"  Reference P/n (elig): {P_over_n_elig:.3f}\n")

# ============================================================
# 4  Save outputs
# ============================================================
config = {
    "metadata": {
        "dataset_path": NPZ_PATH,
        "n_examples": n_examples,
        "n_assets": n_assets,
        "date_range": f"{sample_dates.min().date()} to {sample_dates.max().date()}",
        "feature_block": FEATURE_BLOCK,
        "lookback_L": L,
        "first_test_year": FIRST_TEST_YEAR,
        "last_test_year": LAST_TEST_YEAR,
        "last_test_date": LAST_TEST_DATE,
        "min_train_per_asset": MIN_TRAIN_PER_ASSET,
    },
    "capacity_plan": capacity_plan,
    "windows": windows,
    "id_to_stock": id_to_stock,
}

with open(OUT_PKL, "wb") as f:
    pickle.dump(config, f)

summary_df.to_csv(OUT_SUMMARY_CSV, index=False)

print(f"Saved: {OUT_PKL}, {OUT_SUMMARY_CSV}\n")
print("Window summary:")
print(summary_df.to_string(index=False))