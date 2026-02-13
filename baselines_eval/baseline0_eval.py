"""
Baseline 0: Zero Predictor
==================================================
Evaluates the trivial y_hat = 0 baseline across all feature blocks
using the expanding-window setup.

For each feature block (F1, F2, F3) and each test window, the script
computes micro MSE (observation-weighted) and macro MSE (equal-weighted
across assets), then saves per-window results and diagnostics.

Inputs
------
- evaluation_config_<FB>_L30.pkl : expanding-window split definitions
- dataset_<FB>_L30.npz           : y, asset_id, date arrays

Outputs
-------
- baseline0_zero_<FB>_L30_results.csv : per-window metrics table
- baseline0_zero_<FB>_L30.pkl         : full results including per-asset
                                         MSE components and optional
                                         y_true / date arrays
"""

import pickle
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

# ============================================================
# Configuration
# ============================================================
FEATURE_BLOCKS = ["F1", "F2", "F3"]
L = 30

# When True the per-window PKL includes y_true / date arrays
# (useful for downstream diagnostics but increases file size).
STORE_YTRUE_AND_DATES = True


# ============================================================
# Metrics
# ============================================================
def micro_mse_zero(y_true: np.ndarray) -> float:
    """Micro MSE for y_hat = 0: simply mean(y^2)."""
    return float(np.mean(np.square(y_true), dtype=np.float64))


def macro_mse_zero(
    y_true: np.ndarray, asset_id: np.ndarray, n_assets: int
) -> float:
    """Macro MSE for y_hat = 0: equal-weighted average of per-asset
    mean(y_i^2), computed only over assets present in the test set."""
    sq = np.square(y_true)
    sums = np.bincount(asset_id, weights=sq, minlength=n_assets).astype(np.float64)
    counts = np.bincount(asset_id, minlength=n_assets).astype(np.int64)

    present = counts > 0
    per_asset = np.zeros(n_assets, dtype=np.float64)
    per_asset[present] = sums[present] / counts[present]
    return float(per_asset[present].mean(dtype=np.float64))


# ============================================================
# Single-block runner
# ============================================================
def run_one_block(
    feature_block: str, L: int
) -> Tuple[pd.DataFrame, Dict[int, Dict[str, Any]], Dict[str, Any]]:
    """Evaluate the zero baseline on every expanding window for one
    feature block.  Returns (results_df, per_window_details, metadata)."""

    eval_pkl = f"evaluation_config_{feature_block}_L{L}.pkl"
    npz_path = f"dataset_{feature_block}_L{L}.npz"
    out_csv = f"baseline0_zero_{feature_block}_L{L}_results.csv"
    out_pkl = f"baseline0_zero_{feature_block}_L{L}.pkl"

    print(f"\n{'=' * 60}")
    print(f"Baseline 0: Zero predictor | {feature_block}, L={L}")
    print(f"{'=' * 60}")

    # ---- Load evaluation config and dataset ----
    print(f"Loading eval config : {eval_pkl}")
    with open(eval_pkl, "rb") as f:
        cfg = pickle.load(f)

    print(f"Loading dataset     : {npz_path}")
    data = np.load(npz_path, allow_pickle=True)
    y_all = data["y"].astype(np.float32)
    asset_all = data["asset_id"].astype(np.int32)
    date_all = pd.DatetimeIndex(pd.to_datetime(data["date"]))

    if isinstance(cfg, dict) and "metadata" in cfg and "n_assets" in cfg["metadata"]:
        n_assets = int(cfg["metadata"]["n_assets"])
    else:
        n_assets = int(asset_all.max()) + 1

    # ---- Iterate over expanding windows ----
    rows = []
    per_window_details: Dict[int, Dict[str, Any]] = {}

    for w in cfg["windows"]:
        year = int(w["test_year"])
        idx = w["idx_test_elig"].astype(np.int64)

        y_true = y_all[idx].astype(np.float64)
        a_true = asset_all[idx].astype(np.int32)
        d_true = date_all[idx]

        mse_micro = micro_mse_zero(y_true)
        mse_macro = macro_mse_zero(y_true, a_true, n_assets)

        n_test = len(idx)
        n_assets_test = int(np.unique(a_true).size) if n_test else 0
        test_start = str(d_true.min().date()) if n_test else None
        test_end = str(d_true.max().date()) if n_test else None

        rows.append(dict(
            test_year=year,
            test_start=test_start,
            test_end=test_end,
            n_test=n_test,
            n_assets_in_test=n_assets_test,
            mse_micro=mse_micro,
            mse_macro=mse_macro,
        ))

        # Per-asset MSE breakdown (for macro diagnostics / plots)
        sq = np.square(y_true)
        sums = np.bincount(a_true, weights=sq, minlength=n_assets).astype(np.float64)
        counts = np.bincount(a_true, minlength=n_assets).astype(np.int64)
        present = counts > 0
        per_asset_mse = np.full(n_assets, np.nan, dtype=np.float64)
        per_asset_mse[present] = sums[present] / counts[present]

        details: Dict[str, Any] = dict(
            idx_test_elig=idx,
            per_asset_mse=per_asset_mse,
            per_asset_counts=counts,
        )
        if STORE_YTRUE_AND_DATES:
            details["y_true"] = y_true.astype(np.float32)
            details["asset_id"] = a_true
            details["date"] = d_true.to_numpy(dtype="datetime64[ns]")

        per_window_details[year] = details

        print(f"  [{year}]  n_test={n_test:>7,}  assets={n_assets_test:>3}  "
              f"micro={mse_micro:.6g}  macro={mse_macro:.6g}")

    # ---- Save results ----
    df = pd.DataFrame(rows).sort_values("test_year")
    df.to_csv(out_csv, index=False)

    metadata = dict(
        eval_config_path=eval_pkl,
        dataset_path=npz_path,
        baseline="zero",
        y_hat="0",
        feature_block=feature_block,
        lookback_L=L,
        n_assets=n_assets,
        store_ytrue_and_dates=bool(STORE_YTRUE_AND_DATES),
    )
    with open(out_pkl, "wb") as f:
        pickle.dump(dict(
            metadata=metadata,
            window_results=df,
            window_details=per_window_details,
        ), f)

    print(f"\nSaved: {out_csv}, {out_pkl}")
    print(f"Avg across windows — micro MSE: {df['mse_micro'].mean():.6g}, "
          f"macro MSE: {df['mse_macro'].mean():.6g}")

    return df, per_window_details, metadata


# ============================================================
# Main: run all feature blocks
# ============================================================
def main():
    all_summaries = []
    for fb in FEATURE_BLOCKS:
        df, _, _ = run_one_block(fb, L=L)
        all_summaries.append(dict(
            feature_block=fb,
            lookback_L=L,
            n_windows=int(df.shape[0]),
            avg_micro_mse=float(df["mse_micro"].mean()),
            avg_macro_mse=float(df["mse_macro"].mean()),
        ))

    print(f"\n{'=' * 60}")
    print("Zero-baseline summary across feature blocks")
    print(f"{'=' * 60}")
    print(pd.DataFrame(all_summaries).to_string(index=False))


if __name__ == "__main__":
    main()