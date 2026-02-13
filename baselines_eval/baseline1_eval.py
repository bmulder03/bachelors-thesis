"""
Baseline 1: Separate Historical Mean
=====================================================================
For each expanding-window split, estimate a per-asset mean return on
the training set and use it as the constant prediction on the test set.
This is the simplest baseline: it captures whether an asset's
unconditional average return has any out-of-sample predictive power.

Inputs
------
- evaluation_config_F1_L30.pkl : expanding-window split definitions
- dataset_F1_L30.npz           : feature matrix, targets, asset IDs, dates

Outputs
-------
- baseline1_sepmean_F1_L30_results.csv : per-window MSE summary
- baseline1_sepmean_F1_L30.pkl         : full results + per-window detail
"""

import pickle

import numpy as np
import pandas as pd

# ============================================================
# Configuration
# ============================================================
EVAL_PKL = "evaluation_config_F1_L30.pkl"
NPZ_PATH = "dataset_F1_L30.npz"

OUT_CSV = "baseline1_sepmean_F1_L30_results.csv"
OUT_PKL = "baseline1_sepmean_F1_L30.pkl"


# ============================================================
# Metrics
# ============================================================
def micro_mse(y_true: np.ndarray, y_hat: np.ndarray) -> float:
    """Pool all observations and compute a single MSE."""
    return float(np.mean((y_true - y_hat) ** 2, dtype=np.float64))


def macro_mse(y_true: np.ndarray, y_hat: np.ndarray,
              asset_id: np.ndarray, n_assets: int) -> float:
    """Compute per-asset MSE, then equal-weight across assets present
    in the test set.  This prevents large-cap stocks with many
    observations from dominating the loss."""
    sq_err = (y_true - y_hat) ** 2

    sums   = np.bincount(asset_id, weights=sq_err, minlength=n_assets).astype(np.float64)
    counts = np.bincount(asset_id,                  minlength=n_assets).astype(np.int64)

    present = counts > 0
    per_asset = np.zeros(n_assets, dtype=np.float64)
    per_asset[present] = sums[present] / counts[present]

    return float(per_asset[present].mean(dtype=np.float64))


# ============================================================
# Main
# ============================================================
def main():
    print("=== Baseline 1: Separate Historical Mean ===\n")

    # ============================================================
    # 1  Load evaluation config & dataset
    # ============================================================
    with open(EVAL_PKL, "rb") as f:
        cfg = pickle.load(f)

    data = np.load(NPZ_PATH, allow_pickle=True)
    y_all     = data["y"].astype(np.float32)
    asset_all = data["asset_id"].astype(np.int32)
    date_all  = pd.DatetimeIndex(pd.to_datetime(data["date"]))

    n_assets = int(cfg["metadata"]["n_assets"])

    print(f"Dataset: {len(y_all):,} obs, {n_assets} assets, "
          f"{len(cfg['windows'])} windows\n")

    # ============================================================
    # 2  Evaluate each expanding window
    # ============================================================
    rows = []
    per_window_details = {}

    for w in cfg["windows"]:
        Y = int(w["test_year"])

        idx_train = w["idx_train_elig"].astype(np.int64)
        idx_test  = w["idx_test_elig"].astype(np.int64)

        # --- Train: per-asset historical mean ---
        y_train = y_all[idx_train].astype(np.float64)
        a_train = asset_all[idx_train]

        train_sums   = np.bincount(a_train, weights=y_train, minlength=n_assets).astype(np.float64)
        train_counts = np.bincount(a_train,                   minlength=n_assets).astype(np.int64)

        mu = np.zeros(n_assets, dtype=np.float64)
        mu[train_counts > 0] = train_sums[train_counts > 0] / train_counts[train_counts > 0]

        # --- Test: predict the training mean for each asset ---
        y_true = y_all[idx_test].astype(np.float64)
        a_test = asset_all[idx_test]
        d_test = date_all[idx_test]
        y_hat  = mu[a_test]

        mse_mi = micro_mse(y_true, y_hat)
        mse_ma = macro_mse(y_true, y_hat, a_test, n_assets)

        n_test       = len(idx_test)
        n_assets_test = int(np.unique(a_test).size)

        rows.append(dict(
            test_year=Y,
            test_start=str(d_test.min().date()) if n_test else None,
            test_end=str(d_test.max().date()) if n_test else None,
            n_test=n_test,
            n_assets_in_test=n_assets_test,
            mse_micro=mse_mi,
            mse_macro=mse_ma,
        ))

        per_window_details[Y] = dict(
            mu=mu.astype(np.float32),
            idx_test_elig=idx_test,
            y_true=y_true.astype(np.float32),
            y_hat=y_hat.astype(np.float32),
            asset_id=a_test.astype(np.int32),
            date=d_test.to_numpy(dtype="datetime64[ns]"),
            train_counts=train_counts,
        )

        print(f"  [{Y}]  n_test={n_test:>7,}  assets={n_assets_test:>3}  "
              f"micro={mse_mi:.6g}  macro={mse_ma:.6g}")

    # ============================================================
    # 3  Save results
    # ============================================================
    df = pd.DataFrame(rows).sort_values("test_year")
    df.to_csv(OUT_CSV, index=False)

    out = dict(
        metadata=dict(
            eval_config_path=EVAL_PKL,
            dataset_path=NPZ_PATH,
            baseline="separate_mean",
            description="Per-asset historical mean estimated on expanding training window",
            n_assets=n_assets,
        ),
        window_results=df,
        window_details=per_window_details,
    )
    with open(OUT_PKL, "wb") as f:
        pickle.dump(out, f)

    print(f"\nSaved: {OUT_CSV}, {OUT_PKL}")
    print(f"\nAverage across windows:")
    print(f"  micro MSE: {df['mse_micro'].mean():.6g}")
    print(f"  macro MSE: {df['mse_macro'].mean():.6g}")


if __name__ == "__main__":
    main()