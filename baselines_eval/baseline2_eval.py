"""
Baseline 2: Fully Pooled Mean Predictor
===============================================================
For each expanding-window split, estimates a single grand mean on the
training set and predicts that constant for every test observation.
Serves as the simplest "unconditional expectation" benchmark.

Inputs
------
- evaluation_config_F1_L30.pkl : expanding-window split definitions
- dataset_F1_L30.npz          : targets, asset IDs, and dates

Outputs
-------
- baseline2_pooledmean_F1_L30_results.csv : per-window summary table
- baseline2_pooledmean_F1_L30.pkl         : full results + per-window detail arrays
"""

import pickle

import numpy as np
import pandas as pd

# ============================================================
# Configuration
# ============================================================
EVAL_PKL = "evaluation_config_F1_L30.pkl"
NPZ_PATH = "dataset_F1_L30.npz"

OUT_CSV = "baseline2_pooledmean_F1_L30_results.csv"
OUT_PKL = "baseline2_pooledmean_F1_L30.pkl"


# ============================================================
# Metrics
# ============================================================
def micro_mse(y_true: np.ndarray, y_hat: np.ndarray) -> float:
    """MSE computed over all pooled observations (observation-weighted)."""
    return float(np.mean((y_true - y_hat) ** 2, dtype=np.float64))


def macro_mse(
    y_true: np.ndarray,
    y_hat: np.ndarray,
    asset_id: np.ndarray,
    n_assets: int,
) -> float:
    """Per-asset MSE averaged equally across assets present in the test set."""
    sq_err = (y_true - y_hat) ** 2
    sums = np.bincount(asset_id, weights=sq_err, minlength=n_assets).astype(np.float64)
    counts = np.bincount(asset_id, minlength=n_assets).astype(np.int64)

    present = counts > 0
    per_asset = np.zeros(n_assets, dtype=np.float64)
    per_asset[present] = sums[present] / counts[present]

    return float(per_asset[present].mean(dtype=np.float64))


# ============================================================
# Main
# ============================================================
def main():
    print("=== Baseline 2: Fully Pooled Mean ===\n")

    # ============================================================
    # 1  Load evaluation config & dataset
    # ============================================================
    with open(EVAL_PKL, "rb") as f:
        cfg = pickle.load(f)

    data = np.load(NPZ_PATH, allow_pickle=True)
    y_all = data["y"].astype(np.float32)
    asset_all = data["asset_id"].astype(np.int32)
    date_all = pd.DatetimeIndex(pd.to_datetime(data["date"]))

    n_assets = int(cfg["metadata"]["n_assets"])

    print(f"Dataset: {len(y_all):,} obs, {n_assets} assets, "
          f"{len(cfg['windows'])} windows\n")

    # ============================================================
    # 2  Expanding-window evaluation loop
    # ============================================================
    rows = []
    per_window_details = {}

    for w in cfg["windows"]:
        yr = int(w["test_year"])
        idx_train = w["idx_train_elig"].astype(np.int64)
        idx_test = w["idx_test_elig"].astype(np.int64)

        # Train: estimate pooled mean
        y_train = y_all[idx_train].astype(np.float64)
        mu_pool = float(np.mean(y_train, dtype=np.float64))

        # Test: constant prediction
        y_true = y_all[idx_test].astype(np.float64)
        a_test = asset_all[idx_test].astype(np.int32)
        d_test = date_all[idx_test]
        y_hat = np.full_like(y_true, mu_pool, dtype=np.float64)

        mse_mi = micro_mse(y_true, y_hat)
        mse_ma = macro_mse(y_true, y_hat, a_test, n_assets)

        n_test = len(idx_test)
        n_assets_test = int(np.unique(a_test).size)

        rows.append(dict(
            test_year=yr,
            test_start=str(d_test.min().date()) if n_test else None,
            test_end=str(d_test.max().date()) if n_test else None,
            n_test=n_test,
            n_assets_in_test=n_assets_test,
            mu_pool=mu_pool,
            mse_micro=mse_mi,
            mse_macro=mse_ma,
        ))

        per_window_details[yr] = dict(
            mu_pool=np.float32(mu_pool),
            idx_test_elig=idx_test,
            y_true=y_true.astype(np.float32),
            y_hat=y_hat.astype(np.float32),
            asset_id=a_test.astype(np.int32),
            date=d_test.to_numpy(dtype="datetime64[ns]"),
        )

        print(f"  [{yr}]  mu={mu_pool:+.3e}  n={n_test:,}  assets={n_assets_test:3d}  "
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
            baseline="pooled_mean",
            description="Fully pooled historical mean estimated on training window",
            n_assets=n_assets,
        ),
        window_results=df,
        window_details=per_window_details,
    )
    with open(OUT_PKL, "wb") as f:
        pickle.dump(out, f)

    print(f"\nSaved: {OUT_CSV}, {OUT_PKL}")
    print(f"\nSummary (avg across windows):")
    print(f"  micro MSE: {df['mse_micro'].mean():.6g}")
    print(f"  macro MSE: {df['mse_macro'].mean():.6g}")


if __name__ == "__main__":
    main()