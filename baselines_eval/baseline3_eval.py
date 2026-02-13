"""
Baseline 3: James-Stein Shrinkage Mean
==============================================================
Empirical-Bayes James-Stein shrinkage of per-asset mean returns toward
the pooled cross-sectional mean.  Includes diagnostics that quantify how
strongly the estimator collapses to the pooled mean (tau², shrinkage
weights, fraction of fully-pooled assets).

Inputs
------
- evaluation_config_F1_L30.pkl : evaluation windows (train/test indices)
- dataset_F1_L30.npz           : returns, asset IDs, dates

Outputs
-------
- baseline3_jamesstein_F1_L30_results.csv : per-window metrics & diagnostics
- baseline3_jamesstein_F1_L30.pkl         : full results incl. per-window arrays
"""

import pickle

import numpy as np
import pandas as pd

# ============================================================
# Configuration
# ============================================================
EVAL_PKL = "evaluation_config_F1_L30.pkl"
NPZ_PATH = "dataset_F1_L30.npz"

OUT_CSV = "baseline3_jamesstein_F1_L30_results.csv"
OUT_PKL = "baseline3_jamesstein_F1_L30.pkl"


# ============================================================
# Evaluation metrics
# ============================================================
def micro_mse(y_true: np.ndarray, y_hat: np.ndarray) -> float:
    """MSE computed over all observations equally (observation-weighted)."""
    return float(np.mean((y_true - y_hat) ** 2, dtype=np.float64))


def macro_mse(y_true: np.ndarray, y_hat: np.ndarray,
              asset_id: np.ndarray, n_assets: int) -> float:
    """MSE averaged across per-asset MSEs (equal weight per asset)."""
    sq_err = (y_true - y_hat) ** 2
    sums = np.bincount(asset_id, weights=sq_err, minlength=n_assets).astype(np.float64)
    counts = np.bincount(asset_id, minlength=n_assets).astype(np.int64)
    present = counts > 0
    per_asset = np.zeros(n_assets, dtype=np.float64)
    per_asset[present] = sums[present] / counts[present]
    return float(per_asset[present].mean(dtype=np.float64))


# ============================================================
# James–Stein estimator
# ============================================================
def fit_js_means(y_train: np.ndarray, a_train: np.ndarray, n_assets: int):
    """
    Shrink per-asset sample means toward the pooled mean.

    The between-asset variance of true means (tau²) is estimated as
    Var(mu_i) − mean(se²_i), floored at zero.  Each asset's shrinkage
    weight toward the pooled mean is  b_i = se²_i / (se²_i + tau²),
    so assets with noisier estimates shrink more.

    Returns
    -------
    mu_pool   : scalar pooled mean
    mu_sep    : (n_assets,) raw per-asset means
    mu_js     : (n_assets,) shrunken means
    n_i       : (n_assets,) observation counts
    s2_i      : (n_assets,) within-asset sample variance (ddof=1)
    se2_i     : (n_assets,) estimated variance of the sample mean (s2_i / n_i)
    tau2_raw  : scalar  Var(mu_sep) − mean(se2_i)  (can be negative)
    tau2      : scalar  max(tau2_raw, 0)
    shrink_i  : (n_assets,) shrinkage weight on pooled mean, in [0, 1]
    var_mu    : scalar  cross-sectional variance of mu_sep (ddof=1)
    mean_se2  : scalar  mean of se2_i across assets
    """
    y_train = y_train.astype(np.float64)
    a_train = a_train.astype(np.int32)

    # Per-asset sufficient statistics
    n_i = np.bincount(a_train, minlength=n_assets).astype(np.int64)
    sum_i = np.bincount(a_train, weights=y_train, minlength=n_assets).astype(np.float64)
    sumsq_i = np.bincount(a_train, weights=y_train ** 2, minlength=n_assets).astype(np.float64)

    mu_pool = float(y_train.mean(dtype=np.float64))

    # Raw per-asset means
    present = n_i > 0
    mu_sep = np.full(n_assets, np.nan, dtype=np.float64)
    mu_sep[present] = sum_i[present] / n_i[present]

    # Within-asset variance and variance of the sample mean
    ok_var = n_i > 1
    s2_i = np.full(n_assets, np.nan, dtype=np.float64)
    se2_i = np.full(n_assets, np.nan, dtype=np.float64)
    s2_i[ok_var] = (sumsq_i[ok_var] - sum_i[ok_var] ** 2 / n_i[ok_var]) / (n_i[ok_var] - 1)
    se2_i[ok_var] = s2_i[ok_var] / n_i[ok_var]

    # Between-asset variance of true means (empirical-Bayes moment estimate)
    mu_vec = mu_sep[ok_var]
    se2_vec = se2_i[ok_var]

    if mu_vec.size <= 1:
        var_mu = 0.0
        mean_se2 = float(np.mean(se2_vec)) if se2_vec.size else 0.0
    else:
        var_mu = float(np.var(mu_vec, ddof=1))
        mean_se2 = float(np.mean(se2_vec))

    tau2_raw = var_mu - mean_se2
    tau2 = max(tau2_raw, 0.0)

    # Shrinkage weights: b_i = 1 → full pooling, b_i = 0 → raw mean
    shrink_i = np.ones(n_assets, dtype=np.float64)
    valid = ok_var & np.isfinite(se2_i)

    if tau2 > 0.0:
        shrink_i[valid] = np.clip(se2_i[valid] / (se2_i[valid] + tau2), 0.0, 1.0)

    # Shrunken means (defaults to pooled mean for assets without data)
    mu_js = np.full(n_assets, mu_pool, dtype=np.float64)
    mu_js[present] = (1.0 - shrink_i[present]) * mu_sep[present] + shrink_i[present] * mu_pool

    return (mu_pool, mu_sep, mu_js, n_i, s2_i, se2_i,
            tau2_raw, tau2, shrink_i, var_mu, mean_se2)


# ============================================================
# Main
# ============================================================
def main():
    print("=== Baseline 3: James–Stein shrinkage mean ===\n")

    # ============================================================
    # 1  Load evaluation config & dataset
    # ============================================================
    print(f"Loading eval config: {EVAL_PKL}")
    with open(EVAL_PKL, "rb") as f:
        cfg = pickle.load(f)

    print(f"Loading dataset:     {NPZ_PATH}")
    data = np.load(NPZ_PATH, allow_pickle=True)
    y_all = data["y"].astype(np.float32)
    asset_all = data["asset_id"].astype(np.int32)
    date_all = pd.DatetimeIndex(pd.to_datetime(data["date"]))

    n_assets = int(cfg["metadata"]["n_assets"])

    # ============================================================
    # 2  Evaluate each rolling window
    # ============================================================
    rows = []
    per_window_details = {}

    for w in cfg["windows"]:
        year = int(w["test_year"])
        idx_train = w["idx_train_elig"].astype(np.int64)
        idx_test = w["idx_test_elig"].astype(np.int64)

        # --- Fit on training data ---
        y_train = y_all[idx_train]
        a_train = asset_all[idx_train]

        (mu_pool, mu_sep, mu_js, n_i, s2_i, se2_i,
         tau2_raw, tau2, shrink_i, var_mu, mean_se2) = fit_js_means(
            y_train, a_train, n_assets,
        )

        # --- Predict on test data ---
        y_true = y_all[idx_test].astype(np.float64)
        a_test = asset_all[idx_test].astype(np.int32)
        d_test = date_all[idx_test]
        y_hat = mu_js[a_test].astype(np.float64)

        mse_micro = micro_mse(y_true, y_hat)
        mse_macro = macro_mse(y_true, y_hat, a_test, n_assets)

        # --- Shrinkage diagnostics ---
        eligible = np.flatnonzero(n_i > 1)
        if eligible.size:
            sh = shrink_i[eligible]
            mean_shrink = float(np.mean(sh))
            med_shrink = float(np.median(sh))
            p10_shrink = float(np.quantile(sh, 0.10))
            p90_shrink = float(np.quantile(sh, 0.90))
            min_shrink = float(np.min(sh))
            max_shrink = float(np.max(sh))
            frac_full_pool = float(np.mean(np.isclose(sh, 1.0)))
        else:
            mean_shrink = med_shrink = p10_shrink = p90_shrink = np.nan
            min_shrink = max_shrink = frac_full_pool = np.nan

        n_test = len(idx_test)
        test_start = str(d_test.min().date()) if n_test else None
        test_end = str(d_test.max().date()) if n_test else None

        rows.append(dict(
            test_year=year,
            test_start=test_start,
            test_end=test_end,
            n_test=n_test,
            n_assets_in_test=int(np.unique(a_test).size),
            mu_pool=mu_pool,
            var_mu=var_mu,
            mean_se2=mean_se2,
            tau2_raw=tau2_raw,
            tau2=tau2,
            mean_shrink=mean_shrink,
            median_shrink=med_shrink,
            p10_shrink=p10_shrink,
            p90_shrink=p90_shrink,
            min_shrink=min_shrink,
            max_shrink=max_shrink,
            frac_full_pool=frac_full_pool,
            mse_micro=mse_micro,
            mse_macro=mse_macro,
        ))

        per_window_details[year] = dict(
            mu_pool=np.float32(mu_pool),
            mu_sep=mu_sep.astype(np.float32),
            mu_js=mu_js.astype(np.float32),
            n_i=n_i.astype(np.int32),
            s2_i=s2_i.astype(np.float32),
            se2_i=se2_i.astype(np.float32),
            tau2_raw=float(tau2_raw),
            tau2=float(tau2),
            shrink_i=shrink_i.astype(np.float32),
            idx_test_elig=idx_test,
            y_true=y_true.astype(np.float32),
            y_hat=y_hat.astype(np.float32),
            asset_id=a_test.astype(np.int32),
            date=d_test.to_numpy(dtype="datetime64[ns]"),
        )

        print(f"[{year}] var(mu_i)={var_mu:.3e}  mean(se²_i)={mean_se2:.3e}  "
              f"tau²={tau2:.3e}  mean_shrink={mean_shrink:.4f}  "
              f"frac_pool={frac_full_pool:.3f}  "
              f"micro={mse_micro:.6g}  macro={mse_macro:.6g}")

    # ============================================================
    # 3  Save results
    # ============================================================
    df = pd.DataFrame(rows).sort_values("test_year")
    df.to_csv(OUT_CSV, index=False)

    out = dict(
        metadata=dict(
            eval_config_path=EVAL_PKL,
            dataset_path=NPZ_PATH,
            baseline="james_stein_mean",
            description=(
                "Empirical-Bayes James–Stein shrinkage of per-asset means "
                "toward pooled mean (with collapse diagnostics)"
            ),
            n_assets=n_assets,
            min_train_per_asset=int(cfg["metadata"].get("min_train_per_asset", -1)),
        ),
        window_results=df,
        window_details=per_window_details,
    )
    with open(OUT_PKL, "wb") as f:
        pickle.dump(out, f)

    print(f"\nSaved: {OUT_CSV}, {OUT_PKL}")

    # ============================================================
    # 4  Summary
    # ============================================================
    print(f"\nAverage across windows:")
    print(f"  micro MSE: {df['mse_micro'].mean():.6g}")
    print(f"  macro MSE: {df['mse_macro'].mean():.6g}")

    share_tau2_zero = float(np.mean(np.isclose(df["tau2"].to_numpy(), 0.0)))
    avg_frac_pool = float(np.nanmean(df["frac_full_pool"].to_numpy()))
    print(f"\nCollapse diagnostics:")
    print(f"  Windows with tau² = 0:              {share_tau2_zero:.3f}")
    print(f"  Avg fraction fully pooled (b_i=1):  {avg_frac_pool:.3f}")
    print(f"  Avg mean shrinkage weight:          {df['mean_shrink'].mean():.4f}")


if __name__ == "__main__":
    main()