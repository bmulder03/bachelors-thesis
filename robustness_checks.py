"""
Robustness Checks
=========================================
Paired robustness comparisons between the joint (MTL) and separate
(single-task) models, using saved prediction files. No retraining is
performed; all checks operate on previously generated NPZ outputs.

Checks performed
----------------
(A) Paired year-level inference on MSE deltas (joint - separate):
    win rate, one-sided sign test, and bootstrap 95 % CI on the mean delta.
(B) Tail-robust evaluation: MAE and trimmed MSE (drop top q % of squared
    errors) to verify that results are not driven by outlier predictions.

Inputs
------
- joint_mtl_{fb}_L30/joint_mtl_preds_Y{year}.npz
      keys: idx_test_elig, y_true, y_hat
- sep_deep_{fb}_L30_cpupar/sep_deep_preds_Y{year}.npz
      keys: idx_test_elig, y_test, pred_test

Outputs
-------
- robustness_summary/{fb}_robust_metrics_by_year.csv
- robustness_summary/ALL_robust_metrics_by_year.csv
- robustness_summary/ALL_year_level_inference.csv
- robustness_summary/ALL_robust_avg_summary.csv
"""

from __future__ import annotations

from math import lgamma
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

try:
    from scipy.stats import binomtest
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False

# ============================================================
# Configuration
# ============================================================
L = 30
FEATURE_BLOCKS = ["F1", "F2", "F3"]
YEARS = list(range(2011, 2025 + 1))

JOINT_DIR_TPL = "joint_mtl_{fb}_L30"
SEP_DIR_TPL = "sep_deep_{fb}_L30_cpupar"

TRIM_Q = 0.01       # drop top 1 % of squared errors for trimmed MSE
N_BOOT = 50_000     # bootstrap replications for CI on mean delta
BOOT_SEED = 0

OUT_DIR = Path("robustness_summary")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# NPZ loading
# ============================================================
def joint_npz_path(fb: str, year: int) -> Path:
    return Path(JOINT_DIR_TPL.format(fb=fb)) / f"joint_mtl_preds_Y{year}.npz"


def sep_npz_path(fb: str, year: int) -> Path:
    return Path(SEP_DIR_TPL.format(fb=fb)) / f"sep_deep_preds_Y{year}.npz"


def load_joint_npz(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    z = np.load(path, allow_pickle=False)
    idx = z["idx_test_elig"].astype(np.int64)
    y = z["y_true"].astype(np.float64)
    yhat = z["y_hat"].astype(np.float64)
    return idx, y, yhat


def load_sep_npz(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    z = np.load(path, allow_pickle=False)
    idx = z["idx_test_elig"].astype(np.int64)
    y = z["y_test"].astype(np.float64)
    yhat = z["pred_test"].astype(np.float64)
    ok = np.isfinite(yhat)
    return idx[ok], y[ok], yhat[ok]


# ============================================================
# Sample alignment
# ============================================================
def align_on_idx(
    idx_j: np.ndarray, y_j: np.ndarray, yhat_j: np.ndarray,
    idx_s: np.ndarray, y_s: np.ndarray, yhat_s: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Intersect joint and separate predictions on their shared global
    test indices and return aligned (y, e_joint, e_sep) arrays."""
    pos_j = {int(i): k for k, i in enumerate(idx_j)}
    jj, ss = [], []
    for k, i in enumerate(idx_s):
        pj = pos_j.get(int(i))
        if pj is not None:
            jj.append(pj)
            ss.append(k)

    if len(jj) == 0:
        raise RuntimeError("No overlap between joint and separate indices.")

    jj = np.asarray(jj, dtype=np.int64)
    ss = np.asarray(ss, dtype=np.int64)

    y = y_s[ss]
    e_joint = y - yhat_j[jj]
    e_sep = y - yhat_s[ss]
    return y, e_joint, e_sep


# ============================================================
# Metrics
# ============================================================
def mse(e: np.ndarray) -> float:
    return float(np.mean(e * e, dtype=np.float64))


def mae(e: np.ndarray) -> float:
    return float(np.mean(np.abs(e), dtype=np.float64))


def trimmed_mse(e: np.ndarray, trim_q: float) -> float:
    """MSE after dropping the top `trim_q` fraction of squared errors,
    reducing sensitivity to extreme outlier predictions."""
    se = e * e
    if trim_q <= 0:
        return float(np.mean(se, dtype=np.float64))
    cutoff = np.quantile(se, 1.0 - trim_q)
    keep = se <= cutoff
    if not np.any(keep):
        return float(np.mean(se, dtype=np.float64))
    return float(np.mean(se[keep], dtype=np.float64))


# ============================================================
# Statistical inference helpers
# ============================================================
def sign_test_p_one_sided(wins: int, n: int) -> float:
    """One-sided sign-test p-value: P(X >= wins) under H0: p = 0.5.
    Uses scipy if available, otherwise an exact log-combinatorial sum."""
    if HAVE_SCIPY:
        return float(binomtest(wins, n, p=0.5, alternative="greater").pvalue)

    def log_comb(n_, k_):
        return lgamma(n_ + 1) - lgamma(k_ + 1) - lgamma(n_ - k_ + 1)

    logs = np.array([log_comb(n, k) for k in range(wins, n + 1)], dtype=np.float64)
    m = float(np.max(logs))
    tail = float(np.sum(np.exp(logs - m)))
    p = np.exp(m) * tail / (2.0 ** n)
    return float(min(max(p, 0.0), 1.0))


def bootstrap_ci_mean(d: np.ndarray, n_boot: int, seed: int) -> Tuple[float, float]:
    """Non-parametric bootstrap 95 % CI for the mean of `d`."""
    rng = np.random.default_rng(seed)
    n = len(d)
    means = rng.choice(d, size=(n_boot, n), replace=True).mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def year_level_summary(df: pd.DataFrame, delta_col: str, seed: int) -> Dict[str, float]:
    """Aggregate year-level deltas into win rate, sign-test p-value,
    and a bootstrap CI on the mean delta."""
    d = df[delta_col].to_numpy(dtype=float)
    n = len(d)
    wins = int(np.sum(d < 0.0))
    lo, hi = bootstrap_ci_mean(d, n_boot=N_BOOT, seed=seed)
    return {
        "n_years": n,
        "mean_delta": float(np.mean(d)),
        "median_delta": float(np.median(d)),
        "win_rate": float(wins / n),
        "sign_test_p_one_sided": sign_test_p_one_sided(wins, n),
        "ci95_mean_delta_lo": lo,
        "ci95_mean_delta_hi": hi,
    }


# ============================================================
# Per-feature-block evaluation
# ============================================================
def run_block(fb: str) -> pd.DataFrame:
    """Load joint and separate predictions for every test year, align
    on shared indices, and compute MSE / MAE / trimmed-MSE deltas."""
    rows = []
    for year in YEARS:
        pj = joint_npz_path(fb, year)
        ps = sep_npz_path(fb, year)
        if not pj.exists():
            raise FileNotFoundError(f"Missing joint NPZ: {pj}")
        if not ps.exists():
            raise FileNotFoundError(f"Missing sep NPZ: {ps}")

        idx_j, y_j, yhat_j = load_joint_npz(pj)
        idx_s, y_s, yhat_s = load_sep_npz(ps)
        y, e_j, e_s = align_on_idx(idx_j, y_j, yhat_j, idx_s, y_s, yhat_s)

        mse_j, mse_s = mse(e_j), mse(e_s)
        mae_j, mae_s = mae(e_j), mae(e_s)
        tmse_j, tmse_s = trimmed_mse(e_j, TRIM_Q), trimmed_mse(e_s, TRIM_Q)

        rows.append({
            "feature_block": fb,
            "test_year": int(year),
            "n_common": int(len(y)),
            "mse_joint": mse_j,
            "mse_sep": mse_s,
            "delta_mse": mse_j - mse_s,
            "mae_joint": mae_j,
            "mae_sep": mae_s,
            "delta_mae": mae_j - mae_s,
            "mse_trim_joint": tmse_j,
            "mse_trim_sep": tmse_s,
            "delta_mse_trim": tmse_j - tmse_s,
            "trim_q": float(TRIM_Q),
        })

    df = pd.DataFrame(rows).sort_values("test_year")
    df.to_csv(OUT_DIR / f"{fb}_robust_metrics_by_year.csv", index=False)
    return df


# ============================================================
# Main
# ============================================================
def main():
    all_blocks = []
    inf_rows = []

    for fb in FEATURE_BLOCKS:
        df = run_block(fb)
        all_blocks.append(df)

        # Paired year-level inference for each metric
        for metric, dcol in [
            ("mse", "delta_mse"),
            ("mae", "delta_mae"),
            ("mse_trim", "delta_mse_trim"),
        ]:
            seed = BOOT_SEED + (hash((fb, metric)) % 10_000)
            summ = year_level_summary(df, dcol, seed=seed)
            summ.update({"feature_block": fb, "metric": metric, "trim_q": float(TRIM_Q)})
            inf_rows.append(summ)

    # Combined year-level metrics across all feature blocks
    df_all = pd.concat(all_blocks, ignore_index=True)
    df_all.to_csv(OUT_DIR / "ALL_robust_metrics_by_year.csv", index=False)

    # Year-level inference table (win rates, sign tests, bootstrap CIs)
    df_inf = pd.DataFrame(inf_rows).sort_values(["feature_block", "metric"])
    df_inf.to_csv(OUT_DIR / "ALL_year_level_inference.csv", index=False)

    # Equal-weight-per-year averages
    avg_rows = []
    for fb in FEATURE_BLOCKS:
        d = df_all[df_all["feature_block"] == fb]
        avg_rows.append({
            "feature_block": fb,
            "avg_delta_mse": float(d["delta_mse"].mean()),
            "avg_delta_mae": float(d["delta_mae"].mean()),
            "avg_delta_mse_trim": float(d["delta_mse_trim"].mean()),
            "avg_mse_joint": float(d["mse_joint"].mean()),
            "avg_mse_sep": float(d["mse_sep"].mean()),
            "avg_mae_joint": float(d["mae_joint"].mean()),
            "avg_mae_sep": float(d["mae_sep"].mean()),
            "avg_mse_trim_joint": float(d["mse_trim_joint"].mean()),
            "avg_mse_trim_sep": float(d["mse_trim_sep"].mean()),
            "trim_q": float(TRIM_Q),
        })
    df_avg = pd.DataFrame(avg_rows).sort_values("feature_block")
    df_avg.to_csv(OUT_DIR / "ALL_robust_avg_summary.csv", index=False)

    print(f"[OK] Wrote tables to: {OUT_DIR.resolve()}")
    print(f"Key file: {(OUT_DIR / 'ALL_year_level_inference.csv').resolve()}")


if __name__ == "__main__":
    main()