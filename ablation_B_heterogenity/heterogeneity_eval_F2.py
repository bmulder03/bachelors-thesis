"""
Ablation B - Heterogeneity Analysis (F2)
=====================================================================
Tests whether the joint (MTL) model helps some assets more than others,
bucketed by a training-set characteristic (sample size or zero-return
fraction).

For each expanding-window split, assets are re-bucketed into quintiles
using that window's training-set characteristics.  ΔMSE (joint - separate)
is computed per asset, averaged within each bucket per window, then
aggregated across windows (equal weight per window).  Negative ΔMSE means
the joint model outperforms the separate model.

Inputs
------
- evaluation_config_F2_L30.pkl    : window definitions & metadata
- dataset_F2_L30.npz              : full target / asset-id arrays
- sep_deep_F2_L30_cpupar/         : per-window separate-model predictions
- joint_mtl_F2_L30/               : per-window joint-model predictions

Outputs
-------
- heterogeneity_analysis_F2_L30/
    delta_loss_by_bucket_window_averaged.csv
    who_benefits_window_averaged.png
"""

import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

# ============================================================
# Configuration
# ============================================================
EVAL_PKL = "evaluation_config_F2_L30.pkl"
NPZ_DATASET = "dataset_F2_L30.npz"

SEP_DIR = Path("sep_deep_F2_L30_cpupar")
JOINT_DIR = Path("joint_mtl_F2_L30")

OUT_DIR = Path("heterogeneity_analysis_F2_L30")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_BY_BUCKET_CSV = OUT_DIR / "delta_loss_by_bucket_window_averaged.csv"
OUT_PLOT = OUT_DIR / "who_benefits_window_averaged.png"

BUCKET_BY = "sample_size"   # "sample_size" or "zero_returns"
N_BUCKETS = 5               # quintiles
EPS = 1e-12

# F2 ΔMSE values are very small; hard-code the y-axis scaling exponent
# rather than auto-detecting (F1 script auto-detects).
EXP_POWER = -6


# ============================================================
# Helper functions
# ============================================================
def compute_per_asset_mse(
    y_true: np.ndarray,
    y_hat: np.ndarray,
    asset_id: np.ndarray,
    n_assets: int,
) -> np.ndarray:
    """Return MSE per asset; assets with no test samples get NaN."""
    sq_err = (y_true - y_hat) ** 2
    sums = np.bincount(asset_id, weights=sq_err, minlength=n_assets).astype(np.float64)
    counts = np.bincount(asset_id, minlength=n_assets).astype(np.int64)

    out = np.full(n_assets, np.nan, dtype=np.float64)
    present = counts > 0
    out[present] = sums[present] / counts[present]
    return out


def load_window_preds(
    directory: Path,
    prefix: str,
    test_year: int,
    keys: Tuple[str, str, str],
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """Load y_true, y_hat, asset_id from a per-window .npz file."""
    path = directory / f"{prefix}{test_year}.npz"
    if not path.exists():
        return None, None, None
    d = np.load(path, allow_pickle=True)
    return (
        d[keys[0]].astype(np.float64),
        d[keys[1]].astype(np.float64),
        d[keys[2]].astype(np.int32),
    )


def compute_bucket_assignments(
    idx_train: np.ndarray,
    y_all: np.ndarray,
    asset_all: np.ndarray,
    n_assets: int,
    bucket_by: str,
    n_buckets: int,
) -> Tuple[np.ndarray, np.ndarray, str]:
    """Assign each asset to a quintile bucket based on training-set
    characteristics for the current window.

    Returns (stock_buckets, bucket_var, bucket_var_name) where
    stock_buckets[i] is in {-1, 0, ..., n_buckets-1} (-1 = not in train).
    """
    a_train = asset_all[idx_train].astype(np.int32)
    y_train = y_all[idx_train].astype(np.float64)

    train_counts = np.bincount(a_train, minlength=n_assets).astype(np.int64)

    # Zero-return fraction: proxy for illiquidity / data sparsity
    zero_counts = np.bincount(
        a_train[np.isclose(y_train, 0.0)], minlength=n_assets,
    ).astype(np.int64)
    zero_frac = np.full(n_assets, np.nan, dtype=np.float64)
    present = train_counts > 0
    zero_frac[present] = zero_counts[present] / (train_counts[present] + EPS)

    if bucket_by == "sample_size":
        bucket_var = train_counts.astype(np.float64)
        bucket_var_name = "Training sample size (window train)"
    elif bucket_by == "zero_returns":
        bucket_var = zero_frac
        bucket_var_name = "Zero-return fraction (window train)"
    else:
        raise ValueError(f"Unknown BUCKET_BY: {bucket_by}")

    # Only bucket assets that actually appear in the training set
    stocks_used = np.where(train_counts > 0)[0]
    stock_buckets = np.full(n_assets, -1, dtype=np.int32)
    if len(stocks_used) == 0:
        return stock_buckets, bucket_var, bucket_var_name

    vals = bucket_var[stocks_used]
    good = np.isfinite(vals)
    stocks_used = stocks_used[good]
    if len(stocks_used) == 0:
        return stock_buckets, bucket_var, bucket_var_name

    # Percentile-based bucket edges, recomputed fresh for this window
    edges = np.percentile(vals[good], np.linspace(0, 100, n_buckets + 1))
    edges[0] = -np.inf
    edges[-1] = np.inf

    stock_buckets[stocks_used] = np.clip(
        np.digitize(bucket_var[stocks_used], edges) - 1, 0, n_buckets - 1,
    )
    return stock_buckets, bucket_var, bucket_var_name


# ============================================================
# Main
# ============================================================
def main():
    print("=== Ablation B: Heterogeneity Analysis (Who Benefits?) — F2 ===\n")

    # ============================================================
    # 1  Load evaluation config and dataset arrays
    # ============================================================
    with open(EVAL_PKL, "rb") as f:
        cfg = pickle.load(f)

    n_assets = int(cfg["metadata"]["n_assets"])
    windows = cfg["windows"]
    if len(windows) == 0:
        raise RuntimeError("No windows found in eval config.")

    data = np.load(NPZ_DATASET, allow_pickle=True)
    y_all = data["y"].astype(np.float64)
    asset_all = data["asset_id"].astype(np.int32)

    # ============================================================
    # 2  Loop over windows: bucket assets, compute ΔMSE per bucket
    # ============================================================
    # Each list accumulates one value per window for a given bucket.
    bucket_means: Dict[int, List[float]] = {b: [] for b in range(N_BUCKETS)}
    bucket_nstocks: Dict[int, List[int]] = {b: [] for b in range(N_BUCKETS)}
    bucket_varmean: Dict[int, List[float]] = {b: [] for b in range(N_BUCKETS)}
    bucket_varmin: Dict[int, List[float]] = {b: [] for b in range(N_BUCKETS)}
    bucket_varmax: Dict[int, List[float]] = {b: [] for b in range(N_BUCKETS)}

    bucket_var_name_global = None
    used_years: List[int] = []

    for w in windows:
        year = int(w["test_year"])

        # Bucket assets using this window's training set
        idx_train = w["idx_train_elig"].astype(np.int64)
        stock_buckets, bucket_var, bucket_var_name = compute_bucket_assignments(
            idx_train, y_all, asset_all, n_assets, BUCKET_BY, N_BUCKETS,
        )
        bucket_var_name_global = bucket_var_name_global or bucket_var_name

        # Load separate and joint predictions for this test year
        yT_s, yH_s, a_s = load_window_preds(
            SEP_DIR, "sep_deep_preds_Y", year, ("y_test", "pred_test", "asset_id_test"),
        )
        yT_j, yH_j, a_j = load_window_preds(
            JOINT_DIR, "joint_mtl_preds_Y", year, ("y_true", "y_hat", "asset_id"),
        )
        if yT_s is None or yT_j is None:
            print(f"[{year}] Missing predictions — skipping.")
            continue

        # Truncate to common length if test sets differ in size
        if len(a_s) != len(a_j):
            print(f"[{year}] WARNING: test lengths differ "
                  f"(sep={len(a_s)}, joint={len(a_j)}); truncating to min.")
            m = min(len(a_s), len(a_j))
            yT_s, yH_s, a_s = yT_s[:m], yH_s[:m], a_s[:m]
            yT_j, yH_j, a_j = yT_j[:m], yH_j[:m], a_j[:m]

        # Keep only samples that are finite in both models and share the same asset
        ok = (
            np.isfinite(yH_s) & np.isfinite(yH_j)
            & np.isfinite(yT_s) & np.isfinite(yT_j)
            & (a_s == a_j)
        )
        if not ok.any():
            print(f"[{year}] WARNING: no valid overlapping samples — skipping.")
            continue

        y_true = yT_j[ok]
        a_true = a_j[ok]

        mse_sep = compute_per_asset_mse(y_true, yH_s[ok], a_true, n_assets)
        mse_joint = compute_per_asset_mse(y_true, yH_j[ok], a_true, n_assets)
        delta = mse_joint - mse_sep

        used_years.append(year)
        print(f"[{year}] ΔMSE computed from {int(ok.sum()):,} aligned test samples.")

        # Accumulate bucket-level summaries for this window
        for b in range(N_BUCKETS):
            stocks_b = np.where(stock_buckets == b)[0]
            if len(stocks_b) == 0:
                continue

            vals = delta[stocks_b]
            vals = vals[np.isfinite(vals)]
            if len(vals) == 0:
                continue

            bucket_means[b].append(float(np.mean(vals)))
            bucket_nstocks[b].append(len(stocks_b))

            bv = bucket_var[stocks_b]
            bv = bv[np.isfinite(bv)]
            if bv.size > 0:
                bucket_varmean[b].append(float(np.mean(bv)))
                bucket_varmin[b].append(float(np.min(bv)))
                bucket_varmax[b].append(float(np.max(bv)))

    if len(used_years) == 0:
        raise RuntimeError("No usable windows (no aligned predictions across models).")

    print(f"\nUsed {len(used_years)} windows: {min(used_years)}–{max(used_years)}")
    print(f"Bucketing by: {bucket_var_name_global}")

    # ============================================================
    # 3  Aggregate across windows
    # ============================================================
    # For each bucket: mean of per-window means ± SE across windows.
    rows = []
    for b in range(N_BUCKETS):
        means = np.array(bucket_means[b], dtype=np.float64)
        if means.size == 0:
            continue

        se = (
            float(np.std(means, ddof=1) / np.sqrt(len(means)))
            if len(means) > 1 else np.nan
        )

        rows.append({
            "bucket": b + 1,
            "n_windows": len(means),
            "avg_n_stocks_in_bucket": (
                float(np.mean(bucket_nstocks[b])) if bucket_nstocks[b] else np.nan
            ),
            "bucket_var_mean": (
                float(np.mean(bucket_varmean[b])) if bucket_varmean[b] else np.nan
            ),
            "bucket_var_min_mean": (
                float(np.mean(bucket_varmin[b])) if bucket_varmin[b] else np.nan
            ),
            "bucket_var_max_mean": (
                float(np.mean(bucket_varmax[b])) if bucket_varmax[b] else np.nan
            ),
            "mean_delta_mse": float(np.mean(means)),
            "se_delta_mse": se,
        })

    results_df = pd.DataFrame(rows).sort_values("bucket").reset_index(drop=True)
    results_df.to_csv(OUT_BY_BUCKET_CSV, index=False)

    print(f"\nSaved: {OUT_BY_BUCKET_CSV}")
    print("\n" + "=" * 80)
    print("HETEROGENEITY RESULTS (WINDOW-AVERAGED)")
    print("=" * 80)
    print(results_df.to_string(index=False))

    # ============================================================
    # 4  Plot: bucket-level ΔMSE with 90 % CI across windows
    # ============================================================
    ACCENT = "#1A5FDB"
    NEUTRAL_DARK = "#6B7280"
    NEUTRAL_LIGHT = "#B0B7C3"

    n_bars = len(results_df)
    if n_bars == 0:
        print("\nNo buckets to plot.")
        return

    # First bar accented, rest on a dark-to-light grey gradient
    palette = [ACCENT]
    if n_bars > 1:
        dark = np.array(mpl.colors.to_rgba(NEUTRAL_DARK)[:3])
        light = np.array(mpl.colors.to_rgba(NEUTRAL_LIGHT)[:3])
        for i in range(n_bars - 1):
            t = i / max(n_bars - 2, 1)
            palette.append(mpl.colors.rgb2hex(dark * (1 - t) + light * t))

    x_pos = np.arange(n_bars)
    ci_half = 1.645 * results_df["se_delta_mse"].values

    scale = 10 ** (-EXP_POWER)
    bar_vals = results_df["mean_delta_mse"].values * scale
    ci_scaled = ci_half * scale

    fig, ax = plt.subplots(figsize=(9, 5.5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ax.bar(
        x_pos, bar_vals, width=0.58,
        color=palette, edgecolor="white", linewidth=0.8, zorder=2,
    )

    # 90 % CI whiskers with caps
    for i in range(n_bars):
        ci = float(ci_scaled[i]) if np.isfinite(ci_scaled[i]) else 0.0
        if ci == 0.0:
            continue
        yc = float(bar_vals[i])
        ax.plot(
            [x_pos[i], x_pos[i]], [yc - ci, yc + ci],
            color="#374151", linewidth=1.2, solid_capstyle="butt", zorder=3,
        )
        for y_end in (yc - ci, yc + ci):
            ax.plot(
                [x_pos[i] - 0.08, x_pos[i] + 0.08], [y_end, y_end],
                color="#374151", linewidth=1.2, solid_capstyle="butt", zorder=3,
            )

    # X-tick labels: quintile number + approximate bucket range
    x_labels = []
    for _, r in results_df.iterrows():
        b = int(r["bucket"])
        if BUCKET_BY == "sample_size":
            lo = int(np.round(r["bucket_var_min_mean"])) if np.isfinite(r["bucket_var_min_mean"]) else 0
            hi = int(np.round(r["bucket_var_max_mean"])) if np.isfinite(r["bucket_var_max_mean"]) else 0
            x_labels.append(f"Q{b}\n[{lo}–{hi}]")
        else:
            lo = r["bucket_var_min_mean"] if np.isfinite(r["bucket_var_min_mean"]) else np.nan
            hi = r["bucket_var_max_mean"] if np.isfinite(r["bucket_var_max_mean"]) else np.nan
            x_labels.append(f"Q{b}\n[{lo:.2f}–{hi:.2f}]")

    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels, fontsize=9.5, color="#374151")
    ax.axhline(0, color="#9CA3AF", linewidth=0.9, linestyle="--", zorder=1)

    # Axis styling
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#D1D5DB")
        ax.spines[spine].set_linewidth(0.8)

    ax.set_ylabel(
        f"ΔMSE  (Joint − Separate)  ×  $10^{{{EXP_POWER}}}$",
        fontsize=11, color="#374151", labelpad=10,
    )
    ax.set_xlabel(
        ("Training sample size quintile (window-specific)" if BUCKET_BY == "sample_size"
         else "Zero-return fraction quintile (window-specific)"),
        fontsize=11, color="#374151", labelpad=12,
    )

    ax.tick_params(axis="y", colors="#6B7280", labelsize=9.5, length=3)
    ax.tick_params(axis="x", colors="#6B7280", length=0)
    ax.yaxis.grid(False)
    ax.set_axisbelow(True)

    ax.text(
        0.98, 0.04, "90 % CI across windows", transform=ax.transAxes,
        fontsize=8.5, color="#9CA3AF", va="bottom", ha="right", style="italic",
    )

    # Expand y-limits slightly so CI caps and labels don't clip
    ymin_cur, ymax_cur = ax.get_ylim()
    top = float(np.max(bar_vals + np.nan_to_num(ci_scaled, nan=0.0)))
    bot = float(np.min(bar_vals - np.nan_to_num(ci_scaled, nan=0.0)))
    pad = (ymax_cur - ymin_cur) * 0.08
    ax.set_ylim(min(ymin_cur, bot - pad * 0.6), max(ymax_cur, top + pad * 1.4))

    plt.tight_layout(pad=1.4)
    plt.savefig(OUT_PLOT, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"\nSaved plot: {OUT_PLOT}")
    print(f"Done. Outputs in: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()