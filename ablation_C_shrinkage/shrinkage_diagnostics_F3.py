"""
Ablation C - Shrinkage / Variance-Decomposition Diagnostics (F3)
================================================================
Tests whether the joint (MTL) model behaves like a shrinkage estimator
relative to separately trained models: do predictions become less
dispersed across and within assets, and does that variance reduction
translate into lower MSE?

Per asset-year the script computes mean(pred), std(pred), and MSE for
both models, then summarises two dispersion channels:

  Cross-asset dispersion : std over assets of mean(pred), each year.
  Within-asset dispersion: average std(pred) across assets, each year.

Sign convention — negative deltas mean the joint model is smaller/better:
  ΔMSE      = MSE_joint - MSE_sep
  Δstd_pred = std_pred_joint - std_pred_sep

Inputs
------
- evaluation_config_F3_L30.pkl    : window definitions & metadata
- sep_deep_F3_L30_cpupar/         : per-window separate-model predictions
- joint_mtl_F3_L30/               : per-window joint-model predictions

Outputs
-------
- shrinkage_diagnostics_F3_L30/
    asset_year_stats.csv           : per asset-year statistics
    year_summary.csv               : year-level aggregates
    overall_summary.csv            : single-row grand summary
    plots/*.png                    : diagnostic figures
"""

import pickle
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# ============================================================
# Configuration
# ============================================================
FEATURE_BLOCK = "F3"
L = 30

EVAL_PKL = f"evaluation_config_{FEATURE_BLOCK}_L{L}.pkl"

SEP_DIR = Path(f"sep_deep_{FEATURE_BLOCK}_L{L}_cpupar")
JOINT_DIR = Path(f"joint_mtl_{FEATURE_BLOCK}_L{L}")

OUT_DIR = Path(f"shrinkage_diagnostics_{FEATURE_BLOCK}_L{L}")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_ASSET_YEAR_CSV = OUT_DIR / "asset_year_stats.csv"
OUT_YEAR_SUMMARY_CSV = OUT_DIR / "year_summary.csv"
OUT_OVERALL_CSV = OUT_DIR / "overall_summary.csv"

PLOT_DIR = OUT_DIR / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

MIN_TEST_OBS = 30   # minimum test observations per asset-year
EPS = 1e-12

# Threshold for "effectively zero" within-asset prediction std
NEAR_ZERO_THRESHOLD = 1e-12

# Plotting: multiply raw values by 10^PLOT_SCALE_POWER for readability,
# then label axes with "× 10^{-PLOT_SCALE_POWER}".
PLOT_SCALE_POWER = 4
PLOT_SCALE = 10 ** PLOT_SCALE_POWER

# Frozen orchestra: test year to visualize
ORCHESTRA_YEAR = 2020


# ============================================================
# I/O helpers
# ============================================================
def load_preds(
    directory: Path,
    prefix: str,
    test_year: int,
    keys: Tuple[str, str, str],
) -> Optional[Dict[str, np.ndarray]]:
    """Load y_true, y_hat, asset_id from a per-window .npz file."""
    path = directory / f"{prefix}{test_year}.npz"
    if not path.exists():
        return None
    d = np.load(path, allow_pickle=True)
    return {
        "y_true": d[keys[0]].astype(np.float64),
        "y_hat": d[keys[1]].astype(np.float64),
        "asset_id": d[keys[2]].astype(np.int32),
    }


def align_predictions(
    sep: Dict[str, np.ndarray],
    joint: Dict[str, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Align separate and joint predictions to the same valid samples.

    Returns (y_true, y_hat_sep, y_hat_joint, asset_id) restricted to
    samples that are finite in both models and share the same asset id.
    """
    yT_s, yH_s, a_s = sep["y_true"], sep["y_hat"], sep["asset_id"]
    yT_j, yH_j, a_j = joint["y_true"], joint["y_hat"], joint["asset_id"]

    # Truncate to common length if test sets differ in size
    m = min(len(yT_s), len(yT_j))
    yT_s, yH_s, a_s = yT_s[:m], yH_s[:m], a_s[:m]
    yT_j, yH_j, a_j = yT_j[:m], yH_j[:m], a_j[:m]

    ok = (
        np.isfinite(yT_s) & np.isfinite(yT_j)
        & np.isfinite(yH_s) & np.isfinite(yH_j)
        & (a_s == a_j)
    )
    return yT_j[ok], yH_s[ok], yH_j[ok], a_j[ok]


# ============================================================
# Statistical helpers
# ============================================================
def per_asset_pred_stats(
    y_true: np.ndarray,
    y_hat: np.ndarray,
    asset_id: np.ndarray,
    n_assets: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return per-asset MSE, mean(pred), std(pred), and count.
    Assets with no observations get NaN."""
    cnt = np.bincount(asset_id, minlength=n_assets).astype(np.int64)
    present = cnt > 0

    err2 = (y_true - y_hat) ** 2
    mse = np.full(n_assets, np.nan, dtype=np.float64)
    mse[present] = (
        np.bincount(asset_id, weights=err2, minlength=n_assets)[present]
        / cnt[present]
    )

    sum_pred = np.bincount(asset_id, weights=y_hat, minlength=n_assets).astype(np.float64)
    mean_pred = np.full(n_assets, np.nan, dtype=np.float64)
    mean_pred[present] = sum_pred[present] / cnt[present]

    sum_pred2 = np.bincount(asset_id, weights=y_hat ** 2, minlength=n_assets).astype(np.float64)
    var_pred = np.full(n_assets, np.nan, dtype=np.float64)
    var_pred[present] = np.maximum(0.0, sum_pred2[present] / cnt[present] - mean_pred[present] ** 2)
    std_pred = np.sqrt(var_pred)

    return mse, mean_pred, std_pred, cnt


def per_asset_true_stats(
    y_true: np.ndarray,
    asset_id: np.ndarray,
    n_assets: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return per-asset mean(y_true) and std(y_true)."""
    cnt = np.bincount(asset_id, minlength=n_assets).astype(np.int64)
    present = cnt > 0

    sum_y = np.bincount(asset_id, weights=y_true, minlength=n_assets).astype(np.float64)
    mean_y = np.full(n_assets, np.nan, dtype=np.float64)
    mean_y[present] = sum_y[present] / cnt[present]

    sum_y2 = np.bincount(asset_id, weights=y_true ** 2, minlength=n_assets).astype(np.float64)
    var_y = np.full(n_assets, np.nan, dtype=np.float64)
    var_y[present] = np.maximum(0.0, sum_y2[present] / cnt[present] - mean_y[present] ** 2)

    return mean_y, np.sqrt(var_y)


def safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson correlation, returning NaN if fewer than 5 finite pairs."""
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 5:
        return np.nan
    return float(np.corrcoef(x[ok], y[ok])[0, 1])


# ============================================================
# Plot helpers
# ============================================================
def style_ax(ax):
    """Apply the shared thesis plot style."""
    ax.set_facecolor("white")
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.7, color="#b0b0b0")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_linewidth(0.8)
    ax.tick_params(labelsize=10, length=4)


def scaled_ylabel(base_tex: str) -> str:
    """Format a y-axis label with the shared scaling convention."""
    return rf"${base_tex}\;\times 10^{{-{PLOT_SCALE_POWER}}}$"


def savefig(path: Path):
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


# ============================================================
# Frozen Orchestra Plot
# ============================================================
def plot_frozen_orchestra(
    y_true: np.ndarray,
    y_hat_sep: np.ndarray,
    y_hat_joint: np.ndarray,
    asset_id: np.ndarray,
    test_year: int,
    outpath: Path,
    xlim_clip: Optional[Tuple[float, float]] = None,
):
    """
    Frozen orchestra plot: for each asset, the separate model's predictions
    are shown as a horizontal violin (the temporal distribution of its
    forecasts), while the joint model's prediction is a single dot (because
    within-asset variance is essentially zero).

    All eligible assets are shown, sorted by the separate model's prediction
    dispersion (widest violins at the top).
    """
    # --- Collect per-asset prediction arrays ---
    unique_assets = np.unique(asset_id)
    asset_data = []
    for a in unique_assets:
        mask = asset_id == a
        preds_sep = y_hat_sep[mask]
        preds_joint = y_hat_joint[mask]
        if len(preds_sep) < MIN_TEST_OBS:
            continue
        asset_data.append({
            "asset_id": int(a),
            "preds_sep": preds_sep,
            "preds_joint": preds_joint,
            "mean_sep": float(np.mean(preds_sep)),
            "mean_joint": float(np.mean(preds_joint)),
            "std_sep": float(np.std(preds_sep)),
            "std_joint": float(np.std(preds_joint)),
        })

    # Sort by separate model dispersion (largest first)
    asset_data.sort(key=lambda x: -x["std_sep"])
    n = len(asset_data)

    if n == 0:
        print(f"  [Orchestra] No eligible assets for year {test_year}.")
        return

    # --- Auto-clip x-axis ---
    if xlim_clip is None:
        all_sep = np.concatenate([ad["preds_sep"] for ad in asset_data])
        p = np.percentile(np.abs(all_sep), 97)
        xlim_clip = (-p * 1.15, p * 1.15)

    n_truncated = sum(
        1 for ad in asset_data
        if ad["preds_sep"].min() < xlim_clip[0]
        or ad["preds_sep"].max() > xlim_clip[1]
    )

    # --- Adaptive sizing ---
    row_height = max(0.08, min(0.22, 12.0 / n))
    violin_half_height = row_height * 0.75
    dot_size = max(20, min(50, 1500.0 / n))
    fig_height = max(5, n * row_height + 1.5)

    fig, ax = plt.subplots(figsize=(9, fig_height))
    fig.patch.set_facecolor("white")

    CLR_SEP_FILL = "#1f77b4"
    CLR_SEP_EDGE = "#0b4f8a"
    CLR_JOINT = "#c0392b"
    CLR_ZERO = "#aaaaaa"
    CLR_RULE = "#e8e8e8"

    y_positions = np.arange(n)

    # Thin ruled lines for each row
    for ypos in y_positions:
        ax.axhline(
            n - 1 - ypos, color=CLR_RULE, linewidth=0.5,
            zorder=0, alpha=0.7,
        )

    # Zero line
    ax.axvline(0, color=CLR_ZERO, linewidth=0.7, linestyle="--", alpha=1, zorder=0)

    for i, ad in enumerate(asset_data):
        ypos = n - 1 - i
        preds = ad["preds_sep"]

        # Clip predictions for KDE visualization
        preds_clipped = np.clip(preds, xlim_clip[0], xlim_clip[1])

        # --- Separate model: violin ---
        try:
            kde = gaussian_kde(preds_clipped, bw_method=0.3)
            pad = 0.3 * ad["std_sep"]
            x_lo = max(xlim_clip[0], preds_clipped.min() - pad)
            x_hi = min(xlim_clip[1], preds_clipped.max() + pad)
            x_grid = np.linspace(x_lo, x_hi, 200)
            density = kde(x_grid)
            density_norm = density / (density.max() + EPS) * violin_half_height

            ax.fill_between(
                x_grid, ypos - density_norm, ypos + density_norm,
                color=CLR_SEP_FILL, alpha=0.22, linewidth=0, zorder=1,
            )
            ax.plot(x_grid, ypos + density_norm, color=CLR_SEP_EDGE,
                    linewidth=0.5, alpha=0.7, zorder=2)
            ax.plot(x_grid, ypos - density_norm, color=CLR_SEP_EDGE,
                    linewidth=0.5, alpha=0.7, zorder=2)
        except Exception:
            ax.plot(
                [max(xlim_clip[0], preds.min()), min(xlim_clip[1], preds.max())],
                [ypos, ypos],
                color=CLR_SEP_FILL, linewidth=1.0, alpha=0.6, zorder=1,
            )

        # --- Joint model: single dot ---
        joint_x = np.clip(ad["mean_joint"], xlim_clip[0], xlim_clip[1])
        ax.scatter(
            joint_x, ypos,
            s=dot_size, color=CLR_JOINT, edgecolors="white",
            linewidths=1, zorder=5,
        )

    # --- Clean y-axis ---
    ax.set_yticks([])
    ax.set_ylim(-0.8, n - 0.2)
    ax.set_xlim(xlim_clip)

    ax.set_xlabel("Prediction value", fontsize=10.5, labelpad=6)

    # --- Annotation box ---
    n_collapsed = sum(1 for ad in asset_data if ad["std_joint"] < NEAR_ZERO_THRESHOLD)
    collapse_pct = 100 * n_collapsed / n

    note_lines = [
        f"All {n} eligible assets (ranked by separate-model dispersion)",
        f"Joint model collapsed: {n_collapsed}/{n} ({collapse_pct:.0f}%)",
    ]
    if n_truncated > 0:
        note_lines.append(f"{n_truncated} asset(s) truncated at x-axis limits")

    ax.text(
        0.99, -0.025,
        "\n".join(note_lines),
        transform=ax.transAxes, ha="right", va="top",
        fontsize=7.5, color="#555555",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                edgecolor="#cccccc", alpha=0.9),
    )

    # --- Legend ---
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color=CLR_SEP_FILL, linewidth=8, alpha=0.22,
               label="Separate: temporal prediction distribution"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=CLR_JOINT,
               markersize=7, label="Joint: constant prediction"),
    ]
    ax.legend(
        handles=legend_elements, loc="upper right",
        fontsize=8.5, frameon=True, fancybox=True,
        framealpha=0.95, edgecolor="#cccccc",
    )

    style_ax(ax)
    ax.grid(axis="y", visible=False)

    savefig(outpath)
    print(f"  [Orchestra] Saved: {outpath} ({n} assets)")


# ============================================================
# Main
# ============================================================
def main():
    print(f"=== Ablation C: Shrinkage Diagnostics ({FEATURE_BLOCK}, L={L}) ===\n")

    with open(EVAL_PKL, "rb") as f:
        cfg = pickle.load(f)

    n_assets = int(cfg["metadata"]["n_assets"])
    windows = cfg["windows"]

    asset_year_rows = []
    year_rows = []

    # Store raw predictions for the orchestra plot
    orchestra_data = None

    # ============================================================
    # 1  Per-window computation
    # ============================================================
    for w in windows:
        year = int(w["test_year"])

        sep = load_preds(
            SEP_DIR, "sep_deep_preds_Y", year,
            ("y_test", "pred_test", "asset_id_test"),
        )
        joint = load_preds(
            JOINT_DIR, "joint_mtl_preds_Y", year,
            ("y_true", "y_hat", "asset_id"),
        )
        if sep is None or joint is None:
            print(f"[{year}] Missing predictions — skipping.")
            continue

        y_true, y_hat_sep, y_hat_joint, asset_id = align_predictions(sep, joint)

        # Store data for frozen orchestra plot
        if year == ORCHESTRA_YEAR:
            orchestra_data = (y_true.copy(), y_hat_sep.copy(),
                              y_hat_joint.copy(), asset_id.copy())

        mse_s, mp_s, sp_s, cnt = per_asset_pred_stats(y_true, y_hat_sep, asset_id, n_assets)
        mse_j, mp_j, sp_j, _   = per_asset_pred_stats(y_true, y_hat_joint, asset_id, n_assets)
        mt, st = per_asset_true_stats(y_true, asset_id, n_assets)

        eligible = np.where(cnt >= MIN_TEST_OBS)[0]

        # --- Per-asset OOS R² relative to zero predictor ---
        err2_zero = y_true ** 2
        mse_zero = np.full(n_assets, np.nan, dtype=np.float64)
        present = cnt > 0
        mse_zero[present] = (
            np.bincount(asset_id, weights=err2_zero, minlength=n_assets)[present]
            / cnt[present]
        )

        r2_joint = np.full(n_assets, np.nan, dtype=np.float64)
        r2_sep = np.full(n_assets, np.nan, dtype=np.float64)
        nonzero_denom = mse_zero > EPS
        r2_joint[nonzero_denom] = 1.0 - mse_j[nonzero_denom] / mse_zero[nonzero_denom]
        r2_sep[nonzero_denom] = 1.0 - mse_s[nonzero_denom] / mse_zero[nonzero_denom]

        # Asset-year level rows
        for a in eligible:
            asset_year_rows.append({
                "test_year": year, "asset_id": int(a), "n_obs": int(cnt[a]),
                "mse_sep": float(mse_s[a]), "mse_joint": float(mse_j[a]),
                "mse_zero": float(mse_zero[a]),
                "delta_mse": float(mse_j[a] - mse_s[a]),
                "r2_oos_joint": float(r2_joint[a]),
                "r2_oos_sep": float(r2_sep[a]),
                "mean_pred_sep": float(mp_s[a]), "mean_pred_joint": float(mp_j[a]),
                "delta_mean_pred": float(mp_j[a] - mp_s[a]),
                "std_pred_sep": float(sp_s[a]), "std_pred_joint": float(sp_j[a]),
                "delta_std_pred": float(sp_j[a] - sp_s[a]),
                "mean_true": float(mt[a]), "std_true": float(st[a]),
            })

        # Year-level summary
        e_mp_s, e_mp_j = mp_s[eligible], mp_j[eligible]
        e_sp_s, e_sp_j = sp_s[eligible], sp_j[eligible]
        e_ms_s, e_ms_j = mse_s[eligible], mse_j[eligible]
        e_r2_j, e_r2_s = r2_joint[eligible], r2_sep[eligible]

        frac_zero_std_joint_yr = float(
            (e_sp_j < NEAR_ZERO_THRESHOLD).sum() / len(e_sp_j)
        )
        frac_zero_std_sep_yr = float(
            (e_sp_s < NEAR_ZERO_THRESHOLD).sum() / len(e_sp_s)
        )

        yr = {
            "test_year": year,
            "n_assets": len(eligible),
            "mean_mse_sep": float(np.nanmean(e_ms_s)),
            "mean_mse_joint": float(np.nanmean(e_ms_j)),
            "delta_mean_mse": float(np.nanmean(e_ms_j - e_ms_s)),
            "cross_asset_std_mean_pred_sep": float(np.nanstd(e_mp_s)),
            "cross_asset_std_mean_pred_joint": float(np.nanstd(e_mp_j)),
            "delta_cross_asset_std_mean_pred": float(np.nanstd(e_mp_j) - np.nanstd(e_mp_s)),
            "avg_within_std_pred_sep": float(np.nanmean(e_sp_s)),
            "avg_within_std_pred_joint": float(np.nanmean(e_sp_j)),
            "delta_avg_within_std_pred": float(np.nanmean(e_sp_j - e_sp_s)),
            "corr_delta_stdpred_delta_mse": safe_corr(e_sp_j - e_sp_s, e_ms_j - e_ms_s),
            "frac_zero_std_joint": frac_zero_std_joint_yr,
            "frac_zero_std_sep": frac_zero_std_sep_yr,
        }
        year_rows.append(yr)

        print(f"[{year}] assets={len(eligible)} | "
              f"Δmean_mse={yr['delta_mean_mse']:.3e} | "
              f"Δcross_std={yr['delta_cross_asset_std_mean_pred']:.3e} | "
              f"Δwithin_std={yr['delta_avg_within_std_pred']:.3e} | "
              f"zero_std_joint={frac_zero_std_joint_yr:.1%} | "
              f"zero_std_sep={frac_zero_std_sep_yr:.1%}")

    asset_year_df = pd.DataFrame(asset_year_rows).sort_values(["test_year", "asset_id"])
    year_df = pd.DataFrame(year_rows).sort_values("test_year")

    # ============================================================
    # 2  Year-level correlations
    # ============================================================
    corr_cross = safe_corr(
        year_df["delta_cross_asset_std_mean_pred"].to_numpy(),
        year_df["delta_mean_mse"].to_numpy(),
    )
    corr_within = safe_corr(
        year_df["delta_avg_within_std_pred"].to_numpy(),
        year_df["delta_mean_mse"].to_numpy(),
    )
    print(f"\nYear-level corr(Δcross_std, Δmean_mse) = {corr_cross:.4f}")
    print(f"Year-level corr(Δwithin_std, Δmean_mse) = {corr_within:.4f}\n")

    # ============================================================
    # 3  Non-collapsed asset-years: R² analysis
    # ============================================================
    nonzero_mask = asset_year_df["std_pred_joint"] >= NEAR_ZERO_THRESHOLD
    collapsed_mask = asset_year_df["std_pred_joint"] < NEAR_ZERO_THRESHOLD

    n_nonzero = nonzero_mask.sum()
    n_collapsed = collapsed_mask.sum()

    print("\n" + "=" * 90)
    print("R² ANALYSIS: NON-COLLAPSED vs COLLAPSED ASSET-YEARS (joint model)")
    print("=" * 90)

    if n_nonzero > 0:
        nonzero_df = asset_year_df[nonzero_mask]
        r2_nz = nonzero_df["r2_oos_joint"]
        frac_pos_r2_nz = float((r2_nz > 0).mean())
        mean_r2_nz = float(r2_nz.mean())
        median_r2_nz = float(r2_nz.median())

        print(f"\nNon-collapsed asset-years (std_pred_joint >= {NEAR_ZERO_THRESHOLD}):")
        print(f"  Count:              {n_nonzero}")
        print(f"  Mean R²_oos:        {mean_r2_nz:.6f}")
        print(f"  Median R²_oos:      {median_r2_nz:.6f}")
        print(f"  Frac R²_oos > 0:    {frac_pos_r2_nz:.1%}")
        print(f"\n  R²_oos distribution:")
        print(f"  {r2_nz.describe().to_string()}")
    else:
        print("\n  No non-collapsed asset-years found.")

    if n_collapsed > 0:
        collapsed_df = asset_year_df[collapsed_mask]
        r2_c = collapsed_df["r2_oos_joint"]
        frac_pos_r2_c = float((r2_c > 0).mean())
        mean_r2_c = float(r2_c.mean())
        median_r2_c = float(r2_c.median())

        print(f"\nCollapsed asset-years (std_pred_joint < {NEAR_ZERO_THRESHOLD}):")
        print(f"  Count:              {n_collapsed}")
        print(f"  Mean R²_oos:        {mean_r2_c:.6f}")
        print(f"  Median R²_oos:      {median_r2_c:.6f}")
        print(f"  Frac R²_oos > 0:    {frac_pos_r2_c:.1%}")

    print(f"\nSeparate model (all asset-years):")
    r2_sep_all = asset_year_df["r2_oos_sep"]
    print(f"  Mean R²_oos:        {float(r2_sep_all.mean()):.6f}")
    print(f"  Median R²_oos:      {float(r2_sep_all.median()):.6f}")
    print(f"  Frac R²_oos > 0:    {float((r2_sep_all > 0).mean()):.1%}")

    print("=" * 90)

    # ============================================================
    # 4  Save tables
    # ============================================================
    asset_year_df.to_csv(OUT_ASSET_YEAR_CSV, index=False)
    year_df.to_csv(OUT_YEAR_SUMMARY_CSV, index=False)

    frac_zero_std_joint = float(
        (asset_year_df["std_pred_joint"] < NEAR_ZERO_THRESHOLD).mean()
    )
    frac_zero_std_sep = float(
        (asset_year_df["std_pred_sep"] < NEAR_ZERO_THRESHOLD).mean()
    )

    overall = {
        "n_asset_year_rows": len(asset_year_df),
        "mean_delta_mse": float(asset_year_df["delta_mse"].mean()),
        "median_delta_mse": float(asset_year_df["delta_mse"].median()),
        "frac_assets_joint_better": float((asset_year_df["delta_mse"] < 0).mean()),
        "mean_delta_std_pred": float(asset_year_df["delta_std_pred"].mean()),
        "median_delta_std_pred": float(asset_year_df["delta_std_pred"].median()),
        "frac_std_pred_reduced": float((asset_year_df["delta_std_pred"] < 0).mean()),
        "corr_delta_stdpred_delta_mse_all": safe_corr(
            asset_year_df["delta_std_pred"].to_numpy(),
            asset_year_df["delta_mse"].to_numpy(),
        ),
        "mean_mse_sep": float(asset_year_df["mse_sep"].mean()),
        "mean_std_pred_sep": float(asset_year_df["std_pred_sep"].mean()),
        "frac_zero_std_joint": frac_zero_std_joint,
        "frac_zero_std_sep": frac_zero_std_sep,
        "n_noncollapsed": int(n_nonzero),
        "n_collapsed": int(n_collapsed),
        "frac_pos_r2_noncollapsed": float((asset_year_df.loc[nonzero_mask, "r2_oos_joint"] > 0).mean()) if n_nonzero > 0 else np.nan,
        "mean_r2_noncollapsed": float(asset_year_df.loc[nonzero_mask, "r2_oos_joint"].mean()) if n_nonzero > 0 else np.nan,
    }
    overall_df = pd.DataFrame([overall])
    overall_df.to_csv(OUT_OVERALL_CSV, index=False)

    print("\n" + "=" * 90)
    print("OVERALL SHRINKAGE DIAGNOSTICS")
    print("=" * 90)
    print(overall_df.to_string(index=False))
    print("=" * 90)
    print(f"\nZero-std fraction (joint):    {frac_zero_std_joint:.1%}")
    print(f"Zero-std fraction (separate): {frac_zero_std_sep:.1%}")

    # ============================================================
    # 5  Descriptive stats for threshold calibration
    # ============================================================
    print("\n--- std_pred_joint distribution ---")
    print(asset_year_df["std_pred_joint"].describe())
    print("\n--- std_pred_sep distribution ---")
    print(asset_year_df["std_pred_sep"].describe())

    # ============================================================
    # 6  Plots
    # ============================================================
    CLR_SEP = "#1f77b4"
    CLR_JOINT = "#d6604d"

    years = year_df["test_year"].to_numpy()

    cross_sep = year_df["cross_asset_std_mean_pred_sep"].to_numpy() * PLOT_SCALE
    cross_jnt = year_df["cross_asset_std_mean_pred_joint"].to_numpy() * PLOT_SCALE
    within_sep = year_df["avg_within_std_pred_sep"].to_numpy() * PLOT_SCALE
    within_jnt = year_df["avg_within_std_pred_joint"].to_numpy() * PLOT_SCALE

    # Plot 1: cross-asset dispersion of mean forecasts
    fig, ax = plt.subplots(figsize=(10, 4.5))
    fig.patch.set_facecolor("white")
    ax.plot(years, cross_sep, marker="o", markersize=4, linewidth=1.2, color=CLR_SEP, label="Separate")
    ax.plot(years, cross_jnt, marker="o", markersize=4, linewidth=1.2, color=CLR_JOINT, label="Joint")
    ax.fill_between(years, cross_jnt, cross_sep, color=CLR_SEP, alpha=0.12)
    ax.set_xlabel("Test year", fontsize=11)
    ax.set_xticks(years)
    ax.tick_params(axis="x", rotation=45, labelsize=9)
    ax.set_ylabel(scaled_ylabel(r"\mathrm{std}(\hat{\mu}_i)"), fontsize=11, labelpad=8)
    ax.legend(fontsize=9.5, frameon=False, loc="upper right")
    style_ax(ax)
    savefig(PLOT_DIR / "cross_asset_dispersion_mean_pred.png")

    # Plot 2: average within-asset prediction dispersion
    fig, ax = plt.subplots(figsize=(10, 4.5))
    fig.patch.set_facecolor("white")
    ax.plot(years, within_sep, marker="o", markersize=4, linewidth=1.2, color=CLR_SEP, label="Separate")
    ax.plot(years, within_jnt, marker="o", markersize=4, linewidth=1.2, color=CLR_JOINT, label="Joint")
    ax.fill_between(years, within_jnt, within_sep, color=CLR_SEP, alpha=0.12)
    ax.set_xlabel("Test year", fontsize=11)
    ax.set_xticks(years)
    ax.tick_params(axis="x", rotation=45, labelsize=9)
    ax.set_ylabel(scaled_ylabel(r"\mathrm{avg}_i\;\mathrm{std}(\hat{y}_i)"), fontsize=11, labelpad=8)
    ax.legend(fontsize=9.5, frameon=False, loc="upper right")
    style_ax(ax)
    savefig(PLOT_DIR / "avg_within_asset_std_pred.png")

    # Plot 3: scatter of Δstd(pred) vs ΔMSE
    fig, ax = plt.subplots(figsize=(6, 5))
    fig.patch.set_facecolor("white")
    ax.scatter(asset_year_df["delta_std_pred"], asset_year_df["delta_mse"], s=10, alpha=0.5)
    ax.axvline(0, linewidth=1)
    ax.axhline(0, linewidth=1)
    ax.set_title(f"{FEATURE_BLOCK} | Δstd(pred) vs ΔMSE (joint − sep)\n"
                 f"corr = {overall['corr_delta_stdpred_delta_mse_all']:.3f}")
    ax.set_xlabel("Δ std(pred)  (joint − sep)")
    ax.set_ylabel("Δ MSE  (joint − sep)")
    style_ax(ax)
    savefig(PLOT_DIR / "scatter_delta_std_pred_vs_delta_mse.png")

    # Plot 4: histogram of Δstd(pred)
    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor("white")
    ax.hist(asset_year_df["delta_std_pred"].dropna().to_numpy(), bins=40)
    ax.axvline(0, linewidth=1)
    ax.set_title(f"{FEATURE_BLOCK} | Distribution of Δ std(pred) (joint − sep)")
    ax.set_xlabel("Δ std(pred)")
    ax.set_ylabel("Count (asset-year)")
    style_ax(ax)
    savefig(PLOT_DIR / "hist_delta_std_pred.png")

    # Plot 5: Frozen orchestra
    if orchestra_data is not None:
        print(f"\nGenerating frozen orchestra plot for {ORCHESTRA_YEAR}...")
        plot_frozen_orchestra(
            *orchestra_data,
            test_year=ORCHESTRA_YEAR,
            outpath=PLOT_DIR / f"frozen_orchestra_{ORCHESTRA_YEAR}.png",
        )
    else:
        print(f"\n[Orchestra] No data for year {ORCHESTRA_YEAR} — skipping.")

    # ============================================================
    # 7  Summary
    # ============================================================
    print("\nSaved:")
    print(f"  - {OUT_ASSET_YEAR_CSV}")
    print(f"  - {OUT_YEAR_SUMMARY_CSV}")
    print(f"  - {OUT_OVERALL_CSV}")
    print(f"  - plots in: {PLOT_DIR.resolve()}")
    print(f"\nDone. Outputs in: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()