"""
Comparative Evaluation
===================================================
Aggregates per-window evaluation metrics for joint vs. separate deep
models across feature blocks (F1/F2/F3), computes out-of-sample skill
scores relative to a zero-forecast baseline, and produces summary
tables and diagnostic plots.

Inputs
------
- joint_mtl_{fb}_L30/joint_mtl_window_metrics.csv      : per-window MSE from the joint (MTL) model
- sep_deep_{fb}_L30_cpupar/sep_deep_window_metrics.csv  : per-window MSE from separate models
- baseline0_zero_{fb}_L30_results.csv                   : zero-forecast baseline MSE
- baseline{1,2,3}_*_F1_L30_results.csv                  : F1-only mean-based baselines (optional)

Outputs
-------
Tables (CSV)
  - {F1,F2,F3}_joint_vs_sep_with_zero_and_skill.csv : per-window comparison for each feature block
  - featureblocks_avg_summary.csv                    : cross-window averages by feature block
  - F1_main_with_all_baselines.csv                   : F1 table including mean-based baselines
  - F1_baselines_avg_summary.csv                     : F1 baseline ranking

Plots (PNG)
  - Per-block skill time series (micro & macro)
  - Feature-block bar charts (MSE & skill)
  - F1 baseline bar charts (MSE & skill)
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# Configuration
# ============================================================
LOOKBACK = 30
FEATURE_BLOCKS = ["F1", "F2", "F3"]

JOINT_DIR_TPL = "joint_mtl_{fb}_L30"
SEP_DIR_TPL = "sep_deep_{fb}_L30_cpupar"

JOINT_METRICS_NAME = "joint_mtl_window_metrics.csv"
SEP_METRICS_NAME = "sep_deep_window_metrics.csv"

ZERO_BASELINE_CSV_TPL = "baseline0_zero_{fb}_L30_results.csv"

F1_OTHER_BASELINES = {
    "sepmean":    "baseline1_sepmean_F1_L30_results.csv",
    "pooledmean": "baseline2_pooledmean_F1_L30_results.csv",
    "jamesstein": "baseline3_jamesstein_F1_L30_results.csv",
}

OUT_ROOT = Path("results_summary")
OUT_TABLES = OUT_ROOT / "tables"
OUT_PLOTS = OUT_ROOT / "plots"
OUT_TABLES.mkdir(parents=True, exist_ok=True)
OUT_PLOTS.mkdir(parents=True, exist_ok=True)

DPI = 160
Y_PAD_FRAC = 0.08  # 8 % vertical padding around data range in plots
Y_PAD_ABS = 1e-8   # floor padding when the data range is near-zero


# ============================================================
# Formatting helpers
# ============================================================
def fmt_sci(x: float, sig: int = 3) -> str:
    """Format as mantissa×10^exp, e.g. '5.280×10^-4'."""
    try:
        x = float(x)
    except (TypeError, ValueError):
        return "nan"
    if not np.isfinite(x):
        return "nan"
    if x == 0.0:
        return "0"
    exp = int(np.floor(np.log10(abs(x))))
    return f"{x / 10 ** exp:.{sig}f}×10^{exp}"


def fmt_fixed(x: float, decimals: int = 6) -> str:
    """Format as fixed-point decimal."""
    try:
        x = float(x)
    except (TypeError, ValueError):
        return "nan"
    if not np.isfinite(x):
        return "nan"
    return f"{x:.{decimals}f}"


def format_summary_for_print(df: pd.DataFrame) -> pd.DataFrame:
    """Apply human-readable formatting to a summary DataFrame for console output."""
    out = df.copy()
    for col in out.columns:
        if col.endswith("_pct"):
            out[col] = out[col].apply(lambda v: fmt_fixed(v, 3))
        elif col.startswith("skill_") or col.startswith("avg_skill_") or col.endswith("_vs_zero"):
            out[col] = out[col].apply(lambda v: fmt_fixed(v, 6))
        elif col.startswith("delta_"):
            out[col] = out[col].apply(fmt_sci)
        elif any(col.endswith(s) for s in ("_micro", "_macro", "_mse")):
            out[col] = out[col].apply(fmt_sci)
    return out


# ============================================================
# I/O helpers
# ============================================================
def _read_csv_safe(path: str | Path,
                   required_cols: Optional[List[str]] = None) -> Optional[pd.DataFrame]:
    """Read a CSV, returning None (with a warning) if the file is
    missing or lacks required columns."""
    path = Path(path)
    if not path.exists():
        print(f"[WARN] Missing file: {path}")
        return None
    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"[WARN] Failed reading {path}: {e}")
        return None
    if required_cols:
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            print(f"[WARN] {path} missing columns: {missing}")
            return None
    return df


def _standardise_year_col(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure a consistent integer 'test_year' column exists."""
    df = df.copy()
    if "test_year" not in df.columns and "year" in df.columns:
        df = df.rename(columns={"year": "test_year"})
    df["test_year"] = df["test_year"].astype(int)
    return df


# ============================================================
# Data loaders (one per result type)
# ============================================================
_MSE_COLS = ["test_year", "mse_micro", "mse_macro"]
_EXTRA_COLS = ["n_test", "n_val", "n_train"]


def _load_and_rename(path: Path, prefix: str) -> Optional[pd.DataFrame]:
    """Shared loader: read metrics CSV, standardise year column, rename
    mse_micro/macro to {prefix}_micro/macro, and carry over any extra
    sample-size columns."""
    df = _read_csv_safe(path, required_cols=_MSE_COLS)
    if df is None:
        return None
    df = _standardise_year_col(df).sort_values("test_year")
    rename = {"mse_micro": f"{prefix}_micro", "mse_macro": f"{prefix}_macro"}
    keep = ["test_year", "mse_micro", "mse_macro"]
    for c in _EXTRA_COLS:
        if c in df.columns:
            keep.append(c)
            rename[c] = f"{prefix}_{c}" if prefix != "" else c
    return df[keep].rename(columns=rename)


def load_joint_metrics(fb: str) -> Optional[pd.DataFrame]:
    path = Path(JOINT_DIR_TPL.format(fb=fb)) / JOINT_METRICS_NAME
    return _load_and_rename(path, "joint")


def load_sep_metrics(fb: str) -> Optional[pd.DataFrame]:
    path = Path(SEP_DIR_TPL.format(fb=fb)) / SEP_METRICS_NAME
    return _load_and_rename(path, "sep")


def load_zero_baseline(fb: str) -> Optional[pd.DataFrame]:
    path = Path(ZERO_BASELINE_CSV_TPL.format(fb=fb))
    return _load_and_rename(path, "zero")


def load_f1_other_baseline(name: str, csv_path: str) -> Optional[pd.DataFrame]:
    return _load_and_rename(Path(csv_path), name)


# ============================================================
# Metrics engineering
# ============================================================
def add_deltas_and_skill(df: pd.DataFrame) -> pd.DataFrame:
    """Append comparison columns to a per-window table:

    - delta_{k}          : joint MSE − separate MSE  (negative = joint wins)
    - delta_{k}_pct      : percentage version of the above
    - skill_{k}_sep      : OOS R²-style skill of separate vs. zero forecast
    - skill_{k}_joint    : OOS R²-style skill of joint vs. zero forecast
    - skill_{k}_delta    : skill improvement from separate to joint
    """
    out = df.copy()
    for k in ("micro", "macro"):
        out[f"delta_{k}"] = out[f"joint_{k}"] - out[f"sep_{k}"]
        out[f"delta_{k}_pct"] = (
            100.0 * (out[f"joint_{k}"] - out[f"sep_{k}"])
            / out[f"sep_{k}"].replace(0.0, np.nan)
        )
        out[f"skill_{k}_sep"] = 1.0 - out[f"sep_{k}"] / out[f"zero_{k}"]
        out[f"skill_{k}_joint"] = 1.0 - out[f"joint_{k}"] / out[f"zero_{k}"]
        out[f"skill_{k}_delta"] = out[f"skill_{k}_joint"] - out[f"skill_{k}_sep"]
    return out


# ============================================================
# Plot helpers
# ============================================================
def _save_fig(path: Path):
    plt.tight_layout()
    plt.savefig(path, dpi=DPI)
    plt.close()


def _autoscale_y(ax, arrays: List[np.ndarray]):
    """Set y-limits to tightly fit *arrays* with a small padding margin."""
    vals = np.concatenate([
        a[np.isfinite(a)] for a in (np.asarray(a, dtype=np.float64) for a in arrays)
        if np.isfinite(a).any()
    ])
    if vals.size == 0:
        return
    lo, hi = float(vals.min()), float(vals.max())
    pad = max(Y_PAD_ABS, (hi - lo if hi != lo else abs(lo)) * Y_PAD_FRAC)
    ax.set_ylim(lo - pad, hi + pad)


# ============================================================
# Plot functions
# ============================================================
def _plot_skill_timeseries_single(years, sep, joint, title, ylabel, path):
    """One skill-vs-zero time-series panel (shared by micro & macro)."""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(years, sep, marker="o", label="Separate")
    ax.plot(years, joint, marker="o", label="Joint")
    ax.axhline(0.0, linewidth=1)
    ax.set_title(title)
    ax.set_xlabel("Test year")
    ax.set_ylabel(ylabel)
    ax.set_xticks(years)
    ax.tick_params(axis="x", rotation=90)
    ax.legend()
    _autoscale_y(ax, [sep, joint, np.zeros_like(sep)])
    _save_fig(path)


def plot_skill_timeseries(df: pd.DataFrame, fb: str, out_dir: Path):
    """Skill-vs-zero time series for one feature block (micro + macro)."""
    years = df["test_year"].to_numpy()
    for k in ("micro", "macro"):
        _plot_skill_timeseries_single(
            years,
            sep=df[f"skill_{k}_sep"].to_numpy(),
            joint=df[f"skill_{k}_joint"].to_numpy(),
            title=f"{fb} | Skill vs Zero ({k.title()}): 1 − MSE / MSE_zero",
            ylabel=f"Skill ({k}, higher is better)",
            path=out_dir / f"{fb}_skill_{k}_timeseries.png",
        )


def plot_featureblock_bars(avg_df: pd.DataFrame, out_dir: Path):
    """Grouped bar charts comparing feature blocks on MSE and skill."""
    fbs = avg_df.index.tolist()
    x = np.arange(len(fbs))
    w = 0.25

    for k in ("micro", "macro"):
        # MSE bars: zero / separate / joint
        fig, ax = plt.subplots(figsize=(8, 4))
        zero_vals = avg_df[f"zero_{k}"].to_numpy()
        sep_vals = avg_df[f"sep_{k}"].to_numpy()
        joint_vals = avg_df[f"joint_{k}"].to_numpy()
        ax.bar(x - w, zero_vals, width=w, label="Zero")
        ax.bar(x, sep_vals, width=w, label="Separate")
        ax.bar(x + w, joint_vals, width=w, label="Joint")
        ax.set_title(f"Avg {k.title()} MSE by Feature Block")
        ax.set_xlabel("Feature block")
        ax.set_ylabel(f"Avg {k} MSE")
        ax.set_xticks(x)
        ax.set_xticklabels(fbs)
        ax.legend()
        _autoscale_y(ax, [zero_vals, sep_vals, joint_vals])
        _save_fig(out_dir / f"featureblocks_avg_mse_{k}.png")

        # Skill bars: separate / joint
        fig, ax = plt.subplots(figsize=(8, 4))
        s_sep = avg_df[f"skill_{k}_sep"].to_numpy()
        s_joint = avg_df[f"skill_{k}_joint"].to_numpy()
        ax.bar(x - w / 2, s_sep, width=w, label="Separate")
        ax.bar(x + w / 2, s_joint, width=w, label="Joint")
        ax.axhline(0.0, linewidth=1)
        ax.set_title(f"Avg Skill vs Zero ({k.title()}) by Feature Block")
        ax.set_xlabel("Feature block")
        ax.set_ylabel(f"Avg skill ({k})")
        ax.set_xticks(x)
        ax.set_xticklabels(fbs)
        ax.legend()
        _autoscale_y(ax, [s_sep, s_joint, np.zeros_like(s_sep)])
        _save_fig(out_dir / f"featureblocks_avg_skill_{k}.png")


def plot_f1_baselines(f1_tbl: pd.DataFrame, out_dir: Path):
    """Bar charts ranking all F1 methods by MSE and skill vs. zero."""
    methods = [
        n for n in ["zero", "sepmean", "pooledmean", "jamesstein", "sep", "joint"]
        if f"{n}_micro" in f1_tbl.columns and f"{n}_macro" in f1_tbl.columns
    ]
    if "zero" not in methods:
        print("[WARN] Cannot plot F1 baselines: zero baseline missing.")
        return

    x = np.arange(len(methods))
    for k in ("micro", "macro"):
        # MSE
        avg_mse = [float(f1_tbl[f"{m}_{k}"].mean()) for m in methods]
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(x, avg_mse)
        ax.set_title(f"F1: Avg {k.title()} MSE (All Methods)")
        ax.set_xlabel("Method")
        ax.set_ylabel(f"Avg {k} MSE")
        ax.set_xticks(x)
        ax.set_xticklabels(methods)
        _autoscale_y(ax, [np.array(avg_mse)])
        _save_fig(out_dir / f"F1_baselines_avg_mse_{k}.png")

        # Skill (excludes zero, which is the reference)
        methods_nz = [m for m in methods if m != "zero"]
        x_nz = np.arange(len(methods_nz))
        skill = [
            float((1.0 - f1_tbl[f"{m}_{k}"] / f1_tbl[f"zero_{k}"]).mean())
            for m in methods_nz
        ]
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(x_nz, skill)
        ax.axhline(0.0, linewidth=1)
        ax.set_title(f"F1: Avg Skill vs Zero ({k.title()})")
        ax.set_xlabel("Method")
        ax.set_ylabel(f"Avg skill ({k})")
        ax.set_xticks(x_nz)
        ax.set_xticklabels(methods_nz)
        _autoscale_y(ax, [np.array(skill), np.zeros(len(skill))])
        _save_fig(out_dir / f"F1_baselines_avg_skill_{k}.png")


# ============================================================
# Table builders
# ============================================================
def build_featureblock_table(fb: str) -> Optional[pd.DataFrame]:
    """Merge zero / separate / joint metrics for one feature block and
    compute deltas and skill scores."""
    zero = load_zero_baseline(fb)
    sep = load_sep_metrics(fb)
    joint = load_joint_metrics(fb)
    if any(d is None for d in (zero, sep, joint)):
        print(f"[WARN] Skipping {fb}: missing one of (zero, sep, joint).")
        return None
    df = (
        zero
        .merge(sep, on="test_year", how="inner")
        .merge(joint, on="test_year", how="inner")
    )
    df.insert(0, "feature_block", fb)
    return add_deltas_and_skill(df)


def build_f1_main_table() -> Optional[pd.DataFrame]:
    """F1 per-window table extended with all mean-based baselines."""
    base = build_featureblock_table("F1")
    if base is None:
        return None
    for name, path in F1_OTHER_BASELINES.items():
        dfb = load_f1_other_baseline(name, path)
        if dfb is not None:
            base = base.merge(dfb, on="test_year", how="left")
    return base


# ============================================================
# Main
# ============================================================
AVG_COLS = [
    "zero_micro", "zero_macro",
    "sep_micro", "sep_macro",
    "joint_micro", "joint_macro",
    "delta_micro", "delta_macro",
    "delta_micro_pct", "delta_macro_pct",
    "skill_micro_sep", "skill_micro_joint", "skill_micro_delta",
    "skill_macro_sep", "skill_macro_joint", "skill_macro_delta",
]


def main():
    # --------------------------------------------------------
    # Per-feature-block tables & plots
    # --------------------------------------------------------
    per_block: Dict[str, pd.DataFrame] = {}

    for fb in FEATURE_BLOCKS:
        df = build_featureblock_table(fb)
        if df is None:
            continue
        per_block[fb] = df

        out_path = OUT_TABLES / f"{fb}_joint_vs_sep_with_zero_and_skill.csv"
        df.to_csv(out_path, index=False)
        print(f"[OK] Wrote {out_path}")

        plot_skill_timeseries(df, fb, OUT_PLOTS)

    # --------------------------------------------------------
    # Cross-feature-block summary
    # --------------------------------------------------------
    avg_df = None
    if per_block:
        avg_df = pd.DataFrame({
            fb: df[AVG_COLS].mean() for fb, df in per_block.items()
        }).T.rename_axis("feature_block")

        out_avg = OUT_TABLES / "featureblocks_avg_summary.csv"
        avg_df.to_csv(out_avg)
        print(f"[OK] Wrote {out_avg}")

        plot_featureblock_bars(avg_df, OUT_PLOTS)

    # --------------------------------------------------------
    # F1: full baseline comparison (adds mean-based methods)
    # --------------------------------------------------------
    f1_tbl = build_f1_main_table()
    f1_avg = None

    if f1_tbl is not None:
        out_f1 = OUT_TABLES / "F1_main_with_all_baselines.csv"
        f1_tbl.to_csv(out_f1, index=False)
        print(f"[OK] Wrote {out_f1}")

        methods = ["zero", "sepmean", "pooledmean", "jamesstein", "sep", "joint"]
        rows = []
        for m in methods:
            mc, mc2 = f"{m}_micro", f"{m}_macro"
            if mc not in f1_tbl.columns or mc2 not in f1_tbl.columns:
                continue
            row = {
                "method": m,
                "avg_micro_mse": float(f1_tbl[mc].mean()),
                "avg_macro_mse": float(f1_tbl[mc2].mean()),
            }
            if m != "zero":
                row["avg_skill_micro_vs_zero"] = float(
                    (1.0 - f1_tbl[mc] / f1_tbl["zero_micro"]).mean()
                )
                row["avg_skill_macro_vs_zero"] = float(
                    (1.0 - f1_tbl[mc2] / f1_tbl["zero_macro"]).mean()
                )
            rows.append(row)

        f1_avg = pd.DataFrame(rows).sort_values("avg_micro_mse")
        out_f1_avg = OUT_TABLES / "F1_baselines_avg_summary.csv"
        f1_avg.to_csv(out_f1_avg, index=False)
        print(f"[OK] Wrote {out_f1_avg}")

        plot_f1_baselines(f1_tbl, OUT_PLOTS)

    # --------------------------------------------------------
    # Console summary
    # --------------------------------------------------------
    print("\n" + "=" * 90)
    print("SUMMARY (avg across windows; equal weight per year)")
    print("=" * 90)

    if avg_df is not None:
        print("\nCross-feature averages (MSE in ×10^exp, skill in decimals):")
        print(format_summary_for_print(avg_df).to_string())

    if f1_avg is not None:
        print("\nF1 baseline averages (MSE in ×10^exp, skill in decimals):")
        print(format_summary_for_print(f1_avg).to_string(index=False))

    print("\nOutputs written to:")
    print(f"  tables: {OUT_TABLES.resolve()}")
    print(f"  plots:  {OUT_PLOTS.resolve()}")


if __name__ == "__main__":
    main()