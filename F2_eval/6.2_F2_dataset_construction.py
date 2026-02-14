"""
Thesis Data Pipeline - Step 6.2 (Feature Engineering & Dataset Build)
=====================================================================
Construct the F2 supervised dataset from daily excess returns and prices.

Each sample j consists of:
    X[j]  — shape (L, F) = (30, 10): a 30-day lookback window of 10 features
    y[j]  — scalar: the excess return at date t (already next-day in Y_excess)

No-leakage guarantee:
    For a sample at date t, inputs use features from [t-L, …, t-1] only.
    Every rolling feature at day τ uses data up to and including τ.

Inputs
------
- Y_excess_daily_target.parquet  : daily excess returns
- mask_membership_daily.parquet  : boolean SPI-membership mask
- prices_daily.parquet           : forward-filled daily prices

Outputs
-------
- usable_F2_L30.parquet                : boolean mask of usable (date, stock) pairs
- dataset_F2_L30.npz                   : compressed arrays (X, y, asset_id, date)
- dataset_F2_L30_feature_names.csv     : ordered feature name list
- dataset_F2_L30_id_to_stock.csv       : integer asset-id → RIC_FULL mapping
- f2_correlation_matrix.csv            : feature correlation matrix
- f2_correlation_matrix.pdf / .png     : publication-quality heatmap
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# Configuration
# ============================================================
Y_EXCESS_PATH = "Y_excess_daily_target.parquet"
MASK_PATH     = "mask_membership_daily.parquet"
PRICES_PATH   = "prices_daily.parquet"

OUT_USABLE = "usable_F2_L30.parquet"
OUT_NPZ    = "dataset_F2_L30.npz"
OUT_FEATS  = "dataset_F2_L30_feature_names.csv"
OUT_MAP    = "dataset_F2_L30_id_to_stock.csv"

L = 30                          # lookback window length (days)

W_MEAN = [5, 10, 20]           # rolling-mean windows
W_VOL  = [5, 10, 20]           # rolling-volatility windows
W_MOM  = [10, 20]              # momentum windows
W_DD   = 20                    # drawdown window

# ============================================================
# 1  Load & align data
# ============================================================
Y_excess   = pd.read_parquet(Y_EXCESS_PATH)
mask_final = pd.read_parquet(MASK_PATH)
prices     = pd.read_parquet(PRICES_PATH)

keep_cols = Y_excess.columns.tolist()
Y = Y_excess[keep_cols].copy()
M = mask_final[keep_cols].astype(bool).copy()
P = prices[keep_cols].copy()

assert Y.index.equals(M.index) and Y.index.equals(P.index), "Indices must match"
assert list(Y.columns) == list(M.columns) == list(P.columns), "Columns must match"

print("Loaded:")
print(f"  Y_excess: {Y.shape}  {Y.index.min().date()} → {Y.index.max().date()}")
print(f"  mask:     {M.shape}")
print(f"  prices:   {P.shape}")

# ============================================================
# 2  Feature engineering (F2 = 10 features)
# ============================================================
# Every feature at day τ is computed from data ≤ τ.  The supervised
# sample at date t later slices τ ∈ [t-L, …, t-1], so no future
# information ever enters the input tensor.


def compute_f2_features(Y: pd.DataFrame, P: pd.DataFrame) -> dict:
    """Return an ordered dict of 10 DataFrames, one per F2 feature."""
    feats = {}

    # (1) raw excess return
    feats["ret"] = Y

    # (2-4) rolling mean return
    for w in W_MEAN:
        feats[f"mean_{w}"] = Y.rolling(w, min_periods=w).mean()

    # (5-7) rolling return volatility
    for w in W_VOL:
        feats[f"vol_{w}"] = Y.rolling(w, min_periods=w).std(ddof=0)

    # (8-9) momentum: cumulative return over window
    # Computed in log-space for numerical stability, then mapped back.
    log1p = np.log1p(Y)
    for w in W_MOM:
        feats[f"mom_{w}"] = np.expm1(log1p.rolling(w, min_periods=w).sum())

    # (10) drawdown from cumulated returns within the W_DD-day window (starting at 1)
    # Use cumulative log-returns S_t = sum_{τ≤t} log(1+r_τ).
    # Within-window drawdown at t is:
    #   exp(S_t - max_{u in window} S_u) - 1
    # This is strictly return-information only (no price level information).
    log1p = np.log1p(Y)
    S = log1p.cumsum()
    S_peak = S.rolling(W_DD, min_periods=W_DD).max()
    feats[f"drawdown_{W_DD}"] = np.expm1(S - S_peak)


    return feats


features = compute_f2_features(Y, P)
feature_names = list(features.keys())
assert len(feature_names) == 10, f"Expected 10 F2 features, got {len(feature_names)}"
print(f"\nF2 features ({len(feature_names)}): {feature_names}")

# Per-cell validity: every feature must be finite
feat_ok = None
for name in feature_names:
    ok = features[name].notna() & np.isfinite(features[name].to_numpy())
    feat_ok = ok if feat_ok is None else (feat_ok & ok)

# ============================================================
# 3  Determine usable (date, stock) pairs
# ============================================================
# A pair (t, i) is usable iff:
#   1) target y_{t,i} is finite
#   2) stock i is in the SPI at date t
#   3) stock i has continuous SPI membership over [t-L, …, t-1]
#   4) all 10 features are finite over that same lag window

Y_ok = Y.notna()

M_lags_ok = (
    M.rolling(L, min_periods=L).min()
    .shift(1).fillna(False).astype(bool)
)
F_lags_ok = (
    feat_ok.rolling(L, min_periods=L).min()
    .shift(1).fillna(False).astype(bool)
)

usable = M & Y_ok & M_lags_ok & F_lags_ok
usable.to_parquet(OUT_USABLE)

usable_per_date = usable.sum(axis=1)
first_date = usable_per_date[usable_per_date > 0].index.min()
last_date  = usable_per_date[usable_per_date > 0].index.max()

print(f"\nUsable supervised examples:")
print(f"  Date range:         {first_date.date()} → {last_date.date()}")
print(f"  Total (date,stock): {int(usable.sum().sum()):,}")
print(f"  Max per day:        {int(usable_per_date.max())}")

# ============================================================
# 4  Build supervised arrays
# ============================================================
# Output shapes:
#   X        — (N_obs, L, F)    float32
#   y        — (N_obs,)         float32
#   asset_id — (N_obs,)         int32
#   date     — (N_obs,)         datetime64[ns]


def build_f2_dataset(Y, features, usable, L):
    """Assemble supervised tensors by iterating over target dates."""
    stocks = list(Y.columns)
    stock_to_id = {s: i for i, s in enumerate(stocks)}
    id_to_stock = {i: s for s, i in stock_to_id.items()}

    Y_arr = Y.to_numpy(dtype=np.float32)
    U_arr = usable.to_numpy(dtype=bool)
    dates = Y.index.to_numpy(dtype="datetime64[ns]")

    # Pre-stack features into a single (T, N_stocks, F) tensor
    feat_stack = np.stack(
        [features[n].to_numpy(dtype=np.float32) for n in features],
        axis=2,
    )

    T = Y_arr.shape[0]
    X_list, y_list, a_list, d_list = [], [], [], []

    for t in range(L, T):
        ok = U_arr[t, :]
        if not ok.any():
            continue

        idx = np.where(ok)[0]
        w = slice(t - L, t)

        # (L, N_stocks, F) → select usable stocks → (k, L, F)
        X_t = feat_stack[w, :, :][:, idx, :].transpose(1, 0, 2)
        y_t = Y_arr[t, idx]

        X_list.append(X_t)
        y_list.append(y_t)
        a_list.append(idx.astype(np.int32))
        d_list.append(np.full(len(idx), dates[t], dtype="datetime64[ns]"))

    X_out = np.concatenate(X_list, axis=0)
    y_out = np.concatenate(y_list, axis=0)
    a_out = np.concatenate(a_list, axis=0)
    d_out = np.concatenate(d_list, axis=0)

    return X_out, y_out, a_out, d_out, id_to_stock


X, y_arr, asset_id, sample_dates, id_to_stock = build_f2_dataset(
    Y, features, usable, L,
)

print(f"\nDataset shapes:")
print(f"  X:        {X.shape}  (expected: N, {L}, {len(feature_names)})")
print(f"  y:        {y_arr.shape}")
print(f"  asset_id: {asset_id.shape}")
print(f"  dates:    {sample_dates.shape}")
print(f"  Unique assets in samples: {len(np.unique(asset_id))}")

# ============================================================
# 5  Save dataset
# ============================================================
np.savez_compressed(
    OUT_NPZ,
    X=X.astype(np.float32),
    y=y_arr.astype(np.float32),
    asset_id=asset_id.astype(np.int32),
    date=sample_dates,
)
pd.Series(feature_names).to_csv(OUT_FEATS, index=False, header=["feature_name"])
pd.Series(id_to_stock).to_csv(OUT_MAP, header=["RIC_FULL"])

print(f"\nSaved: {OUT_USABLE}, {OUT_NPZ}, {OUT_FEATS}, {OUT_MAP}")
print(f"NaNs in X: {int(np.isnan(X).sum())}  |  NaNs in y: {int(np.isnan(y_arr).sum())}")

# Quick spot-check on a random sample
j = np.random.randint(0, len(y_arr))
print(f"\nRandom sample j={j}:")
print(f"  asset={id_to_stock[int(asset_id[j])]}  "
      f"date={pd.to_datetime(sample_dates[j]).date()}  "
      f"y={float(y_arr[j]):.6f}")
print(f"  X[j, -1, :] (features at t−1): {X[j, -1, :]}")

# ============================================================
# 6  Audit block — training-readiness checks
# ============================================================


def describe_feat(name, df):
    """Compute summary statistics over finite values of a feature DataFrame."""
    arr = df.to_numpy()
    finite = np.isfinite(arr)
    n_finite = int(finite.sum())
    if n_finite == 0:
        return {"name": name, "finite_pct": 0.0,
                "min": np.nan, "max": np.nan, "mean": np.nan, "std": np.nan}
    vals = arr[finite].astype(np.float64)
    return {
        "name": name,
        "finite_pct": 100.0 * n_finite / arr.size,
        "min": float(vals.min()),
        "max": float(vals.max()),
        "mean": float(vals.mean()),
        "std": float(vals.std()),
    }


print("\n" + "=" * 80)
print("F2 DATASET AUDIT (TRAINING READINESS)")
print("=" * 80)

# A) Input alignment
print("\n[A] Input alignment")
print(f"  Y={Y.shape}  M={M.shape}  P={P.shape}")
print(f"  Date range: {Y.index.min().date()} → {Y.index.max().date()}")
print(f"  Columns identical: {list(Y.columns) == list(M.columns) == list(P.columns)}")

# B) Feature integrity
print(f"\n[B] Feature integrity (F={len(feature_names)})")
feat_stats = pd.DataFrame([describe_feat(n, features[n]) for n in feature_names])
print(feat_stats.to_string(index=False, float_format=lambda x: f"{x: .4e}"))

suspicious = feat_stats[np.abs(feat_stats["max"]) > 50]
if len(suspicious):
    print(f"\n  WARNING: potentially extreme ranges (|max| > 50):")
    print(suspicious[["name", "min", "max"]].to_string(index=False))
else:
    print("\n  OK: no extreme feature ranges flagged.")

# C) Usable examples over time
print(f"\n[C] Usable supervised examples")
print(f"  Date range: {first_date.date()} → {last_date.date()}")
print(f"  Total: {int(usable.sum().sum()):,}  |  Max/day: {int(usable_per_date.max())}")
print("\n  Per-year totals:")
print(usable_per_date.resample("YE").sum().to_string())

# D) Array sanity
print(f"\n[D] Array sanity")
print(f"  X.shape={X.shape}  y.shape={y_arr.shape}")
print(f"  dtypes: X={X.dtype}  y={y_arr.dtype}  asset_id={asset_id.dtype}")
print(f"  NaNs — X: {int(np.isnan(X).sum())}  y: {int(np.isnan(y_arr).sum())}")
print(f"  Infs — X: {int(np.isinf(X).sum())}  y: {int(np.isinf(y_arr).sum())}")
print(f"  asset_id range: [{asset_id.min()}, {asset_id.max()}]  "
      f"n_assets={len(id_to_stock)}")

# E) No-leakage spot checks
print(f"\n[E] No-leakage spot checks (k=5)")
rng = np.random.default_rng(123)
for j in rng.integers(0, len(y_arr), size=5):
    t_date = pd.to_datetime(sample_dates[j])
    ric = id_to_stock[int(asset_id[j])]
    tpos = Y.index.get_indexer([t_date])[0]
    lag_dates = Y.index[tpos - L : tpos]

    m_t = bool(M.loc[t_date, ric])
    m_lags = bool(M.loc[lag_dates, ric].all())
    y_t = float(Y.loc[t_date, ric])

    # Verify that the last feature value in the input tensor matches the
    # independently computed feature at the final lag date.
    last_ret_true = float(features["ret"].loc[lag_dates[-1], ric])
    last_ret_in_X = float(X[j, -1, feature_names.index("ret")])
    ret_match = np.isclose(last_ret_in_X, last_ret_true, atol=1e-6)

    print(f"  j={j}  asset={ric}  target_date={t_date.date()}")
    print(f"    membership at t: {m_t}  |  all lags: {m_lags}")
    print(f"    y(t) finite: {np.isfinite(y_t)}  |  y(t)={y_t:.6f}")
    print(f"    last input day: {lag_dates[-1].date()}  |  ret match: {ret_match}")

# F) Eligibility preview for an example cutoff
print(f"\n[F] Eligibility preview")
CUTOFF = pd.Timestamp("2010-01-01")
MIN_REQUIRED = 1000
counts_pre = usable.loc[usable.index < CUTOFF].sum(axis=0).astype(int)
n_ok = int((counts_pre >= MIN_REQUIRED).sum())
print(f"  Assets with ≥{MIN_REQUIRED} usable samples before {CUTOFF.date()}: "
      f"{n_ok} / {len(counts_pre)}")
print(f"  Counts summary: {counts_pre.describe().to_string()}")

print("\n" + "=" * 80)
print("AUDIT COMPLETE")
print("=" * 80)

# ============================================================
# 7  Feature correlation matrix (Appendix C)
# ============================================================
# Pool across all samples and time steps to get the unconditional
# correlation structure of the 10 input features.

print("\n" + "=" * 80)
print("FEATURE CORRELATION MATRIX (APPENDIX C)")
print("=" * 80)

X_flat = X.reshape(-1, len(feature_names))
corr_f2 = pd.DataFrame(X_flat, columns=feature_names).corr()

print(f"\n{corr_f2.round(3).to_string()}")
corr_f2.to_csv("f2_correlation_matrix.csv")
print("\nSaved: f2_correlation_matrix.csv")

# ============================================================
# 8  Correlation heatmap
# ============================================================
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.linewidth": 0.5,
})

fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(
    corr_f2,
    annot=True,
    fmt=".2f",
    cmap="RdBu_r",
    center=0,
    vmin=-1, vmax=1,
    square=True,
    linewidths=0.5,
    linecolor="white",
    cbar_kws={"shrink": 0.8, "label": "Correlation",
              "ticks": [-1, -0.5, 0, 0.5, 1]},
    ax=ax,
)
ax.set_xlabel("")
ax.set_ylabel("")
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()

plt.savefig("f2_correlation_matrix.pdf", dpi=300, bbox_inches="tight")
plt.savefig("f2_correlation_matrix.png", dpi=300, bbox_inches="tight")
plt.close()

print("Saved: f2_correlation_matrix.pdf and f2_correlation_matrix.png")