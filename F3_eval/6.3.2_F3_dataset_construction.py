"""
Thesis Data Pipeline - Step 6.3 (Feature Engineering & Dataset Build: F3)
=========================================================================
Construct the F3 supervised dataset from excess returns, prices, and
LSEG market-microstructure fields (turnover, bid/ask, VWAP).

F3 extends the return-based F2 feature set with liquidity and friction
measures while maintaining the same tensor shape: X[j] = (L, F) = (30, 10).

F3 feature channels (10 total):
    0  ret           : excess return
    1  log_turn      : log1p(turnover)
    2  turn_mean5    : rolling mean of log_turn (5 d)
    3  turn_z20      : z-score of log_turn (20 d)
    4  turn_vol20    : rolling std of log_turn (20 d)
    5  rel_spread    : relative bid-ask spread (ASK-BID)/midpoint
    6  spread_vol20  : rolling std of rel_spread (20 d)
    7  vwap_dev5     : rolling mean of (Close-VWAP)/VWAP (5 d)
    8  vwap_mom5     : VWAP momentum via log-differences (5 d)
    9  amihud20      : Amihud illiquidity ratio, 20-day rolling mean

No-leakage guarantee:
    Features at day τ use only data ≤ τ.  The supervised sample at date t
    draws its input window from [t-L, …, t-1].  The target y(t) is the
    excess return at date t (already next-day aligned in Y_excess).

Inputs
------
- Y_excess_daily_target.parquet  : daily excess returns
- mask_membership_daily.parquet  : boolean SPI-membership mask
- prices_daily.parquet           : forward-filled daily prices
- spi_TRNOVR_UNS.csv            : unsigned turnover (LSEG)
- spi_BID.csv                   : best bid price (LSEG)
- spi_ASK.csv                   : best ask price (LSEG)
- spi_VWAP.csv                  : volume-weighted average price (LSEG)

Outputs
-------
- usable_F3_L30.parquet              : boolean mask of usable (date, stock) pairs
- dataset_F3_L30.npz                 : compressed arrays (X, y, asset_id, date)
- dataset_F3_L30_feature_names.csv   : ordered feature name list
- dataset_F3_L30_id_to_stock.csv     : integer asset-id → RIC_FULL mapping
- f3_correlation_matrix.csv          : feature correlation matrix
- f3_correlation_matrix.pdf / .png   : publication-quality heatmap
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# Configuration
# ============================================================
SEED = 42
L = 30

# Input paths
Y_EXCESS_PATH = "Y_excess_daily_target.parquet"
MASK_PATH     = "mask_membership_daily.parquet"
PRICES_PATH   = "prices_daily.parquet"

CSV_TRN  = "spi_TRNOVR_UNS.csv"
CSV_BID  = "spi_BID.csv"
CSV_ASK  = "spi_ASK.csv"
CSV_VWAP = "spi_VWAP.csv"

# Output paths
OUT_USABLE = f"usable_F3_L{L}.parquet"
OUT_NPZ    = f"dataset_F3_L{L}.npz"
OUT_FEATS  = f"dataset_F3_L{L}_feature_names.csv"
OUT_MAP    = f"dataset_F3_L{L}_id_to_stock.csv"

# Rolling-window lengths
W_TURN_MEAN = 5
W_TURN_Z    = 20
W_SPREADVOL = 20
W_VWAP_DEV  = 5
W_VWAP_MOM  = 5
W_AMIHUD    = 20

EPS = 1e-12

# Per-feature clipping bounds (set to None to disable)
CLIP_ABS_RET = 1.0
CLIP_TURNZ   = 10.0
CLIP_RELSPRD = 1.0
CLIP_VWAPDEV = 1.0
CLIP_VWAPMOM = 10.0
CLIP_AMIHUD  = 1e3


# ============================================================
# Helpers
# ============================================================
def load_lseg_csv(path: str) -> pd.DataFrame:
    """Read an LSEG-format CSV, coerce to numeric, and set a DatetimeIndex."""
    df = pd.read_csv(path)
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.set_index("Date")
    else:
        df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0], errors="coerce")
        df = df.set_index(df.columns[0])
        df.index.name = "Date"
    df = df[~df.index.isna()].sort_index()
    df = df.apply(pd.to_numeric, errors="coerce")
    return df


def clip_symmetric(df: pd.DataFrame, bound: float | None) -> pd.DataFrame:
    """Clip values to [−bound, +bound].  No-op if bound is None."""
    if bound is None:
        return df
    return df.clip(lower=-bound, upper=bound)


def ffill_within_membership(df: pd.DataFrame, mask: pd.DataFrame) -> pd.DataFrame:
    """Forward-fill each column only inside its membership window.
    Gaps outside the mask stay NaN; no back-fill is applied."""
    out = df.reindex(columns=mask.columns).copy()
    for col in out.columns:
        m = mask[col].astype(bool).fillna(False)
        if m.any():
            out.loc[m, col] = out.loc[m, col].ffill()
    return out


def compute_usable_mask(Y, M, feat_ok, L):
    """Determine which (date, stock) pairs yield valid supervised samples.

    A pair (t, i) is usable iff:
      1) target y(t, i) is finite
      2) stock i is in the SPI at date t
      3) stock i has continuous membership over [t−L, …, t−1]
      4) all 10 features are finite over [t−L, …, t−1]
    """
    M_ok = M.astype(bool).fillna(False)
    M_lags_ok = (
        M_ok.rolling(L, min_periods=L).min()
        .shift(1).fillna(False).astype(bool)
    )
    F_lags_ok = (
        feat_ok.rolling(L, min_periods=L).min()
        .shift(1).fillna(False).astype(bool)
    )
    return M_ok & Y.notna() & M_lags_ok & F_lags_ok


# ============================================================
# Dataset builder
# ============================================================
def build_f3_dataset(Y, F_all, usable, L, feature_names):
    """Assemble supervised tensors by iterating over target dates.

    Returns X (N, L, F), y (N,), asset_id (N,), sample_dates (N,),
    and the id_to_stock mapping dictionary.
    """
    stocks = list(Y.columns)
    stock_to_id = {s: i for i, s in enumerate(stocks)}
    id_to_stock = {i: s for s, i in stock_to_id.items()}

    Y_arr = Y.to_numpy(dtype=np.float32)
    U_arr = usable.to_numpy(dtype=bool)
    dates = Y.index.to_numpy(dtype="datetime64[ns]")

    # Pre-stack features into (T, N_stocks, F)
    F_ordered = F_all.loc[
        :, pd.MultiIndex.from_product([stocks, feature_names], names=["RIC", "FEATURE"])
    ]
    T, N = Y_arr.shape
    Fdim = len(feature_names)
    F_arr = F_ordered.to_numpy(dtype=np.float32).reshape(T, N, Fdim)

    # Tracking counters for diagnostics
    drop_by_feature = np.zeros(Fdim, dtype=np.int64)
    drop_by_year: dict[int, int] = {}
    examples_printed = 0
    kept_total, dropped_total = 0, 0

    X_list, y_list, a_list, d_list = [], [], [], []

    for t in range(L, T):
        ok = U_arr[t, :]
        if not ok.any():
            continue

        idx = np.where(ok)[0]
        w = slice(t - L, t)
        Xw = F_arr[w, :, :][:, idx, :]           # (L, k, F)

        # Drop samples with any non-finite value in the lag window
        finite_k = np.isfinite(Xw).all(axis=(0, 2))
        if not finite_k.all():
            n_bad = int((~finite_k).sum())
            dropped_total += n_bad

            for jcol in np.where(~finite_k)[0]:
                bad_feats = ~np.isfinite(Xw[:, jcol, :]).all(axis=0)
                drop_by_feature += bad_feats.astype(np.int64)

            yr = int(pd.Timestamp(dates[t]).year)
            drop_by_year[yr] = drop_by_year.get(yr, 0) + n_bad

            if examples_printed < 3:
                j0 = np.where(~finite_k)[0][0]
                bad_names = [
                    feature_names[i]
                    for i in np.where(~np.isfinite(Xw[:, j0, :]).all(axis=0))[0]
                ]
                print(f"[drop example] date={pd.Timestamp(dates[t]).date()}  "
                      f"dropped={n_bad}  bad_features={bad_names}")
                examples_printed += 1

            idx = idx[finite_k]
            Xw = Xw[:, finite_k, :]

        if len(idx) == 0:
            continue

        X_t = Xw.transpose(1, 0, 2)
        y_t = Y_arr[t, idx]

        # Drop samples whose target is non-finite
        finite_y = np.isfinite(y_t)
        if not finite_y.all():
            dropped_total += int((~finite_y).sum())
            idx, X_t, y_t = idx[finite_y], X_t[finite_y], y_t[finite_y]

        if len(idx) == 0:
            continue

        kept_total += len(idx)
        X_list.append(X_t)
        y_list.append(y_t)
        a_list.append(idx.astype(np.int32))
        d_list.append(np.full(len(idx), dates[t], dtype="datetime64[ns]"))

    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    asset_id = np.concatenate(a_list, axis=0)
    sample_dates = np.concatenate(d_list, axis=0)

    # Print build diagnostics
    print(f"\n[build] kept: {kept_total:,}  |  dropped (non-finite): {dropped_total:,}")
    print("\n[drop diagnostics] per-feature flags:")
    for i, name in enumerate(feature_names):
        print(f"  {name:12s}: {drop_by_feature[i]:,}")
    if drop_by_year:
        top = sorted(drop_by_year.items(), key=lambda x: x[1], reverse=True)[:10]
        print("\n[drop diagnostics] top years by drop count:")
        for yr, cnt in top:
            print(f"  {yr}: {cnt:,}")

    return X, y, asset_id, sample_dates, id_to_stock


# ============================================================
# Main
# ============================================================
def main():
    np.random.seed(SEED)

    # ============================================================
    # 1  Load base data (final universe from pipeline)
    # ============================================================
    Y_excess = pd.read_parquet(Y_EXCESS_PATH)
    M = pd.read_parquet(MASK_PATH)
    P = pd.read_parquet(PRICES_PATH)

    keep_cols = list(Y_excess.columns)
    Y = Y_excess[keep_cols].copy()
    M = M.reindex(index=Y.index, columns=keep_cols).astype(bool).fillna(False)
    P = P.reindex(index=Y.index, columns=keep_cols)

    print("Loaded:")
    print(f"  Y_excess: {Y.shape}  {Y.index.min().date()} → {Y.index.max().date()}")
    print(f"  mask:     {M.shape}")
    print(f"  prices:   {P.shape}")

    # ============================================================
    # 2  Load & align LSEG microstructure fields
    # ============================================================
    trn  = load_lseg_csv(CSV_TRN)
    bid  = load_lseg_csv(CSV_BID)
    ask  = load_lseg_csv(CSV_ASK)
    vwap = load_lseg_csv(CSV_VWAP)

    def align(df, name):
        """Reindex to the common calendar/columns and apply membership mask."""
        df = df.reindex(index=Y.index, columns=keep_cols)
        df = df.apply(pd.to_numeric, errors="coerce").where(M)
        print(f"  Aligned {name}: {df.shape}  |  "
              f"median non-missing/stock={int(df.notna().sum().median())}")
        return df

    print("\nLSEG field alignment:")
    trn  = align(trn,  "TRNOVR_UNS")
    bid  = align(bid,  "BID")
    ask  = align(ask,  "ASK")
    vwap = align(vwap, "VWAP")

    # Forward-fill within membership, then enforce positivity for
    # fields that enter log or ratio calculations.
    trn  = ffill_within_membership(trn,  M)
    bid  = ffill_within_membership(bid,  M)
    ask  = ffill_within_membership(ask,  M)
    vwap = ffill_within_membership(vwap, M)

    trn  = trn.where(trn > 0)
    vwap = vwap.where(vwap > 0)

    # ============================================================
    # 3  Feature construction (all causal: data at day τ uses only ≤ τ)
    # ============================================================
    # (0) ret — excess return, symmetrically clipped
    ret = clip_symmetric(Y.copy(), CLIP_ABS_RET)

    # (1) log_turn — log-transformed turnover
    log_turn = np.log1p(trn)

    # (2) turn_mean5 — short-term average turnover level
    turn_mean5 = log_turn.rolling(W_TURN_MEAN, min_periods=W_TURN_MEAN).mean()

    # (3–4) turn_z20 / turn_vol20 — turnover z-score and volatility
    mu20 = log_turn.rolling(W_TURN_Z, min_periods=W_TURN_Z).mean()
    sd20 = log_turn.rolling(W_TURN_Z, min_periods=W_TURN_Z).std(ddof=0)
    turn_z20   = clip_symmetric((log_turn - mu20) / (sd20 + EPS), CLIP_TURNZ)
    turn_vol20 = sd20

    # (5) rel_spread — relative bid-ask spread; inverted quotes set to NaN
    good_quote = (ask >= bid) & ask.notna() & bid.notna()
    ask_clean = ask.where(good_quote)
    bid_clean = bid.where(good_quote)
    mid = (ask_clean + bid_clean) / 2.0
    rel_spread = ((ask_clean - bid_clean) / (mid.abs() + EPS)).clip(
        lower=0.0, upper=CLIP_RELSPRD,
    )

    # (6) spread_vol20 — spread volatility
    spread_vol20 = rel_spread.rolling(W_SPREADVOL, min_periods=W_SPREADVOL).std(ddof=0)

    # (7) vwap_dev5 — rolling mean deviation of close price from VWAP
    vwap_dev = clip_symmetric((P - vwap) / (vwap.abs() + EPS), CLIP_VWAPDEV)
    vwap_dev5 = vwap_dev.rolling(W_VWAP_DEV, min_periods=W_VWAP_DEV).mean()

    # (8) vwap_mom5 — VWAP momentum via cumulative log-differences
    log_vwap = np.log(vwap)
    vwap_mom5 = np.expm1(
        log_vwap.diff().rolling(W_VWAP_MOM, min_periods=W_VWAP_MOM).sum()
    ).clip(lower=-0.99, upper=CLIP_VWAPMOM)

    # (9) amihud20 — Amihud (2002) illiquidity ratio, 20-day rolling mean
    amihud20 = (
        (ret.abs() / (trn + EPS))
        .rolling(W_AMIHUD, min_periods=W_AMIHUD).mean()
        .clip(upper=CLIP_AMIHUD)
    )

    feature_names = [
        "ret", "log_turn", "turn_mean5", "turn_z20", "turn_vol20",
        "rel_spread", "spread_vol20", "vwap_dev5", "vwap_mom5", "amihud20",
    ]
    feat_frames = [
        ret, log_turn, turn_mean5, turn_z20, turn_vol20,
        rel_spread, spread_vol20, vwap_dev5, vwap_mom5, amihud20,
    ]
    print(f"\nF3 features ({len(feature_names)}): {feature_names}")

    # Combine into a single DataFrame with MultiIndex columns (RIC, FEATURE)
    F_parts = []
    for name, df in zip(feature_names, feat_frames):
        tmp = df.reindex(index=Y.index, columns=keep_cols).astype(np.float32, copy=False)
        tmp.columns = pd.MultiIndex.from_product(
            [tmp.columns, [name]], names=["RIC", "FEATURE"],
        )
        F_parts.append(tmp)

    F_all = (
        pd.concat(F_parts, axis=1)
        .reindex(columns=pd.MultiIndex.from_product(
            [keep_cols, feature_names], names=["RIC", "FEATURE"],
        ))
        .sort_index(axis=1)
    )

    # ============================================================
    # 4  Per-day, per-stock feature validity (AND across all 10 channels)
    # ============================================================
    feat_ok = None
    for fn in feature_names:
        ok = pd.DataFrame(
            np.isfinite(
                F_all.xs(fn, level="FEATURE", axis=1)
                .reindex(columns=keep_cols)
                .to_numpy(dtype=np.float32)
            ),
            index=Y.index,
            columns=keep_cols,
        )
        feat_ok = ok if feat_ok is None else (feat_ok & ok)

    # ============================================================
    # 5  Usable-sample mask
    # ============================================================
    usable = compute_usable_mask(Y, M, feat_ok, L)
    usable.to_parquet(OUT_USABLE)

    usable_per_date = usable.sum(axis=1)
    first_date = usable_per_date[usable_per_date > 0].index.min()
    last_date  = usable_per_date[usable_per_date > 0].index.max()

    print(f"\nUsable supervised examples:")
    print(f"  Date range:         {first_date.date()} → {last_date.date()}")
    print(f"  Total (date,stock): {int(usable.sum().sum()):,}")
    print(f"  Max per day:        {int(usable_per_date.max())}")

    # ============================================================
    # 6  Build supervised arrays
    # ============================================================
    X, y_arr, asset_id, sample_dates, id_to_stock = build_f3_dataset(
        Y, F_all, usable, L, feature_names,
    )

    # Final safety net: drop any residual non-finite samples
    ok = np.isfinite(y_arr) & np.isfinite(X).all(axis=(1, 2))
    if not ok.all():
        n_drop = int((~ok).sum())
        print(f"WARNING: dropping {n_drop} residual non-finite samples.")
        X, y_arr, asset_id, sample_dates = (
            X[ok], y_arr[ok], asset_id[ok], sample_dates[ok],
        )

    print(f"\nDataset shapes:")
    print(f"  X:        {X.shape}  (expected: N, {L}, {len(feature_names)})")
    print(f"  y:        {y_arr.shape}")
    print(f"  asset_id: {asset_id.shape}")
    print(f"  dates:    {sample_dates.shape}")
    print(f"  Unique assets in samples: {len(np.unique(asset_id))}")

    # ============================================================
    # 7  Save dataset
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

    # ============================================================
    # 8  Spot-check: print one supervised sample
    # ============================================================
    j = 0
    a = int(asset_id[j])
    ric = id_to_stock[a]
    t_date = pd.Timestamp(sample_dates[j])
    tpos = Y.index.get_loc(t_date)
    last_lag_day = Y.index[tpos - 1]

    print(f"\n{'=' * 40}")
    print(f"SAMPLE SPOT-CHECK (j={j})")
    print(f"{'=' * 40}")
    print(f"  asset_id={a}  RIC={ric}  target_date={t_date.date()}")
    print(f"  y(t)={float(y_arr[j]):.6f}")
    print(f"  X[j] shape: {X[j].shape}  (L, F)")
    print(f"  X[j, 0, :]  (t−L): {X[j, 0, :]}")
    print(f"  X[j, -1, :] (t−1): {X[j, -1, :]}")

    ret_in_X = float(X[j, -1, 0])
    ret_true = float(Y.loc[last_lag_day, ric])
    print(f"  last lag day: {last_lag_day.date()}")
    print(f"  ret(t−1) in X: {ret_in_X}  |  in Y: {ret_true}  |  "
          f"match: {abs(ret_in_X - ret_true) < 1e-6}")

    # ============================================================
    # 9  Audit block — training-readiness checks
    # ============================================================
    print(f"\n{'=' * 80}")
    print("F3 DATASET AUDIT (TRAINING READINESS)")
    print("=" * 80)

    # A) Input alignment
    print("\n[A] Input alignment")
    print(f"  Y={Y.shape}  M={M.shape}  P={P.shape}")
    print(f"  Date range: {Y.index.min().date()} → {Y.index.max().date()}")
    print(f"  Columns identical: "
          f"{list(Y.columns) == list(M.columns) == list(P.columns)}")

    # B) Feature integrity
    print(f"\n[B] Feature integrity (F={len(feature_names)})")
    stats = []
    for fn in feature_names:
        arr = F_all.xs(fn, level="FEATURE", axis=1).to_numpy(dtype=np.float32)
        finite = np.isfinite(arr)
        pct = 100.0 * finite.sum() / max(1, finite.size)
        if finite.any():
            v = arr[finite].astype(np.float64)
            stats.append([fn, pct, float(v.min()), float(v.max()),
                          float(v.mean()), float(v.std())])
        else:
            stats.append([fn, pct, np.nan, np.nan, np.nan, np.nan])
    stats_df = pd.DataFrame(
        stats, columns=["name", "finite_pct", "min", "max", "mean", "std"],
    )
    print(stats_df.to_string(index=False))

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
    print(f"  asset_id range: [{int(asset_id.min())}, {int(asset_id.max())}]  "
          f"n_assets={len(id_to_stock)}")

    print(f"\n{'=' * 80}")
    print("AUDIT COMPLETE")
    print("=" * 80)

    # ============================================================
    # 10  Feature correlation matrix (Appendix C)
    # ============================================================
    print(f"\n{'=' * 80}")
    print("FEATURE CORRELATION MATRIX (APPENDIX C)")
    print("=" * 80)

    X_flat = X.reshape(-1, len(feature_names))
    corr_f3 = pd.DataFrame(X_flat, columns=feature_names).corr()

    print(f"\n{corr_f3.round(3).to_string()}")
    corr_f3.to_csv("f3_correlation_matrix.csv")
    print("\nSaved: f3_correlation_matrix.csv")

    # ============================================================
    # 11  Correlation heatmap
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
        corr_f3,
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

    plt.savefig("f3_correlation_matrix.pdf", dpi=300, bbox_inches="tight")
    plt.savefig("f3_correlation_matrix.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved: f3_correlation_matrix.pdf and f3_correlation_matrix.png")


if __name__ == "__main__":
    main()