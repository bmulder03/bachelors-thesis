"""
F1 Evaluation - Separate Deep Models (CPU Parallel)
====================================================
Expanding-window evaluation with one CNN-Transformer model per asset,
trained and predicted in parallel across CPU workers.

Each test year gets freshly initialised per-asset models trained on all
preceding data (strict expanding window).  The architecture mirrors the
Joint MTL trunk but is scaled down to ~70k parameters per model.

Training uses balanced early stopping per asset.  Aggregated validation
curves (carry-last weighted average) are stored for the first
window for diagnostics.

Inputs
------
- evaluation_config_F1_L30.pkl : window definitions (train/val/test indices)
- dataset_F1_L30.npz          : feature matrix X, targets y, asset IDs

Outputs
-------
- sep_deep_F1_L30_cpupar/sep_deep_window_metrics.csv : per-window MSE summary
- sep_deep_F1_L30_cpupar/sep_deep_preds_Y{year}.npz  : test-set predictions
- sep_deep_F1_L30_cpupar/sep_deep_report_Y{year}.pkl  : per-window metadata
"""

import os
import pickle
import time
import multiprocessing as mp
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ============================================================
# Configuration – Paths
# ============================================================
EVAL_PKL = "evaluation_config_F1_L30.pkl"
NPZ_PATH = "dataset_F1_L30.npz"

OUT_DIR = Path("sep_deep_F1_L30_cpupar")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_METRICS_CSV = OUT_DIR / "sep_deep_window_metrics.csv"

# ============================================================
# Configuration – Parallelism
# ============================================================
N_WORKERS = max(1, (os.cpu_count() or 4) // 2)
MP_START_METHOD = "spawn"

# ============================================================
# Configuration – Training
# ============================================================
SEED = 42
BATCH_SIZE = 64
LR = 3e-4
WEIGHT_DECAY = 1e-6
MAX_EPOCHS = 30
MIN_EPOCHS = 5
PATIENCE = 10
GRAD_CLIP = None

# ============================================================
# Configuration – Architecture (~70k params per model)
# ============================================================
# 80 % trunk, 20 % head — head size matches the per-asset head
# budget in the Joint MTL model for a fair comparison.
SEP_D_MODEL = 56
SEP_N_CONV_BLOCKS = 1
SEP_N_TRANSFORMER_LAYERS = 2
SEP_N_HEADS = 4
SEP_DIM_FF = 48
SEP_HEAD_HIDDEN = 242
DROPOUT = 0.0


# ============================================================
# Worker-global data (loaded once per spawned process)
# ============================================================
G_X: np.ndarray = None
G_y: np.ndarray = None
G_asset_id: np.ndarray = None
G_L: int = None


def _worker_init(npz_path: str, seed: int):
    """Load the dataset inside each worker to avoid pickling large arrays.
    Also pins torch to a single thread so workers don't oversubscribe."""
    global G_X, G_y, G_asset_id, G_L

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    np.random.seed(seed)
    torch.manual_seed(seed)

    data = np.load(npz_path, allow_pickle=True)
    G_X = data["X"].astype(np.float32)
    G_y = data["y"].astype(np.float32)
    G_asset_id = data["asset_id"].astype(np.int32)
    G_L = int(G_X.shape[1])


# ============================================================
# Utilities
# ============================================================
def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def micro_mse(y_true: np.ndarray, y_hat: np.ndarray) -> float:
    return float(np.mean((y_true - y_hat) ** 2, dtype=np.float64))


def macro_mse(
    y_true: np.ndarray,
    y_hat: np.ndarray,
    asset_id: np.ndarray,
    n_assets: int,
) -> float:
    """Compute MSE per asset, then average across assets (macro-averaged)."""
    sqe = (y_true - y_hat) ** 2
    sums = np.bincount(asset_id, weights=sqe, minlength=n_assets).astype(np.float64)
    counts = np.bincount(asset_id, minlength=n_assets).astype(np.int64)
    present = counts > 0
    per_asset = np.zeros(n_assets, dtype=np.float64)
    per_asset[present] = sums[present] / counts[present]
    return float(per_asset[present].mean(dtype=np.float64))


def aggregate_curves_carry_last(
    results: List[Tuple],
) -> Tuple[List[float], List[float], int]:
    """Build an aggregated train/val loss curve across all per-asset models.

    Assets that early-stopped carry their last recorded value forward so
    that epochs after stopping still contribute to the average.  Training
    curves are weighted by n_train, validation curves by n_val."""

    max_len = max(
        (len(info.get("val_hist", [])) for _, _, _, info in results),
        default=0,
    )
    if max_len == 0:
        return [], [], 0

    agg_train, agg_val = [], []

    for e in range(max_len):
        num_tr = den_tr = 0.0
        num_va = den_va = 0.0

        for _, _, _, info in results:
            th = info.get("train_hist")
            vh = info.get("val_hist")
            if not th or not vh:
                continue

            # Carry-last: clamp index to the last available epoch
            tr_val = float(th[min(e, len(th) - 1)])
            va_val = float(vh[min(e, len(vh) - 1)])
            w_tr = float(info.get("n_train", 0))
            w_va = float(info.get("n_val", 0))

            if w_tr > 0 and np.isfinite(tr_val):
                num_tr += w_tr * tr_val
                den_tr += w_tr
            if w_va > 0 and np.isfinite(va_val):
                num_va += w_va * va_val
                den_va += w_va

        agg_train.append(num_tr / den_tr if den_tr > 0 else float("nan"))
        agg_val.append(num_va / den_va if den_va > 0 else float("nan"))

    return agg_train, agg_val, max_len


# ============================================================
# Dataset & data loading (worker-side, reads from globals)
# ============================================================
class IndexDataset(Dataset):
    """Thin wrapper that indexes into the worker-global arrays."""

    def __init__(self, idx: np.ndarray):
        self.idx = idx.astype(np.int64)

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, k: int):
        j = self.idx[k]
        return torch.from_numpy(G_X[j]).float(), torch.tensor(G_y[j], dtype=torch.float32)


def make_loader(idx: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(IndexDataset(idx), batch_size=batch_size, shuffle=shuffle, drop_last=False)


# ============================================================
# Model components
# ============================================================
class ResidualConvBlock(nn.Module):
    """Conv1d → GroupNorm → ReLU → Conv1d → GroupNorm, with a residual
    shortcut (1×1 projection when channel counts differ)."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        pad = kernel_size // 2
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=pad)
        self.gn1 = nn.GroupNorm(1, out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=pad)
        self.gn2 = nn.GroupNorm(1, out_channels)
        self.relu = nn.ReLU()
        self.proj = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.proj(x)
        out = self.relu(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))
        return self.relu(out + identity)


class CNNTransformerTrunk(nn.Module):
    """Shared feature extractor: linear input projection → residual Conv1d
    blocks → Transformer encoder → mean-pooled embedding."""

    def __init__(
        self,
        seq_len: int = 30,
        d_model: int = 56,
        n_conv_blocks: int = 1,
        n_transformer_layers: int = 2,
        n_heads: int = 4,
        dim_feedforward: int = 96,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.input_proj = nn.Linear(1, d_model)
        self.conv_blocks = nn.Sequential(
            *[ResidualConvBlock(d_model, d_model) for _ in range(n_conv_blocks)]
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="relu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_transformer_layers,
            norm=nn.LayerNorm(d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        x = self.input_proj(x)            # (B, L, d)
        x = self.conv_blocks(              # conv expects (B, d, L)
            x.transpose(1, 2)
        ).transpose(1, 2)                  # back to (B, L, d)
        x = self.transformer(x)           # (B, L, d)
        return x.mean(dim=1)              # (B, d)


class MLPHead(nn.Module):
    """Single-asset two-layer MLP: embedding → hidden → scalar prediction."""

    def __init__(self, d_model: int = 56, head_hidden: int = 48):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, head_hidden),
            nn.ReLU(),
            nn.Linear(head_hidden, 1),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h).squeeze(-1)


class SeparateModel(nn.Module):
    """Full per-asset model: trunk → head."""

    def __init__(self, L: int):
        super().__init__()
        self.trunk = CNNTransformerTrunk(
            seq_len=L,
            d_model=SEP_D_MODEL,
            n_conv_blocks=SEP_N_CONV_BLOCKS,
            n_transformer_layers=SEP_N_TRANSFORMER_LAYERS,
            n_heads=SEP_N_HEADS,
            dim_feedforward=SEP_DIM_FF,
            dropout=DROPOUT,
        )
        self.head = MLPHead(d_model=SEP_D_MODEL, head_hidden=SEP_HEAD_HIDDEN)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.trunk(x))


# ============================================================
# Per-asset training & prediction (CPU, called inside workers)
# ============================================================
def train_one_asset_cpu(
    idx_train: np.ndarray,
    idx_val: np.ndarray,
    collect_curves: bool = False,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """Train a single SeparateModel with early stopping, return the model
    and a metadata dict (optionally including per-epoch loss curves)."""

    device = torch.device("cpu")
    model = SeparateModel(L=G_L).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.MSELoss()

    train_loader = make_loader(idx_train, BATCH_SIZE, shuffle=True)
    val_loader = make_loader(idx_val, BATCH_SIZE, shuffle=False)

    best_val = float("inf")
    best_state = None
    bad_epochs = 0
    stopped_epoch = 0
    train_hist, val_hist = [], []

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            if GRAD_CLIP is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()
            if collect_curves:
                train_losses.append(float(loss.item()))

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                val_losses.append(float(loss_fn(model(xb), yb).item()))
        val_loss = float(np.mean(val_losses)) if val_losses else float("inf")

        if collect_curves:
            train_hist.append(float(np.mean(train_losses)) if train_losses else float("nan"))
            val_hist.append(val_loss)

        if val_loss < best_val - 1e-12:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1

        stopped_epoch = epoch
        if epoch >= MIN_EPOCHS and bad_epochs >= PATIENCE:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    info: Dict[str, Any] = {
        "best_val": best_val,
        "stopped_epoch": stopped_epoch,
        "n_train": len(idx_train),
        "n_val": len(idx_val),
    }
    if collect_curves:
        info["train_hist"] = train_hist
        info["val_hist"] = val_hist

    return model, info


@torch.no_grad()
def predict_cpu(model: nn.Module, idx_test: np.ndarray) -> np.ndarray:
    """Run inference on test indices and return predictions as float32."""
    model.eval()
    loader = make_loader(idx_test, BATCH_SIZE, shuffle=False)
    parts = [model(xb).cpu().numpy() for xb, _ in loader]
    return np.concatenate(parts, axis=0).astype(np.float32)


def _task_train_predict(args: Tuple) -> Tuple[int, np.ndarray, np.ndarray, Dict]:
    """Worker entry point: train one asset and return predictions + metadata."""
    asset_i, idx_tr, idx_va, idx_te, pos, collect_curves = args
    t0 = time.time()
    model, info = train_one_asset_cpu(idx_tr, idx_va, collect_curves=collect_curves)
    preds = predict_cpu(model, idx_te)
    info["seconds"] = time.time() - t0
    return int(asset_i), pos, preds, info


# ============================================================
# Main
# ============================================================
def main():
    print(f"=== Separate models (CPU parallel) | workers={N_WORKERS} ===")

    with open(EVAL_PKL, "rb") as f:
        cfg = pickle.load(f)

    parent_data = np.load(NPZ_PATH, allow_pickle=True)
    y_all = parent_data["y"].astype(np.float32)
    asset_all = parent_data["asset_id"].astype(np.int32)
    n_assets = int(cfg["metadata"]["n_assets"])
    last_test_year = int(cfg["metadata"]["last_test_year"])
    first_test_year = int(cfg["windows"][0]["test_year"])

    # Print architecture summary once
    tmp = SeparateModel(L=30)
    trunk_p = count_parameters(tmp.trunk)
    head_p = count_parameters(tmp.head)
    total_p = trunk_p + head_p
    print("=" * 80)
    print("Individual model parameter breakdown:")
    print(f"  Trunk: {trunk_p:>12,}  ({trunk_p / total_p * 100:.1f}%)")
    print(f"  Head:  {head_p:>12,}  ({head_p / total_p * 100:.1f}%)")
    print(f"  Total: {total_p:>12,}")
    print("=" * 80 + "\n")
    del tmp

    ctx = mp.get_context(MP_START_METHOD)
    pool = ctx.Pool(processes=N_WORKERS, initializer=_worker_init, initargs=(NPZ_PATH, SEED))

    rows = []

    try:
        for w in cfg["windows"]:
            test_year = int(w["test_year"])
            idx_train = w["idx_train_elig"].astype(np.int64)
            idx_val = w["idx_val_elig"].astype(np.int64)
            idx_test = w["idx_test_elig"].astype(np.int64)
            IY = w["I_Y"].astype(bool)

            # Store per-epoch curves for the first window only
            is_first = (test_year == first_test_year)
            store_curves = is_first

            print(f"\n=== Window {test_year} ===")
            print(f"  train={len(idx_train):,}  val={len(idx_val):,}  "
                  f"test={len(idx_test):,}  eligible_assets={int(IY.sum())} "
                  f"| store_curves={store_curves}")

            # Build per-asset task list
            a_train = asset_all[idx_train]
            a_val = asset_all[idx_val]
            a_test = asset_all[idx_test]

            tasks = []
            for i in range(n_assets):
                if not IY[i]:
                    continue
                tr_i = idx_train[a_train == i]
                va_i = idx_val[a_val == i]
                te_i = idx_test[a_test == i]
                if len(tr_i) == 0 or len(va_i) == 0 or len(te_i) == 0:
                    continue
                pos = np.flatnonzero(a_test == i)
                tasks.append((i, tr_i, va_i, te_i, pos, store_curves))

            t0 = time.time()
            results = pool.map(_task_train_predict, tasks)
            dt = time.time() - t0

            # Assemble predictions and per-asset metadata
            pred_test = np.full(len(idx_test), np.nan, dtype=np.float32)
            y_test = y_all[idx_test].astype(np.float32)
            a_test_ids = asset_all[idx_test].astype(np.int32)

            report: Dict[str, Any] = {
                "test_year": test_year,
                "n_assets_eligible": int(IY.sum()),
                "store_curves": store_curves,
                "assets": {},
            }

            for asset_i, pos, preds, info in results:
                if len(pos) != len(preds):
                    raise RuntimeError(
                        f"Alignment mismatch: asset {asset_i}, year {test_year} "
                        f"(pos={len(pos)}, preds={len(preds)})"
                    )
                pred_test[pos] = preds

                entry = {
                    "best_val": float(info["best_val"]),
                    "stopped_epoch": int(info["stopped_epoch"]),
                    "seconds": float(info["seconds"]),
                    "n_train": int(info.get("n_train", 0)),
                    "n_val": int(info.get("n_val", 0)),
                }
                if store_curves:
                    entry["train_hist"] = info["train_hist"]
                    entry["val_hist"] = info["val_hist"]
                report["assets"][int(asset_i)] = entry

            # Aggregate validation curves (carry-last) for saved windows
            if store_curves:
                agg_train, agg_val, max_len = aggregate_curves_carry_last(results)
                report["aggregate"] = {
                    "train_hist": agg_train,
                    "val_hist": agg_val,
                    "max_len": max_len,
                    "note": "Carry-last aggregation; train weighted by n_train, val by n_val.",
                }

                if is_first:
                    print(f"\n[{test_year}] Aggregated validation loss per epoch "
                          f"(carry-last, weighted by n_val):")
                    n_total = len(results)
                    for e, v in enumerate(agg_val, start=1):
                        active = sum(
                            1 for _, _, _, info in results
                            if info.get("val_hist") and len(info["val_hist"]) >= e
                        )
                        print(f"  epoch {e:02d}: val={v:.12e}  "
                              f"(active_assets={active}/{n_total})")

            # Score predictions
            ok = ~np.isnan(pred_test)
            if not ok.all():
                print(f"[{test_year}] WARNING: {int((~ok).sum())} missing preds; "
                      f"scoring predicted samples only.")

            y_true = y_test[ok].astype(np.float64)
            y_hat = pred_test[ok].astype(np.float64)
            a_true = a_test_ids[ok].astype(np.int32)

            mse_micro = micro_mse(y_true, y_hat)
            mse_macro = macro_mse(y_true, y_hat, a_true, n_assets)

            # Save predictions and report
            pred_path = OUT_DIR / f"sep_deep_preds_Y{test_year}.npz"
            np.savez_compressed(
                pred_path,
                idx_test_elig=idx_test,
                pred_test=pred_test,
                y_test=y_test,
                asset_id_test=a_test_ids,
            )

            report_path = OUT_DIR / f"sep_deep_report_Y{test_year}.pkl"
            with open(report_path, "wb") as f:
                pickle.dump(report, f)

            rows.append({
                "test_year": test_year,
                "n_test": len(idx_test),
                "n_assets_in_test": int(np.unique(a_test_ids).size),
                "mse_micro": mse_micro,
                "mse_macro": mse_macro,
                "seconds_window": dt,
                "pred_file": pred_path.name,
                "report_file": report_path.name,
            })

            print(f"[{test_year}] done in {dt / 60:.1f} min | "
                  f"micro={mse_micro:.6g}  macro={mse_macro:.6g} | "
                  f"saved {pred_path.name}")
            if store_curves:
                print(f"[{test_year}] NOTE: Stored per-epoch curves in {report_path.name}.")

    finally:
        pool.close()
        pool.join()

    # -- Save aggregate results --
    df = pd.DataFrame(rows).sort_values("test_year")
    df.to_csv(OUT_METRICS_CSV, index=False)

    print(f"\nSaved: {OUT_METRICS_CSV}")
    print(f"Avg micro MSE: {df['mse_micro'].mean():.6g}")
    print(f"Avg macro MSE: {df['mse_macro'].mean():.6g}")
    print(f"Outputs dir: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    mp.set_start_method(MP_START_METHOD, force=True)
    main()


# ============================================================
# Results
# ============================================================

# === Separate models (CPU parallel) | workers=5 ===

# ================================================================================
# Individual model parameter breakdown:
#   Trunk: 56,320 (80.0%)
#   Head: 14,037 (20.0%)
#   Total: 70,357
#   Target: 70k (56k trunk + 14k head)
# ================================================================================

# === Window 2011 ===
#   train=308,984 val=29,780 test=29,537 eligible_assets=118 | store_curves=True
#
# [2011] Aggregated validation loss per epoch (carry-last, weighted by n_val):
#   epoch 01: val=4.687815739056e-04  (active_assets=118/118)
#   epoch 02: val=3.882892223904e-04  (active_assets=118/118)
#   epoch 03: val=3.878979271356e-04  (active_assets=118/118)
#   epoch 04: val=3.847561888776e-04  (active_assets=118/118)
#   epoch 05: val=3.903453180111e-04  (active_assets=118/118)
#   epoch 06: val=4.026349479255e-04  (active_assets=118/118)
#   epoch 07: val=4.109601990103e-04  (active_assets=118/118)
#   epoch 08: val=4.046721970559e-04  (active_assets=118/118)
#   epoch 09: val=4.153067991566e-04  (active_assets=118/118)
#   epoch 10: val=4.030963863946e-04  (active_assets=118/118)
#   epoch 11: val=4.064478325898e-04  (active_assets=118/118)
#   epoch 12: val=4.140674755419e-04  (active_assets=113/118)
#   epoch 13: val=4.139296207908e-04  (active_assets=103/118)
#   epoch 14: val=4.031378724800e-04  (active_assets=92/118)
#   epoch 15: val=4.215292278903e-04  (active_assets=85/118)
#   epoch 16: val=4.144851218127e-04  (active_assets=78/118)
#   epoch 17: val=4.211294230839e-04  (active_assets=69/118)
#   epoch 18: val=4.126157936375e-04  (active_assets=61/118)
#   epoch 19: val=4.088474674257e-04  (active_assets=57/118)
#   epoch 20: val=4.154532833719e-04  (active_assets=51/118)
#   epoch 21: val=4.315955908204e-04  (active_assets=43/118)
#   epoch 22: val=4.157727226233e-04  (active_assets=35/118)
#   epoch 23: val=4.164206367997e-04  (active_assets=28/118)
#   epoch 24: val=4.216830214183e-04  (active_assets=26/118)
#   epoch 25: val=4.238279916788e-04  (active_assets=21/118)
#   epoch 26: val=4.212708418363e-04  (active_assets=19/118)
#   epoch 27: val=4.258712513166e-04  (active_assets=16/118)
#   epoch 28: val=4.231161527038e-04  (active_assets=15/118)
#   epoch 29: val=4.239792612397e-04  (active_assets=14/118)
#   epoch 30: val=4.191005742745e-04  (active_assets=10/118)
# [2011] done in 12.4 min | micro=0.000580454 macro=0.000581455
# [2011] NOTE: Stored per-epoch train/val curves in sep_deep_report_Y2011.pkl.
#
# [2012] done in 13.2 min | micro=0.000347286 macro=0.000345463
# [2013] done in 13.6 min | micro=0.000322351 macro=0.000321981
# [2014] done in 16.2 min | micro=0.000519382 macro=0.000520197
# [2015] done in 15.5 min | micro=0.000508252 macro=0.000508252
# [2016] done in 17.4 min | micro=0.000386352 macro=0.000384428
# [2017] done in 17.2 min | micro=0.000277665 macro=0.000279093
# [2018] done in 17.8 min | micro=0.000428097 macro=0.000445819
# [2019] done in 21.6 min | micro=0.000446961 macro=0.000448506
# [2020] done in 21.4 min | micro=0.000773205 macro=0.000821235
# [2021] done in 21.6 min | micro=0.000338251 macro=0.000338251
# [2022] done in 22.4 min | micro=0.000590118 macro=0.000589839
# [2023] done in 24.1 min | micro=0.000513923 macro=0.000530112
# [2024] done in 22.9 min | micro=0.00121118  macro=0.00121287
# [2025] done in 24.0 min | micro=0.000704262 macro=0.000700588
# [2025] NOTE: Stored per-epoch train/val curves in sep_deep_report_Y2025.pkl.
#
# Saved: sep_deep_F1_L30_cpupar/sep_deep_window_metrics.csv
# Avg micro MSE: 0.000529850
# Avg macro MSE: 0.000535206
# Outputs dir: /Users/benmulder/Documents/Dokumente/Python/sep_deep_F1_L30_cpupar