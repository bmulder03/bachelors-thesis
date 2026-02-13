"""
F3 Evaluation - Joint MTL Model
================================
Expanding-window evaluation of the joint multi-task learning model on
the full SPI universe using feature block F3 (multivariate, L=30, F=10).

F3 has the same input dimensions as F2 so the architecture and code are
identical — only the dataset and output paths differ.

Inputs
------
- evaluation_config_F3_L30.pkl : window definitions (train/val/test indices)
- dataset_F3_L30.npz          : feature matrix X (N,L,F), targets y, asset IDs

Outputs
-------
- joint_mtl_F3_L30/joint_mtl_window_metrics.csv : per-window MSE summary
- joint_mtl_F3_L30/joint_mtl_preds_Y{year}.npz  : test-set predictions
- joint_mtl_F3_L30/joint_mtl_report_Y{year}.pkl  : per-window metadata
"""

import pickle
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Sampler

# ============================================================
# Configuration – Paths
# ============================================================
EVAL_PKL = "evaluation_config_F3_L30.pkl"
NPZ_PATH = "dataset_F3_L30.npz"

OUT_DIR = Path("joint_mtl_F3_L30")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_METRICS_CSV = OUT_DIR / "joint_mtl_window_metrics.csv"

# ============================================================
# Configuration – Training
# ============================================================
SEED = 42
BATCH_SIZE = 512
LR = 3e-4
WEIGHT_DECAY = 1e-6
MAX_EPOCHS = 30
MIN_EPOCHS = 5
PATIENCE = 10
GRAD_CLIP = None
GRAD_ACCUM = 1
CAP_STEPS_PER_EPOCH: Optional[int] = None
SAVE_TEST_PREDS = True

NUM_WORKERS = 4
PERSISTENT_WORKERS = True
PREFETCH_FACTOR = 4

# ============================================================
# Configuration – Architecture
# ============================================================
SEQ_LEN = 30

JOINT_D_MODEL = 376
JOINT_N_CONV_BLOCKS = 3
JOINT_N_TRANSFORMER_LAYERS = 4
JOINT_N_HEADS = 8
JOINT_DIM_FF = 1024

HEAD_HIDDEN = 36
DROPOUT = 0.0


# ============================================================
# Utilities
# ============================================================
def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def micro_mse_np(y_true: np.ndarray, y_hat: np.ndarray) -> float:
    """Pool all observations and compute a single MSE (micro-averaged)."""
    return float(np.mean((y_true - y_hat) ** 2, dtype=np.float64))


def macro_mse_np(
    y_true: np.ndarray,
    y_hat: np.ndarray,
    asset_id: np.ndarray,
    n_assets: int,
) -> float:
    """Compute MSE per asset, then average across assets (macro-averaged).
    This gives equal weight to every stock regardless of observation count."""
    sqe = (y_true - y_hat) ** 2
    sums = np.bincount(asset_id, weights=sqe, minlength=n_assets).astype(np.float64)
    counts = np.bincount(asset_id, minlength=n_assets).astype(np.int64)
    present = counts > 0
    per_asset = np.zeros(n_assets, dtype=np.float64)
    per_asset[present] = sums[present] / counts[present]
    return float(per_asset[present].mean(dtype=np.float64))


# ============================================================
# Datasets & data loading
# ============================================================
class FullIndexDataset(Dataset):
    """Wraps the complete (X, y, asset_id) arrays as pre-converted tensors
    so that individual __getitem__ calls are pure tensor indexing.
    X is stored as (N, L, F); a 2-D input is promoted to (N, L, 1)."""

    def __init__(self, X: np.ndarray, y: np.ndarray, asset_id: np.ndarray):
        X_t = torch.from_numpy(X).float()
        if X_t.dim() == 2:
            X_t = X_t.unsqueeze(-1)
        self.X = X_t
        self.y = torch.from_numpy(y).float()
        self.asset_id = torch.from_numpy(asset_id).long()

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, j: int):
        return self.X[j], self.y[j], self.asset_id[j]


class SubsetByIndexDataset(Dataset):
    """Lightweight view into a FullIndexDataset for val/test splits."""

    def __init__(self, ds_full: FullIndexDataset, idx: np.ndarray):
        self.ds = ds_full
        self.idx = idx.astype(np.int64)

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, k: int):
        return self.ds[int(self.idx[k])]


class BalancedAssetBatchSampler(Sampler[List[int]]):
    """Yields batches by first sampling assets uniformly, then drawing one
    observation per sampled asset.  This prevents high-observation stocks
    from dominating each batch."""

    def __init__(
        self,
        train_indices: np.ndarray,
        asset_id_full: np.ndarray,
        assets_used: np.ndarray,
        batch_size: int,
        steps_per_epoch: int,
        seed: int,
    ):
        self.batch_size = int(batch_size)
        self.steps_per_epoch = int(steps_per_epoch)

        a_train = asset_id_full[train_indices]
        self.pool_by_asset: Dict[int, np.ndarray] = {}
        for a in assets_used:
            idx_a = train_indices[a_train == a]
            if len(idx_a) > 0:
                self.pool_by_asset[int(a)] = idx_a

        self.assets_available = np.array(
            sorted(self.pool_by_asset.keys()), dtype=np.int32
        )
        if len(self.assets_available) == 0:
            raise RuntimeError("No training indices for the requested assets.")

        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return self.steps_per_epoch

    def __iter__(self):
        for _ in range(self.steps_per_epoch):
            assets = self.rng.choice(
                self.assets_available, size=self.batch_size, replace=True
            )
            batch = []
            for a in assets:
                pool = self.pool_by_asset[int(a)]
                batch.append(int(pool[self.rng.integers(0, len(pool))]))
            yield batch


def _worker_kwargs() -> dict:
    """Shared DataLoader keyword arguments for parallel loading."""
    if NUM_WORKERS == 0:
        return {}
    return dict(
        num_workers=NUM_WORKERS,
        persistent_workers=PERSISTENT_WORKERS,
        prefetch_factor=PREFETCH_FACTOR,
    )


def make_train_loader_balanced(
    ds_full: FullIndexDataset,
    idx_train: np.ndarray,
    asset_id_full: np.ndarray,
    assets_used: np.ndarray,
    batch_size: int,
    seed: int,
) -> DataLoader:
    steps = max(1, len(idx_train) // batch_size)
    if CAP_STEPS_PER_EPOCH is not None:
        steps = min(steps, int(CAP_STEPS_PER_EPOCH))

    sampler = BalancedAssetBatchSampler(
        train_indices=idx_train.astype(np.int64),
        asset_id_full=asset_id_full,
        assets_used=assets_used.astype(np.int32),
        batch_size=batch_size,
        steps_per_epoch=steps,
        seed=seed,
    )
    return DataLoader(ds_full, batch_sampler=sampler, **_worker_kwargs())


def make_eval_loader(ds_subset: Dataset, batch_size: int) -> DataLoader:
    return DataLoader(
        ds_subset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        **_worker_kwargs(),
    )


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
    """Shared feature extractor: linear input projection → stacked residual
    Conv1d blocks → Transformer encoder → mean-pooled embedding.

    The input projection maps F raw features to d_model dimensions,
    making the trunk agnostic to the number of input features."""

    def __init__(
        self,
        seq_len: int = 30,
        n_features: int = 1,
        d_model: int = 512,
        n_conv_blocks: int = 3,
        n_transformer_layers: int = 4,
        n_heads: int = 8,
        dim_feedforward: int = 1024,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.n_features = n_features
        self.input_proj = nn.Linear(n_features, d_model)
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
        return x.mean(dim=1)              # (B, d) — mean-pool over sequence


class BatchedMLPHeads(nn.Module):
    """Vectorised per-asset two-layer MLP heads.

    For each sample with asset index a the output is:
        y = W2[a] · relu(W1[a] · h + b1[a]) + b2[a]

    All assets are stored in a single parameter tensor and indexed with
    advanced indexing — no Python loop over assets at forward time."""

    def __init__(self, n_assets: int, d_model: int, hidden: int):
        super().__init__()
        self.n_assets = n_assets
        self.d_model = d_model
        self.hidden = hidden

        self.W1 = nn.Parameter(torch.empty(n_assets, hidden, d_model))
        self.b1 = nn.Parameter(torch.zeros(n_assets, hidden))
        self.W2 = nn.Parameter(torch.empty(n_assets, 1, hidden))
        self.b2 = nn.Parameter(torch.zeros(n_assets, 1))

        nn.init.kaiming_uniform_(self.W1, a=np.sqrt(5))
        nn.init.kaiming_uniform_(self.W2, a=np.sqrt(5))

    def forward(self, h: torch.Tensor, asset_ids: torch.Tensor) -> torch.Tensor:
        # Layer 1: (B, hidden, d) @ (B, d, 1) → (B, hidden)
        z1 = torch.bmm(self.W1[asset_ids], h.unsqueeze(-1)).squeeze(-1) + self.b1[asset_ids]
        a1 = torch.relu(z1)

        # Layer 2: (B, 1, hidden) @ (B, hidden, 1) → scalar
        z2 = torch.bmm(self.W2[asset_ids], a1.unsqueeze(-1))
        return z2.squeeze(-1).squeeze(-1) + self.b2[asset_ids].squeeze(-1)


class JointMTL(nn.Module):
    """Full model: shared trunk → per-asset heads."""

    def __init__(self, n_assets: int, n_features: int):
        super().__init__()
        self.trunk = CNNTransformerTrunk(
            seq_len=SEQ_LEN,
            n_features=n_features,
            d_model=JOINT_D_MODEL,
            n_conv_blocks=JOINT_N_CONV_BLOCKS,
            n_transformer_layers=JOINT_N_TRANSFORMER_LAYERS,
            n_heads=JOINT_N_HEADS,
            dim_feedforward=JOINT_DIM_FF,
            dropout=DROPOUT,
        )
        self.heads = BatchedMLPHeads(
            n_assets=n_assets, d_model=JOINT_D_MODEL, hidden=HEAD_HIDDEN
        )

    def forward(self, x: torch.Tensor, asset_ids: torch.Tensor) -> torch.Tensor:
        h = self.trunk(x)
        return self.heads(h, asset_ids)


# ============================================================
# Training & evaluation for a single window
# ============================================================
def train_eval_one_window(
    ds_full: FullIndexDataset,
    model: JointMTL,
    idx_train: np.ndarray,
    idx_val: np.ndarray,
    idx_test: np.ndarray,
    assets_used: np.ndarray,
    device: torch.device,
    seed: int,
    test_year: int,
    is_first_window: bool = False,
) -> Tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
    """Train one freshly initialised model, evaluate on the held-out test
    year, and return micro/macro MSE plus raw predictions."""

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.MSELoss()

    asset_id_full_np = ds_full.asset_id.cpu().numpy().astype(np.int32)

    train_loader = make_train_loader_balanced(
        ds_full, idx_train, asset_id_full_np, assets_used, BATCH_SIZE, seed
    )
    val_loader = make_eval_loader(
        SubsetByIndexDataset(ds_full, idx_val), BATCH_SIZE
    )
    test_loader = make_eval_loader(
        SubsetByIndexDataset(ds_full, idx_test), BATCH_SIZE
    )

    # -- Training loop with early stopping --
    best_val = float("inf")
    best_state = None
    bad_epochs = 0

    if is_first_window:
        print(f"\n[{test_year}] Per-epoch validation loss (first window):")

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        opt.zero_grad(set_to_none=True)

        for step, (xb, yb, ab) in enumerate(train_loader, start=1):
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            ab = ab.to(device, non_blocking=True)

            loss = loss_fn(model(xb, ab), yb) / GRAD_ACCUM
            loss.backward()

            if step % GRAD_ACCUM == 0:
                if GRAD_CLIP is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                opt.step()
                opt.zero_grad(set_to_none=True)

        # Validation loss (batch-averaged micro MSE, no per-batch sync)
        model.eval()
        val_sum = torch.zeros((), device=device)
        val_n = 0
        with torch.no_grad():
            for xb, yb, ab in val_loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                ab = ab.to(device, non_blocking=True)
                val_sum += loss_fn(model(xb, ab), yb).detach()
                val_n += 1
        val_loss = float((val_sum / max(1, val_n)).cpu())

        if is_first_window:
            print(f"  epoch {epoch:3d}: val_loss={val_loss:.6e}")

        if val_loss < best_val - 1e-12:
            best_val = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1

        if epoch >= MIN_EPOCHS and bad_epochs >= PATIENCE:
            if is_first_window:
                print(f"  Early stopped at epoch {epoch} (best_val={best_val:.6e})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    # -- Test-set inference --
    model.eval()
    y_true_parts, y_hat_parts, a_parts = [], [], []
    with torch.no_grad():
        for xb, yb, ab in test_loader:
            xb = xb.to(device, non_blocking=True)
            ab = ab.to(device, non_blocking=True)
            y_true_parts.append(yb.numpy())
            y_hat_parts.append(model(xb, ab).cpu().numpy())
            a_parts.append(ab.cpu().numpy())

    y_true = np.concatenate(y_true_parts).astype(np.float64)
    y_hat = np.concatenate(y_hat_parts).astype(np.float64)
    a_true = np.concatenate(a_parts).astype(np.int32)

    return (
        micro_mse_np(y_true, y_hat),
        macro_mse_np(y_true, y_hat, a_true, model.heads.n_assets),
        y_true.astype(np.float32),
        y_hat.astype(np.float32),
        a_true,
    )


# ============================================================
# Main
# ============================================================
def main():
    set_seed(SEED)
    device = get_device()
    print(f"Device: {device}\n")

    with open(EVAL_PKL, "rb") as f:
        cfg = pickle.load(f)

    data = np.load(NPZ_PATH, allow_pickle=True)
    X = data["X"].astype(np.float32)
    y = data["y"].astype(np.float32)
    asset_id = data["asset_id"].astype(np.int32)

    # Auto-detect sequence length and feature dimension
    if X.ndim == 2:
        L, F = X.shape[1], 1
        X = X[:, :, None]
    elif X.ndim == 3:
        L, F = X.shape[1], X.shape[2]
    else:
        raise RuntimeError(f"Unexpected X.ndim={X.ndim}, expected 2 or 3.")

    if L != SEQ_LEN:
        print(f"WARNING: SEQ_LEN={SEQ_LEN} but data has L={L}.")

    ds_full = FullIndexDataset(X, y, asset_id)
    n_assets = int(cfg["metadata"]["n_assets"])
    assert n_assets == int(asset_id.max()) + 1

    # Print architecture summary once
    tmp = JointMTL(n_assets=n_assets, n_features=F)
    trunk_p = count_parameters(tmp.trunk)
    heads_p = count_parameters(tmp.heads)
    total_p = trunk_p + heads_p
    print("=" * 80)
    print("JointMTL parameter breakdown:")
    print(f"  Input:              L={L}, F={F}")
    print(f"  Trunk:              {trunk_p:>12,}  ({trunk_p / total_p * 100:.1f}%)")
    print(f"  Heads (vectorised): {heads_p:>12,}  ({heads_p / total_p * 100:.1f}%)")
    print(f"  Total:              {total_p:>12,}")
    print("=" * 80 + "\n")
    del tmp

    # -- Expanding-window loop --
    rows = []
    t_global = time.time()

    for wi, w in enumerate(cfg["windows"], start=1):
        test_year = int(w["test_year"])
        t0 = time.time()

        # Fresh model & deterministic seed per window
        window_seed = SEED + 10_000 * test_year
        torch.manual_seed(window_seed)
        np.random.seed(window_seed % (2**32 - 1))
        model = JointMTL(n_assets=n_assets, n_features=F).to(device)

        idx_train = w["idx_train_elig"].astype(np.int64)
        idx_val = w["idx_val_elig"].astype(np.int64)
        idx_test = w["idx_test_elig"].astype(np.int64)
        assets_used = np.unique(asset_id[idx_train]).astype(np.int32)

        mse_micro, mse_macro, y_true_t, y_hat_t, a_true_t = train_eval_one_window(
            ds_full=ds_full,
            model=model,
            idx_train=idx_train,
            idx_val=idx_val,
            idx_test=idx_test,
            assets_used=assets_used,
            device=device,
            seed=window_seed,
            test_year=test_year,
            is_first_window=(wi == 1),
        )

        dt_min = (time.time() - t0) / 60.0
        rows.append({
            "test_year": test_year,
            "n_train": len(idx_train),
            "n_val": len(idx_val),
            "n_test": len(idx_test),
            "n_assets_train": len(assets_used),
            "mse_micro": mse_micro,
            "mse_macro": mse_macro,
            "minutes": dt_min,
        })

        if SAVE_TEST_PREDS:
            np.savez_compressed(
                OUT_DIR / f"joint_mtl_preds_Y{test_year}.npz",
                idx_test_elig=idx_test,
                y_true=y_true_t,
                y_hat=y_hat_t,
                asset_id=a_true_t,
            )

        with open(OUT_DIR / f"joint_mtl_report_Y{test_year}.pkl", "wb") as f:
            pickle.dump({
                "test_year": test_year,
                "note": "Fresh init per window (strict expanding-window).",
                "assets_used": assets_used.tolist(),
            }, f)

        print(f"[{test_year}] micro={mse_micro:.6g}  macro={mse_macro:.6g} "
              f"| {dt_min:.1f} min | assets={len(assets_used)}")

        del model
        if device.type == "mps":
            torch.mps.empty_cache()

    # -- Save aggregate results --
    df = pd.DataFrame(rows).sort_values("test_year")
    df.to_csv(OUT_METRICS_CSV, index=False)

    print(f"\nSaved: {OUT_METRICS_CSV}")
    print(f"Avg micro MSE: {df['mse_micro'].mean():.6g}")
    print(f"Avg macro MSE: {df['mse_macro'].mean():.6g}")
    print(f"TOTAL runtime: {(time.time() - t_global) / 60:.1f} min")
    print(f"Outputs dir: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()


# ============================================================
# Results
# ============================================================
#
# Device: mps
#
# ================================================================================
# JointMTL parameter breakdown:
#   Input: L=30, F=10
#   Trunk: 7,916,264 (80.2%)
#   Heads (vectorized): 1,959,696 (19.8%)
#   Total: 9,875,960
#   Target: 10M (8M trunk + 2M heads)
# ================================================================================
#
# [2011] Per-epoch validation loss (first window):
#   epoch   1: val_loss=3.810780e-04
#   epoch   2: val_loss=3.676829e-04
#   epoch   3: val_loss=3.629232e-04
#   epoch   4: val_loss=3.712306e-04
#   epoch   5: val_loss=3.625143e-04
#   epoch   6: val_loss=3.569529e-04
#   epoch   7: val_loss=3.576735e-04
#   epoch   8: val_loss=3.580537e-04
#   epoch   9: val_loss=3.564023e-04
#   epoch  10: val_loss=3.565887e-04
#   epoch  11: val_loss=3.639842e-04
#   epoch  12: val_loss=3.555316e-04
#   epoch  13: val_loss=3.584958e-04
#   epoch  14: val_loss=3.555313e-04
#   epoch  15: val_loss=3.565424e-04
#   epoch  16: val_loss=3.553280e-04
#   epoch  17: val_loss=3.555960e-04
#   epoch  18: val_loss=3.550315e-04
#   epoch  19: val_loss=3.550944e-04
#   epoch  20: val_loss=3.557137e-04
#   epoch  21: val_loss=3.557034e-04
#   epoch  22: val_loss=3.548560e-04
#   epoch  23: val_loss=3.555181e-04
#   epoch  24: val_loss=3.545867e-04
#   epoch  25: val_loss=3.548179e-04
#   epoch  26: val_loss=3.556628e-04
#   epoch  27: val_loss=3.553449e-04
#   epoch  28: val_loss=3.551297e-04
#   epoch  29: val_loss=3.552914e-04
#   epoch  30: val_loss=3.553510e-04
# [2011] micro=0.000575093 macro=0.000574724 | 85.1 min | assets=117
# [2012] micro=0.000341858 macro=0.000340048 | 77.0 min | assets=120
# [2013] micro=0.000323325 macro=0.000322951 | 72.6 min | assets=123
# [2014] micro=0.000520065 macro=0.000521073 | 119.0 min | assets=129
# [2015] micro=0.000508782 macro=0.000508782 | 83.3 min | assets=126
# [2016] micro=0.000386671 macro=0.000384662 | 98.9 min | assets=129
# [2017] micro=0.000278498 macro=0.000280013 | 114.4 min | assets=130
# [2018] micro=0.000424801 macro=0.000444634 | 95.8 min | assets=130
# [2019] micro=0.00043676 macro=0.000448259 | 96.2 min | assets=129
# [2020] micro=0.000770514 macro=0.000830063 | 168.6 min | assets=128
# [2021] micro=0.000338348 macro=0.000338533 | 137.2 min | assets=126
# [2022] micro=0.00058867 macro=0.000596858 | 166.7 min | assets=126
# [2023] micro=0.000513527 macro=0.000532617 | 185.8 min | assets=123
# [2024] micro=0.00120932 macro=0.0012179 | 194.9 min | assets=119
# [2025] micro=0.000695983 macro=0.000710069 | 187.6 min | assets=118
#
# Saved: joint_mtl_F3_L30/joint_mtl_window_metrics.csv
# Avg micro MSE: 0.000527481
# Avg macro MSE: 0.000536746
# TOTAL runtime: 1883.0 min
# Outputs dir: /Users/benmulder/Documents/Dokumente/Python/joint_mtl_F3_L30