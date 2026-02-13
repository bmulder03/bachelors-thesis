# Multi-Task Learning in Swiss Equity Markets

**An Empirical Test of Joint Estimation Efficiency**

Bachelor's Thesis — University of St. Gallen, February 2026

**Author:** Ben Mulder
**Supervisor:** Prof. Dr. Despoina Makariou, Institute of Insurance Economics

---

## Overview

This repository contains the full codebase for an empirical comparison of separate
per-asset deep learning estimation versus joint multi-task estimation for forecasting
daily excess returns of Swiss Performance Index (SPI) constituents over 1998–2025.

The central question is whether coupling parameter learning across assets through a
shared representation reduces out-of-sample forecasting error, holding the information
set, model architecture, and effective capacity fixed. Two deep learning estimators —
one training a separate CNN–Transformer model per asset, the other sharing a common
encoder across assets with asset-specific prediction heads — are compared under
identical training protocols and expanding-window evaluation across three feature
regimes of increasing dimensionality.

All models are benchmarked against naïve baselines including the zero predictor,
historical means, and a James–Stein shrinkage estimator.

## Execution Order

Scripts are designed to be run sequentially in three phases.

**Phase 1 — Data preparation** (run once, in order):

1. `data_preparation/1.1_create_events.py`
2. `data_preparation/1.2_create_current.py`
3. `data_preparation/2_spi_historical_reconstruction.py`
4. `data_preparation/3_SPI_RICs_to_pull.py`
5. `data_preparation/4_combine_initial_pull.py`
6. `data_preparation/5_clean_mask.py`

**Phase 2 — Dataset construction & validation** (per feature block):

- `data_validation/1_fragmentation_data_val.py` → `2_general_data_val.py` → `3_dataset_verification_(F1).py`
- `F1_eval/6_determine_usable_pairs(F1).py`
- `F2_eval/6.2_F2_dataset_construction.py`
- `F3_eval/6.3.1_combine_final_pull_F3.py` → `6.3.2_F3_dataset_construction.py`

**Phase 3 — Model training, evaluation & analysis** (independent across blocks):

- Within each `F*_eval/` folder: expanding window setup → joint model → separate model
- `baselines_eval/baseline0_eval.py` through `baseline3_eval.py`
- Ablations A, B, C (in any order, after Phase 3 models have been trained)
- `robustness_checks.py`
- `corr_SPI_study.py` (standalone, requires only Phase 1 outputs)

## Data

Raw data is not included in this repository due to licensing restrictions.
The following sources are required:

| Data | Source | Access |
|------|--------|--------|
| Daily adjusted closing prices (SPI constituents) | LSEG Workspace | Licensed |
| Daily volume, bid/ask, VWAP (for F3) | LSEG Workspace | Licensed |
| SPI membership events (joiners/leavers) | LSEG Workspace | Licensed |
| Risk-free rates (SARON, call money, 3M confed bills) | [SNB Data Portal](https://data.snb.ch) | Public |

Place the raw data files in the working directory as expected by the data
preparation scripts. File names and expected formats are documented in the
scripts themselves.

## Requirements

Install dependencies with:

    pip install -r requirements.txt

Core dependencies: Python ≥ 3.10, pandas, numpy, scikit-learn, torch,
matplotlib, pyarrow.

## Feature Blocks

| Block | Description | Dimensions | Information |
|-------|-------------|-----------|-------------|
| **F1** | Lagged excess returns | 30 | Minimal baseline |
| **F2** | Rolling means, volatilities, momentum, drawdown | 300 | Information-equivalent to F1 (deterministic transforms) |
| **F3** | Returns + turnover, spreads, VWAP, Amihud illiquidity | 300 | Genuinely new liquidity/microstructure information |

## Evaluation Design

- **Expanding window:** 15 annual test periods (2011–2025). Training ≤ Y−2, validation = Y−1, test = Y.
- **Metrics:** Micro- and macro-averaged MSE; out-of-sample R² relative to the zero predictor.
- **Capacity control:** Joint and separate models use identical CNN–Transformer architecture; aggregate parameter counts matched at the design target.
- **Baselines:** Zero predictor, separate historical means, pooled historical mean, James–Stein shrinkage.

## License

This code is provided for academic reference. Please contact the author before reuse.

## Contact

Ben Mulder — ben.mulder@student.unisg.ch
