# Seed Protocol

## Overview

All experiments use deterministic random seeds to ensure reproducibility. Seeds control:
- Data splitting (train/val/test)
- Detector model initialization
- Autoencoder weight initialization
- Bootstrap sampling for confidence intervals

## Seed Ranges

### Phase 1: Target-Encoder Ablation
- **Seeds**: 42-61 (20 seeds)
- **Scope**: Per-dataset, per-encoder
- **Total runs**: 3 datasets x 3 encoders x 20 seeds = 180

### Phase 2: Synthetic Ground-Truth Benchmark
- **Seeds**: 10 per configuration (seed 42 for LHS sampling)
- **Scope**: Per-configuration
- **Total runs**: 200 configs x 10 seeds = 2,000

### Phase 3: Real-Data Multi-Dataset Evaluation
- **Seeds**: 42-61 (20 seeds)
- **Scope**: Per-dataset, per-encoder
- **Total runs**: 3 datasets x 3 encoders x 20 seeds = 180

## Seed Assignment

Seeds are assigned sequentially:
- Seed 42: first run
- Seed 43: second run
- ...
- Seed 61: last run

Each seed independently controls:
1. `train_test_split(random_state=seed)` for val/test split
2. `set_seed(seed)` before detector training
3. Bootstrap sampling for confidence intervals

## Determinism Verification

- **sklearn detectors** (IF, LOF, OCSVM): Bit-identical across runs (verified K4.14)
- **Autoencoder**: Weight-identical with same seed (verified K4.14)
- **GPU autoencoder**: May diverge from CPU due to CUDA non-determinism (verified K4.15)

## Why 20 Seeds?

- 10 seeds: insufficient for stable variance estimates (pre-v3)
- 20 seeds: 95% CI width < 0.01 for mean AUROC across all datasets
- Sensitivity analysis: 175/240 = 72.9% primary invariant, 1118/1200 = 93.2% secondary invariant
