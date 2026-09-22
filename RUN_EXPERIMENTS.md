# Reproducing Experiments

## Prerequisites

1. Install dependencies: `pip install -r requirements.txt`
2. Download data: `kaggle datasets download -d ofrabby/anomaly-detector-data -p data/ --unzip`

## Phase 1: Target-Encoder Ablation

Validates encoder sensitivity on a single dataset (CICIDS2017) with 4 detectors.

```bash
python experiments/phase1/ablation_targetencoder.py
```

- **Datasets**: CICIDS2017
- **Detectors**: IF, LOF, OCSVM, AE
- **Encoders**: target, hybrid, ordinal
- **Seeds**: 42-61 (20 seeds)
- **Runtime**: ~10 minutes (CPU)
- **Output**: Per-seed AUROC, sign-conflict rates, encoder comparison

## Phase 2: Synthetic Ground-Truth Benchmark

Validates LOO diagnostic claims with known detector roles.

```bash
python experiments/phase2/run_phase2.py --n_configs 200 --n_seeds 10
```

- **Configurations**: 200 (Latin Hypercube Sampling)
- **Seeds**: 10 per configuration
- **Total runs**: 2,000
- **Runtime**: ~2 hours (CPU)
- **Output**: Confusion matrix (H/R/B), sign-conflict rates, epsilon-VRG validation

### Hypotheses

| Hypothesis | Claim | Criterion |
|-----------|-------|-----------|
| H1 | LOO detects harmful detectors | Recall >= 0.80 |
| H2 | LOO prunes more redundant than beneficial | R_recall > B_recall |
| H3 | Sign-conflict predicts harm | Spearman rho > 0.3 |
| H4 | epsilon-VRG reduces false-pruning | R-fp reduction >= 50% |

## Phase 3: Real-Data Multi-Dataset Evaluation

Full evaluation across 3 datasets, 3 encoders, 20 seeds.

```bash
python experiments/phase3/run_phase3.py \
    --datasets CICIDS2017,UNSW-NB15,NSL-KDD \
    --encoders target,hybrid,ordinal \
    --seeds 42-61 \
    --outdir results/phase3
```

- **Datasets**: CICIDS2017, UNSW-NB15, NSL-KDD
- **Encoders**: target, hybrid, ordinal
- **Detectors**: IF, LOF, OCSVM, AE, ECOD, HBOS
- **Seeds**: 42-61 (20 seeds)
- **Total configurations**: 3 x 3 x 20 = 180
- **Runtime**: ~8-14 hours (GPU recommended)
- **Output**: Per-dataset method means, Holm-corrected comparisons, seed-fragility analysis

### Statistical Analysis

- Wilcoxon signed-rank test (two-sided)
- Holm correction for multiple comparisons
- Hierarchical linear model for seed variance estimation
- Kruskal-Wallis for cross-dataset comparison

## Tests

```bash
pytest tests/ -v
```

## Kaggle Execution

For GPU-accelerated execution, use the Kaggle notebook:
- URL: https://www.kaggle.com/code/ofrabby/redundancy-paradox-phase3
- Requires GPU (T4 or better)
- Datasets auto-mounted from `ofrabby/anomaly-detector-data` and `ofrabby/redundancy-paradox-code-v2`
