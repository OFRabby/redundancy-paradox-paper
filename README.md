# Fragile Diagnostics: Leave-One-Out Ensemble Selection is Seed-Sensitive, Detector-Dependent, and Universally Unreliable

**Md. Omar Faruk Rabby**
Department of Computer Science and Engineering, Southeast University, Dhaka 1208, Bangladesh

---

## Abstract

Leave-one-out (LOO) contribution is widely used for ensemble pruning in anomaly detection, yet its reliability as a diagnostic tool has never been validated against known ground truth. We present a two-phase evaluation: (1) a controlled synthetic benchmark with 2,000 ensemble configurations where detector roles (harmful, redundant, beneficial) are known, and (2) a real-data multi-dataset evaluation across 3 NIDS datasets, 3 encoders, and 20 seeds (180 configurations). We find that LOO detects harmful detectors with perfect recall (1.000) but misclassifies 67% of redundant detectors as beneficial (recall 0.327), a pattern we term the *redundancy paradox*. Sign-conflict between LOO's validation signal and held-out test effect occurs at 48%, independent of ensemble correlation structure. We propose epsilon-VRG, a guardrail that reduces redundancy false-pruning by 66% without retaining harmful detectors, and provide practical guidelines for NIDS ensemble construction. All code and data are publicly available.

## Installation

```bash
git clone https://github.com/ofrabby/redundancy-paradox-paper.git
cd redundancy-paradox-paper
pip install -r requirements.txt
```

## Data

Download the NIDS datasets from Kaggle:

```bash
pip install kaggle
kaggle datasets download -d ofrabby/anomaly-detector-data -p data/ --unzip
```

See [data/README.md](data/README.md) for details.

## Reproduction

```bash
# Phase 1: Target-encoder ablation (4 detectors, 20 seeds, 1 dataset)
python experiments/phase1/ablation_targetencoder.py

# Phase 2: Synthetic ground-truth benchmark (200 configs, 10 seeds)
python experiments/phase2/run_phase2.py --n_configs 200 --n_seeds 10

# Phase 3: Real-data multi-dataset evaluation (3 datasets, 3 encoders, 20 seeds)
python experiments/phase3/run_phase3.py \
    --datasets CICIDS2017,UNSW-NB15,NSL-KDD \
    --encoders target,hybrid,ordinal \
    --seeds 42-61

# Run all tests
pytest tests/ -v
```

## Repository Structure

```
├── src/                          # Core library
│   ├── load_dataset.py           # Dataset loading and feature engineering
│   ├── detector_factory.py       # 6 detector families (IF, LOF, OCSVM, AE, ECOD, HBOS)
│   ├── ensemble_eval.py          # EW/NL/EV ensemble evaluation
│   ├── stats_analysis.py         # Statistical tests (Holm, Kruskal-Wallis, Spearman)
│   └── configs/features.yaml     # Feature extraction configuration
├── experiments/                  # Experiment scripts
│   ├── phase1/                   # Target-encoder ablation
│   ├── phase2/                   # Synthetic ground-truth benchmark
│   └── phase3/                   # Real-data multi-dataset evaluation
├── tests/                        # Unit tests
├── data/                         # Dataset directory (download separately)
├── results/                      # Experiment results
├── figures/                      # Generated figures
├── manuscript/                   # LaTeX manuscript
└── .github/workflows/test.yml    # CI configuration
```

## Reproducibility Badge

[![Code Ocean](https://codeocean.com/assets/images/badge.svg)](https://codeocean.com/capsule/PLACEHOLDER)

## Interactive Notebook

Run the full Phase 3 experiment on Kaggle:
[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/ofrabby/redundancy-paradox-phase3)

## Citation

```bibtex
@article{rabby2026fragile,
  title={Fragile Diagnostics: Leave-One-Out Ensemble Selection is Seed-Sensitive, Detector-Dependent, and Universally Unreliable},
  author={Rabby, Md. Omar Faruk},
  journal={PLACEHOLDER},
  year={2026}
}
```

## License

MIT License
