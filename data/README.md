# Datasets

## Primary Datasets

The three NIDS datasets used in this study are available on Kaggle:

### Download

```bash
pip install kaggle
kaggle datasets download -d ofrabby/anomaly-detector-data -p . --unzip
```

### Contents

| Dataset | File | Rows | Features | Anomaly Ratio |
|---------|------|------|----------|---------------|
| CICIDS2017 | CICIDS2017.parquet | ~100K | Flow-based | ~15% |
| UNSW-NB15 | UNSW-NB15.parquet | ~100K | Flow-based | ~3.8% |
| NSL-KDD | NSL-KDD.parquet | ~100K | Connection-based | ~47% |

### Preprocessing

All datasets undergo identical preprocessing:
1. Categorical encoding (target, hybrid, ordinal)
2. Missing value imputation (median)
3. Feature engineering (aggregation of flow features)
4. Train/val/test split (70/15/15)
5. Standard scaling (fit on train only)

### Known Limitations

- **CICIDS2017**: Chronological split (no random shuffling)
- **UNSW-NB15**: Chronological split; 100K subsample from 1.58M preprocessed
- **NSL-KDD**: Random split (no timestamps available)

## Code Dataset

The source code for feature extraction is available at:
- Kaggle: `ofrabby/redundancy-paradox-code-v2`
- Contains: `anomaly_detector-main/` with full preprocessing pipeline

## Secondary Datasets (Optional)

Two additional datasets are available for extended evaluation:
- **CSE-CIC-IDS2018**: Requires separate download
- **TON_IoT**: Requires separate download

These are not included in the primary analysis but can be used for additional experiments.

## License

Datasets are derived from publicly available NIDS benchmarks. See individual dataset licenses for details.
