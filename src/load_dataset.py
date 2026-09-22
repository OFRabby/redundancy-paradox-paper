"""
Phase 3: Dataset Loading and Feature Extraction
================================================
Loads 5 NIDS datasets, splits them, and extracts features using Phase 1's
FeatureExtractor. All loaders return (df_train, df_val, df_test, y_train, y_val, y_test).

Usage:
    from load_dataset import load_dataset, build_features, DATASET_LOADERS
    result = load_dataset("CICIDS2017", seed=42)
    X_train, X_val, X_test, extractor = build_features(*result, encoder_type="target")
"""

import os
import sys
import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import polars as pl
from sklearn.model_selection import train_test_split

# ============================================================================
# CONSTANTS
# ============================================================================

DATA_DIR_KAGGLE = "/kaggle/input/anomaly-detector-data"
DATA_DIR_KAGGLE_ALT = "/kaggle/input/datasets/ofrabby/anomaly-detector-data"
DATA_DIR_LOCAL = "data/processed"

SEEDS = list(range(42, 62))  # 20 seeds
SPLIT = (0.70, 0.15, 0.15)  # train/val/test — matches Phase 1

SUBSAMPLE_100K = {"CSE-CIC-IDS2018", "TON_IoT"}

FE_CONFIG = {
    "host_windows": ["1h", "6h", "24h"],
    "exclude_columns": [
        "ts", "uid", "id.orig_h", "id.orig_p", "id.resp_h",
        "id.resp_p", "label", "attack_category", "is_anomaly",
    ],
}

CANONICAL_SCHEMA = [
    "proto", "service", "conn_state", "duration", "orig_bytes", "resp_bytes",
    "orig_pkts", "resp_pkts", "ts", "id.orig_h", "id.orig_p", "id.resp_h",
    "id.resp_p", "uid", "label", "attack_category", "is_anomaly",
]

logger = logging.getLogger(__name__)

# ============================================================================
# PROJECT ROOT AND PATHS
# ============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent / "anomaly_detector-main" / "anomaly_detector-main"
if not PROJECT_ROOT.exists():
    PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent

sys.path.insert(0, str(PROJECT_ROOT))

from src.features.extractor import FeatureExtractor  # noqa: E402

# ============================================================================
# PATH RESOLUTION
# ============================================================================


def find_data_dir() -> Path:
    """Locate data directory: prefer Kaggle mount, fallback to local."""
    for kaggle_path in [Path(DATA_DIR_KAGGLE), Path(DATA_DIR_KAGGLE_ALT)]:
        if kaggle_path.is_dir():
            logger.info(f"Using Kaggle data dir: {kaggle_path}")
            return kaggle_path

    local_path = PROJECT_ROOT / DATA_DIR_LOCAL
    if local_path.is_dir():
        logger.info(f"Using local data dir: {local_path}")
        return local_path

    raise FileNotFoundError(
        f"Neither Kaggle ({DATA_DIR_KAGGLE}, {DATA_DIR_KAGGLE_ALT}) nor local ({local_path}) data dir found"
    )


# ============================================================================
# SPLITTING
# ============================================================================


def split_dataset(
    df: pl.DataFrame,
    strategy: str,
    seed: int = 42,
    time_col: str = "ts",
    train_frac: float = SPLIT[0],
    val_frac: float = SPLIT[1],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Produce exact train/val/test splits.

    Parameters
    ----------
    df : pl.DataFrame
        Full dataset.
    strategy : str
        "chronological" (sort by time_col) or "random" (stratified shuffle).
    seed : int
        Random seed for reproducibility.
    time_col : str
        Column name for chronological splitting.
    train_frac, val_frac : float
        Split fractions. Test = 1 - train - val.

    Returns
    -------
    train_idx, val_idx, test_idx : np.ndarray
        Non-overlapping index arrays.
    """
    n = len(df)
    train_n = int(n * train_frac)
    val_n = int(n * val_frac)
    test_n = n - train_n - val_n

    if strategy == "chronological" and time_col in df.columns:
        time_vals = df[time_col].to_numpy()
        sorted_idx = np.argsort(time_vals)
        train_idx = sorted_idx[:train_n]
        val_idx = sorted_idx[train_n: train_n + val_n]
        test_idx = sorted_idx[train_n + val_n:]
    elif strategy == "random":
        rng = np.random.RandomState(seed)
        indices = rng.permutation(n)
        train_idx = indices[:train_n]
        val_idx = indices[train_n: train_n + val_n]
        test_idx = indices[train_n + val_n:]
    else:
        raise ValueError(f"Unknown split strategy: {strategy}")

    # Verify no overlap
    assert len(set(train_idx) & set(val_idx)) == 0, "Train/val overlap"
    assert len(set(train_idx) & set(test_idx)) == 0, "Train/test overlap"
    assert len(set(val_idx) & set(test_idx)) == 0, "Val/test overlap"
    assert len(train_idx) + len(val_idx) + len(test_idx) == n, "Split size mismatch"

    return train_idx, val_idx, test_idx


def _subsample_stratified(df: pl.DataFrame, n_samples: int, seed: int = 42) -> pl.DataFrame:
    """Stratified subsample to n_samples rows, preserving class ratios."""
    if len(df) <= n_samples:
        return df

    rng = np.random.RandomState(seed)
    labels = df["is_anomaly"].to_numpy()
    indices = np.arange(len(df))

    # Stratified split: keep n_samples
    kept, _ = train_test_split(
        indices, train_size=n_samples, stratify=labels, random_state=seed
    )
    kept.sort()
    return df[kept]


# ============================================================================
# DATASET LOADERS
# ============================================================================


def load_cicids2017(seed: int = 42) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame,
                                               np.ndarray, np.ndarray, np.ndarray]:
    """Load CICIDS2017. Chronological split on ts.

    Returns (df_train, df_val, df_test, y_train, y_val, y_test).
    """
    data_dir = find_data_dir()
    path = data_dir / "cicids2017.parquet"
    if not path.exists():
        raise FileNotFoundError(f"CICIDS2017 not found: {path}")

    df = pl.read_parquet(path)
    logger.info(f"CICIDS2017 loaded: {len(df)} rows")

    train_idx, val_idx, test_idx = split_dataset(df, strategy="chronological", seed=seed,
                                                  time_col="ts")

    df_train = df[train_idx]
    df_val = df[val_idx]
    df_test = df[test_idx]

    y_train = df_train["is_anomaly"].to_numpy().astype(int)
    y_val = df_val["is_anomaly"].to_numpy().astype(int)
    y_test = df_test["is_anomaly"].to_numpy().astype(int)

    logger.info(f"CICIDS2017 split: train={len(df_train)}, val={len(df_val)}, test={len(df_test)}")
    return df_train, df_val, df_test, y_train, y_val, y_test


def load_unsw_nb15(seed: int = 42) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame,
                                              np.ndarray, np.ndarray, np.ndarray]:
    """Load UNSW-NB15. Chronological split on ts.

    Returns (df_train, df_val, df_test, y_train, y_val, y_test).
    """
    data_dir = find_data_dir()
    path = data_dir / "unsw_nb15.parquet"
    if not path.exists():
        raise FileNotFoundError(f"UNSW-NB15 not found: {path}")

    df = pl.read_parquet(path)
    logger.info(f"UNSW-NB15 loaded: {len(df)} rows")

    train_idx, val_idx, test_idx = split_dataset(df, strategy="chronological", seed=seed,
                                                  time_col="ts")

    df_train = df[train_idx]
    df_val = df[val_idx]
    df_test = df[test_idx]

    y_train = df_train["is_anomaly"].to_numpy().astype(int)
    y_val = df_val["is_anomaly"].to_numpy().astype(int)
    y_test = df_test["is_anomaly"].to_numpy().astype(int)

    logger.info(f"UNSW-NB15 split: train={len(df_train)}, val={len(df_val)}, test={len(df_test)}")
    return df_train, df_val, df_test, y_train, y_val, y_test


def load_nsl_kdd(seed: int = 42) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame,
                                            np.ndarray, np.ndarray, np.ndarray]:
    """Load NSL-KDD. Random stratified split (no timestamps).

    Returns (df_train, df_val, df_test, y_train, y_val, y_test).
    """
    data_dir = find_data_dir()
    path = data_dir / "nsl_kdd.parquet"
    if not path.exists():
        raise FileNotFoundError(f"NSL-KDD not found: {path}")

    df = pl.read_parquet(path)
    logger.info(f"NSL-KDD loaded: {len(df)} rows")

    train_idx, val_idx, test_idx = split_dataset(df, strategy="random", seed=seed)

    df_train = df[train_idx]
    df_val = df[val_idx]
    df_test = df[test_idx]

    y_train = df_train["is_anomaly"].to_numpy().astype(int)
    y_val = df_val["is_anomaly"].to_numpy().astype(int)
    y_test = df_test["is_anomaly"].to_numpy().astype(int)

    logger.info(f"NSL-KDD split: train={len(df_train)}, val={len(df_val)}, test={len(df_test)}")
    return df_train, df_val, df_test, y_train, y_val, y_test


def load_cse_cic_ids2018(seed: int = 42) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame,
                                                    np.ndarray, np.ndarray, np.ndarray]:
    """Load CSE-CIC-IDS2018 (subsampled to 100K). Random stratified split.

    Returns (df_train, df_val, df_test, y_train, y_val, y_test).
    """
    data_dir = find_data_dir()
    path = data_dir / "cse_cic_ids2018.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"CSE-CIC-IDS2018 not found: {path}. Run prepare_new_datasets.py first."
        )

    df = pl.read_parquet(path)
    logger.info(f"CSE-CIC-IDS2018 loaded: {len(df)} rows")

    train_idx, val_idx, test_idx = split_dataset(df, strategy="random", seed=seed)

    df_train = df[train_idx]
    df_val = df[val_idx]
    df_test = df[test_idx]

    y_train = df_train["is_anomaly"].to_numpy().astype(int)
    y_val = df_val["is_anomaly"].to_numpy().astype(int)
    y_test = df_test["is_anomaly"].to_numpy().astype(int)

    logger.info(
        f"CSE-CIC-IDS2018 split: train={len(df_train)}, val={len(df_val)}, test={len(df_test)}"
    )
    return df_train, df_val, df_test, y_train, y_val, y_test


def load_ton_iot(seed: int = 42) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame,
                                           np.ndarray, np.ndarray, np.ndarray]:
    """Load TON_IoT (subsampled to 100K). Random stratified split.

    Returns (df_train, df_val, df_test, y_train, y_val, y_test).
    """
    data_dir = find_data_dir()
    path = data_dir / "ton_iot.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"TON_IoT not found: {path}. Run prepare_new_datasets.py first."
        )

    df = pl.read_parquet(path)
    logger.info(f"TON_IoT loaded: {len(df)} rows")

    train_idx, val_idx, test_idx = split_dataset(df, strategy="random", seed=seed)

    df_train = df[train_idx]
    df_val = df[val_idx]
    df_test = df[test_idx]

    y_train = df_train["is_anomaly"].to_numpy().astype(int)
    y_val = df_val["is_anomaly"].to_numpy().astype(int)
    y_test = df_test["is_anomaly"].to_numpy().astype(int)

    logger.info(f"TON_IoT split: train={len(df_train)}, val={len(df_val)}, test={len(df_test)}")
    return df_train, df_val, df_test, y_train, y_val, y_test


# ============================================================================
# DISPATCH TABLE
# ============================================================================

DATASET_LOADERS = {
    "CICIDS2017": load_cicids2017,
    "UNSW-NB15": load_unsw_nb15,
    "NSL-KDD": load_nsl_kdd,
    "CSE-CIC-IDS2018": load_cse_cic_ids2018,
    "TON_IoT": load_ton_iot,
}


# ============================================================================
# FEATURE EXTRACTION
# ============================================================================


def build_features(
    df_train: pl.DataFrame,
    df_val: pl.DataFrame,
    df_test: pl.DataFrame,
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    encoder_type: str = "target",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, "FeatureExtractor"]:
    """Extract features using Phase 1's FeatureExtractor.

    Parameters
    ----------
    df_train, df_val, df_test : pl.DataFrame
        Raw DataFrames from load_dataset.
    y_train, y_val, y_test : np.ndarray
        Binary labels (0=normal, 1=attack).
    encoder_type : str
        "target", "hybrid", or "ordinal".

    Returns
    -------
    X_train, X_val, X_test : np.ndarray
        Transputed feature matrices.
    extractor : FeatureExtractor
        Fitted extractor (for inverse transforms or saving).
    """
    extractor = FeatureExtractor(FE_CONFIG, encoder_type=encoder_type)

    if encoder_type == "target":
        X_train = extractor.fit_transform(df_train, labels=y_train)
    else:
        X_train = extractor.fit_transform(df_train, labels=None)

    X_val = extractor.transform(df_val)
    X_test = extractor.transform(df_test)

    logger.info(
        f"Features extracted: X_train={X_train.shape}, X_val={X_val.shape}, "
        f"X_test={X_test.shape}, encoder={encoder_type}"
    )
    return X_train, X_val, X_test, extractor


def load_dataset(
    name: str,
    seed: int = 42,
) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame,
           np.ndarray, np.ndarray, np.ndarray]:
    """Load a dataset by name.

    Parameters
    ----------
    name : str
        Dataset name (key in DATASET_LOADERS).
    seed : int
        Random seed for splitting.

    Returns
    -------
    df_train, df_val, df_test : pl.DataFrame
    y_train, y_val, y_test : np.ndarray
    """
    if name not in DATASET_LOADERS:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(DATASET_LOADERS.keys())}")

    return DATASET_LOADERS[name](seed=seed)
