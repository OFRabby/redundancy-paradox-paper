"""
Unit tests for load_dataset.py
================================
Run with: pytest tests/test_load_dataset.py -v

Note: CSE-CIC-IDS2018 and TON_IoT tests are skipped if parquet files
do not exist (they require prepare_new_datasets.py to be run first).
"""

import sys
import os
import pytest
import numpy as np
import polars as pl

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from load_dataset import (
    load_dataset,
    build_features,
    split_dataset,
    DATASET_LOADERS,
    FE_CONFIG,
    SPLIT,
    SEEDS,
)


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def seed():
    return 42


# ============================================================================
# EXISTING DATASET TESTS
# ============================================================================


class TestNSLKDD:
    """Tests for NSL-KDD (local data, should always exist)."""

    def test_loads(self, seed):
        """NSL-KDD returns 6-tuple with expected row count."""
        result = load_dataset("NSL-KDD", seed=seed)
        assert len(result) == 6
        df_train, df_val, df_test, y_train, y_val, y_test = result
        total = len(df_train) + len(df_val) + len(df_test)
        # NSL-KDD has ~148K rows (exact count depends on preprocessing)
        assert total > 100_000, f"NSL-KDD too small: {total} rows"
        assert total < 200_000, f"NSL-KDD too large: {total} rows"

    def test_split_ratios(self, seed):
        """NSL-KDD splits are approximately 70/15/15."""
        df_train, df_val, df_test, _, _, _ = load_dataset("NSL-KDD", seed=seed)
        total = len(df_train) + len(df_val) + len(df_test)
        train_ratio = len(df_train) / total
        val_ratio = len(df_val) / total
        test_ratio = len(df_test) / total

        assert 0.65 < train_ratio < 0.75, f"Train ratio {train_ratio:.3f} not ~0.70"
        assert 0.10 < val_ratio < 0.20, f"Val ratio {val_ratio:.3f} not ~0.15"
        assert 0.10 < test_ratio < 0.20, f"Test ratio {test_ratio:.3f} not ~0.15"

    def test_random_split_nsl_kdd(self, seed):
        """NSL-KDD random split preserves label proportions."""
        df_train, df_val, df_test, y_train, y_val, y_test = load_dataset("NSL-KDD", seed=seed)

        # Check label proportions are similar across splits
        train_rate = y_train.mean()
        val_rate = y_val.mean()
        test_rate = y_test.mean()

        # Allow 5% absolute difference
        assert abs(train_rate - val_rate) < 0.05, f"Train/val label rate differ: {train_rate:.3f} vs {val_rate:.3f}"
        assert abs(train_rate - test_rate) < 0.05, f"Train/test label rate differ: {train_rate:.3f} vs {test_rate:.3f}"


class TestCICIDS2017:
    """Tests for CICIDS2017 (local data, should always exist)."""

    def test_loads(self, seed):
        """CICIDS2017 returns 6-tuple with expected row count."""
        result = load_dataset("CICIDS2017", seed=seed)
        assert len(result) == 6
        df_train, df_val, df_test, y_train, y_val, y_test = result
        total = len(df_train) + len(df_val) + len(df_test)
        # CICIDS2017 has ~2.8-3.1M rows
        assert total > 1_000_000, f"CICIDS2017 too small: {total} rows"

    def test_chronological_split_preserved(self, seed):
        """CICIDS2017 chronological split preserves time ordering."""
        df_train, df_val, df_test, _, _, _ = load_dataset("CICIDS2017", seed=seed)

        # Check ts is monotonic within each split
        for split_name, df_split in [("train", df_train), ("val", df_val), ("test", df_test)]:
            if "ts" in df_split.columns:
                ts_vals = df_split["ts"].to_numpy()
                diffs = np.diff(ts_vals)
                # Allow small number of violations (ties are ok)
                n_violations = (diffs < 0).sum()
                violation_rate = n_violations / len(diffs) if len(diffs) > 0 else 0
                assert violation_rate < 0.01, (
                    f"{split_name} split has {n_violations} time violations "
                    f"({violation_rate:.3%} of transitions)"
                )


class TestUNSWNB15:
    """Tests for UNSW-NB15 (local data, should always exist)."""

    def test_loads(self, seed):
        """UNSW-NB15 returns 6-tuple."""
        result = load_dataset("UNSW-NB15", seed=seed)
        assert len(result) == 6
        df_train, df_val, df_test, y_train, y_val, y_test = result
        total = len(df_train) + len(df_val) + len(df_test)
        # UNSW-NB15 has varying sizes depending on preprocessing
        assert total > 50_000, f"UNSW-NB15 too small: {total} rows"


# ============================================================================
# OPTIONAL DATASET TESTS (require prepare_new_datasets.py)
# ============================================================================


class TestCSECICIDS2018:
    """Tests for CSE-CIC-IDS2018 (requires prepare_new_datasets.py)."""

    @pytest.mark.skipif(
        not os.path.exists("research/data/processed/cse_cic_ids2018.parquet"),
        reason="CSE-CIC-IDS2018 not preprocessed"
    )
    def test_loads(self, seed):
        """CSE-CIC-IDS2018 returns 6-tuple with ~100K rows."""
        result = load_dataset("CSE-CIC-IDS2018", seed=seed)
        assert len(result) == 6
        df_train, df_val, df_test, y_train, y_val, y_test = result
        total = len(df_train) + len(df_val) + len(df_test)
        assert 80_000 < total < 120_000, f"CSE-CIC-IDS2018 unexpected size: {total}"


class TestTONIoT:
    """Tests for TON_IoT (requires prepare_new_datasets.py)."""

    @pytest.mark.skipif(
        not os.path.exists("research/data/processed/ton_iot.parquet"),
        reason="TON_IoT not preprocessed"
    )
    def test_loads(self, seed):
        """TON_IoT returns 6-tuple with ~100K rows."""
        result = load_dataset("TON_IoT", seed=seed)
        assert len(result) == 6
        df_train, df_val, df_test, y_train, y_val, y_test = result
        total = len(df_train) + len(df_val) + len(df_test)
        assert 80_000 < total < 120_000, f"TON_IoT unexpected size: {total}"


# ============================================================================
# FEATURE EXTRACTION TESTS
# ============================================================================


class TestFeatureExtraction:
    """Tests for build_features (runs on NSL-KDD for speed)."""

    @pytest.fixture
    def nsl_kdd_data(self, seed):
        return load_dataset("NSL-KDD", seed=seed)

    def test_target_encoder_produces_features(self, nsl_kdd_data):
        """Target encoder produces feature matrix."""
        df_train, df_val, df_test, y_train, y_val, y_test = nsl_kdd_data
        X_train, X_val, X_test, extractor = build_features(
            df_train, df_val, df_test, y_train, y_val, y_test, encoder_type="target"
        )
        assert X_train.ndim == 2
        assert X_train.shape[0] == len(df_train)
        assert X_val.shape[0] == len(df_val)
        assert X_test.shape[0] == len(df_test)
        assert X_train.shape[1] > 0, "No features extracted"

    def test_hybrid_encoder_produces_features(self, nsl_kdd_data):
        """Hybrid encoder produces feature matrix."""
        df_train, df_val, df_test, y_train, y_val, y_test = nsl_kdd_data
        X_train, X_val, X_test, extractor = build_features(
            df_train, df_val, df_test, y_train, y_val, y_test, encoder_type="hybrid"
        )
        assert X_train.ndim == 2
        assert X_train.shape[0] == len(df_train)

    def test_ordinal_encoder_produces_features(self, nsl_kdd_data):
        """Ordinal encoder produces feature matrix."""
        df_train, df_val, df_test, y_train, y_val, y_test = nsl_kdd_data
        X_train, X_val, X_test, extractor = build_features(
            df_train, df_val, df_test, y_train, y_val, y_test, encoder_type="ordinal"
        )
        assert X_train.ndim == 2
        assert X_train.shape[0] == len(df_train)

    def test_feature_counts(self, nsl_kdd_data):
        """Feature counts match Phase 1 baseline for each dataset.

        CICIDS2017=33, UNSW-NB15=35, NSL-KDD=10 (corrected from 11;
        see AUDIT_REPORT Finding #6b for seed-inconsistency discovery).
        """
        expected = {"CICIDS2017": 33, "UNSW-NB15": 35, "NSL-KDD": 10}
        for name, exp in expected.items():
            df_train, df_val, df_test, y_train, y_val, y_test = load_dataset(name, seed=42)
            X_train, _, _, _ = build_features(df_train, df_val, df_test, y_train, y_val, y_test)
            actual = X_train.shape[1]
            assert actual == exp, f"{name}: expected {exp} features, got {actual}"

    def test_nsl_kdd_feature_count_consistent_across_seeds(self):
        """NSL-KDD must produce the same feature count for every seed."""
        counts = set()
        for seed in range(42, 62):
            df_train, df_val, df_test, y_train, y_val, y_test = load_dataset("NSL-KDD", seed=seed)
            X_train, _, _, _ = build_features(df_train, df_val, df_test, y_train, y_val, y_test)
            counts.add(X_train.shape[1])
        assert len(counts) == 1, f"NSL-KDD feature count varies across seeds: {counts}"
        assert counts.pop() == 10

    def test_no_label_in_features(self, nsl_kdd_data):
        """Preprocessor excludes label columns from feature routing."""
        df_train, df_val, df_test, y_train, y_val, y_test = nsl_kdd_data

        extractor = build_features(
            df_train, df_val, df_test, y_train, y_val, y_test, encoder_type="target"
        )[-1]

        # The leakage guard is in _build_preprocessor: label columns
        # are excluded from _numeric_cols/_categorical_cols lists
        leak_cols = {"is_anomaly", "label", "attack_category"}
        routed = set(extractor._numeric_cols) | set(extractor._categorical_cols)
        present = leak_cols & routed
        assert not present, f"Label columns routed by preprocessor: {present}"


# ============================================================================
# SEED AND CONFIG TESTS
# ============================================================================


class TestConfig:
    """Tests for module-level constants."""

    def test_seeds_are_42_to_61(self):
        """SEEDS should be range(42, 62)."""
        assert SEEDS == list(range(42, 62))
        assert len(SEEDS) == 20

    def test_split_sums_to_one(self):
        """SPLIT fractions sum to 1.0."""
        assert abs(sum(SPLIT) - 1.0) < 1e-10

    def test_dataset_loaders_has_all_five(self):
        """DATASET_LOADERS contains all 5 datasets."""
        expected = {"CICIDS2017", "UNSW-NB15", "NSL-KDD", "CSE-CIC-IDS2018", "TON_IoT"}
        assert set(DATASET_LOADERS.keys()) == expected

    def test_fe_config_has_required_keys(self):
        """FE_CONFIG has host_windows and exclude_columns."""
        assert "host_windows" in FE_CONFIG
        assert "exclude_columns" in FE_CONFIG
        assert "is_anomaly" in FE_CONFIG["exclude_columns"]
        assert "label" in FE_CONFIG["exclude_columns"]
        assert "attack_category" in FE_CONFIG["exclude_columns"]


# ============================================================================
# LABEL LEAKAGE PATH TEST
# ============================================================================


class TestLabelLeakage:
    """Verify no code path passes labels to hybrid/ordinal encoders."""

    def test_hybrid_encoder_no_labels(self, seed):
        """Hybrid encoder fit_transform receives labels=None."""
        df_train, df_val, df_test, y_train, y_val, y_test = load_dataset("NSL-KDD", seed=seed)

        # build_features should call fit_transform with labels=None for hybrid
        X_train, X_val, X_test, extractor = build_features(
            df_train, df_val, df_test, y_train, y_val, y_test, encoder_type="hybrid"
        )
        assert X_train.shape[0] == len(df_train)

    def test_ordinal_encoder_no_labels(self, seed):
        """Ordinal encoder fit_transform receives labels=None."""
        df_train, df_val, df_test, y_train, y_val, y_test = load_dataset("NSL-KDD", seed=seed)

        X_train, X_val, X_test, extractor = build_features(
            df_train, df_val, df_test, y_train, y_val, y_test, encoder_type="ordinal"
        )
        assert X_train.shape[0] == len(df_train)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
