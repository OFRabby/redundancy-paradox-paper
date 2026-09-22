"""Leakage-guard unit tests for FeatureExtractor.

Run:  pytest tests/test_leakage_guard.py -v
"""

import sys
from pathlib import Path

# Add project root to sys.path so src.features.extractor is importable
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent.parent.parent / "anomaly_detector-main" / "anomaly_detector-main")
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import pandas as pd
import polars as pl
import pytest
from sklearn.base import BaseEstimator, TransformerMixin

from src.features.extractor import (
    FeatureExtractor,
    FrequencyEncoder,
    HIGH_CARDINALITY_THRESHOLD,
    META_COLUMNS,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_toy_df(n: int = 100) -> pl.DataFrame:
    """Minimal connection-log DataFrame that passes _compute_* methods."""
    rng = np.random.RandomState(0)
    ts_base = 1_700_000_000
    return pl.DataFrame({
        "ts": [ts_base + i for i in range(n)],
        "id.orig_h": [f"10.0.0.{i % 5}" for i in range(n)],
        "id.orig_p": rng.randint(1024, 65535, n).tolist(),
        "id.resp_h": [f"10.0.0.{(i + 1) % 5}" for i in range(n)],
        "id.resp_p": rng.choice([80, 443, 8080], n).tolist(),
        "proto": rng.choice(["tcp", "udp"], n).tolist(),
        "service": rng.choice(["http", "dns", "-"], n).tolist(),
        "conn_state": rng.choice(["SF", "S0", "REJ"], n).tolist(),
        "duration": rng.exponential(1.0, n).tolist(),
        "orig_bytes": rng.randint(0, 10000, n).tolist(),
        "resp_bytes": rng.randint(0, 10000, n).tolist(),
        "orig_pkts": rng.randint(1, 100, n).tolist(),
        "resp_pkts": rng.randint(1, 100, n).tolist(),
        "orig_ip_bytes": rng.randint(0, 10000, n).tolist(),
        "resp_ip_bytes": rng.randint(0, 10000, n).tolist(),
        "label": ["Benign"] * n,
        "attack_category": ["Normal"] * n,
        "is_anomaly": [0] * n,
    })


# ---------------------------------------------------------------------------
# Test 1: Structural leakage guard — label columns must not reach encoder
# ---------------------------------------------------------------------------

def test_no_label_columns_in_encoder_input():
    """is_anomaly, label, attack_category must be absent from X fed to encoder."""
    df = _make_toy_df()
    labels = df["is_anomaly"].to_numpy()

    for enc_type in ("target", "hybrid", "ordinal"):
        ext = FeatureExtractor(config={
            "host_windows": [],
            "exclude_columns": [
                "ts", "uid", "id.orig_h", "id.orig_p", "id.resp_h",
                "id.resp_p", "label", "attack_category", "is_anomaly",
            ],
        }, encoder_type=enc_type)
        X = ext.fit_transform(df, labels=labels)

        # None of the output feature names should be label-like
        leak_names = {"is_anomaly", "label", "attack_category"}
        out_names = set(ext.feature_names)
        overlap = leak_names & out_names
        assert not overlap, f"[{enc_type}] Label columns leaked into output: {overlap}"


# ---------------------------------------------------------------------------
# Test 2: FrequencyEncoder is sklearn-compatible
# ---------------------------------------------------------------------------

def test_frequency_encoder_sklearn_compat():
    """FrequencyEncoder must inherit BaseEstimator + TransformerMixin."""
    assert issubclass(FrequencyEncoder, BaseEstimator)
    assert issubclass(FrequencyEncoder, TransformerMixin)


def test_frequency_encoder_fit_uses_X_only():
    """FrequencyEncoder.fit must accept (X, y=None) and ignore y."""
    enc = FrequencyEncoder()
    X = pd.DataFrame({"a": ["x", "y", "x", "z"], "b": [1, 2, 1, 3]})
    enc.fit(X, y=np.array([0, 1, 0, 1]))  # y ignored
    assert hasattr(enc, "freq_maps_")
    assert "a" in enc.freq_maps_
    assert enc.freq_maps_["a"]["x"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Test 3: Hybrid encoder — OneHot for ≤20, Frequency for >20
# ---------------------------------------------------------------------------

def test_hybrid_encoder_policy():
    """Hybrid: ≤20 unique → OneHot; >20 unique → FrequencyEncoder."""
    ext = FeatureExtractor(config={
        "host_windows": [],
        "exclude_columns": [
            "ts", "uid", "id.orig_h", "id.orig_p", "id.resp_h",
            "id.resp_p", "label", "attack_category", "is_anomaly",
            "orig_ip_bytes", "resp_ip_bytes",
        ],
    }, encoder_type="hybrid")

    df = _make_toy_df()
    X = ext.fit_transform(df)

    # Check that the ColumnTransformer contains both 'onehot' and 'frequency' keys
    ct = ext.preprocessor
    transformer_names = [name for name, _, _ in ct.transformers]
    # If proto > 20 unique in training data, 'frequency' should appear
    # proto has 3 unique values, so 'onehot' should appear
    # service has 3 unique values, so should also be onehot
    # conn_state has 3 unique values
    assert "onehot" in transformer_names, f"Expected 'onehot' in {transformer_names}"


# ---------------------------------------------------------------------------
# Test 4: Encoder type propagated correctly
# ---------------------------------------------------------------------------

def test_encoder_type_propagated():
    """encoder_type must be stored on the instance."""
    for enc in ("target", "hybrid", "ordinal"):
        ext = FeatureExtractor(encoder_type=enc)
        assert ext.encoder_type == enc


# ---------------------------------------------------------------------------
# Test 5: Default encoder_type is "target" (backward compat)
# ---------------------------------------------------------------------------

def test_default_encoder_is_target():
    """FeatureExtractor() with no encoder_type arg should default to 'target'."""
    ext = FeatureExtractor()
    assert ext.encoder_type == "target"
