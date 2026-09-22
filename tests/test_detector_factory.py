"""Unit tests for detector_factory.py — 12 tests total."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from detector_factory import (
    DETECTOR_REGISTRY, DETECTOR_NAMES, PYOD_AVAILABLE, TORCH_AVAILABLE,
    build_all_detectors, fit_all_detectors, score_all_detectors,
    get_detector_info,
    HBOSFallback, ECODFallback,
)

if TORCH_AVAILABLE:
    from detector_factory import AEWrapper


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def random_data():
    rng = np.random.RandomState(42)
    X_train = rng.randn(1000, 10).astype(np.float32)
    X_val = rng.randn(200, 10).astype(np.float32)
    X_test = rng.randn(500, 10).astype(np.float32)
    return X_train, X_val, X_test


# ============================================================================
# TESTS 1–10: LOCAL-Runnable (no torch gate)
# ============================================================================

class TestRegistryHasAllDetectors:
    def test_registry_has_all_detectors(self):
        if TORCH_AVAILABLE:
            assert len(DETECTOR_REGISTRY) == 6
            expected = {"IF", "LOF", "OCSVM", "AE", "HBOS", "ECOD"}
        else:
            assert len(DETECTOR_REGISTRY) == 5
            expected = {"IF", "LOF", "OCSVM", "HBOS", "ECOD"}
        assert set(DETECTOR_REGISTRY.keys()) == expected


class TestBuildAllDetectors:
    def test_build_all_detectors(self):
        detectors = build_all_detectors(seed=42)
        assert set(detectors.keys()) == set(DETECTOR_NAMES)


class TestFitSklearnDetectors:
    def test_fit_sklearn_detectors(self, random_data):
        X_train, _, _ = random_data
        X_small = X_train[:100]
        for name in ["IF", "LOF", "OCSVM"]:
            det = build_all_detectors(seed=42)[name]
            result = det.fit(X_small)
            assert result is det, f"{name}.fit() must return self"


class TestScoreShapes:
    def test_score_shapes(self, random_data):
        X_train, X_val, X_test = random_data
        X_small_train = X_train[:100]
        detectors = build_all_detectors(seed=42)
        for name in DETECTOR_NAMES:
            if name == "AE":
                detectors[name].fit(X_train, X_val=X_val)
            else:
                detectors[name].fit(X_small_train)
        scores = score_all_detectors(detectors, X_test)
        for name, s in scores.items():
            assert s.shape == (500,), f"{name}: expected shape (500,), got {s.shape}"


class TestScoresFinite:
    def test_scores_finite(self, random_data):
        X_train, X_val, X_test = random_data
        X_small_train = X_train[:100]
        detectors = build_all_detectors(seed=42)
        for name in DETECTOR_NAMES:
            if name == "AE":
                detectors[name].fit(X_train, X_val=X_val)
            else:
                detectors[name].fit(X_small_train)
        scores = score_all_detectors(detectors, X_test)
        for name, s in scores.items():
            assert np.isfinite(s).all(), f"{name}: non-finite scores"


class TestDeterministicPerSeed:
    def test_deterministic_per_seed(self, random_data):
        X_train, X_val, X_test = random_data
        X_small = X_train[:100]

        # Same seed → same scores
        d1 = build_all_detectors(seed=42)["IF"]
        d2 = build_all_detectors(seed=42)["IF"]
        d1.fit(X_small)
        d2.fit(X_small)
        s1 = d1.decision_function(X_test)
        s2 = d2.decision_function(X_test)
        np.testing.assert_array_equal(s1, s2, err_msg="Same seed produced different IF scores")

        # Different seed → scores may differ (not guaranteed, but usually)
        d3 = build_all_detectors(seed=99)["IF"]
        d3.fit(X_small)
        s3 = d3.decision_function(X_test)
        # Just verify it runs without error
        assert s3.shape == s1.shape


class TestWrapperNameProperty:
    def test_wrapper_name_property(self):
        for name in DETECTOR_NAMES:
            det = DETECTOR_REGISTRY[name](seed=42)
            assert det.name == name, f"{name} wrapper .name returned '{det.name}', expected '{name}'"


class TestPyodAvailabilityFlag:
    def test_pyod_availability_flag(self):
        assert isinstance(PYOD_AVAILABLE, bool)
        info = get_detector_info()
        assert info["pyod_available"] == PYOD_AVAILABLE


class TestHBOSFallbackScoreShape:
    def test_hbos_fallback_score_shape(self, random_data):
        X_train, _, X_test = random_data
        det = HBOSFallback(seed=42)
        det.fit(X_train)
        scores = det.decision_function(X_test)
        assert scores.shape == (500,)
        assert np.isfinite(scores).all()


class TestECODFallbackScoreShape:
    def test_ecod_fallback_score_shape(self, random_data):
        X_train, _, X_test = random_data
        det = ECODFallback(seed=42)
        det.fit(X_train)
        scores = det.decision_function(X_test)
        assert scores.shape == (500,)
        assert np.isfinite(scores).all()


# ============================================================================
# TESTS 11–12: TORCH-Gated
# ============================================================================

@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch unavailable")
class TestAEWrapper:
    def test_ae_requires_x_val(self, random_data):
        X_train, _, _ = random_data
        X_small = X_train[:100]
        det = AEWrapper(seed=42)
        with pytest.raises(ValueError, match="requires X_val"):
            det.fit(X_small, X_val=None)

    def test_ae_with_x_val(self, random_data):
        X_train, X_val, X_test = random_data
        det = AEWrapper(seed=42)
        det.fit(X_train, X_val=X_val)
        scores = det.decision_function(X_test)
        assert scores.shape == (500,)
        assert np.isfinite(scores).all()
