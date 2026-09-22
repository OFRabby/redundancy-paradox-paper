"""
Unified Detector Factory for Phase 3
======================================
Wraps 4 Phase 1 detectors + 2 PyOD detectors into a common interface.

Phase 1 detectors (IF, LOF, OCSVM, AE) are reused EXACTLY for H5 comparability.
HBOS and ECOD are from PyOD with local fallback implementations.

Usage:
    from detector_factory import build_all_detectors, fit_all_detectors, score_all_detectors

    detectors = build_all_detectors(seed=42)
    fit_all_detectors(X_train, X_val, seed=42)  # AE uses X_val; others ignore it
    scores = score_all_detectors(detectors, X_test)
"""

import logging
import numpy as np
import os
import sys
from typing import Protocol

logger = logging.getLogger(__name__)

# ============================================================================
# UNIFIED DETECTOR PROTOCOL
# ============================================================================

class DetectorProtocol(Protocol):
    """Unified interface for all detectors.

    Convention: higher score = more anomalous.
    fit() accepts optional X_val for AE's early stopping; other detectors ignore it.
    """
    def fit(self, X_train: np.ndarray, X_val: np.ndarray | None = None) -> "DetectorProtocol": ...
    def decision_function(self, X: np.ndarray) -> np.ndarray: ...
    @property
    def name(self) -> str: ...

# ============================================================================
# PHASE 1 PATH RESOLUTION (FIX 2: Kaggle-compatible)
# ============================================================================

def _find_phase1_root() -> str:
    """Find Phase 1 project root (parent of src/). Works on both local and Kaggle."""
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        # Kaggle: code copied into /kaggle/working/code/
        os.path.join(here, "anomaly_detector-main"),
        # Kaggle alternate: dataset mount
        os.path.join("/kaggle/input/redundancy-paradox-code-v2", "anomaly_detector-main"),
        os.path.join("/kaggle/input/datasets/ofrabby/redundancy-paradox-code-v2", "anomaly_detector-main"),
        # Local: research/research/kaggle_output/real_data_validation -> research/anomaly_detector-main/...
        os.path.join(here, "..", "..", "..", "anomaly_detector-main", "anomaly_detector-main"),
        # Local alternate
        os.path.join(here, "..", "..", "..", "..", "anomaly_detector-main", "anomaly_detector-main"),
    ]
    for c in candidates:
        if os.path.isdir(os.path.join(c, "src", "models")):
            return os.path.abspath(c)
    raise FileNotFoundError(
        f"Cannot find Phase 1 project root (src/models/). Tried: {candidates}"
    )

_PHASE1_ROOT = _find_phase1_root()
sys.path.insert(0, _PHASE1_ROOT)

# ============================================================================
# PHASE 1 IMPORTS (DO NOT MODIFY)
# ============================================================================

from src.models.isolation_forest import IsolationForestDetector
from src.models.lof import LOFDetector
from src.models.ocsvm import OCSVMDetector

# AE is optional — torch may be unavailable locally (e.g. Windows SAC blocks torch._C)
try:
    from src.models.autoencoder import Autoencoder, train_autoencoder
    TORCH_AVAILABLE = True
except (ImportError, OSError) as e:
    Autoencoder = None
    train_autoencoder = None
    TORCH_AVAILABLE = False
    logger.warning(f"PyTorch not available — AEWrapper will be disabled: {e}")

# ============================================================================
# PyOD IMPORT (OPTIONAL)
# ============================================================================

try:
    from pyod.models.hbos import HBOS as _PyOD_HBOS
    from pyod.models.ecod import ECOD as _PyOD_ECOD
    PYOD_AVAILABLE = True
except ImportError:
    _PyOD_HBOS = None
    _PyOD_ECOD = None
    PYOD_AVAILABLE = False

# ============================================================================
# WRAPPER CLASSES — PHASE 1 DETECTORS
# ============================================================================

class IFWrapper:
    """Isolation Forest — delegates to Phase 1 IsolationForestDetector."""

    def __init__(self, seed: int = 42, **kwargs):
        self.seed = seed
        self.kwargs = kwargs
        self._model: IsolationForestDetector | None = None

    @property
    def name(self) -> str:
        return "IF"

    def fit(self, X_train: np.ndarray, X_val: np.ndarray | None = None) -> "IFWrapper":
        params = {
            "n_estimators": 200,
            "max_samples": 256,
            "contamination": 0.01,
            "max_features": 1.0,
            "bootstrap": False,
            "n_jobs": -1,
            "random_state": self.seed,
        }
        params.update(self.kwargs)
        self._model = IsolationForestDetector(**params)
        self._model.fit(X_train)
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        return self._model.score_samples(X)


class LOFWrapper:
    """Local Outlier Factor — delegates to Phase 1 LOFDetector."""

    def __init__(self, seed: int = 42, **kwargs):
        self.seed = seed
        self.kwargs = kwargs
        self._model: LOFDetector | None = None

    @property
    def name(self) -> str:
        return "LOF"

    def fit(self, X_train: np.ndarray, X_val: np.ndarray | None = None) -> "LOFWrapper":
        params = {
            "n_neighbors": 20,
            "novelty": True,
            "n_jobs": -1,
        }
        params.update(self.kwargs)
        self._model = LOFDetector(**params)
        self._model.fit(X_train)
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        return self._model.score_samples(X)


class OCSVMWrapper:
    """One-Class SVM — delegates to Phase 1 OCSVMDetector."""

    def __init__(self, seed: int = 42, **kwargs):
        self.seed = seed
        self.kwargs = kwargs
        self._model: OCSVMDetector | None = None

    @property
    def name(self) -> str:
        return "OCSVM"

    def fit(self, X_train: np.ndarray, X_val: np.ndarray | None = None) -> "OCSVMWrapper":
        params = {
            "kernel": "rbf",
            "nu": 0.01,
        }
        params.update(self.kwargs)
        self._model = OCSVMDetector(**params)
        self._model.fit(X_train)
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        return self._model.score_samples(X)


class AEWrapper:
    """Autoencoder — delegates to Phase 1 train_autoencoder.

    CRITICAL: X_val is REQUIRED for early stopping. If X_val is None,
    a ValueError is raised. This preserves Phase 1's exact training path.
    """

    def __init__(self, seed: int = 42, **kwargs):
        self.seed = seed
        self.kwargs = kwargs
        self._model: Autoencoder | None = None

    @property
    def name(self) -> str:
        return "AE"

    def fit(self, X_train: np.ndarray, X_val: np.ndarray | None = None) -> "AEWrapper":
        if not TORCH_AVAILABLE:
            raise RuntimeError(
                "AEWrapper requires PyTorch, which is not available in this environment. "
                "AE testing must run on Kaggle or an environment with torch installed."
            )
        if X_val is None:
            raise ValueError(
                "AEWrapper requires X_val for early stopping. "
                "Pass X_val to fit_all_detectors()."
            )
        params = {
            "hidden_dims": [128, 64, 32, 16],
            "epochs": 30,
            "batch_size": 1024,
            "learning_rate": 1e-3,
            "dropout": 0.1,
            "patience": 5,
            "device": "cpu",  # Match Phase 1 for H5 comparability
        }
        params.update(self.kwargs)
        # train_autoencoder returns (model, train_losses, val_losses)
        result = train_autoencoder(X_train, X_val, **params)
        self._model = result[0]  # Extract Autoencoder model
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        return self._model.score_samples(X)


# ============================================================================
# WRAPPER CLASSES — PyOD DETECTORS (FIX 1: correct decision_function)
# ============================================================================

class HBOSWrapper:
    """Histogram-Based Outlier Score — PyOD implementation."""

    def __init__(self, seed: int = 42, **kwargs):
        self.seed = seed
        self.kwargs = kwargs
        self._model = None

    @property
    def name(self) -> str:
        return "HBOS"

    def fit(self, X_train: np.ndarray, X_val: np.ndarray | None = None) -> "HBOSWrapper":
        params = {"n_bins": 20, "contamination": 0.01}
        params.update(self.kwargs)
        self._model = _PyOD_HBOS(**params)
        self._model.fit(X_train)
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        return self._model.decision_function(X)


class ECODWrapper:
    """ECOD (Empirical CDF) — PyOD implementation."""

    def __init__(self, seed: int = 42, **kwargs):
        self.seed = seed
        self.kwargs = kwargs
        self._model = None

    @property
    def name(self) -> str:
        return "ECOD"

    def fit(self, X_train: np.ndarray, X_val: np.ndarray | None = None) -> "ECODWrapper":
        params = {"contamination": 0.01}
        params.update(self.kwargs)
        self._model = _PyOD_ECOD(**params)
        self._model.fit(X_train)
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        return self._model.decision_function(X)


# ============================================================================
# FALLBACK CLASSES (when PyOD unavailable)
# ============================================================================

class HBOSFallback:
    """Fallback HBOS via per-feature histogram log-density."""

    def __init__(self, seed: int = 42, n_bins: int = 20):
        self.seed = seed
        self.n_bins = n_bins
        self._bins = None
        self._log_density = None

    @property
    def name(self) -> str:
        return "HBOS"

    def fit(self, X_train: np.ndarray, X_val: np.ndarray | None = None) -> "HBOSFallback":
        n_features = X_train.shape[1]
        self._bins = []
        self._log_density = []
        for j in range(n_features):
            counts, bin_edges = np.histogram(X_train[:, j], bins=self.n_bins, density=True)
            counts = np.maximum(counts, 1e-10)  # avoid log(0)
            self._bins.append(bin_edges)
            self._log_density.append(np.log(counts))
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        n_samples = X.shape[0]
        scores = np.zeros(n_samples)
        for j in range(X.shape[1]):
            bin_idx = np.searchsorted(self._bins[j][1:-1], X[:, j])
            bin_idx = np.clip(bin_idx, 0, len(self._log_density[j]) - 1)
            scores += self._log_density[j][bin_idx]
        return -scores  # negate: higher = more anomalous


class ECODFallback:
    """Fallback ECOD via per-feature empirical CDF distance from median.

    LIMITATION: This simplified implementation uses sum(|ECDF_i - 0.5|) which
    degrades on high-anomaly-rate datasets (e.g., NSL-KDD with 40% anomalies).
    For production use, PyOD's ECOD (which handles left/right tails separately)
    is preferred. On NSL-KDD:
    - ECOD fallback: AUROC 0.234 (sub-random)
    - ECOD PyOD (expected): > 0.5 (verify on Kaggle)
    """

    def __init__(self, seed: int = 42):
        self.seed = seed
        self._ecdf_x = None
        self._ecdf_y = None

    @property
    def name(self) -> str:
        return "ECOD"

    def fit(self, X_train: np.ndarray, X_val: np.ndarray | None = None) -> "ECODFallback":
        n_features = X_train.shape[1]
        self._ecdf_x = []
        self._ecdf_y = []
        for j in range(n_features):
            sorted_vals = np.sort(X_train[:, j])
            n = len(sorted_vals)
            self._ecdf_x.append(sorted_vals)
            self._ecdf_y.append(np.arange(1, n + 1) / n)
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        n_samples = X.shape[0]
        scores = np.zeros(n_samples)
        for j in range(X.shape[1]):
            ecdf_vals = np.searchsorted(self._ecdf_x[j], X[:, j], side='right') / len(self._ecdf_x[j])
            ecdf_vals = np.clip(ecdf_vals, 1e-10, 1 - 1e-10)
            # Score: distance from 0.5 (uniform); higher = more extreme
            scores += np.abs(ecdf_vals - 0.5)
        return scores  # higher = more anomalous


# ============================================================================
# REGISTRY
# ============================================================================

DETECTOR_REGISTRY: dict[str, type] = {
    "IF":    IFWrapper,
    "LOF":   LOFWrapper,
    "OCSVM": OCSVMWrapper,
    "HBOS":  HBOSWrapper if PYOD_AVAILABLE else HBOSFallback,
    "ECOD":  ECODWrapper if PYOD_AVAILABLE else ECODFallback,
}
if TORCH_AVAILABLE:
    DETECTOR_REGISTRY["AE"] = AEWrapper

DETECTOR_NAMES = list(DETECTOR_REGISTRY.keys())

# ============================================================================
# FACTORY FUNCTIONS (FIX 4: no TypeError fallback)
# ============================================================================

def build_detector(name: str, seed: int = 42, **kwargs) -> DetectorProtocol:
    """Build a single detector by name."""
    if name not in DETECTOR_REGISTRY:
        raise ValueError(f"Unknown detector: {name}. Available: {DETECTOR_NAMES}")
    return DETECTOR_REGISTRY[name](seed=seed, **kwargs)


def build_all_detectors(seed: int = 42) -> dict[str, DetectorProtocol]:
    """Build all 6 detectors."""
    return {name: build_detector(name, seed=seed) for name in DETECTOR_REGISTRY}


def fit_all_detectors(
    X_train: np.ndarray,
    X_val: np.ndarray | None = None,
    seed: int = 42,
) -> dict[str, DetectorProtocol]:
    """Fit all 6 detectors. X_val is required for AE; ignored by others.

    No try/except — AEWrapper raises ValueError if X_val is None.
    All other wrappers accept X_val=None and ignore it.
    """
    detectors = build_all_detectors(seed=seed)
    for name, det in detectors.items():
        det.fit(X_train, X_val=X_val)
    return detectors


def score_all_detectors(
    detectors: dict[str, DetectorProtocol],
    X: np.ndarray,
) -> dict[str, np.ndarray]:
    """Score all detectors on X. Returns dict of score arrays."""
    return {name: det.decision_function(X) for name, det in detectors.items()}


# ============================================================================
# VERIFICATION HELPERS
# ============================================================================

def verify_pyod_available() -> bool:
    """Check if PyOD is available and functional."""
    if not PYOD_AVAILABLE:
        logger.warning("PyOD not available — using fallback HBOS/ECOD")
        return False
    # Functional test
    X = np.random.RandomState(42).randn(100, 5)
    for name, cls in [("HBOS", _PyOD_HBOS), ("ECOD", _PyOD_ECOD)]:
        try:
            m = cls()
            m.fit(X)
            scores = m.decision_function(X)
            assert len(scores) == 100, f"{name}: expected 100 scores, got {len(scores)}"
        except Exception as e:
            logger.warning(f"PyOD {name} test failed: {e}")
            return False
    logger.info("PyOD verified OK")
    return True


def assert_all_scores_finite(scores: dict[str, np.ndarray]) -> None:
    """Assert all score arrays contain only finite values."""
    for name, s in scores.items():
        if not np.all(np.isfinite(s)):
            bad = ~np.isfinite(s)
            raise ValueError(f"{name}: {bad.sum()} non-finite scores")


def get_detector_info() -> dict[str, str]:
    """Return detector name -> source (Phase1 or PyOD/fallback) mapping."""
    info = {"torch_available": TORCH_AVAILABLE, "pyod_available": PYOD_AVAILABLE}
    for name in DETECTOR_REGISTRY:
        cls = DETECTOR_REGISTRY[name]
        if "Fallback" in cls.__name__:
            info[name] = "fallback"
        elif name in ("HBOS", "ECOD") and PYOD_AVAILABLE:
            info[name] = "pyod"
        else:
            info[name] = "phase1"
    return info
