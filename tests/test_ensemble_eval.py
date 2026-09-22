"""Unit tests for ensemble_eval.py — 17 tests."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from ensemble_eval import (
    rank_average_scores, compute_loo_contribution, compute_test_effect,
    compute_sign_conflict, compute_material_conflict,
    epsilon_vrg_selection, naive_loo_selection, evaluate_ensemble,
    ALPHA,
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def synthetic_scores():
    rng = np.random.RandomState(42)
    n = 2000
    y = (rng.random(n) < 0.1).astype(int)
    s_helpful = y * 2.0 + rng.randn(n) * 0.5
    s_harmful = -y * 2.0 + rng.randn(n) * 0.5
    s_noise = rng.randn(n)
    scores = {"helpful": s_helpful, "harmful": s_harmful, "noise": s_noise}
    return scores, y


@pytest.fixture
def minimal_scores():
    """Minimal 3-detector setup for deterministic tests."""
    rng = np.random.RandomState(99)
    n = 500
    y = np.zeros(n, dtype=int)
    y[:50] = 1
    s1 = y.astype(float) + rng.randn(n) * 0.1
    s2 = -y.astype(float) + rng.randn(n) * 0.1
    s3 = rng.randn(n)
    return {"A": s1, "B": s2, "C": s3}, y


# ============================================================================
# TESTS 1–3: RANK AVERAGING
# ============================================================================

class TestRankAverage:
    def test_rank_average_shape(self, synthetic_scores):
        scores, _ = synthetic_scores
        result = rank_average_scores(list(scores.values()))
        assert result.shape == (2000,)

    def test_rank_average_range(self, synthetic_scores):
        scores, _ = synthetic_scores
        result = rank_average_scores(list(scores.values()))
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_rank_average_equal_ranks(self):
        """Identical scores → ranks are all equal → combined = 0.505 (scipy average rank)."""
        s = np.ones(100)
        result = rank_average_scores([s, s, s])
        # scipy.stats.rankdata with ties assigns average rank.
        # For 100 identical scores, all get rank 50.5 → normalized 0.505.
        assert np.allclose(result, result[0]), "Not all elements equal"
        assert np.isclose(result[0], 0.505, atol=1e-10), f"Expected 0.505, got {result[0]}"


# ============================================================================
# TESTS 4–6: LOO CONTRIBUTION
# ============================================================================

class TestLOOContribution:
    def test_loo_contribution_keys(self, synthetic_scores):
        scores, y = synthetic_scores
        c_i, auroc_full = compute_loo_contribution(scores, y)
        assert set(c_i.keys()) == set(scores.keys())
        assert isinstance(auroc_full, float)

    def test_loo_contribution_helpful_positive(self, synthetic_scores):
        scores, y = synthetic_scores
        c_i, _ = compute_loo_contribution(scores, y)
        assert c_i["helpful"] > 0, "Helpful detector should have C_i > 0"

    def test_loo_contribution_harmful_negative(self, synthetic_scores):
        scores, y = synthetic_scores
        c_i, _ = compute_loo_contribution(scores, y)
        assert c_i["harmful"] < 0, "Harmful detector should have C_i < 0"


# ============================================================================
# TESTS 7–8: TEST EFFECT
# ============================================================================

class TestTestEffect:
    def test_test_effect_keys(self, synthetic_scores):
        scores, y = synthetic_scores
        te, auroc_full = compute_test_effect(scores, y)
        assert set(te.keys()) == set(scores.keys())
        assert isinstance(auroc_full, float)

    def test_test_effect_sign(self, synthetic_scores):
        scores, y = synthetic_scores
        te, _ = compute_test_effect(scores, y)
        assert te["harmful"] > 0, "Harmful detector → TestEffect > 0"
        assert te["helpful"] < 0, "Helpful detector → TestEffect < 0"


# ============================================================================
# TESTS 9–11: CONFLICT DETECTION
# ============================================================================

class TestConflictDetection:
    def test_sign_conflict_opposite_semantics(self):
        c_i = {"A": 0.05, "B": -0.03}
        te = {"A": 0.04, "B": 0.02}
        sc = compute_sign_conflict(c_i, te)
        assert sc["A"] is True, "Same sign → conflict"
        assert sc["B"] is False, "Opposite sign → no conflict"

    def test_material_conflict_large_same_sign(self):
        c_i = {"A": 0.05}
        te = {"A": 0.05}
        mc = compute_material_conflict(c_i, te, threshold=0.01)
        assert mc["A"] is True, "Same sign + large magnitude → material conflict"

    def test_material_conflict_small_same_sign(self):
        c_i = {"A": 0.001}
        te = {"A": 0.001}
        mc = compute_material_conflict(c_i, te, threshold=0.01)
        assert mc["A"] is False, "Same sign but below threshold → not material"


# ============================================================================
# TESTS 12–15: SELECTION LOGIC
# ============================================================================

class TestSelectionLogic:
    def test_epsilon_vrg_retains_negligible(self):
        c_i = {"A": -0.001, "B": 0.01}
        epsilon = 0.008
        selected, eps = epsilon_vrg_selection(c_i, auroc_full_val=0.8, alpha=epsilon / 0.8)
        assert "A" in selected, "C_i = -0.001 > -epsilon should be retained"

    def test_epsilon_vrg_excludes_material(self):
        c_i = {"A": -0.05, "B": 0.01}
        epsilon = 0.008
        selected, eps = epsilon_vrg_selection(c_i, auroc_full_val=0.8, alpha=epsilon / 0.8)
        assert "A" not in selected, "C_i = -0.05 < -epsilon should be excluded"

    def test_naive_loo_only_positive(self):
        c_i = {"A": -0.01, "B": 0.01, "C": -0.001}
        selected = naive_loo_selection(c_i)
        assert selected == ["B"], "Only positive C_i should be selected"

    def test_naive_loo_fallback(self):
        c_i = {"A": -0.05, "B": -0.03}
        selected = naive_loo_selection(c_i)
        assert selected == ["B"], "Fallback: max C_i = -0.03"


# ============================================================================
# TESTS 16–17: ORCHESTRATOR
# ============================================================================

class TestEvaluateEnsemble:
    def test_evaluate_ensemble_keys(self, synthetic_scores):
        scores, y = synthetic_scores
        result = evaluate_ensemble(scores, scores, y, y)
        expected_keys = {
            "auroc_ew_val", "auroc_ew_test",
            "auroc_nl_val", "auroc_nl_test", "nl_selected",
            "auroc_ev_val", "auroc_ev_test", "ev_selected", "epsilon",
            "c_i", "test_effect", "sign_conflict", "material_conflict",
            "standalone_auroc_val",
        }
        assert set(result.keys()) == expected_keys

    def test_evaluate_ew_matches_manual(self, synthetic_scores):
        scores, y = synthetic_scores
        result = evaluate_ensemble(scores, scores, y, y)
        manual = rank_average_scores([scores[n] for n in scores])
        manual_auroc = float(roc_auc_score(y, manual))
        np.testing.assert_allclose(result["auroc_ew_val"], manual_auroc, atol=1e-10)
