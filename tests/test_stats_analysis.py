"""Unit tests for stats_analysis.py — 8 tests."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import pytest

from stats_analysis import (
    bootstrap_ci,
    apply_holm_correction,
    spearman_correlation,
    paired_comparison,
)


# ============================================================================
# FIXTURES
# ============================================================================

def _make_df(dataset, method_a_scores, method_b_scores, seed_start=42):
    """Build a minimal DataFrame for paired_comparison tests."""
    n = len(method_a_scores)
    rows = []
    for i in range(n):
        rows.append({
            "dataset": dataset,
            "seed": seed_start + i,
            "encoder": "target",
            "method": "EW",
            "auroc_test": method_a_scores[i],
        })
        rows.append({
            "dataset": dataset,
            "seed": seed_start + i,
            "encoder": "target",
            "method": "NL",
            "auroc_test": method_b_scores[i],
        })
    return pd.DataFrame(rows)


# ============================================================================
# TESTS 1–2: BOOTSTRAP CI
# ============================================================================

class TestBootstrapCI:
    def test_bootstrap_ci_shape(self):
        """Returns (float, float) with low <= high."""
        rng = np.random.default_rng(0)
        diffs = rng.normal(0, 1, size=100)
        lo, hi = bootstrap_ci(diffs)
        assert isinstance(lo, float)
        assert isinstance(hi, float)
        assert lo <= hi

    def test_bootstrap_ci_degenerate(self):
        """n=1 returns (diff, diff)."""
        diffs = np.array([0.05])
        lo, hi = bootstrap_ci(diffs)
        assert lo == 0.05
        assert hi == 0.05


# ============================================================================
# TESTS 3–4: HOLM CORRECTION
# ============================================================================

class TestHolmCorrection:
    def test_apply_holm_single_p(self):
        """Single p-value unchanged by Holm."""
        adjusted, rejected = apply_holm_correction([0.03])
        assert len(adjusted) == 1
        assert adjusted[0] == pytest.approx(0.03)

    def test_apply_holm_ordering(self):
        """p_holm >= p_raw for all."""
        raw = [0.001, 0.01, 0.05, 0.1, 0.5]
        adjusted, _ = apply_holm_correction(raw)
        for r, a in zip(raw, adjusted):
            assert a >= r, f"Adjusted {a} < raw {r}"


# ============================================================================
# TESTS 5–6: SPEARMAN CORRELATION
# ============================================================================

class TestSpearmanCorrelation:
    def test_spearman_perfect_correlation(self):
        """Monotone x, y -> rho ~ 1.0."""
        x = np.arange(50, dtype=float)
        y = x * 2.0
        result = spearman_correlation(x, y)
        assert result["rho"] == pytest.approx(1.0, abs=1e-10)
        assert result["n"] == 50

    def test_spearman_no_correlation(self):
        """Random x, y -> |rho| < 0.2."""
        rng = np.random.default_rng(99)
        x = rng.normal(0, 1, size=500)
        y = rng.normal(0, 1, size=500)
        result = spearman_correlation(x, y)
        assert abs(result["rho"]) < 0.2


# ============================================================================
# TESTS 7–8: PAIRED COMPARISON
# ============================================================================

class TestPairedComparison:
    def test_paired_comparison_synthetic(self):
        """EW higher than NL by +0.05 -> mean_diff ~ +0.05, p_raw small."""
        rng = np.random.default_rng(42)
        n = 20
        ew = 0.80 + rng.normal(0, 0.01, n)
        nl = ew - 0.05 + rng.normal(0, 0.01, n)
        df = _make_df("CICIDS2017", ew, nl)
        result = paired_comparison(df, "CICIDS2017", "EW", "NL")
        assert result["n_pairs"] == n
        assert result["mean_diff"] == pytest.approx(0.05, abs=0.01)
        assert result["cohens_d"] > 1.0
        assert result["p_raw"] < 0.05

    def test_paired_comparison_alignment(self):
        """Shuffled seed order in one method -> still correct mean_diff."""
        rng = np.random.default_rng(42)
        n = 20
        ew = 0.80 + rng.normal(0, 0.01, n)
        nl = ew - 0.05 + rng.normal(0, 0.01, n)

        # Build EW rows normally, NL rows with shuffled seeds
        ew_rows = []
        nl_rows = []
        seeds = list(range(42, 42 + n))
        shuffled_seeds = seeds.copy()
        rng.shuffle(shuffled_seeds)
        for i in range(n):
            ew_rows.append({
                "dataset": "CICIDS2017", "seed": seeds[i],
                "encoder": "target", "method": "EW", "auroc_test": ew[i],
            })
            nl_rows.append({
                "dataset": "CICIDS2017", "seed": shuffled_seeds[i],
                "encoder": "target", "method": "NL", "auroc_test": nl[i],
            })
        df = pd.DataFrame(ew_rows + nl_rows)

        result = paired_comparison(df, "CICIDS2017", "EW", "NL")
        assert result["n_pairs"] == n
        assert result["mean_diff"] == pytest.approx(0.05, abs=0.01)
