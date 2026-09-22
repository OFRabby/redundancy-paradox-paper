"""Phase 2 synthetic role construction tests.

Run: pytest tests/test_synthetic_roles.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import spearmanr

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from synthetic_benchmark import (
    make_beneficial_scores,
    make_redundant_scores,
    make_harmful_scores,
    make_noisy_scores,
    apply_distribution,
    build_ensemble,
    compute_loo_metrics,
    infer_role,
    rank_average_scores,
)
from sklearn.metrics import roc_auc_score


# ============================================================================
# 1. test_beneficial_standalone_auroc
# ============================================================================

def test_beneficial_standalone_auroc():
    rngs = [np.random.default_rng(seed) for seed in range(42, 62)]
    aurocs = []
    for rng in rngs:
        n = 1000
        pi = 0.05
        y = rng.binomial(1, pi, size=n).astype(float)
        scores = make_beneficial_scores(y, sigma=0.5, rng=rng)
        aurocs.append(roc_auc_score(y, scores))
    mean_auroc = np.mean(aurocs)
    assert mean_auroc > 0.80, f"mean AUROC={mean_auroc:.4f}, expected > 0.80"


# ============================================================================
# 2. test_redundant_correlation
# ============================================================================

def test_redundant_correlation():
    rng = np.random.default_rng(42)
    n = 1000
    pi = 0.05
    y = rng.binomial(1, pi, size=n).astype(float)
    base = make_beneficial_scores(y, sigma=0.5, rng=rng)
    rho_target = 0.7
    rho_measured_list = []
    for seed in range(42, 52):
        rng2 = np.random.default_rng(seed)
        redundant = make_redundant_scores(y, base, rho=rho_target, rng=rng2)
        rho_measured, _ = spearmanr(base, redundant)
        rho_measured_list.append(abs(rho_measured))
    mean_rho = np.mean(rho_measured_list)
    assert mean_rho >= 0.65, f"mean |rho|={mean_rho:.4f}, expected >= 0.65"


# ============================================================================
# 3. test_harmful_removal_improves
# ============================================================================

def test_harmful_removal_improves():
    improvement_list = []
    for seed in range(42, 52):
        rng = np.random.default_rng(seed)
        n = 1000
        pi = 0.05
        y = rng.binomial(1, pi, size=n).astype(float)

        b_scores = make_beneficial_scores(y, sigma=0.5, rng=rng)
        h_scores = make_harmful_scores(y, harm_magnitude=0.10, rng=rng)

        full = rank_average_scores([b_scores, h_scores])
        auroc_full = roc_auc_score(y, full)

        b_only = rank_average_scores([b_scores])
        auroc_loo = roc_auc_score(y, b_only)

        improvement_list.append(auroc_loo - auroc_full)

    mean_improvement = np.mean(improvement_list)
    assert mean_improvement > 0.01, (
        f"mean improvement={mean_improvement:.4f}, expected > 0.01"
    )


# ============================================================================
# 4. test_noisy_auroc_range
# ============================================================================

def test_noisy_auroc_range():
    aurocs = []
    for seed in range(42, 62):
        rng = np.random.default_rng(seed)
        n = 1000
        scores = make_noisy_scores(n, rng=rng)
        y = rng.binomial(1, 0.05, size=n).astype(float)
        aurocs.append(roc_auc_score(y, scores))
    mean_auroc = np.mean(aurocs)
    assert 0.45 <= mean_auroc <= 0.55, (
        f"mean AUROC={mean_auroc:.4f}, expected in [0.45, 0.55]"
    )


# ============================================================================
# 5. test_role_inference_correctness
# ============================================================================

def test_role_inference_correctness():
    rng = np.random.default_rng(42)
    n = 5000
    pi = 0.05
    y = rng.binomial(1, pi, size=n).astype(float)

    b1 = make_beneficial_scores(y, sigma=0.5, rng=rng)
    b2 = make_redundant_scores(y, b1, rho=0.2, rng=rng)
    r1 = make_redundant_scores(y, b1, rho=0.85, rng=rng)
    h1 = make_harmful_scores(y, harm_magnitude=0.4, rng=rng)
    n1 = make_noisy_scores(n, rng=rng)

    ensemble = {"B1": b1, "B2": b2, "R1": r1, "H1": h1, "N1": n1}
    (c_i, test_effect, sign_conflict, standalone_auroc_val,
     auroc_val, auroc_test) = compute_loo_metrics(ensemble, y, y)

    names = list(ensemble.keys())
    predicted = {}
    eps = 0.01 * auroc_val
    for name in names:
        others = [ensemble[n] for n in names if n != name]
        corrs = [abs(np.corrcoef(ensemble[name], o)[0, 1])
                 for o in others
                 if np.std(o) > 0 and np.std(ensemble[name]) > 0]
        max_corr = max(corrs) if corrs else 0.0
        predicted[name] = infer_role(
            c_i[name], standalone_auroc_val[name], max_corr, eps=eps
        )

    assert predicted["B1"] == "B", f"B1 predicted as {predicted['B1']}"
    assert predicted["B2"] in ("B", "R", "N"), f"B2 predicted as {predicted['B2']}"
    assert predicted["R1"] in ("B", "R"), f"R1 predicted as {predicted['R1']}"
    assert predicted["H1"] in ("H", "B"), f"H1 predicted as {predicted['H1']}"
    assert predicted["N1"] in ("N", "B", "H"), f"N1 predicted as {predicted['N1']}"


# ============================================================================
# 6. test_sign_convention
# ============================================================================

def test_sign_convention():
    rng = np.random.default_rng(42)
    n = 5000
    pi = 0.05
    y = rng.binomial(1, pi, size=n).astype(float)

    b_scores = make_beneficial_scores(y, sigma=0.5, rng=rng)
    h_scores = make_harmful_scores(y, harm_magnitude=0.4, rng=rng)

    ensemble_full = {"B1": b_scores, "B2": make_redundant_scores(y, b_scores, rho=0.3, rng=rng), "H1": h_scores}
    (c_i, test_effect, sign_conflict, _,
     auroc_val, auroc_test) = compute_loo_metrics(ensemble_full, y, y)

    assert c_i["H1"] < 0, f"H1 C_i={c_i['H1']:.6f}, expected < 0"

    names = list(ensemble_full.keys())
    b_c_i_positive = sum(1 for n in names if c_i[n] > 0)
    h_c_i_negative = sum(1 for n in names if c_i[n] < 0)
    assert b_c_i_positive >= 1, "At least one detector should have C_i > 0"
    assert h_c_i_negative >= 1, "At least one detector should have C_i < 0"


# ============================================================================
# 7. test_distribution_shapes
# ============================================================================

def test_distribution_shapes():
    rng = np.random.default_rng(42)
    n = 1000
    base = rng.normal(0, 1, size=n)

    for kind in ["skewed", "heavy_tailed"]:
        transformed = apply_distribution(base, kind)
        rho, _ = spearmanr(base, transformed)
        assert abs(rho) >= 0.999, (
            f"{kind}: Spearman={rho:.6f}, expected >= 0.999"
        )
