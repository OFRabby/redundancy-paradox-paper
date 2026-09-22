"""
Ensemble Evaluation Module for Phase 3
========================================
Implements LOO contribution, TestEffect, sign-conflict detection,
epsilon-VRG selection, and full ensemble evaluation.

Phase 1 parity: all core functions match run_task5.py exactly.
Phase 3 addition: material_conflict (pre-specified in PHASE3_DESIGN.md).
"""

import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

# ============================================================================
# CONSTANTS (Phase 1 parity)
# ============================================================================

ALPHA = 0.01  # epsilon-VRG scaling: epsilon = ALPHA * AUROC_full(D_val)


# ============================================================================
# RANK AVERAGING (Phase 1 parity)
# ============================================================================

def rank_average_scores(score_list: list[np.ndarray]) -> np.ndarray:
    """Equal-weight rank average across detectors. Phase 1 parity.

    For each detector's scores, compute rankdata(s)/n (ranks normalized to [0,1]),
    then average across detectors.
    """
    n = len(score_list[0])
    combined = np.zeros(n)
    for s in score_list:
        combined += rankdata(s) / n
    return combined / len(score_list)


# ============================================================================
# LOO CONTRIBUTION (Phase 1 parity)
# ============================================================================

def compute_loo_contribution(
    scores_val: dict[str, np.ndarray],
    y_val: np.ndarray,
) -> tuple[dict[str, float], float]:
    """C_i = AUROC_full - AUROC_without_i. Higher = more helpful on val.

    Returns:
        (c_i, auroc_full_val) where c_i[name] = contribution of detector name.
    """
    names = list(scores_val.keys())
    full = rank_average_scores([scores_val[n] for n in names])
    auroc_full = float(roc_auc_score(y_val, full))
    c_i = {}
    for name in names:
        loo_names = [n for n in names if n != name]
        loo = rank_average_scores([scores_val[n] for n in loo_names])
        c_i[name] = auroc_full - float(roc_auc_score(y_val, loo))
    return c_i, auroc_full


# ============================================================================
# TEST EFFECT (Phase 1 parity)
# ============================================================================

def compute_test_effect(
    scores_test: dict[str, np.ndarray],
    y_test: np.ndarray,
) -> tuple[dict[str, float], float]:
    """TestEffect_i = AUROC_without_i - AUROC_full. Positive = test-harmful.

    Returns:
        (test_effect, auroc_full_test) where test_effect[name] = effect of removing detector name.
    """
    names = list(scores_test.keys())
    full = rank_average_scores([scores_test[n] for n in names])
    auroc_full = float(roc_auc_score(y_test, full))
    te = {}
    for name in names:
        loo_names = [n for n in names if n != name]
        loo = rank_average_scores([scores_test[n] for n in loo_names])
        te[name] = float(roc_auc_score(y_test, loo)) - auroc_full
    return te, auroc_full


# ============================================================================
# SIGN CONFLICT (Phase 1 parity)
# ============================================================================

def compute_sign_conflict(
    c_i: dict[str, float],
    test_effect: dict[str, float],
) -> dict[str, bool]:
    """Conflict = C_i and TestEffect have SAME sign (opposite semantics).

    C_i > 0 means detector helps val ensemble.
    TestEffect > 0 means detector harms test ensemble.
    Same sign = paradoxical (sign conflict).
    """
    return {name: (c_i[name] > 0) == (test_effect[name] > 0) for name in c_i}


# ============================================================================
# MATERIAL CONFLICT (Phase 3 addition, pre-specified)
# ============================================================================

def compute_material_conflict(
    c_i: dict[str, float],
    test_effect: dict[str, float],
    threshold: float = 0.01,
) -> dict[str, bool]:
    """Material conflict: sign conflict with at least one metric exceeding
    threshold in magnitude. Filters out numerical-noise conflicts at zero.

    Args:
        c_i: LOO contribution per detector (positive = val-helpful).
        test_effect: held-out effect per detector (positive = test-harmful).
        threshold: minimum magnitude for one side of the conflict.

    Returns:
        dict[name, bool] where True = material conflict.
    """
    out = {}
    for name in c_i:
        ci = c_i[name]
        te = test_effect[name]
        sign_conflict = (ci > 0) == (te > 0)
        magnitude = max(abs(ci), abs(te)) >= threshold
        out[name] = sign_conflict and magnitude
    return out


# ============================================================================
# EPSILON-VRG SELECTION (Phase 1 parity)
# ============================================================================

def epsilon_vrg_selection(
    c_i: dict[str, float],
    auroc_full_val: float,
    alpha: float = ALPHA,
) -> tuple[list[str], float]:
    """Retain detectors with C_i > -epsilon where epsilon = alpha * AUROC_full_val.

    Fallback: if no detector passes, retain the one with max C_i.

    Returns:
        (sorted_selected, epsilon)
    """
    epsilon = alpha * auroc_full_val
    selected = [name for name, c in c_i.items() if c > -epsilon]
    if not selected:
        selected = [max(c_i, key=c_i.get)]
    return sorted(selected), float(epsilon)


# ============================================================================
# NAIVE LOO SELECTION (Phase 1 parity)
# ============================================================================

def naive_loo_selection(c_i: dict[str, float]) -> list[str]:
    """Retain detectors with C_i >= 0.

    Fallback: if no detector passes, retain the one with max C_i.
    """
    selected = [name for name, c in c_i.items() if c >= 0]
    if not selected:
        selected = [max(c_i, key=c_i.get)]
    return sorted(selected)


# ============================================================================
# FULL EVALUATION (orchestrator)
# ============================================================================

def evaluate_ensemble(
    scores_val: dict[str, np.ndarray],
    scores_test: dict[str, np.ndarray],
    y_val: np.ndarray,
    y_test: np.ndarray,
    alpha: float = ALPHA,
) -> dict:
    """Orchestrator: compute EW, NL, EV AUROC + all per-detector metrics.

    Returns dict with keys:
        auroc_ew_val, auroc_ew_test
        auroc_nl_val, auroc_nl_test, nl_selected
        auroc_ev_val, auroc_ev_test, ev_selected, epsilon
        c_i, test_effect, sign_conflict, material_conflict
        standalone_auroc_val (per detector)
    """
    names = list(scores_val.keys())

    # LOO contribution
    c_i, auroc_full_val = compute_loo_contribution(scores_val, y_val)

    # TestEffect
    test_effect, auroc_full_test = compute_test_effect(scores_test, y_test)

    # Conflict detection
    sign_conflict = compute_sign_conflict(c_i, test_effect)
    material_conflict = compute_material_conflict(c_i, test_effect)

    # Selection
    nl_selected = naive_loo_selection(c_i)
    ev_selected, epsilon = epsilon_vrg_selection(c_i, auroc_full_val, alpha=alpha)

    # Ensemble AUROC: Equal Weight (EW)
    ew_val = rank_average_scores([scores_val[n] for n in names])
    ew_test = rank_average_scores([scores_test[n] for n in names])
    auroc_ew_val = float(roc_auc_score(y_val, ew_val))
    auroc_ew_test = float(roc_auc_score(y_test, ew_test))

    # Ensemble AUROC: Naive LOO (NL)
    if nl_selected:
        nl_val = rank_average_scores([scores_val[n] for n in nl_selected])
        nl_test = rank_average_scores([scores_test[n] for n in nl_selected])
    else:
        nl_val = ew_val
        nl_test = ew_test
    auroc_nl_val = float(roc_auc_score(y_val, nl_val))
    auroc_nl_test = float(roc_auc_score(y_test, nl_test))

    # Ensemble AUROC: Epsilon-VRG (EV)
    if ev_selected:
        ev_val = rank_average_scores([scores_val[n] for n in ev_selected])
        ev_test = rank_average_scores([scores_test[n] for n in ev_selected])
    else:
        ev_val = ew_val
        ev_test = ew_test
    auroc_ev_val = float(roc_auc_score(y_val, ev_val))
    auroc_ev_test = float(roc_auc_score(y_test, ev_test))

    # Standalone AUROC per detector
    standalone_auroc_val = {}
    for name in names:
        standalone_auroc_val[name] = float(roc_auc_score(y_val, scores_val[name]))

    return {
        "auroc_ew_val": auroc_ew_val,
        "auroc_ew_test": auroc_ew_test,
        "auroc_nl_val": auroc_nl_val,
        "auroc_nl_test": auroc_nl_test,
        "nl_selected": nl_selected,
        "auroc_ev_val": auroc_ev_val,
        "auroc_ev_test": auroc_ev_test,
        "ev_selected": ev_selected,
        "epsilon": epsilon,
        "c_i": c_i,
        "test_effect": test_effect,
        "sign_conflict": sign_conflict,
        "material_conflict": material_conflict,
        "standalone_auroc_val": standalone_auroc_val,
    }
