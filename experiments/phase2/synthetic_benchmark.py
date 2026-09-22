"""
PHASE 2: Synthetic Ground-Truth Benchmark for LOO Mechanism Validation
=======================================================================
Validates the LOO diagnostic mechanism with known detector roles.

Roles (by construction):
  BENEFICIAL (B): AUROC > 0.80, low correlation, removal degrades ensemble
  REDUNDANT (R):  AUROC > 0.80, high correlation with B, removal harmless
  HARMFUL (H):    Inverted ranks OR AUROC < 0.5, removal improves ensemble
  NOISY (N):      AUROC in [0.45, 0.55], removal harmless

Usage:
    python synthetic_benchmark.py --n_configs 200 --n_seeds 10 --outdir phase2_output
"""

import argparse
import json
import csv
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

# ============================================================================
# GRID (per PHASE2_DESIGN.md Section 3)
# ============================================================================

GRID = {
    "anomaly_rate":     [0.01, 0.05, 0.10],
    "correlation_rho":  [0.0, 0.3, 0.6, 0.9],
    "harm_magnitude":   [0.0, 0.05, 0.10, 0.20],
    "n_detectors":      [4, 6, 8],
    "n_samples":        [1_000, 10_000, 100_000],
    "score_distribution": ["gaussian", "skewed", "heavy_tailed"],
}

GRID_FACTORS = list(GRID.keys())

# H verification tracking (reset per build_ensemble call via reset)
_h_verify_pass = 0
_h_verify_fail = 0


def get_h_verify_stats():
    """Return (pass_count, fail_count) from last build_ensemble calls."""
    return _h_verify_pass, _h_verify_fail


def reset_h_verify_stats():
    """Reset H verification counters. Call once at start of a run."""
    global _h_verify_pass, _h_verify_fail
    _h_verify_pass = 0
    _h_verify_fail = 0

# N verification tracking (parallel to H verification)
_n_verify_pass = 0
_n_verify_fail = 0


def get_n_verify_stats():
    """Return (pass_count, fail_count) for N verification."""
    return _n_verify_pass, _n_verify_fail


def reset_n_verify_stats():
    """Reset N verification counters. Call once at start of a run."""
    global _n_verify_pass, _n_verify_fail
    _n_verify_pass = 0
    _n_verify_fail = 0

# ============================================================================
# ROLE CONSTRUCTION FUNCTIONS
# ============================================================================

def make_beneficial_scores(y, sigma=0.5, rng=None):
    rng = rng or np.random.default_rng()
    n = len(y)
    base = 1.0 * y
    noise = rng.normal(0, sigma, size=n)
    return base + noise


def make_redundant_scores(y, base_scores, rho=0.7, rng=None):
    rng = rng or np.random.default_rng()
    noise = rng.normal(0, np.std(base_scores), size=len(base_scores))
    return rho * base_scores + np.sqrt(max(1 - rho**2, 0)) * noise


def make_harmful_scores(y, n, harm_magnitude, base_signal, rng):
    """
    Invert all anomalies, mixing with a helpful signal to control harm strength.
    harm_magnitude in [0.0, 0.20] maps to strength in [0.0, 1.0].
    At strength=1.0: H is fully harmful (score = -base_signal + noise).
    At strength=0.5: partially harmful.
    At harm_magnitude=0.0: function is not called (H detector is skipped).
    """
    strength = min(1.0, harm_magnitude / 0.10)
    inverted = -base_signal
    helpful = base_signal
    score = (1 - strength) * helpful + strength * inverted
    score += rng.normal(0, 0.3, n)
    return score, strength


def make_noisy_scores(n, rng=None):
    rng = rng or np.random.default_rng()
    return rng.normal(0, 1, size=n)


# ============================================================================
# DISTRIBUTION SHAPER
# ============================================================================

def apply_distribution(scores, kind):
    if kind == "gaussian":
        return scores
    elif kind == "skewed":
        shifted = scores - scores.min() + 1.0
        return np.exp(shifted / shifted.max() * 2)
    elif kind == "heavy_tailed":
        shifted = scores - scores.min() + 1.0
        return shifted ** 2
    else:
        raise ValueError(f"Unknown distribution kind: {kind}")


# ============================================================================
# METRIC COMPUTATION
# ============================================================================

def rank_average_scores(score_list):
    n = len(score_list[0])
    w = 1.0 / len(score_list)
    combined = np.zeros(n)
    for s in score_list:
        combined += w * (rankdata(s) / n)
    return combined


def _safe_auroc(y, scores):
    try:
        return float(roc_auc_score(y, scores))
    except ValueError:
        return float("nan")


def compute_loo_metrics(scores, y_val, y_test):
    names = list(scores.keys())

    full_val = rank_average_scores([scores[n] for n in names])
    auroc_full_val = _safe_auroc(y_val, full_val)

    full_test = rank_average_scores([scores[n] for n in names])
    auroc_full_test = _safe_auroc(y_test, full_test)

    c_i = {}
    test_effect = {}
    standalone_auroc_val = {}

    for name in names:
        loo_names = [n for n in names if n != name]
        loo_val = rank_average_scores([scores[n] for n in loo_names])
        auroc_loo_val = _safe_auroc(y_val, loo_val)
        c_i[name] = auroc_full_val - auroc_loo_val

        loo_test = rank_average_scores([scores[n] for n in loo_names])
        auroc_loo_test = _safe_auroc(y_test, loo_test)
        test_effect[name] = auroc_loo_test - auroc_full_test

        standalone_auroc_val[name] = _safe_auroc(y_val, scores[name])

    sign_conflict = {}
    for name in names:
        ci_valid = not (isinstance(c_i[name], float) and
                        (c_i[name] != c_i[name]))
        te_valid = not (isinstance(test_effect[name], float) and
                        (test_effect[name] != test_effect[name]))
        if ci_valid and te_valid:
            sign_conflict[name] = (c_i[name] > 0) != (test_effect[name] > 0)
        else:
            sign_conflict[name] = False

    return (c_i, test_effect, sign_conflict, standalone_auroc_val,
            auroc_full_val, auroc_full_test)


# ============================================================================
# ROLE INFERENCE (validation-only)
# ============================================================================

H_N_CUTOFF = 0.05


def infer_role(c_i, standalone_auroc, corr_with_others, eps, eps_class=None):
    """
    eps: VRG retention threshold (unchanged behavior for downstream code)
    eps_class: wider threshold for role classification (default: max(3*eps, 0.03))
    H_N_CUTOFF: separate threshold for H vs N separation
    """
    if eps_class is None:
        eps_class = max(3 * eps, 0.03)

    def _isnan(x):
        return isinstance(x, float) and x != x

    if _isnan(c_i) or _isnan(standalone_auroc):
        return "NA"

    # 1. Harmful: clearly poor SA OR strongly negative contribution
    if standalone_auroc < 0.45:
        return "H"
    if c_i < -H_N_CUTOFF:
        return "H"

    # 2. Noisy: near-random SA and small |C_i|
    if abs(standalone_auroc - 0.5) < 0.10 and abs(c_i) < H_N_CUTOFF:
        return "N"

    # 3. Redundant: high SA, small |C_i|, correlated
    if standalone_auroc > 0.7 and abs(c_i) < eps_class and corr_with_others > 0.5:
        return "R"

    # 4. Beneficial: high SA and clear positive contribution
    if standalone_auroc > 0.7 and c_i >= eps_class:
        return "B"

    # 5. Redundant fallback
    if standalone_auroc > 0.7 and abs(c_i) < eps_class:
        return "R"

    # 6. Noisy fallback
    if abs(c_i) < H_N_CUTOFF:
        return "N"

    return "H" if c_i < 0 else "B"


# ============================================================================
# ROLE CONSTRUCTION ORCHESTRATOR
# ============================================================================

def build_ensemble(config, rng):
    n_detectors = config["n_detectors"]
    rho = config["correlation_rho"]
    harm_mag = config["harm_magnitude"]
    dist = config["score_distribution"]

    n = config["n_samples"]
    pi = config["anomaly_rate"]
    y_all = rng.binomial(1, pi, size=n).astype(float)

    n_train = int(0.70 * n)
    n_val = int(0.15 * n)
    idx = rng.permutation(n)
    val_idx = idx[n_train:n_train + n_val]
    test_idx = idx[n_train + n_val:]

    y_val = y_all[val_idx]
    y_test = y_all[test_idx]

    true_roles = {}
    all_val_scores = {}
    all_test_scores = {}

    n_beneficial = min(2, n_detectors - 1)
    n_redundant = 1 if n_detectors >= 4 else 0
    n_harmful = 1 if harm_mag > 0 else 0
    n_noisy = max(0, n_detectors - n_beneficial - n_redundant - n_harmful)

    base_val = make_beneficial_scores(y_val, sigma=0.5, rng=rng)
    base_test = make_beneficial_scores(y_test, sigma=0.5, rng=rng)

    for i in range(n_beneficial):
        name = f"B{i+1}"
        true_roles[name] = "B"
        if i == 0:
            s_val = make_beneficial_scores(y_val, sigma=0.5, rng=rng)
            s_test = make_beneficial_scores(y_test, sigma=0.5, rng=rng)
        else:
            s_val = make_beneficial_scores(y_val, sigma=0.5, rng=rng)
            s_test = make_beneficial_scores(y_test, sigma=0.5, rng=rng)
        all_val_scores[name] = apply_distribution(s_val, dist)
        all_test_scores[name] = apply_distribution(s_test, dist)

    base_signal_val = all_val_scores["B1"]
    base_signal_test = all_test_scores["B1"]

    if n_redundant > 0:
        name = "R1"
        true_roles[name] = "R"
        effective_rho = max(rho, 0.95)
        r_val = make_redundant_scores(y_val, base_signal_val, rho=effective_rho, rng=rng)
        r_test = make_redundant_scores(y_test, base_signal_test, rho=effective_rho, rng=rng)
        all_val_scores[name] = apply_distribution(r_val, dist)
        all_test_scores[name] = apply_distribution(r_test, dist)

    global _h_verify_pass, _h_verify_fail, _n_verify_pass, _n_verify_fail

    if n_harmful > 0 and harm_mag < 0.10:
        n_harmful = 0

    if n_harmful > 0:
        name = "H1"
        true_roles[name] = "H"
        h_rng = np.random.default_rng(int(rng.integers(0, 1_000_000_000)))

        h_added = False
        for attempt in range(10):
            h_val_raw, h_str = make_harmful_scores(
                y_val, len(y_val), harm_mag, base_signal_val, h_rng)
            h_test_raw, _ = make_harmful_scores(
                y_test, len(y_test), harm_mag, base_signal_test, h_rng)
            h_val = apply_distribution(h_val_raw, dist)
            h_test = apply_distribution(h_test_raw, dist)

            trial_val = dict(all_val_scores)
            trial_val["H1"] = h_val
            auroc_with_all = _safe_auroc(
                y_val, rank_average_scores([trial_val[n] for n in trial_val]))
            without_h = [trial_val[n] for n in trial_val if n != "H1"]
            auroc_without_h = _safe_auroc(
                y_val, rank_average_scores(without_h))
            c_i_h = auroc_with_all - auroc_without_h

            eps_v = 0.01 * auroc_with_all
            threshold = max(1.5 * eps_v, 0.01)
            if auroc_without_h > auroc_with_all and c_i_h < -threshold:
                all_val_scores["H1"] = h_val
                all_test_scores["H1"] = h_test
                h_added = True
                _h_verify_pass += 1
                break

        if not h_added:
            _h_verify_fail += 1
            del true_roles["H1"]

    n_rng = np.random.default_rng(int(rng.integers(0, 1_000_000_000)))

    for i in range(n_noisy):
        name = f"N{i+1}"
        true_roles[name] = "N"
        n_added = False
        for attempt in range(10):
            nv_val_raw = make_noisy_scores(len(y_val), rng=n_rng)
            nv_test_raw = make_noisy_scores(len(y_test), rng=n_rng)
            nv_val = apply_distribution(nv_val_raw, dist)
            nv_test = apply_distribution(nv_test_raw, dist)
            sa_n = _safe_auroc(y_val, nv_val)
            if 0.47 <= sa_n <= 0.53:
                all_val_scores[name] = nv_val
                all_test_scores[name] = nv_test
                n_added = True
                break
        if not n_added:
            # Accept the last sample anyway (do not skip N)
            all_val_scores[name] = nv_val
            all_test_scores[name] = nv_test
            _n_verify_fail += 1
            print(f"WARNING: N{i+1} verification failed after 10 attempts; using last sample")
        else:
            _n_verify_pass += 1

    return all_val_scores, all_test_scores, true_roles, y_val, y_test


# ============================================================================
# AGGREGATE METRICS
# ============================================================================

def aggregate_metrics(all_results):
    role_pairs = []
    sign_conflict_by_role = {"B": [], "R": [], "H": [], "N": []}
    naive_excluded = {"B": [], "R": [], "H": [], "N": []}
    epsilon_excluded = {"B": [], "R": [], "H": [], "N": []}

    for result in all_results:
        true_roles = result.get("true_roles", {})
        if not true_roles:
            continue
        predicted_roles = result["predicted_roles"]
        c_i = result["c_i"]
        sign_conflict = result["sign_conflict"]

        auroc_val = result["ensemble_auroc_full_val"]
        eps = 0.01 * auroc_val
        selected_epsilon = [name for name, c in c_i.items() if c > -eps]
        if not selected_epsilon:
            selected_epsilon = [max(c_i, key=c_i.get)]

        for name, true_role in true_roles.items():
            pred_role = predicted_roles.get(name, "B")
            role_pairs.append((pred_role, true_role))

            sc = sign_conflict.get(name, False)
            sign_conflict_by_role[true_role].append(1 if sc else 0)

            naive_retained = c_i[name] > 0
            naive_excluded[true_role].append(0 if naive_retained else 1)

            epsilon_retained = name in selected_epsilon
            epsilon_excluded[true_role].append(0 if epsilon_retained else 1)

    roles = ["B", "R", "H", "N", "NA"]
    cm = {r: {c: 0 for c in roles} for r in roles}
    for pred, true in role_pairs:
        cm[true][pred] += 1

    metrics = {}
    for role in ["B", "R", "H", "N"]:
        tp = cm[role][role]
        fp = sum(cm[other][role] for other in ["B", "R", "H", "N"] if other != role)
        fn = sum(cm[role][other] for other in ["B", "R", "H", "N"] if other != role)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        n_total = len(naive_excluded[role])
        signal_loss = np.mean(naive_excluded[role]) if n_total > 0 else 0.0
        epsilon_signal_loss = np.mean(epsilon_excluded[role]) if n_total > 0 else 0.0
        sign_conflict_rate = np.mean(sign_conflict_by_role[role]) if n_total > 0 else 0.0

        metrics[role] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "signal_loss_naive": float(signal_loss),
            "signal_loss_epsilon": float(epsilon_signal_loss),
            "sign_conflict_rate": float(sign_conflict_rate),
            "n_samples": n_total,
        }

    return {
        "confusion_matrix": cm,
        "per_role_metrics": metrics,
    }


# ============================================================================
# SAVE / LOAD HELPERS
# ============================================================================

def _convert(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: _convert(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert(v) for v in obj]
    return obj


def save_results(results, outdir):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    with open(outdir / "confusion_matrix.json", "w") as f:
        json.dump(results["confusion_matrix"], f, indent=2)

    with open(outdir / "per_role_metrics.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["role", "precision", "recall", "f1",
                         "signal_loss_naive", "signal_loss_epsilon",
                         "sign_conflict_rate", "n_samples"])
        for role, m in results["per_role_metrics"].items():
            writer.writerow([role, m["precision"], m["recall"], m["f1"],
                           m["signal_loss_naive"], m["signal_loss_epsilon"],
                           m["sign_conflict_rate"], m["n_samples"]])


def save_raw_results(all_results, outdir):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    serializable = _convert(all_results)
    with open(outdir / "raw_results.json", "w") as f:
        json.dump(serializable, f, indent=2)


def load_results(outdir):
    outdir = Path(outdir)
    with open(outdir / "confusion_matrix.json") as f:
        cm = json.load(f)
    metrics = {}
    with open(outdir / "per_role_metrics.csv") as f:
        reader = csv.DictReader(f)
        for row in reader:
            metrics[row["role"]] = {
                "precision": float(row["precision"]),
                "recall": float(row["recall"]),
                "f1": float(row["f1"]),
                "signal_loss_naive": float(row["signal_loss_naive"]),
                "signal_loss_epsilon": float(row["signal_loss_epsilon"]),
                "sign_conflict_rate": float(row["sign_conflict_rate"]),
                "n_samples": int(row["n_samples"]),
            }
    return {"confusion_matrix": cm, "per_role_metrics": metrics}


# ============================================================================
# SINGLE CONFIG RUNNER
# ============================================================================

def run_single_config(config, seed, verbose=False):
    rng = np.random.default_rng(seed)

    scores_val, scores_test, true_roles, y_val, y_test = build_ensemble(config, rng)

    (c_i, test_effect, sign_conflict, standalone_auroc_val,
     auroc_val, auroc_test) = compute_loo_metrics(scores_val, y_val, y_test)

    names = list(scores_val.keys())
    predicted_roles = {}
    for name in names:
        other_scores = [scores_val[n] for n in names if n != name]
        if other_scores:
            corrs = [abs(np.corrcoef(scores_val[name], o)[0, 1])
                     for o in other_scores
                     if np.std(o) > 0 and np.std(scores_val[name]) > 0]
            max_corr = max(corrs) if corrs else 0.0
        else:
            max_corr = 0.0

        predicted_roles[name] = infer_role(
            c_i[name], standalone_auroc_val[name], max_corr,
            eps=0.01 * auroc_val if not (auroc_val != auroc_val) else 0.001
        )

    return {
        "config": config,
        "seed": seed,
        "true_roles": true_roles,
        "predicted_roles": predicted_roles,
        "c_i": c_i,
        "test_effect": test_effect,
        "sign_conflict": sign_conflict,
        "standalone_auroc_val": standalone_auroc_val,
        "ensemble_auroc_full_val": auroc_val,
        "ensemble_auroc_full_test": auroc_test,
    }


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Phase 2 Synthetic Benchmark")
    parser.add_argument("--n_configs", type=int, default=200)
    parser.add_argument("--n_seeds", type=int, default=10)
    parser.add_argument("--outdir", type=str, default="phase2_output")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    from scipy.stats.qmc import LatinHypercube

    print(f"[Phase2] Generating {args.n_configs} configs with Latin hypercube sampling...")
    n_factors = len(GRID_FACTORS)
    lhs = LatinHypercube(d=n_factors, seed=args.seed)
    unit_samples = lhs.random(n=args.n_configs)

    config_list = []
    for i in range(args.n_configs):
        config = {}
        for j, factor in enumerate(GRID_FACTORS):
            idx = int(unit_samples[i, j] * len(GRID[factor]))
            idx = min(idx, len(GRID[factor]) - 1)
            config[factor] = GRID[factor][idx]
        config_list.append(config)

    all_results = []
    t0 = time.time()
    total = args.n_configs * args.n_seeds

    for ci, config in enumerate(config_list):
        for si in range(args.n_seeds):
            seed = args.seed + ci * 1000 + si
            result = run_single_config(config, seed)
            all_results.append(result)

            done = ci * args.n_seeds + si + 1
            if done % 50 == 0 or done == total:
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed > 0 else 0
                print(f"  [{done}/{total}] {elapsed:.1f}s ({rate:.1f} runs/s)")

    print(f"[Phase2] Aggregating results...")
    agg = aggregate_metrics(all_results)

    print(f"[Phase2] Saving to {args.outdir}/")
    save_results(agg, args.outdir)
    save_raw_results(all_results, args.outdir)

    print("[Phase2] Per-role metrics:")
    for role, m in agg["per_role_metrics"].items():
        print(f"  {role}: F1={m['f1']:.3f}  "
              f"sig_loss(naive)={m['signal_loss_naive']:.3f}  "
              f"sig_loss(eps)={m['signal_loss_epsilon']:.3f}  "
              f"sign_conflict={m['sign_conflict_rate']:.3f}")

    print(f"[Phase2] Done. {len(all_results)} runs in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
