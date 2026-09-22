"""
PHASE 2: Main Runner - Synthetic Ground-Truth Benchmark
=========================================================
Config sampling (LHS), main loop, robustness sub-run, aggregated outputs.

Usage:
    python run_phase2.py --n_configs 200 --n_seeds 10 --outdir phase2_output
"""

import argparse
import json
import csv
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from synthetic_benchmark import (
    GRID, GRID_FACTORS, build_ensemble, compute_loo_metrics,
    infer_role, aggregate_metrics, save_results, save_raw_results,
    rank_average_scores,
    reset_h_verify_stats, reset_n_verify_stats,
    get_h_verify_stats, get_n_verify_stats,
)
from sklearn.metrics import roc_auc_score


# ============================================================================
# CONFIG SAMPLING (LHS + edge doubling)
# ============================================================================

def sample_configs(n_configs, seed):
    from scipy.stats.qmc import LatinHypercube

    n_factors = len(GRID_FACTORS)
    lhs = LatinHypercube(d=n_factors, seed=seed)
    unit_samples = lhs.random(n=n_configs)

    config_list = []
    for i in range(n_configs):
        config = {}
        for j, factor in enumerate(GRID_FACTORS):
            idx = int(unit_samples[i, j] * len(GRID[factor]))
            idx = min(idx, len(GRID[factor]) - 1)
            config[factor] = GRID[factor][idx]
        config_list.append(config)

    edge_configs = []
    for factor, values in GRID.items():
        for v in [values[0], values[-1]]:
            cfg = {f: GRID[f][0] for f in GRID_FACTORS}
            cfg[factor] = v
            edge_configs.append(cfg)

    seen = set()
    deduped = []
    for cfg in edge_configs + config_list:
        key = tuple(sorted(cfg.items()))
        if key not in seen:
            seen.add(key)
            deduped.append(cfg)

    return deduped[:n_configs]


# ============================================================================
# CORRELATION HELPER
# ============================================================================

def max_corr_with_others(name, scores_val, names):
    other_scores = [scores_val[n] for n in names if n != name]
    if not other_scores:
        return 0.0
    corrs = []
    my_std = np.std(scores_val[name])
    if my_std == 0:
        return 0.0
    for o in other_scores:
        if np.std(o) > 0:
            corrs.append(abs(np.corrcoef(scores_val[name], o)[0, 1]))
    return max(corrs) if corrs else 0.0


# ============================================================================
# SINGLE-RUN WRAPPER
# ============================================================================

def run_one(config, seed):
    rng = np.random.default_rng(seed)
    result = build_ensemble(config, rng)
    if result is None:
        return {
            "config": config,
            "seed": seed,
            "status": "skipped_h_verification_failed",
            "true_roles": {},
            "predicted_roles": {},
            "c_i": {},
            "test_effect": {},
            "sign_conflict": {},
            "standalone_auroc_val": {},
            "ensemble_auroc_full_val": float("nan"),
            "ensemble_auroc_full_test": float("nan"),
        }
    scores_val, scores_test, true_roles, y_val, y_test = result
    (c_i, test_effect, sign_conflict, standalone_auroc_val,
     auroc_val, auroc_test) = compute_loo_metrics(scores_val, y_val, y_test)

    names = list(scores_val.keys())
    predicted_roles = {}
    is_nan_auroc = isinstance(auroc_val, float) and auroc_val != auroc_val
    eps = 0.01 * auroc_val if not is_nan_auroc else 0.001
    for name in names:
        mc = max_corr_with_others(name, scores_val, names)
        predicted_roles[name] = infer_role(
            c_i[name], standalone_auroc_val[name], mc, eps=eps
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
# ROBUSTNESS SUB-RUN
# ============================================================================

def run_robustness(configs_20, n_seeds, base_seed, outdir):
    outdir = Path(outdir)
    for n_samples_alt in [1_000, 100_000]:
        robust_results = []
        for ci, config in enumerate(configs_20):
            cfg_alt = dict(config)
            cfg_alt["n_samples"] = n_samples_alt
            for si in range(n_seeds):
                s = base_seed + ci * 1000 + si
                result = run_one(cfg_alt, s)
                robust_results.append(result)

        fname = f"robustness_N{n_samples_alt}.json"
        with open(outdir / fname, "w") as f:
            json.dump(_convert(robust_results), f, indent=2)
        print(f"[Robustness] Saved {fname} ({len(robust_results)} runs)")


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


# ============================================================================
# FALSE-PRUNING + SIGN-CONFLICT AGGREGATION
# ============================================================================

def compute_role_level_rates(all_results):
    naive_signal_loss = {"B": [], "R": [], "H": [], "N": []}
    eps_signal_loss = {"B": [], "R": [], "H": [], "N": []}
    naive_redundancy_fp = {"B": [], "R": [], "H": [], "N": []}
    eps_redundancy_fp = {"B": [], "R": [], "H": [], "N": []}
    harmful_retention_naive = {"B": [], "R": [], "H": [], "N": []}
    harmful_retention_eps = {"B": [], "R": [], "H": [], "N": []}

    sign_conflict_by_role = {"B": [], "R": [], "H": [], "N": []}
    sign_conflict_by_dist = {"gaussian": [], "skewed": [], "heavy_tailed": []}
    sign_conflict_by_rho = {}

    for res in all_results:
        true_roles = res["true_roles"]
        c_i = res["c_i"]
        sign_conflict = res["sign_conflict"]
        config = res["config"]
        auroc_val = res["ensemble_auroc_full_val"]
        eps = 0.01 * auroc_val
        selected_eps = {n for n, c in c_i.items() if c > -eps}

        rho = config.get("correlation_rho", 0.0)
        if rho not in sign_conflict_by_rho:
            sign_conflict_by_rho[rho] = []

        for name, role in true_roles.items():
            sc = sign_conflict.get(name, False)
            sign_conflict_by_role[role].append(1 if sc else 0)
            sign_conflict_by_rho[rho].append(1 if sc else 0)

            dist = config.get("score_distribution", "gaussian")
            if dist in sign_conflict_by_dist:
                sign_conflict_by_dist[dist].append(1 if sc else 0)

            naive_retained = c_i[name] > 0
            eps_retained = name in selected_eps

            naive_signal_loss[role].append(0 if naive_retained else 1)
            eps_signal_loss[role].append(0 if eps_retained else 1)

            if role == "R":
                naive_redundancy_fp["R"].append(0 if naive_retained else 1)
                eps_redundancy_fp["R"].append(0 if eps_retained else 1)
            if role == "H":
                harmful_retention_naive["H"].append(1 if naive_retained else 0)
                harmful_retention_eps["H"].append(1 if eps_retained else 0)

    return (naive_signal_loss, eps_signal_loss,
            naive_redundancy_fp, eps_redundancy_fp,
            harmful_retention_naive, harmful_retention_eps,
            sign_conflict_by_role, sign_conflict_by_dist,
            sign_conflict_by_rho)


def save_rate_csv(data, outdir, filename):
    outdir = Path(outdir)
    with open(outdir / filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "role_or_group", "mean", "std", "n"])
        for key, vals in data.items():
            if vals:
                writer.writerow([filename.replace(".csv", ""), key,
                               f"{np.mean(vals):.6f}", f"{np.std(vals):.6f}",
                               len(vals)])


def save_rates(all_results, outdir):
    outdir = Path(outdir)
    (naive_sl, eps_sl, naive_rfp, eps_rfp,
     naive_hr, eps_hr, sc_role, sc_dist,
     sc_rho) = compute_role_level_rates(all_results)

    with open(outdir / "false_pruning_rates.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "selector", "role", "mean", "std", "n"])
        for role in ["B", "R", "H", "N"]:
            if naive_sl[role]:
                w.writerow(["signal_loss", "naive", role,
                           f"{np.mean(naive_sl[role]):.6f}",
                           f"{np.std(naive_sl[role]):.6f}",
                           len(naive_sl[role])])
            if eps_sl[role]:
                w.writerow(["signal_loss", "epsilon_vrg", role,
                           f"{np.mean(eps_sl[role]):.6f}",
                           f"{np.std(eps_sl[role]):.6f}",
                           len(eps_sl[role])])
            if naive_rfp[role]:
                w.writerow(["redundancy_false_pruning", "naive", role,
                           f"{np.mean(naive_rfp[role]):.6f}",
                           f"{np.std(naive_rfp[role]):.6f}",
                           len(naive_rfp[role])])
            if eps_rfp[role]:
                w.writerow(["redundancy_false_pruning", "epsilon_vrg", role,
                           f"{np.mean(eps_rfp[role]):.6f}",
                           f"{np.std(eps_rfp[role]):.6f}",
                           len(eps_rfp[role])])
            if naive_hr[role]:
                w.writerow(["harmful_retention", "naive", role,
                           f"{np.mean(naive_hr[role]):.6f}",
                           f"{np.std(naive_hr[role]):.6f}",
                           len(naive_hr[role])])
            if eps_hr[role]:
                w.writerow(["harmful_retention", "epsilon_vrg", role,
                           f"{np.mean(eps_hr[role]):.6f}",
                           f"{np.std(eps_hr[role]):.6f}",
                           len(eps_hr[role])])

    with open(outdir / "sign_conflict_rates.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["group", "mean", "std", "n"])
        for role, vals in sc_role.items():
            if vals:
                w.writerow([f"role_{role}", f"{np.mean(vals):.6f}",
                           f"{np.std(vals):.6f}", len(vals)])
        for dist, vals in sc_dist.items():
            if vals:
                w.writerow([f"dist_{dist}", f"{np.mean(vals):.6f}",
                           f"{np.std(vals):.6f}", len(vals)])

    with open(outdir / "sign_conflict_by_rho.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["correlation_rho", "mean_sign_conflict_rate", "std", "n"])
        for rho in sorted(sc_rho.keys()):
            vals = sc_rho[rho]
            if vals:
                w.writerow([f"{rho:.1f}", f"{np.mean(vals):.6f}",
                           f"{np.std(vals):.6f}", len(vals)])


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Phase 2 Main Runner")
    parser.add_argument("--n_configs", type=int, default=200)
    parser.add_argument("--n_seeds", type=int, default=10)
    parser.add_argument("--outdir", type=str, default="phase2_output")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"[Phase2] Sampling {args.n_configs} configs (LHS + edge doubling)...")
    config_list = sample_configs(args.n_configs, args.seed)
    print(f"[Phase2] Actual configs: {len(config_list)}")

    reset_h_verify_stats()
    reset_n_verify_stats()

    all_results = []
    t0 = time.time()
    total = len(config_list) * args.n_seeds

    for ci, config in enumerate(config_list):
        for si in range(args.n_seeds):
            s = args.seed + ci * 1000 + si
            result = run_one(config, s)
            all_results.append(result)

            done = ci * args.n_seeds + si + 1
            if done % 10 == 0 or done == total:
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed > 0 else 0
                print(f"  [{done}/{total}] {elapsed:.1f}s ({rate:.1f} runs/s)")

    print(f"[Phase2] Aggregating...")
    agg = aggregate_metrics(all_results)
    save_results(agg, args.outdir)
    save_raw_results(all_results, args.outdir)
    save_rates(all_results, args.outdir)

    print(f"[Phase2] Robustness sub-run (first 20 configs)...")
    run_robustness(config_list[:20], args.n_seeds, args.seed, args.outdir)

    elapsed = time.time() - t0
    print(f"[Phase2] Done. {len(all_results)} runs in {elapsed:.1f}s")

    print("\n=== Per-role metrics ===")
    for role, m in agg["per_role_metrics"].items():
        print(f"  {role}: P={m['precision']:.3f} R={m['recall']:.3f} "
              f"F1={m['f1']:.3f}  sl(naive)={m['signal_loss_naive']:.3f}  "
              f"sl(eps)={m['signal_loss_epsilon']:.3f}  "
              f"sc={m['sign_conflict_rate']:.3f}")

    hp, hf = get_h_verify_stats()
    np_, nf = get_n_verify_stats()
    print(f"\nH verification: pass={hp}, fail={hf}")
    print(f"N verification: pass={np_}, fail={nf}")

    verification_stats = {
        "h_pass": hp, "h_fail": hf,
        "n_pass": np_, "n_fail": nf,
    }
    with open(f"{args.outdir}/verification_stats.json", "w") as f:
        json.dump(verification_stats, f, indent=2)


if __name__ == "__main__":
    main()
