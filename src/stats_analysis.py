"""Statistical analysis for Phase 3 — hierarchical multi-dataset redundancy paradox.

Primary K=9 family: 3 methods (EW, NL, EV) × 3 datasets (CICIDS2017, UNSW-NB15, NSL-KDD).
All tests pre-specified; no post-hoc family reduction.
"""

import logging
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.formula.api import mixedlm
from statsmodels.stats.multitest import multipletests

logger = logging.getLogger(__name__)

# ============================================================================
# CONSTANTS
# ============================================================================

SIGNIFICANCE_LEVEL = 0.05  # alpha for hypothesis tests

PRIMARY_FAMILY = [
    ("CICIDS2017", "EW", "NL"),
    ("CICIDS2017", "EW", "EV"),
    ("CICIDS2017", "NL", "EV"),
    ("UNSW-NB15",  "EW", "NL"),
    ("UNSW-NB15",  "EW", "EV"),
    ("UNSW-NB15",  "NL", "EV"),
    ("NSL-KDD",    "EW", "NL"),
    ("NSL-KDD",    "EW", "EV"),
    ("NSL-KDD",    "NL", "EV"),
]


# ============================================================================
# PAIRED COMPARISON
# ============================================================================

def paired_comparison(
    df: pd.DataFrame,
    dataset: str,
    method_a: str,
    method_b: str,
    metric: str = "auroc_test",
) -> dict:
    """Wilcoxon signed-rank + Cohen's d + bootstrap 95% CI.

    Compares method_a vs method_b on the given dataset across seeds.

    Returns dict with keys:
        n_pairs, mean_diff, median_diff, cohens_d,
        wilcoxon_stat, p_raw, ci_low, ci_high
    """
    sub = df[df["dataset"] == dataset].copy()
    sub = sub.sort_values(["seed", "encoder"]).reset_index(drop=True)

    a = sub[sub["method"] == method_a][["seed", "encoder", metric]].rename(
        columns={metric: "score_a"}
    )
    b = sub[sub["method"] == method_b][["seed", "encoder", metric]].rename(
        columns={metric: "score_b"}
    )
    merged = pd.merge(a, b, on=["seed", "encoder"], how="inner")

    if len(merged) < 2:
        return {
            "n_pairs": len(merged),
            "mean_diff": float("nan"),
            "median_diff": float("nan"),
            "cohens_d": float("nan"),
            "wilcoxon_stat": float("nan"),
            "p_raw": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
        }

    diffs = merged["score_a"].values - merged["score_b"].values

    mean_diff = float(np.mean(diffs))
    median_diff = float(np.median(diffs))
    std_diff = float(np.std(diffs, ddof=1))
    cohens_d = mean_diff / std_diff if std_diff > 0 else 0.0

    try:
        wilcoxon_stat, p_raw = stats.wilcoxon(diffs, alternative="two-sided")
    except ValueError:
        wilcoxon_stat, p_raw = float("nan"), float("nan")

    ci_low, ci_high = bootstrap_ci(diffs)

    return {
        "n_pairs": len(merged),
        "mean_diff": mean_diff,
        "median_diff": median_diff,
        "cohens_d": cohens_d,
        "wilcoxon_stat": float(wilcoxon_stat),
        "p_raw": float(p_raw),
        "ci_low": ci_low,
        "ci_high": ci_high,
    }


# ============================================================================
# BOOTSTRAP CI
# ============================================================================

def bootstrap_ci(
    diffs: np.ndarray,
    n_boot: int = 10000,
    seed: int = 42,
) -> tuple[float, float]:
    """Percentile bootstrap 95% CI for the mean difference."""
    rng = np.random.default_rng(seed)
    n = len(diffs)
    if n == 0:
        return (float("nan"), float("nan"))
    if n == 1:
        return (float(diffs[0]), float(diffs[0]))
    boot_means = np.array([np.mean(rng.choice(diffs, size=n, replace=True))
                           for _ in range(n_boot)])
    return (float(np.percentile(boot_means, 2.5)),
            float(np.percentile(boot_means, 97.5)))


# ============================================================================
# HOLM-BONFERRONI CORRECTION
# ============================================================================

def apply_holm_correction(p_values: list[float]) -> tuple[list[float], list[bool]]:
    """Holm-Bonferroni correction over the K=9 primary family.

    Returns:
        (adjusted_p, rejected_at_alpha_0.05)
    """
    if not p_values:
        return ([], [])
    reject, p_adj, _, _ = multipletests(p_values, alpha=SIGNIFICANCE_LEVEL, method="holm")
    return (list(p_adj), list(reject))


# ============================================================================
# HIERARCHICAL MODEL
# ============================================================================

def hierarchical_model(
    df: pd.DataFrame,
    formula: str = "auroc_test ~ method",
    groups: str = "dataset",
) -> object:
    """Fit mixed-effects model with dataset as random group.

    Uses method as categorical fixed effect; dataset as random intercept.
    Reports coefficients + SE.
    """
    model = mixedlm(formula, data=df, groups=df[groups]).fit(reml=True)
    return model


# ============================================================================
# KRUSKAL-WALLIS ACROSS DATASETS (H3)
# ============================================================================

def kruskal_wallis_across_datasets(df: pd.DataFrame, metric: str) -> dict:
    """Test if metric varies significantly across datasets.

    Returns dict with keys: statistic, p_value, n_groups, group_sizes
    """
    groups = [group[metric].values for _, group in df.groupby("dataset")]
    if len(groups) < 2:
        return {"statistic": float("nan"), "p_value": float("nan"),
                "n_groups": len(groups), "group_sizes": [len(g) for g in groups]}
    statistic, p_value = stats.kruskal(*groups)
    return {
        "statistic": float(statistic),
        "p_value": float(p_value),
        "n_groups": len(groups),
        "group_sizes": [len(g) for g in groups],
    }


# ============================================================================
# SPEARMAN CORRELATION (H3 + H7)
# ============================================================================

def spearman_correlation(x: np.ndarray, y: np.ndarray) -> dict:
    """Spearman rank correlation.

    Returns dict with keys: rho, p_value, n
    """
    rho, p = stats.spearmanr(x, y)
    return {"rho": float(rho), "p_value": float(p), "n": len(x)}


# ============================================================================
# MAIN ANALYSIS ORCHESTRATOR
# ============================================================================

def run_statistical_analysis(
    results_df: pd.DataFrame,
    output_dir: str,
) -> dict:
    """Full statistical pipeline for Phase 3.

    Steps:
        1. Compute primary K=9 paired comparisons
        2. Holm-Bonferroni correction
        3. Hierarchical model fit
        4. Kruskal-Wallis across datasets
        5. Spearman correlations for H3 and H7
        6. Save outputs:
           - hierarchical_stats.json
           - bootstrap_results.csv
           - holm_k9.csv

    Expected results_df columns:
        dataset, seed, method (EW/NL/EV),
        auroc_val, auroc_test, c_i, test_effect,
        sign_conflict, material_conflict, standalone_auroc_val

    Returns:
        dict with all analysis results
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    # --- 1. K=9 paired comparisons ---
    k9_results = []
    for dataset, ma, mb in PRIMARY_FAMILY:
        res = paired_comparison(results_df, dataset, ma, mb)
        res["dataset"] = dataset
        res["method_a"] = ma
        res["method_b"] = mb
        k9_results.append(res)

    # --- 2. Holm correction ---
    p_raw_list = [r["p_raw"] for r in k9_results]
    p_adj_list, reject_list = apply_holm_correction(p_raw_list)
    for i, res in enumerate(k9_results):
        res["p_holm"] = p_adj_list[i]
        res["reject"] = reject_list[i]

    # --- 3. Hierarchical model ---
    hmodel = hierarchical_model(results_df)

    # --- 4. Kruskal-Wallis across datasets ---
    kw = kruskal_wallis_across_datasets(results_df, "auroc_test")

    # --- 5. Spearman correlations (H3 + H7) ---
    spearman_results = {}
    skipped_reasons = []

    has_sign_conflict = "sign_conflict" in results_df.columns
    has_c_i = "c_i" in results_df.columns
    has_detector = "detector" in results_df.columns

    if has_sign_conflict:
        mean_auroc_per_dataset = results_df.groupby("dataset")["auroc_test"].mean()
        sign_conflict_per_dataset = results_df.groupby("dataset")["sign_conflict"].mean()
        if len(mean_auroc_per_dataset) >= 3:
            spearman_results["h3_auroc_vs_signconflict"] = spearman_correlation(
                mean_auroc_per_dataset.values, sign_conflict_per_dataset.values
            )
            logger.info("Spearman H3: computed")
        else:
            reason = f"fewer than 3 datasets ({len(mean_auroc_per_dataset)})"
            skipped_reasons.append(("h3", reason))
            logger.info("Spearman H3: skipped (%s)", reason)
    else:
        reason = "missing columns: sign_conflict"
        skipped_reasons.append(("h3", reason))
        logger.info("Spearman H3: skipped (%s)", reason)

    if has_c_i and has_sign_conflict and has_detector:
        corr_per_detector = results_df.groupby("detector")[["c_i", "sign_conflict"]].mean()
        if len(corr_per_detector) >= 3:
            spearman_results["h7_civi_vs_signconflict"] = spearman_correlation(
                corr_per_detector["c_i"].values,
                corr_per_detector["sign_conflict"].values,
            )
            logger.info("Spearman H7: computed")
        else:
            reason = f"fewer than 3 detectors ({len(corr_per_detector)})"
            skipped_reasons.append(("h7", reason))
            logger.info("Spearman H7: skipped (%s)", reason)
    else:
        missing = [c for c in ["c_i", "sign_conflict", "detector"] if c not in results_df.columns]
        reason = f"missing columns: {', '.join(missing)}"
        skipped_reasons.append(("h7", reason))
        logger.info("Spearman H7: skipped (%s)", reason)

    if skipped_reasons:
        spearman_results["skipped"] = True
        spearman_results["reasons"] = {k: v for k, v in skipped_reasons}

    # --- 6. Save outputs ---
    holm_df = pd.DataFrame(k9_results)
    holm_df.to_csv(os.path.join(output_dir, "holm_k9.csv"), index=False)

    model_summary = {
        "coefficients": dict(hmodel.params),
        "std_errors": dict(hmodel.bse),
        "p_values": dict(hmodel.pvalues),
    }
    import json
    with open(os.path.join(output_dir, "hierarchical_stats.json"), "w") as f:
        json.dump(model_summary, f, indent=2)

    with open(os.path.join(output_dir, "kruskal_wallis.json"), "w") as f:
        json.dump(kw, f, indent=2)

    with open(os.path.join(output_dir, "spearman.json"), "w") as f:
        json.dump(spearman_results, f, indent=2, default=str)

    return {
        "k9_comparisons": k9_results,
        "holm_adjusted_p": p_adj_list,
        "holm_rejected": reject_list,
        "hierarchical_model": model_summary,
        "kruskal_wallis": kw,
        "spearman": spearman_results,
    }
