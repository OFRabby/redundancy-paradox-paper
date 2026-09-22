"""
PHASE 1A: TargetEncoder Ablation Study
========================================
Compares TargetEncoder (leaked baseline) vs Hybrid (primary) vs Ordinal (robustness)
across all 3 datasets, seeds 42-44.

Hybrid policy (pre-specified):
- OneHotEncoder for categoricals with <= 20 unique values in training split
- FrequencyEncoder (count-based, fit on train only) for > 20 unique values

Material threshold (provisional): |delta_AUROC| >= 0.01

Usage:
    python ablation_targetencoder.py
"""

import os
import sys
import json
import time
import logging
from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score, average_precision_score

# ============================================================================
# LOGGING
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ============================================================================
# CONSTANTS
# ============================================================================
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent / "anomaly_detector-main" / "anomaly_detector-main"
DATA_ROOT = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = SCRIPT_DIR

sys.path.insert(0, str(PROJECT_ROOT))

from src.features.extractor import FeatureExtractor
from src.models.autoencoder import train_autoencoder
from src.models.isolation_forest import train_isolation_forest
from src.models.lof import train_lof
from src.models.ocsvm import train_ocsvm
from src.models.threshold import find_optimal_threshold
from src.utils.helpers import set_seed

DATASETS = {
    "CICIDS2017": {
        "path": "cicids2017.parquet",
        "split_strategy": "chronological",
        "time_column": "ts",
    },
    "UNSW-NB15": {
        "path": "unsw_nb15.parquet",
        "split_strategy": "chronological",
        "time_column": "ts",
    },
    "NSL-KDD": {
        "path": "nsl_kdd.parquet",
        "split_strategy": "random",
    },
}

DETECTORS = ["IF", "LOF", "OCSVM", "AE"]
SEEDS = [42, 43, 44]
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
TEST_FRAC = 0.15
ALPHA = 0.01
MATERIAL_THRESHOLD = 0.01

META_COLUMNS = {
    "ts", "uid", "id.orig_h", "id.orig_p", "id.resp_h", "id.resp_p",
    "timestamp", "label", "is_anomaly", "attack_category",
    "date_partition", "dataset_split",
}

FE_CONFIG = {
    "host_windows": ["1h", "6h", "24h"],
    "exclude_columns": [
        "ts", "uid", "id.orig_h", "id.orig_p", "id.resp_h",
        "id.resp_p", "label", "attack_category", "is_anomaly",
    ],
}

# ============================================================================
# DATA SPLITTING (identical to run_task5.py)
# ============================================================================

def split_dataset(df, strategy, seed=42, time_col="ts"):
    n = len(df)
    train_n = int(n * TRAIN_FRAC)
    val_n = int(n * VAL_FRAC)

    if strategy == "chronological" and time_col in df.columns:
        time_vals = df[time_col].to_numpy()
        sorted_idx = np.argsort(time_vals)
        train_idx = sorted_idx[:train_n]
        val_idx = sorted_idx[train_n:train_n + val_n]
        test_idx = sorted_idx[train_n + val_n:]
    elif strategy == "random":
        rng = np.random.RandomState(seed)
        indices = rng.permutation(n)
        train_idx = indices[:train_n]
        val_idx = indices[train_n:train_n + val_n]
        test_idx = indices[train_n + val_n:]
    else:
        raise ValueError(f"Unknown split strategy: {strategy}")

    assert len(set(train_idx) & set(val_idx)) == 0
    assert len(set(train_idx) & set(test_idx)) == 0
    assert len(set(val_idx) & set(test_idx)) == 0
    return train_idx, val_idx, test_idx


# ============================================================================
# RANK-AVERAGE ENSEMBLE
# ============================================================================

def rank_average_scores(score_list):
    n = len(score_list[0])
    w = 1.0 / len(score_list)
    combined = np.zeros(n)
    for s in score_list:
        combined += w * (rankdata(s) / n)
    return combined


# ============================================================================
# LOO C_i AND TestEffect
# ============================================================================

def compute_c_i(y_val, score_dict, detector_name):
    names = list(score_dict.keys())
    full = rank_average_scores([score_dict[n] for n in names])
    auroc_full = roc_auc_score(y_val, full)
    loo_names = [n for n in names if n != detector_name]
    loo = rank_average_scores([score_dict[n] for n in loo_names])
    auroc_loo = roc_auc_score(y_val, loo)
    return auroc_full, auroc_loo, auroc_full - auroc_loo


def compute_all_c_i(y_val, score_dict):
    results = {}
    for name in DETECTORS:
        auroc_full, auroc_loo, c_i = compute_c_i(y_val, score_dict, name)
        results[name] = {"auroc_full": float(auroc_full), "auroc_loo": float(auroc_loo), "c_i": float(c_i)}
    return results


def compute_testeffect(y_test, score_dict, detector_name):
    names = list(score_dict.keys())
    full = rank_average_scores([score_dict[n] for n in names])
    auroc_full = roc_auc_score(y_test, full)
    loo_names = [n for n in names if n != detector_name]
    loo = rank_average_scores([score_dict[n] for n in loo_names])
    auroc_loo = roc_auc_score(y_test, loo)
    return auroc_full, auroc_loo, auroc_loo - auroc_full


# ============================================================================
# SELECTION RULES
# ============================================================================

def epsilon_vrg_selection(c_i_values, auroc_full_val):
    epsilon = ALPHA * auroc_full_val
    selected = [name for name, c in c_i_values.items() if c > -epsilon]
    if not selected:
        selected = [max(c_i_values, key=c_i_values.get)]
    return sorted(selected), float(epsilon)


def naive_loo_selection(c_i_values):
    selected = [name for name, c in c_i_values.items() if c > 0]
    if not selected:
        selected = [max(c_i_values, key=c_i_values.get)]
    return sorted(selected)


# ============================================================================
# SINGLE SEED RUN
# ============================================================================

def run_single_seed(dataset_name, dataset_cfg, seed, encoder_type, fast=False):
    """Run one seed on one dataset with a specific encoder type.

    If fast=True, AE trains only 2 epochs (for debugging/profiling).
    """
    t0 = time.time()
    set_seed(seed)

    logger.info(f"  [{dataset_name} s={seed} e={encoder_type}] Loading data...")
    data_path = DATA_ROOT / dataset_cfg["path"]
    df = pl.read_parquet(data_path)

    train_idx, val_idx, test_idx = split_dataset(
        df, dataset_cfg["split_strategy"], seed=seed,
        time_col=dataset_cfg.get("time_column", "ts"),
    )
    df_train = df[train_idx, :]
    df_val = df[val_idx, :]
    df_test = df[test_idx, :]
    y_train = df_train["is_anomaly"].to_numpy()
    y_val = df_val["is_anomaly"].to_numpy()
    y_test = df_test["is_anomaly"].to_numpy()

    logger.info(f"  [{dataset_name} s={seed} e={encoder_type}] Preprocessing (encoder={encoder_type})...")
    extractor = FeatureExtractor(config=FE_CONFIG, encoder_type=encoder_type)
    X_train = extractor.fit_transform(df_train, labels=y_train)
    X_val = extractor.transform(df_val)
    X_test = extractor.transform(df_test)
    n_features = X_train.shape[1]
    cat_cols = getattr(extractor, '_categorical_cols', [])
    logger.info(f"  [{dataset_name} s={seed} e={encoder_type}] Features: {n_features}")

    logger.info(f"  [{dataset_name} s={seed} e={encoder_type}] Training IF...")
    set_seed(seed)
    if_model = train_isolation_forest(X_train, {
        "n_estimators": 200, "max_samples": 256, "contamination": 0.01,
        "max_features": 1.0, "bootstrap": False, "n_jobs": -1, "random_state": seed,
    })

    logger.info(f"  [{dataset_name} s={seed} e={encoder_type}] Training LOF...")
    set_seed(seed)
    lof_model = train_lof(X_train, {
        "n_neighbors": 20, "novelty": True, "n_jobs": -1, "max_train_samples": 20000,
    })

    logger.info(f"  [{dataset_name} s={seed} e={encoder_type}] Training OCSVM...")
    set_seed(seed)
    ocsvm_model = train_ocsvm(X_train, {
        "kernel": "rbf", "nu": 0.01, "max_train_samples": 10000,
    })

    ae_epochs = 2 if fast else 30
    logger.info(f"  [{dataset_name} s={seed} e={encoder_type}] Training AE ({ae_epochs} epochs)...")
    set_seed(seed)
    ae_model, _, _ = train_autoencoder(
        X_train, X_val,
        hidden_dims=[128, 64, 32, 16], epochs=ae_epochs, batch_size=1024,
        learning_rate=0.001, dropout=0.1, patience=5, device="cpu",
    )
    logger.info(f"  [{dataset_name} s={seed} e={encoder_type}] AE done ({time.time()-t0:.1f}s elapsed)")

    logger.info(f"  [{dataset_name} s={seed} e={encoder_type}] Scoring...")
    scores_val = {
        "IF": if_model.score_samples(X_val),
        "LOF": lof_model.score_samples(X_val),
        "OCSVM": ocsvm_model.score_samples(X_val),
        "AE": ae_model.score_samples(X_val),
    }
    scores_test = {
        "IF": if_model.score_samples(X_test),
        "LOF": lof_model.score_samples(X_test),
        "OCSVM": ocsvm_model.score_samples(X_test),
        "AE": ae_model.score_samples(X_test),
    }

    val_combined = rank_average_scores([scores_val[d] for d in DETECTORS])
    test_combined = rank_average_scores([scores_test[d] for d in DETECTORS])
    auroc_full_val = float(roc_auc_score(y_val, val_combined))
    auroc_full_test = float(roc_auc_score(y_test, test_combined))

    logger.info(f"  [{dataset_name} s={seed} e={encoder_type}] Computing C_i...")
    c_i_results = compute_all_c_i(y_val, scores_val)
    c_i_values = {d: c_i_results[d]["c_i"] for d in DETECTORS}

    selected_epsilon, epsilon = epsilon_vrg_selection(c_i_values, auroc_full_val)
    selected_naive = naive_loo_selection(c_i_values)

    def method_auroc(selected):
        if not selected:
            return float("nan")
        subset = [scores_test[d] for d in selected]
        return float(roc_auc_score(y_test, rank_average_scores(subset)))

    ew_auroc = method_auroc(DETECTORS)
    nl_auroc = method_auroc(selected_naive)
    ev_auroc = method_auroc(selected_epsilon)

    logger.info(f"  [{dataset_name} s={seed} e={encoder_type}] Computing TestEffect...")
    testeffect = {}
    for d in DETECTORS:
        _, _, te = compute_testeffect(y_test, scores_test, d)
        testeffect[d] = float(te)

    sign_conflicts = 0
    material_conflicts = 0
    conflict_details = []
    for d in DETECTORS:
        ci = c_i_values[d]
        te = testeffect[d]
        if (ci > 0 and te > 0) or (ci < 0 and te < 0):
            sign_conflicts += 1
            if abs(ci - te) >= MATERIAL_THRESHOLD:
                material_conflicts += 1
            conflict_details.append((d, float(ci), float(te)))

    runtime = time.time() - t0
    logger.info(
        f"  [{dataset_name} s={seed} e={encoder_type}] Done: "
        f"features={n_features} EW={ew_auroc:.4f} NL={nl_auroc:.4f} "
        f"EV={ev_auroc:.4f} conflicts={sign_conflicts}/4 runtime={runtime:.1f}s"
    )

    return {
        "dataset": dataset_name,
        "seed": seed,
        "encoder": encoder_type,
        "n_features": n_features,
        "cat_cols": cat_cols,
        "auroc_full_val": auroc_full_val,
        "auroc_full_test": auroc_full_test,
        "ew_auroc_test": ew_auroc,
        "nl_auroc_test": nl_auroc,
        "ev_auroc_test": ev_auroc,
        "c_i": c_i_values,
        "test_effect": testeffect,
        "selected_naive": selected_naive,
        "selected_epsilon": selected_epsilon,
        "epsilon": epsilon,
        "sign_conflicts": sign_conflicts,
        "material_conflicts": material_conflicts,
        "conflict_details": conflict_details,
        "retained_by_nl": sorted(selected_naive),
        "retained_by_ev": sorted(selected_epsilon),
        "runtime": runtime,
    }

# ============================================================================
# MAIN ABLATION
# ============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true", help="AE trains 2 epochs only (debug)")
    args, _ = parser.parse_known_args()
    fast = args.fast

    results = []
    encoders = ["target", "hybrid", "ordinal"]
    
    dataset_order = ["UNSW-NB15", "NSL-KDD", "CICIDS2017"]

    for ds_name in dataset_order:
        ds_cfg = DATASETS[ds_name]
        for encoder_type in encoders:
            for seed in SEEDS:
                logger.info(f"Running {ds_name} seed={seed} encoder={encoder_type}")
                try:
                    r = run_single_seed(ds_name, ds_cfg, seed, encoder_type, fast=fast)
                    results.append(r)
                    logger.info(
                        f"  features={r['n_features']} "
                        f"EW={r['ew_auroc_test']:.4f} NL={r['nl_auroc_test']:.4f} "
                        f"EV={r['ev_auroc_test']:.4f} conflicts={r['sign_conflicts']}/4 "
                        f"runtime={r['runtime']:.1f}s"
                    )
                except Exception as e:
                    logger.error(f"  FAILED: {e}")
                    results.append({
                        "dataset": ds_name, "seed": seed, "encoder": encoder_type,
                        "error": str(e),
                    })
        
        # Save after each dataset (incremental)
        out_path = OUTPUT_DIR / "ablation_targetencoder_results.json"
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"Incremental save after {ds_name}: {out_path}")

    # ============================================================================
    # GENERATE ABLATION_TARGETENCODER.md
    # ============================================================================

    md = []
    md.append("# ABLATION_TARGETENCODER.md")
    md.append("")
    md.append("**Date:** 2026-09-20")
    md.append("**Purpose:** Assess impact of TargetEncoder label leakage on experimental conclusions")
    md.append("**Datasets:** CICIDS2017, UNSW-NB15, NSL-KDD")
    md.append("**Seeds:** 42, 43, 44")
    md.append("**Encoders:** TargetEncoder (leaked baseline), Hybrid (primary), Ordinal (robustness)")
    md.append("**Material threshold (provisional):** |delta_AUROC| >= 0.01")
    md.append("")
    md.append("---")
    md.append("")

    # Per-dataset tables (smaller datasets first)
    dataset_order = ["UNSW-NB15", "NSL-KDD", "CICIDS2017"]
    for ds_name in dataset_order:
        md.append(f"## {ds_name}")
        md.append("")
        ds_results = [r for r in results if r.get("dataset") == ds_name and "error" not in r]
        if not ds_results:
            md.append("No successful results.")
            continue

        # Feature dimensionality
        md.append("### Feature Dimensionality")
        md.append("")
        md.append("| Encoder | Features |")
        md.append("|---------|----------|")
        for enc in encoders:
            enc_results = [r for r in ds_results if r["encoder"] == enc]
            if enc_results:
                md.append(f"| {enc} | {enc_results[0]['n_features']} |")
        md.append("")

        # Method-level AUROC
        md.append("### Method-Level Test AUROC (mean +/- std)")
        md.append("")
        md.append("| Encoder | EW | NL | EV |")
        md.append("|---------|----|----|-----|")
        for enc in encoders:
            enc_results = [r for r in ds_results if r["encoder"] == enc]
            if enc_results:
                ews = [r["ew_auroc_test"] for r in enc_results]
                nls = [r["nl_auroc_test"] for r in enc_results]
                evs = [r["ev_auroc_test"] for r in enc_results]
                md.append(
                    f"| {enc} "
                    f"| {np.mean(ews):.4f} +/- {np.std(ews):.4f} "
                    f"| {np.mean(nls):.4f} +/- {np.std(nls):.4f} "
                    f"| {np.mean(evs):.4f} +/- {np.std(evs):.4f} |"
                )
        md.append("")

        # Sign-conflict rate
        md.append("### Sign-Conflict Rate (out of 4 detectors x 3 seeds = 12 detector-runs)")
        md.append("")
        md.append("| Encoder | Sign Conflicts | Material Conflicts (>= 0.01) |")
        md.append("|---------|---------------|------------------------------|")
        for enc in encoders:
            enc_results = [r for r in ds_results if r["encoder"] == enc]
            if enc_results:
                total_sc = sum(r["sign_conflicts"] for r in enc_results)
                total_mc = sum(r["material_conflicts"] for r in enc_results)
                n_det_runs = len(enc_results) * 4
                md.append(f"| {enc} | {total_sc}/{n_det_runs} ({100*total_sc/n_det_runs:.1f}%) | {total_mc}/{n_det_runs} ({100*total_mc/n_det_runs:.1f}%) |")
        md.append("")

        # Detector retention (epsilon-VRG)
        md.append("### Detector Retention Under Epsilon-VRG (per seed)")
        md.append("")
        md.append("| Encoder | Seed | Retained | Excluded |")
        md.append("|---------|------|----------|----------|")
        for enc in encoders:
            enc_results = [r for r in ds_results if r["encoder"] == enc]
            for r in enc_results:
                md.append(f"| {enc} | {r['seed']} | {r['selected_epsilon']} | {[d for d in DETECTORS if d not in r['selected_epsilon']]} |")
        md.append("")

        # C_i comparison
        md.append("### LOO Contribution (C_i) Comparison")
        md.append("")
        md.append("| Encoder | Detector | Mean C_i | Mean TestEffect | Sign Conflict? |")
        md.append("|---------|----------|----------|-----------------|---------------|")
        for enc in encoders:
            enc_results = [r for r in ds_results if r["encoder"] == enc]
            for d in DETECTORS:
                cis = [r["c_i"][d] for r in enc_results]
                tes = [r["test_effect"][d] for r in enc_results]
                mean_ci = np.mean(cis)
                mean_te = np.mean(tes)
                sc = "YES" if (mean_ci > 0 and mean_te > 0) or (mean_ci < 0 and mean_te < 0) else "NO"
                md.append(f"| {enc} | {d} | {mean_ci:+.4f} | {mean_te:+.4f} | {sc} |")
        md.append("")

        # Runtime
        md.append("### Runtime")
        md.append("")
        md.append("| Encoder | Mean Runtime (s) |")
        md.append("|---------|------------------|")
        for enc in encoders:
            enc_results = [r for r in ds_results if r["encoder"] == enc]
            if enc_results:
                mean_rt = np.mean([r["runtime"] for r in enc_results])
                md.append(f"| {enc} | {mean_rt:.1f} |")
        md.append("")
        md.append("---")
        md.append("")

    # Overall verdict
    md.append("## Overall Verdict")
    md.append("")
    md.append("### Does relative ranking of EW/NL/EV survive the encoder change?")
    md.append("")

    for ds_name in dataset_order:
        ds_results = [r for r in results if r.get("dataset") == ds_name and "error" not in r]
        if not ds_results:
            continue

        target_results = [r for r in ds_results if r["encoder"] == "target"]
        hybrid_results = [r for r in ds_results if r["encoder"] == "hybrid"]

        if target_results and hybrid_results:
            t_ew = np.mean([r["ew_auroc_test"] for r in target_results])
            t_nl = np.mean([r["nl_auroc_test"] for r in target_results])
            t_ev = np.mean([r["ev_auroc_test"] for r in target_results])
            h_ew = np.mean([r["ew_auroc_test"] for r in hybrid_results])
            h_nl = np.mean([r["nl_auroc_test"] for r in hybrid_results])
            h_ev = np.mean([r["ev_auroc_test"] for r in hybrid_results])

            t_ranking = sorted(["EW", "NL", "EV"], key=lambda m: {"EW": t_ew, "NL": t_nl, "EV": t_ev}[m], reverse=True)
            h_ranking = sorted(["EW", "NL", "EV"], key=lambda m: {"EW": h_ew, "NL": h_nl, "EV": h_ev}[m], reverse=True)

            md.append(f"**{ds_name}:**")
            md.append(f"- TargetEncoder ranking: {' > '.join(t_ranking)}")
            md.append(f"- Hybrid ranking: {' > '.join(h_ranking)}")
            md.append(f"- Ranking preserved: **{'YES' if t_ranking == h_ranking else 'NO'}**")
            md.append("")

    # Sign-conflict rate comparison
    md.append("### Sign-conflict rate comparison")
    md.append("")
    md.append("| Dataset | TargetEncoder | Hybrid | Ordinal |")
    md.append("|---------|---------------|--------|---------|")
    for ds_name in dataset_order:
        ds_results = [r for r in results if r.get("dataset") == ds_name and "error" not in r]
        row = f"| {ds_name} |"
        for enc in encoders:
            enc_results = [r for r in ds_results if r["encoder"] == enc]
            if enc_results:
                total_sc = sum(r["sign_conflicts"] for r in enc_results)
                n_det_runs = len(enc_results) * 4
                row += f" {total_sc}/{n_det_runs} ({100*total_sc/n_det_runs:.1f}%) |"
            else:
                row += " N/A |"
        md.append(row)
    md.append("")

    md.append("### One-paragraph verdict")
    md.append("")
    md.append("(To be filled after results are examined)")
    md.append("")

    md.append("---")
    md.append("")
    md.append("## Files")
    md.append("")
    md.append("- `ablation_targetencoder_results.json` — raw per-seed results")
    md.append("- `ablation_targetencoder.py` — ablation script")
    md.append("- This file: `ABLATION_TARGETENCODER.md`")

    report_path = OUTPUT_DIR / "ABLATION_TARGETENCODER.md"
    with open(report_path, "w") as f:
        f.write("\n".join(md))
    logger.info(f"Report saved to {report_path}")


if __name__ == "__main__":
    main()
