"""Phase 3 orchestrator — run full multi-dataset ensemble evaluation.

For each (dataset, encoder, seed) combination:
  1. Load data
  2. Build features
  3. Fit all 6 detectors
  4. Score on val/test
  5. Evaluate ensemble (EW, NL, EV)
  6. Collect results into long-format DataFrame
  7. Run statistical analysis (K=9 paired comparisons, Holm, hierarchical,
     Kruskal-Wallis, Spearman)

H5 verification is NOT performed here. See verify_h5.py for the NSL-KDD
reversal check (EW > NL for target encoder; NL > EW for hybrid/ordinal).
"""

import os
import sys
import time
import logging
import json
from pathlib import Path

import numpy as np
import pandas as pd

from load_dataset import load_dataset, build_features, DATASET_LOADERS
from detector_factory import fit_all_detectors, score_all_detectors, get_detector_info
from ensemble_eval import evaluate_ensemble
from stats_analysis import run_statistical_analysis

logger = logging.getLogger(__name__)

# ============================================================================
# CONSTANTS
# ============================================================================

DATASETS = ["CICIDS2017", "UNSW-NB15", "NSL-KDD", "CSE-CIC-IDS2018", "TON_IoT"]
ENCODERS = ["target", "hybrid", "ordinal"]
SEEDS = list(range(42, 62))  # 20 seeds
METHODS = ["EW", "NL", "EV"]
DEFAULT_OUTDIR = "phase3_output"


# ============================================================================
# SINGLE CONFIG
# ============================================================================

def run_single_config(dataset: str, encoder: str, seed: int) -> dict:
    """Run one (dataset, encoder, seed) combination.

    Steps:
      1. Load raw splits via load_dataset(dataset, seed)
      2. Build features via build_features(df_tr, df_v, df_te, y_tr, y_v, y_te,
         encoder_type=encoder)
      3. Fit all detectors on X_train (with X_val for AE)
      4. Score on val and test
      5. Evaluate ensemble via evaluate_ensemble(scores_val, scores_test, y_val, y_test)
      6. Return dict with all metrics per method (EW, NL, EV)

    Returns dict with keys:
      dataset, encoder, seed, status,
      auroc_ew_val, auroc_ew_test,
      auroc_nl_val, auroc_nl_test, nl_selected,
      auroc_ev_val, auroc_ev_test, ev_selected, epsilon,
      c_i (dict), test_effect (dict), sign_conflict (dict), material_conflict (dict),
      standalone_auroc_val (dict),
      n_detectors (int),
      runtime_sec (float)
    """
    t0 = time.time()

    try:
        df_tr, df_v, df_te, y_tr, y_v, y_te = load_dataset(dataset, seed)
    except Exception as e:
        logger.warning(f"[{dataset}/{encoder}/seed={seed}] load failed: {e}")
        return {
            "dataset": dataset, "encoder": encoder, "seed": seed,
            "status": "load_failed",
            "error": str(e),
            "runtime_sec": time.time() - t0,
        }

    try:
        X_tr, X_v, X_te, extractor = build_features(
            df_tr, df_v, df_te, y_tr, y_v, y_te, encoder_type=encoder,
        )
    except Exception as e:
        logger.warning(f"[{dataset}/{encoder}/seed={seed}] feature extraction failed: {e}")
        return {
            "dataset": dataset, "encoder": encoder, "seed": seed,
            "status": "feature_extraction_failed",
            "error": str(e),
            "runtime_sec": time.time() - t0,
        }

    try:
        detectors = fit_all_detectors(X_tr, X_val=X_v, seed=seed)
    except RuntimeError as e:
        logger.warning(f"[{dataset}/{encoder}/seed={seed}] detector fit failed: {e}")
        return {
            "dataset": dataset, "encoder": encoder, "seed": seed,
            "status": "detector_fit_failed",
            "error": str(e),
            "runtime_sec": time.time() - t0,
        }

    scores_val = score_all_detectors(detectors, X_v)
    scores_test = score_all_detectors(detectors, X_te)

    metrics = evaluate_ensemble(scores_val, scores_test, y_v, y_te)

    return {
        "dataset": dataset,
        "encoder": encoder,
        "seed": seed,
        "status": "ok",
        **metrics,
        "n_detectors": len(detectors),
        "runtime_sec": time.time() - t0,
    }


# ============================================================================
# FULL EXPERIMENT
# ============================================================================

def run_full_experiment(
    datasets: list[str] | None = None,
    encoders: list[str] | None = None,
    seeds: list[int] | None = None,
    outdir: str = DEFAULT_OUTDIR,
    resume: bool = True,
) -> pd.DataFrame:
    """Run all (dataset, encoder, seed) configs with incremental saving.

    - If resume=True, load existing raw_results.jsonl and skip completed configs
    - Save each config result as one JSON line to raw_results.jsonl (crash-safe)
    - Skip configs whose dataset parquet is not present
    - Log progress every 10 configs with elapsed time
    - After all configs complete, consolidate raw_results.jsonl → raw_results.json
    """
    datasets = datasets or DATASETS
    encoders = encoders or ENCODERS
    seeds = seeds or SEEDS

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    jsonl_path = outdir / "raw_results.jsonl"

    # Check dataset availability
    available_datasets = []
    for ds in datasets:
        if ds in DATASET_LOADERS:
            available_datasets.append(ds)
        else:
            logger.warning(f"Dataset '{ds}' not in DATASET_LOADERS — skipping")

    # Load completed configs for resume
    completed = set()
    if resume and jsonl_path.exists():
        with jsonl_path.open() as f:
            for line in f:
                try:
                    r = json.loads(line)
                    completed.add((r["dataset"], r["encoder"], r["seed"]))
                except (json.JSONDecodeError, KeyError):
                    continue
        logger.info(f"Resuming: {len(completed)} configs already completed")

    # Build work list
    work = []
    for ds in available_datasets:
        for enc in encoders:
            for seed in seeds:
                if (ds, enc, seed) not in completed:
                    work.append((ds, enc, seed))

    logger.info(f"Starting {len(work)} configs "
                f"({len(available_datasets)} datasets × {len(encoders)} encoders × {len(seeds)} seeds)")

    t_start = time.time()
    n_done = 0
    n_failed = 0

    with jsonl_path.open("a") as f:
        for ds, enc, seed in work:
            result = run_single_config(ds, enc, seed)
            f.write(json.dumps(result, default=str) + "\n")
            f.flush()

            n_done += 1
            if result["status"] != "ok":
                n_failed += 1

            if n_done % 10 == 0:
                elapsed = time.time() - t_start
                logger.info(f"Progress: {n_done}/{len(work)} configs, "
                            f"{n_failed} failed, {elapsed:.0f}s elapsed")

    # Consolidate JSONL → JSON
    results = []
    with jsonl_path.open() as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))

    json_path = outdir / "raw_results.json"
    with json_path.open("w") as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"Experiment complete: {len(results)} total results "
                f"({n_failed} failures) in {time.time() - t_start:.0f}s")

    return pd.DataFrame(results)


# ============================================================================
# AGGREGATION
# ============================================================================

def aggregate_to_long_format(raw_results: list[dict]) -> pd.DataFrame:
    """Convert raw per-config results to long-format DataFrame for stats_analysis.

    Long format columns: dataset, encoder, seed, method, auroc_val, auroc_test, ...
    Each config yields 3 rows (one per method: EW, NL, EV).
    Only configs with status="ok" are included.
    """
    rows = []
    for r in raw_results:
        if r.get("status") != "ok":
            continue
        for method in METHODS:
            auroc_val = r.get(f"auroc_{method.lower()}_val")
            auroc_test = r.get(f"auroc_{method.lower()}_test")
            if auroc_val is None or auroc_test is None:
                continue
            row = {
                "dataset": r["dataset"],
                "encoder": r["encoder"],
                "seed": r["seed"],
                "method": method,
                "auroc_val": auroc_val,
                "auroc_test": auroc_test,
            }
            rows.append(row)

    return pd.DataFrame(rows)


# ============================================================================
# CLI
# ============================================================================

def main():
    """CLI entry point with argparse:
      --datasets       (default: all 5)
      --encoders       (default: target,hybrid,ordinal)
      --seeds          (default: 42-61)
      --outdir         (default: phase3_output)
      --no-resume      (default: resume enabled)
      --stats-only     (skip experiments, run stats on existing raw_results.json)
    """
    import argparse

    parser = argparse.ArgumentParser(description="Phase 3: multi-dataset ensemble evaluation")
    parser.add_argument("--datasets", nargs="+", default=None,
                        help=f"Datasets to run (default: {DATASETS})")
    parser.add_argument("--encoders", nargs="+", default=None,
                        help=f"Encoders to run (default: {ENCODERS})")
    parser.add_argument("--seeds", nargs="+", type=int, default=None,
                        help=f"Seeds to run (default: {SEEDS})")
    parser.add_argument("--outdir", default=DEFAULT_OUTDIR,
                        help=f"Output directory (default: {DEFAULT_OUTDIR})")
    parser.add_argument("--no-resume", action="store_true",
                        help="Disable resume from existing JSONL")
    parser.add_argument("--stats-only", action="store_true",
                        help="Skip experiments, run stats on existing raw_results.json")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    t_start = time.time()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.stats_only:
        json_path = outdir / "raw_results.json"
        if not json_path.exists():
            logger.error(f"--stats-only requested but {json_path} does not exist")
            sys.exit(1)
        with json_path.open() as f:
            raw_results = json.load(f)
        logger.info(f"Loaded {len(raw_results)} results from {json_path}")
    else:
        raw_df = run_full_experiment(
            datasets=args.datasets,
            encoders=args.encoders,
            seeds=args.seeds,
            outdir=args.outdir,
            resume=not args.no_resume,
        )
        raw_results = raw_df.to_dict("records")

        # Save CSV
        raw_df.to_csv(outdir / "raw_results.csv", index=False)

    # Aggregate to long format
    long_df = aggregate_to_long_format(raw_results)
    long_df.to_csv(outdir / "aggregated_long.csv", index=False)
    logger.info(f"Long format: {len(long_df)} rows")

    # Run statistical analysis
    if len(long_df) > 0:
        stats_summary = run_statistical_analysis(long_df, str(outdir))
        logger.info("Statistical analysis complete")
    else:
        logger.warning("No valid results — skipping statistical analysis")

    # Save run metadata
    elapsed = time.time() - t_start
    total_configs = len(raw_results)
    ok_configs = sum(1 for r in raw_results if r.get("status") == "ok")
    failed_configs = total_configs - ok_configs

    metadata = {
        "total_configs": total_configs,
        "ok_configs": ok_configs,
        "failed_configs": failed_configs,
        "total_runtime_sec": elapsed,
        "datasets_requested": args.datasets or DATASETS,
        "encoders_requested": args.encoders or ENCODERS,
        "seeds_requested": args.seeds or SEEDS,
        "detector_info": get_detector_info(),
        "python_version": sys.version,
    }
    with (outdir / "run_metadata.json").open("w") as f:
        json.dump(metadata, f, indent=2, default=str)

    logger.info(f"Done: {ok_configs}/{total_configs} configs in {elapsed:.0f}s")


if __name__ == "__main__":
    main()
