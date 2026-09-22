# Claim-Evidence Matrix

Critical for MLWA rigor. Every paper claim mapped to actual evidence.

---

## Claims with Evidence

| # | Claim | Location | Evidence Source | Dataset | N | Stat Test | Effect Size | CI | p-value | Supported? | Caveat |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | Sign-conflict at 48% on synthetic | Abstract, Sec 4 | Phase 2 confusion matrix | Synthetic | 2000 runs | descriptive | 48% | ±2% | — | ✅ | Universal finite-sample noise floor |
| 2 | Sign-conflict flat across rho (H3 rejected) | Sec 4.4 | Phase 2 sign_conflict_by_rho.csv | Synthetic | 2000 | Spearman | ρ≈0 | — | >0.05 | ✅ | Correlation-independent |
| 3 | LOO detects harmful (H recall 1.0) | Abstract, Sec 4 | Phase 2 confusion matrix | Synthetic | 225 H detectors | descriptive | 1.000 | [0.98,1.0] | — | ✅ | n limited by H verification threshold |
| 4 | LOO misses 67% redundant (R recall 0.327) | Abstract, Sec 4 | Phase 2 confusion matrix | Synthetic | 1910 R | descriptive | 0.327 | [0.30,0.35] | — | ✅ | |
| 5 | Epsilon-VRG reduces R-fp by 66% | Sec 4.4, 6b | Phase 2 H4 | Synthetic | 2000 | Wilcoxon | d=−0.125 | [−0.152,−0.098] | <0.01 | ✅ | |
| 6 | Phase 1 encoder reversal does not replicate | Sec 5, 6 | Phase 3 seed sweep | NSL-KDD | 20 seeds | descriptive | direction flips 4/20 vs 16/20 | — | — | ✅ (falsified) | honest reporting |
| 7 | Fragility rate > 0.25 across datasets | Sec 6 | Phase 3 aggregated_long.csv | 3 NIDS | 180 runs | descriptive | TBD | — | — | ⏳ (Kaggle) | waiting for Phase 3 completion |
| 8 | 4 uncertainty methods don't beat epsilon-VRG | Sec 6b | Phase 4 comparison.csv | Synthetic | 150 runs | descriptive | bootstrap=1.77 vs ε-VRG=3.19 retained | — | — | ✅ | null result, honestly reported |
| 9 | TargetEncoder leakage propagates to AE | Sec 3 | Phase 1 ablation | CICIDS2017, NSL-KDD | 20 seeds | descriptive | encoder-dependent AUROC shift up to +0.046 | — | — | ✅ | mechanism documented |
| 10 | Epsilon-VRG retains 0 harmful detectors | Sec 4.4 | Phase 2 H4 | Synthetic | 2000 | descriptive | H retention = 0.000 | — | — | ✅ | |
| 11 | Bootstrap-CI incurs 47% higher signal loss | Sec 6b | Phase 4 comparison.csv | Synthetic | 150 runs | descriptive | 0.072 vs 0.049 | — | — | ✅ | |

---

## Not Claimed (Falsified or Explicitly Excluded)

| # | What is NOT claimed | Why | Evidence |
|---|---|---|---|
| F1 | "Epsilon-VRG universally improves AUROC" | Epsilon-VRG reduces false-pruning, not AUROC itself | Phase 2: effect on pruning metrics, not AUROC |
| F2 | "LOO can distinguish redundancy from benefit" | Falsified: R recall 0.327 | Phase 2 confusion matrix |
| F3 | "Encoder choice causes directional reversal" | Falsified: does not replicate across seeds | Phase 3 seed sweep (4/20 vs 16/20) |
| F4 | "Sign-conflict depends on ensemble correlation" | Falsified: H3 rejected | Phase 2: Spearman ρ≈0 |
| F5 | "Single-seed LOO evaluation is reliable" | Falsified: seed-fragility meta-finding | Phase 3: direction flips across seeds |

---

## Pending (Awaiting Kaggle)

| # | Claim | Evidence Needed | Status |
|---|---|---|---|
| P1 | Fragility rate varies by dataset and encoder | Phase 3 aggregated_long.csv | ⏳ Kaggle RUNNING |
| P2 | Holm-corrected comparisons survive correction | Phase 3 stats_analysis.py output | ⏳ Kaggle RUNNING |
| P3 | Hierarchical model random-intercept variance is substantial | Phase 3 hierarchical model output | ⏳ Kaggle RUNNING |

---

## Evidence Provenance

| Evidence Source | Artifact Path | Status |
|---|---|---|
| Phase 2 confusion matrix | `research/kaggle_output/real_data_validation/PHASE2_RESULTS.md` | ✅ VERIFIED |
| Phase 2 sign-conflict by rho | `research/kaggle_output/real_data_validation/kaggle_output_phase2_v3/phase2_output/sign_conflict_by_rho.csv` | ✅ VERIFIED |
| Phase 2 epsilon-VRG | `research/kaggle_output/real_data_validation/PHASE2_RESULTS.md` §6 | ✅ VERIFIED |
| Phase 4 comparison | `research/kaggle_output/real_data_validation/phase4_output/comparison.csv` | ✅ VERIFIED |
| Phase 3 results | `research/kaggle_output/real_data_validation/PHASE3_RESULTS_TEMPLATE.md` | ⏳ Kaggle RUNNING |
| Figure 2 | `redundancy-paradox-paper/figures/output/fig2_signconflict_vs_rho.png` | ✅ GENERATED |
| Graphical abstract | `redundancy-paradox-paper/figures/output/graphical_abstract.png` | ✅ GENERATED |
