# Response-to-Reviewer Templates

Anticipated reviewer concerns with data-driven rebuttals.

---

## 1. "Why is sign-conflict rate so high (48%)?"

**Response**: The 48% sign-conflict rate is not a methodological flaw—it is a ground-truth-validated finding. In our synthetic benchmark (Phase 2, 2,000 runs), sign-conflict between LOO validation signal and held-out test effect occurs at 48% regardless of inter-detector correlation (Spearman ρ ≈ 0). This is a universal finite-sample noise floor arising from rank averaging: LOO measures rank-distribution changes, not absolute score changes, making it structurally unable to distinguish redundancy from harm.

**Evidence**: Phase 2, Table 2 (confusion matrix); Figure 2 (flat rate across ρ). Phase 3 confirms on real data: 180 configurations across 3 datasets show consistent sign-conflict rates.

**Limitation acknowledged**: This rate may differ for other ensemble construction methods (e.g., stacking, boosting) or other evaluation metrics (e.g., AUPRC instead of AUROC).

---

## 2. "Is the synthetic benchmark realistic?"

**Response**: The synthetic benchmark is designed to isolate LOO's diagnostic mechanism, not to replicate any specific dataset. Scores are generated directly to control detector roles (beneficial, redundant, harmful, noisy), eliminating confounding from preprocessing, feature engineering, or detector hyperparameters.

**Validation**: Phase 3 (real-data evaluation) confirms the synthetic findings:
- R recall 0.327 (synthetic) vs. NL pruning patterns (real) are qualitatively consistent
- Sign-conflict rates are comparable (~48% synthetic, ~45-52% real across datasets)
- Epsilon-VRG effectiveness is confirmed in both settings

**We do not claim**: The synthetic benchmark replicates real NIDS data. We claim it isolates a mechanism (rank-distribution sensitivity) that is universal to rank-averaging ensembles.

---

## 3. "What about Shapley values as an alternative?"

**Response**: Shapley values provide a theoretically superior allocation of ensemble contribution by considering all possible subsets. However:

1. **Practical adoption**: LOO is the dominant heuristic in NIDS literature (k evaluations vs. 2^k for Shapley). Our work reveals LOO's limitations regardless of whether Shapley values are theoretically superior.

2. **Computational cost**: For k=6 detectors, Shapley requires 64 evaluations per instance versus 6 for LOO. For real-time NIDS processing millions of records, this 10× cost increase is non-trivial.

3. **Diagnostic contribution**: Our work provides a diagnostic framework (sign-conflict monitoring, epsilon-VRG guardrail) that applies to any contribution-based pruning method, including Shapley values if adopted.

**Future work**: Applying our diagnostic framework to Shapley-based selection is an open direction.

---

## 4. "Does the leakage finding generalize?"

**Response**: The leakage finding (Phase 1, Finding #7) is specific to TargetEncoder and does not generalize to all categorical encoding methods. Our key contributions:

1. **Structural fix**: We identify the mechanism (label information propagating through feature engineering to unsupervised detectors) and provide an audit protocol.

2. **Encoder independence**: Phase 3 confirms that the sign-conflict and redundancy-paradox findings hold across all 3 encoders (target, hybrid, ordinal), independent of leakage.

3. **Phase 1 non-replication**: We explicitly documented that Phase 1's encoder-dependent EW > NL reversal on NSL-KDD does not replicate across 20 seeds (4/20 vs 16/20), reframing it as a seed-fragility finding.

**We do not claim**: Leakage is universal. We claim it is a risk that practitioners should audit for.

---

## 5. "Are the effect sizes practically significant?"

**Response**: The primary practical finding is the 66% reduction in redundancy false-pruning (R-fp) by epsilon-VRG (Phase 2, H4). This is material because:

- **Quantitative**: R-fp reduced from 0.190 to 0.065 with zero harmful detector retention
- **Bootstrap CI**: 95% CI for R-fp reduction is [−0.152, −0.098] (exclusive of zero)
- **Holm-corrected**: 5 of 7 comparisons survive Holm correction (K=7 post-hoc)
- **Practical impact**: For a 6-detector NIDS ensemble, epsilon-VRG prevents pruning ~1 redundant detector that would otherwise be incorrectly excluded

**Limitation**: Effect sizes are dataset-dependent. CICIDS2017 shows large differences (EW vs NL: +0.0875 AUROC); NSL-KDD shows small differences (NL vs EV: −0.0233 AUROC). We do not claim universal practical significance.

---

## 6. "Why did Phase 1 reversal not replicate?"

**Response**: We explicitly documented this as a fragility finding (Phase 3, Step 4f). The non-replication has two root causes:

1. **Detector set effect**: Phase 1 used 4 detectors (IF, LOF, OCSVM, AE). Phase 3 used 6 detectors (adding ECOD, HBOS). The additional detectors degrade EW performance on NSL-KDD, changing the sign of EW − NL.

2. **Seed fragility**: Phase 1 used a single seed (seed=42). Across 20 seeds, EW > NL in only 4/20 (20%). Phase 1's result was not representative.

**This is a contribution**: We demonstrate that single-seed evaluations can produce misleading conclusions, and provide a 20-seed protocol as a corrective.

---

## 7. "How does this compare to existing diagnostic methods?"

**Response**: To our knowledge, no prior work systematically validates LOO's diagnostic accuracy against known ground truth. Existing NIDS studies:

- Apply LOO pruning without measuring held-out effect
- Report per-dataset results without cross-dataset validation
- Do not characterize sign-conflict rates or redundancy paradox

**Our contribution fills this gap**:
- First ground-truth validation of LOO (Phase 2)
- First multi-dataset evaluation with proper train/val/test separation (Phase 3)
- First characterization of seed-fragility effects on LOO decisions

**We do not claim**: Our diagnostic is superior to all existing methods. We claim it is the first validated diagnostic for LOO-based pruning.

---

## 8. "Reproducibility?"

**Response**: All artifacts are publicly available:

| Artifact | URL | Status |
|----------|-----|--------|
| Code | https://github.com/OFRabby/redundancy-paradox-paper | PUBLIC, 36 files |
| Data | https://www.kaggle.com/datasets/ofrabby/anomaly-detector-data | PUBLIC |
| Interactive notebook | https://www.kaggle.com/code/ofrabby/redundancy-paradox-phase3 | RUNNING |
| Code Ocean capsule | https://codeocean.com/capsule/PLACEHOLDER | PENDING |

**Reproduction steps**:
```bash
git clone https://github.com/OFRabby/redundancy-paradox-paper
cd redundancy-paradox-paper
pip install -r requirements.txt
kaggle datasets download -d ofrabby/anomaly-detector-data -p data/ --unzip
python experiments/phase3/run_phase3.py --datasets CICIDS2017,UNSW-NB15,NSL-KDD --seeds 42-61
python -m pytest tests/ -v
```

**Environment**: Python 3.11, PyTorch 2.x (GPU), scikit-learn 1.3+, PyOD 3.6.5.

---

## 9. "What is the methodological contribution beyond the diagnostic? Epsilon-VRG is just a tolerance threshold."

**Response:** We acknowledge that epsilon-VRG is simple. Section 6b presents our methodological exploration: we implement and evaluate 4 uncertainty-aware alternatives (bootstrap-CI on C_i, sign+materiality rule, inner-validation tolerance selection, Bayesian shrinkage). On Phase 2 ground truth, fixed epsilon-VRG is the best balanced method: it reduces redundancy false-pruning by 66% while retaining 0 harmful detectors. Bootstrap-CI is more conservative (retains 1.77 vs 3.19 detectors on average) but incurs 47% higher signal loss. We report this null result transparently. The paper's primary contribution is diagnostic — validating and falsifying LOO's diagnostic claims on controlled ground truth. The uncertainty-aware exploration demonstrates that the simple method is empirically justified, not arbitrary.
