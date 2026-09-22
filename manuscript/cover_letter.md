# Cover Letter

Dear Editor,

We submit "Fragile Diagnostics: Leave-One-Out Ensemble Selection is Seed-Sensitive, Detector-Dependent, and Universally Unreliable" as a Research Methodology paper for consideration.

## Fit with MLWA Scope

This paper demonstrates an improvement to the way existing machine learning research is conducted. We identify a previously unreported failure mode in leave-one-out (LOO) ensemble selection---a widely used heuristic for constructing anomaly detection ensembles---and provide a diagnostic method, practical guidelines, and a reproducibility package. Our work directly addresses the reliability of a fundamental ML methodology, making it well-suited for a venue focused on methodological improvement. This paper's null-result reporting (Phase 4) reflects our commitment to methodological rigor: we test principled alternatives and honestly report when simple heuristics outperform them.

## Four Contributions

1. **Ground-truth diagnostic validation**: Through a controlled synthetic benchmark with 2,000 ensemble configurations where detector roles are known, we demonstrate that LOO conflates redundancy with harm. LOO detects harmful detectors with perfect recall (1.000) but misclassifies 67% of redundant detectors as beneficial (recall 0.327). Sign-conflict between validation LOO and held-out test effect occurs at 48%, independent of ensemble correlation structure (Spearman rho approximately 0).

2. **Mechanism identification**: We identify the root cause---LOO cannot distinguish "removes useful information" from "changes rank distribution"---and propose epsilon-VRG, a guardrail that reduces redundancy false-pruning by 66% without retaining harmful detectors.

3. **Seed-fragility meta-finding**: We show that LOO's directional claims are inherently seed-sensitive. Phase 1's encoder-dependent equal-weight superiority over naive-LOO on NSL-KDD does not replicate across 20 seeds (observed in only 4 of 20), demonstrating that single-seed evaluations can produce misleading conclusions.

4. **Methodological rigor**: We test 4 uncertainty-aware alternatives (bootstrap-CI on C_i, sign+materiality rule, inner-validation tolerance selection, Bayesian shrinkage) against fixed epsilon-VRG. Fixed epsilon-VRG remains the best balanced deployable method; bootstrap-CI is more conservative but incurs 47% higher signal loss. We report this null result transparently.

## Reproducibility

All code, data, and experimental configurations are publicly available:

- **Code**: https://github.com/OFRabby/redundancy-paradox-paper
- **Zenodo DOI**: https://doi.org/10.5281/zenodo.22902922
- **Data**: https://www.kaggle.com/datasets/ofrabby/anomaly-detector-data
- **Interactive notebook**: https://www.kaggle.com/code/ofrabby/redundancy-paradox-phase3

## Prior Submission Note

An earlier version of this work was desk-rejected by Big Data Mining and Analytics due to a technical PDF error. The scientific content has since been substantially expanded with:
- A controlled synthetic ground-truth benchmark (Phase 2)
- A multi-dataset evaluation with 3 encoders and 20 seeds (Phase 3)
- Seed-fragility analysis demonstrating non-replication of prior claims
- Practical guidelines for practitioners

## Suggested Reviewers

1. Dr. Charu Aggarwal (IBM Research)---outlier detection, ensemble methods
2. Dr. Zhi-Hua Zhou (Nanjing University)---ensemble learning, isolation forest
3. Dr. Kai Ming Ting (Monash University)---outlier detection, ensemble methods
4. Dr. Guansong Pang (University of Adelaide)---deep anomaly detection, survey
5. Dr. Nour Moustafa (UNSW Canberra)---network intrusion detection, UNSW-NB15 dataset

Sincerely,

Md. Omar Faruk Rabby
Department of Computer Science and Engineering
Southeast University, Dhaka 1208, Bangladesh
