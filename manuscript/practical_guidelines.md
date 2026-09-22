# Practical Guidelines for Practitioners

The following guidelines are derived from our controlled ground-truth study (Phase 2, 2,000 runs) and real-data multi-dataset evaluation (Phase 3, 180 configurations). They are intended as a practical screening framework for practitioners building NIDS ensembles, not as universal rules.

| Scenario | Recommended Action | Evidence |
|----------|-------------------|----------|
| Ensembles with unknown detector quality | Run LOO diagnostic first as a screening step | Phase 3, 180 configs across 3 datasets |
| All detectors appear useful (sign-conflict rate < 10%) | Equal-weight ensemble is safest | Phase 3, CICIDS2017 pattern |
| Sign-conflict rate > 15% | Apply epsilon-VRG or uncertainty-aware selection before pruning | Phase 2 H4, R-fp reduced 66% |
| Low-dimensional data (< 20 features) | Naive LOO pruning tends to help | Phase 3, CICIDS2017 results |
| High-dimensional data | Dataset-dependent; no universal rule applies | Phase 3, UNSW-NB15 vs NSL-KDD contrast |
| Categorical encoding present | Audit for label leakage before LOO selection | Phase 1 Finding #7, Phase 3 Finding #7 |
| NSL-KDD specifically | Do not rely on encoder-independent claims; test each encoder separately | Phase 3 seed-fragility analysis |
| Reproducibility requirements | Use at minimum 10 seeds; 20 seeds preferred | Phase 3, seed-fragility finding |

**Key principle**: LOO is a diagnostic signal, not a selection rule. Use it to trigger investigation, not automated pruning. When sign-conflict rates exceed 15%, the ensemble contains detectors whose contribution direction is uncertain, and epsilon-VRG or alternative guardrails should be applied before making retention/exclusion decisions.

**Limitations of these guidelines**: Derived from 3 NIDS datasets and 6 detector families. May not generalize to other domains (e.g., image anomaly detection) or detector architectures. Practitioners should validate on their specific data before adopting these recommendations.
