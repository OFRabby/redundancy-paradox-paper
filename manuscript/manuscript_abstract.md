# Abstract

**Word count: 250**

Network intrusion detection systems rely on unsupervised anomaly detection ensembles, and leave-one-out (LOO) contribution is the dominant detector-selection criterion. This paper systematically validates LOO's diagnostic claims on controlled ground truth and real NIDS data, with three findings that question its reliability.

First, we construct a synthetic benchmark with known detector roles (beneficial, redundant, harmful, noisy) and measure sign-conflict between validation LOO contribution and held-out effect across 2,000 runs. Sign-conflict occurs at 48%, independent of detector correlation (H3 rejected), and LOO misses 67% of redundant detectors (R recall 0.327) while detecting harmful ones reliably (H recall 1.000). Second, we evaluate on 3 NIDS datasets with 3 categorical encoders and 20 seeds (180 configs). LOO's directional claims are seed-fragile: the encoder-dependent reversal reported in prior work does not replicate, and direction flips across seeds. Third, we test 4 uncertainty-aware alternatives to fixed epsilon-VRG; fixed epsilon-VRG remains best balanced (66% false-pruning reduction, 0 harmful retention), while bootstrap-CI incurs 47% higher signal loss.

Our findings reframe LOO as a diagnostic signal, not a selection rule. We provide practical guidelines for NIDS ensemble construction, public code, and reproducibility artifacts.
