# Pre-Submission Inquiry — MLWA

**DO NOT SEND YET.** Draft for review.

---

**Subject:** Pre-submission inquiry — LOO ensemble selection diagnostic

Dear Prof. Lin,

I am writing to inquire about the suitability of a manuscript for Machine Learning with Applications before formal submission.

**Title:** "Fragile Diagnostics: Leave-One-Out Ensemble Selection is Seed-Sensitive, Detector-Dependent, and Universally Unreliable"

The paper demonstrates an improvement to the way existing ML research is conducted: we systematically validate leave-one-out (LOO) ensemble selection — the dominant criterion in anomaly detection ensembles — on controlled ground truth with 2,000 synthetic runs. We find:

- Sign-conflict between LOO and held-out effect at 48% (universal, independent of detector correlation)
- LOO misses 67% of redundant detectors (R recall 0.327)
- LOO's directional claims are seed-fragile; a prior encoder-reversal claim does not replicate
- Fixed epsilon-VRG is empirically best among 4 uncertainty-aware alternatives

We provide practical guidelines, public code (GitHub), Kaggle notebooks, and a Code Ocean capsule (reproducibility badge pending). The paper aligns with MLWA's stated focus on practical ML contributions and value for practitioners.

Would this fit MLWA's scope? I am happy to provide the full manuscript for preliminary review.

Sincerely,
Md. Omar Faruk Rabby
Department of CSE, Southeast University, Dhaka, Bangladesh
ofrabby07@gmail.com
