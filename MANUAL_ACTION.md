# Manual Action Required — Code Ocean Setup

Code Ocean is required for the Elsevier Reproducibility Badge, which boosts MLWA acceptance chance.

## Steps

1. **Sign up**: Go to https://codeocean.com/sign-up
   - Use institutional email if possible, else personal
   - Verify email

2. **Create capsule**:
   - Name: `Redundancy Paradox - Reproducibility Package`
   - Type: Computational Reproducibility

3. **Configure environment**:
   - Base: Python 3.11
   - Install: `pip install -r /code/requirements.txt`

4. **Mount data**:
   - Source: https://www.kaggle.com/datasets/ofrabby/anomaly-detector-data
   - Path in capsule: `/data/`

5. **Upload code**:
   ```
   git clone https://github.com/OFRabby/redundancy-paradox-paper /code
   ```

6. **Create run script** (`/code/run.sh`):
   ```bash
   #!/bin/bash
   set -e
   cd /code/redundancy-paradox-paper
   pip install -r requirements.txt
   python experiments/phase3/run_phase3.py \
       --datasets CICIDS2017,UNSW-NB15,NSL-KDD \
       --encoders target,hybrid,ordinal \
       --seeds 42-61 \
       --outdir /results/phase3
   python -m pytest tests/ -v
   ```

7. **Test run** to verify reproducibility

8. **Request badge**: Go to capsule settings → Request Elsevier Reproducibility Badge

## Expected Outcome

- Capsule URL: `https://codeocean.com/capsule/<id>`
- Badge: Elsevier Reproducibility Badge (pending test run)
- manuscript/README.md placeholder to be updated with actual capsule URL

## Blockers

- Requires browser interaction (cannot be automated by agent)
- Test run may take 8-14 hours (Phase 3 full experiment)
- Badge request requires successful test run
