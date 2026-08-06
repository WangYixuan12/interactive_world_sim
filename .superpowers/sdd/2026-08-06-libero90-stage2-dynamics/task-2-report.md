# Task 2 Report: Resumable LIBERO-90 Stage 2 launcher

## Result

Added `jobs/train_iws_libero90_stage2.sbatch` for Stage 2 latent-dynamics
training. It requests one GPU, eight CPUs, 64 GB RAM, and the CIAI GPU
partition/QoS for 12 hours. The launcher preserves the Stage 1 fail-fast
environment checks and adds validation for the required Stage 1 autoencoder
checkpoint plus an optional Stage 2 resume checkpoint.

The launcher uses the approved LIBERO-90 Stage 2 overrides: horizon 10,
validation horizon 200, batch size 4, 1,000,005 steps, bounded batch-one
validation, 30,000-step validation cadence, 10,000-step checkpoint cadence,
uniform loss weighting, terminal-only sampling, disabled validation rendering,
and the verified Stage 1 checkpoint through `algorithm.load_ae`.

## TDD evidence

1. Added `tests/test_libero90_stage2_job.py` before the launcher.
2. Ran the focused test before implementation:

   ```bash
   /home/youssef.ghallab/miniforge3/envs/iws/bin/python -m pytest tests/test_libero90_stage2_job.py -q
   ```

   Result: `2 failed in 0.09s`, because the Stage 2 launcher did not exist.
3. Added the minimal dedicated launcher and reran the focused test.

   Result: `2 passed in 0.02s`.

## Verification

```bash
/home/youssef.ghallab/miniforge3/envs/iws/bin/python -m pytest \
  tests/test_libero90_dataset.py tests/test_libero90_stage1_job.py \
  tests/test_libero90_stage2_job.py -q
```

Result: `9 passed in 6.35s`.

```bash
bash -n jobs/train_iws_libero90_stage2.sbatch
git diff --check
```

Result: both commands exited successfully with no output.

```bash
IWS_ROOT="$PWD" sbatch --test-only jobs/train_iws_libero90_stage2.sbatch
```

Result:

```text
sbatch: Job 153108 to start at 2026-08-06T21:48:28 a using 8 processors on nodes gpu-51 in partition cscc-gpu-p
```

This is a non-submitting Slurm scheduling projection.

## Self-review

Reviewed the dedicated launcher against the Task 2 contract and the Stage 1
guard order. The required Stage 1 checkpoint is rejected before Conda
activation; the optional resume checkpoint, 90-file dataset gate, Conda
activation under `nounset`, CUDA check, and conditional `load=` append all
match the established launcher pattern. No review findings remain.
