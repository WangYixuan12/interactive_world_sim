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
sbatch: Job 153117 to start at 2026-08-06T21:52:42 a using 8 processors on nodes gpu-51 in partition cscc-gpu-p
```

Job 153108 was the implementer's recorded initial command output. Job 153112
appeared only in the short return and was an erroneous, unverified
transcription. This controller-confirmed Job 153117 exact rerun is
authoritative for the acceptance claim; it is a non-submitting Slurm
scheduling projection.

## Self-review

Reviewed the dedicated launcher against the Task 2 contract and the Stage 1
guard order. The required Stage 1 checkpoint is rejected before Conda
activation; the optional resume checkpoint, 90-file dataset gate, Conda
activation under `nounset`, CUDA check, and conditional `load=` append all
match the established launcher pattern. No review findings remain.

## Checkpoint override quoting fix

Smoke job 153124 exited in Hydra before training with `mismatched input '='
expecting <EOF>`. The launcher passed checkpoint paths such as
`epoch=0-step=415000.ckpt` as unquoted Hydra values. Shell double quotes kept
each override in one argv element but did not quote the value for Hydra's
override grammar.

### Red/green evidence

1. Added a regression test using Hydra's `OverridesParser` with a checkpoint
   path containing `=`. It asserts the exact launcher argv forms
   `algorithm.load_ae='$STAGE1_CKPT'` and `load='$RESUME_CKPT'`, then verifies
   both parse back to the original path.
2. Before the launcher change, the focused suite failed as expected:

   ```text
   2 failed, 1 passed in 0.09s
   ```

   The source contained no quotes at the Hydra value layer.
3. Changed only the Stage 1 autoencoder and optional resume `load=` overrides
   to put single quotes around their Hydra values.
4. After the change:

   ```bash
   /home/youssef.ghallab/miniforge3/envs/iws/bin/python -m pytest tests/test_libero90_stage2_job.py -q
   /home/youssef.ghallab/miniforge3/envs/iws/bin/python -m pytest \
     tests/test_libero90_dataset.py tests/test_libero90_stage1_job.py \
     tests/test_libero90_stage2_job.py -q
   bash -n jobs/train_iws_libero90_stage2.sbatch
   git diff --check
   ```

   Results: `3 passed in 0.03s` focused and `10 passed in 6.25s` combined;
   Bash syntax and whitespace checks exited successfully. No additional Slurm
   job was submitted.
