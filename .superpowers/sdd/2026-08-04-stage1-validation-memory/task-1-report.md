# Task 1 Report: Stage 1 validation memory

## Result

Changed the launcher override from `experiment.validation.limit_batch=1.0` to
`experiment.validation.limit_batch=1`. Lightning will now run one 10-sample
validation batch every 6,000 training steps rather than treating `1.0` as the
complete validation set.

## TDD evidence

Added `assert "experiment.validation.limit_batch=1\\n" in script` first.
The prescribed focused pytest command then failed as expected because the
launcher still contained `1.0`. The minimal production change replaced that
value with integer `1`.

## Verification

- `/home/youssef.ghallab/miniforge3/envs/iws/bin/python -m pytest tests/test_libero90_stage1_job.py -q`: `2 passed in 0.02s`.
- `bash -n jobs/train_iws_libero90_stage1.sbatch`: exited successfully.
- `git diff --check`: exited successfully.
- Initial `sbatch --test-only jobs/train_iws_libero90_stage1.sbatch`: accepted Job 147439 as a non-submitting scheduling projection on `gpu-05`.
- A final repeat dry run was blocked by the sandbox with `Error creating slurm stream socket: Operation not permitted` and `Unable to contact slurm controller`; the controller must rerun this external check.

## Review

The production diff is one launcher token. The trailing-newline assertion
prevents `1.0` from satisfying the contract. Validation batch size (10),
cadence (6,000), and checkpoint cadence (5,000) are unchanged. Independent
self-review found no issues.
