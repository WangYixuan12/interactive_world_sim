# Task 1 Report: Stage 1 checkpoint interval

## Result

Changed the Stage 1 launcher to save checkpoints every 5,000 optimizer steps,
before the first validation boundary at step 6,000. The launcher now passes
`experiment.training.checkpointing.every_n_train_steps=5000` to Hydra.

## TDD evidence

1. Changed the existing launcher expectation from `10000` to `5000`.
2. Ran the focused test before the implementation:

   ```bash
   /home/youssef.ghallab/miniforge3/envs/iws/bin/python -m pytest tests/test_libero90_stage1_job.py::test_stage1_job_uses_readme_settings_and_two_views -q
   ```

   It failed as expected because the launcher still contained `10000`.
3. Changed the single launcher override to `5000`.

## Verification

```bash
/home/youssef.ghallab/miniforge3/envs/iws/bin/python -m pytest tests/test_libero90_stage1_job.py -q
```

Result: `2 passed in 0.02s`.

```bash
bash -n jobs/train_iws_libero90_stage1.sbatch
git diff --check
```

Result: both commands exited successfully with no output.

The requested `sbatch --test-only` command was not rerun after its initial
sandbox socket denial. The controller will execute that cluster-side dry run.

## Review

Reviewed the staged scope: the only production change is the checkpoint
interval override, and the focused test checks the new launcher contract. No
unrelated runtime or resource settings changed.
