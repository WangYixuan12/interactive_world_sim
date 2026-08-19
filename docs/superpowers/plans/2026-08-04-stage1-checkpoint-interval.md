# Stage 1 Checkpoint Interval Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Save a resumable LIBERO-90 Stage 1 checkpoint every 5,000 training steps, before validation starts at step 6,000.

**Architecture:** Keep the existing Hydra/Slurm launcher and change only its checkpoint interval override. Update the existing focused launcher test first so the change follows a red-green cycle.

**Tech Stack:** Bash, Slurm `sbatch`, Hydra overrides, pytest.

## Global Constraints

- Preserve two-view channel-wise concatenation and all existing training settings.
- Do not submit a training job; use `sbatch --test-only`.
- Continue using one GPU, eight CPUs, 64 GB RAM, and the existing 24-hour request.

---

### Task 1: Save Stage 1 checkpoints every 5,000 steps

**Files:**
- Modify: `tests/test_libero90_stage1_job.py:20`
- Modify: `jobs/train_iws_libero90_stage1.sbatch:54`

**Interfaces:**
- Consumes: Hydra override `experiment.training.checkpointing.every_n_train_steps`.
- Produces: Stage 1 checkpoints at optimizer steps 5,000, 10,000, and every subsequent 5,000 steps.

- [x] **Step 1: Write the failing test**

Change the existing launcher expectation to:

```python
assert "experiment.training.checkpointing.every_n_train_steps=5000" in script
```

- [x] **Step 2: Run the test to verify it fails**

Run:

```bash
/home/youssef.ghallab/miniforge3/envs/iws/bin/python -m pytest tests/test_libero90_stage1_job.py::test_stage1_job_uses_readme_settings_and_two_views -q
```

Expected: FAIL because the launcher still specifies `10000`.

- [x] **Step 3: Write the minimal implementation**

Change the launcher override to:

```bash
experiment.training.checkpointing.every_n_train_steps=5000
```

- [x] **Step 4: Run verification**

Run:

```bash
/home/youssef.ghallab/miniforge3/envs/iws/bin/python -m pytest tests/test_libero90_stage1_job.py -q
bash -n jobs/train_iws_libero90_stage1.sbatch
git diff --check
IWS_ROOT="$PWD" OUTPUT_DIR=/nfs-stor/youssef.ghallab/Robotics/continual_octo/outputs/real_libero90/stage1_checkpoint_5000_dry_run sbatch --test-only jobs/train_iws_libero90_stage1.sbatch
```

Expected: both focused tests pass, Bash reports no syntax error, Git reports no whitespace errors, and Slurm accepts the dry run.

- [x] **Step 5: Commit**

```bash
git add -- jobs/train_iws_libero90_stage1.sbatch tests/test_libero90_stage1_job.py docs/superpowers/plans/2026-08-04-stage1-checkpoint-interval.md
git commit -m "fix: checkpoint Stage 1 every 5000 steps"
```
