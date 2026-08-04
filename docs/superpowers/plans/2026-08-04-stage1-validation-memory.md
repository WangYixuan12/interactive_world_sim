# Stage 1 Validation Memory Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bound LIBERO-90 Stage 1 validation to one 10-sample batch so validation no longer exhausts the 64 GB Slurm allocation.

**Architecture:** Keep the existing validation implementation and use Lightning's native integer batch limit. Change only the launcher override and its focused regression expectation; do not add metric or buffering code.

**Tech Stack:** Bash, Slurm, Hydra, Lightning, pytest.

## Global Constraints

- Use integer `experiment.validation.limit_batch=1`; float `1.0` means the complete validation set.
- Preserve validation frequency 6,000, validation batch size 10, and checkpoint frequency 5,000.
- Preserve two-view channel-wise concatenation and all other Stage 1 settings.
- Submit only after tests, Bash syntax, Git whitespace, and `sbatch --test-only` pass.

---

### Task 1: Bound validation to one batch

**Files:**
- Modify: `tests/test_libero90_stage1_job.py:17`
- Modify: `jobs/train_iws_libero90_stage1.sbatch:50`

**Interfaces:**
- Consumes: Lightning `Trainer(limit_val_batches=...)` through Hydra key `experiment.validation.limit_batch`.
- Produces: one validation batch of 10 samples every 6,000 training steps.

- [x] **Step 1: Write the failing test**

Add this expectation to the existing launcher-settings test:

```python
assert "experiment.validation.limit_batch=1\n" in script
```

The trailing newline prevents float `1.0` from satisfying the expectation.

- [x] **Step 2: Run the test to verify it fails**

Run:

```bash
/home/youssef.ghallab/miniforge3/envs/iws/bin/python -m pytest tests/test_libero90_stage1_job.py::test_stage1_job_uses_readme_settings_and_two_views -q
```

Expected: FAIL because the launcher specifies `experiment.validation.limit_batch=1.0`.

- [x] **Step 3: Write the minimal implementation**

Change the launcher override to:

```bash
experiment.validation.limit_batch=1
```

- [x] **Step 4: Run verification**

Run:

```bash
/home/youssef.ghallab/miniforge3/envs/iws/bin/python -m pytest tests/test_libero90_stage1_job.py -q
bash -n jobs/train_iws_libero90_stage1.sbatch
git diff --check
sbatch --test-only jobs/train_iws_libero90_stage1.sbatch
```

Expected: both focused tests pass, local checks report no errors, and Slurm accepts the dry run without submitting a job.

- [x] **Step 5: Commit**

```bash
git add -- jobs/train_iws_libero90_stage1.sbatch tests/test_libero90_stage1_job.py docs/superpowers/plans/2026-08-04-stage1-validation-memory.md
git commit -m "fix: bound Stage 1 validation memory"
```

### Task 2: Restart and verify Stage 1 training

**Files:**
- Generated output: `outputs/real_libero90/stage1_validation_bounded`
- Generated logs: `logs/slurm/iws-libero90-s1_<job-id>.{out,err}`

**Interfaces:**
- Consumes: committed launcher from Task 1 and the existing LIBERO-90 dataset.
- Produces: a live Slurm job starting from step zero, with a checkpoint expected at step 5,000 before bounded validation at step 6,000.

- [ ] **Step 1: Submit the job**

Run:

```bash
IWS_ROOT="$PWD" OUTPUT_DIR=/nfs-stor/youssef.ghallab/Robotics/continual_octo/outputs/real_libero90/stage1_validation_bounded sbatch jobs/train_iws_libero90_stage1.sbatch
```

Expected: Slurm prints one submitted job ID.

- [ ] **Step 2: Verify startup**

Query the submitted job with `squeue` and inspect its Slurm logs and resolved Hydra configuration. Confirm CUDA-enabled Lightning training starts and the resolved values are:

```text
experiment.validation.limit_batch: 1
experiment.validation.batch_size: 10
experiment.validation.val_every_n_step: 6000
experiment.training.checkpointing.every_n_train_steps: 5000
```

- [ ] **Step 3: Record the restart**

Update the parent repository's `progress.md` and `bugs.md` with the job ID, output directory, prior OOM root cause, bounded-validation fix, and verified startup state.
