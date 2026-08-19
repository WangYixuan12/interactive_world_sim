# LIBERO-90 Stage 2 Dynamics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train the Stage 2 latent dynamics model on correct LIBERO-90 temporal windows from the verified Stage 1 step-415,000 checkpoint.

**Architecture:** Extend the existing lazy LIBERO dataset with cumulative per-demo window indexing and terminal-padded validation sequences. Add one dedicated, resumable Stage 2 Slurm launcher, prove it with a short GPU smoke run, then submit one production allocation.

**Tech Stack:** Python, PyTorch, HDF5, Hydra, Lightning, pytest, Bash, Slurm.

## Global Constraints

- Preserve `agentview_rgb` and `eye_in_hand_rgb`; the model concatenates them channel-wise to six channels.
- Preserve 128-pixel images, latent dimension 512, eight latent channels, and action dimension 7.
- Load `/nfs-stor/youssef.ghallab/Robotics/continual_octo/outputs/real_libero90/stage1_validation_bounded/checkpoints/epoch=0-step=415000.ckpt` through `algorithm.load_ae`.
- Stage 2 uses horizon 10, validation horizon 200, training batch 4, max steps 1,000,005, validation every 30,000 steps, and checkpointing every 10,000 steps.
- Validation uses integer `limit_batch=1`, batch size 1, and `algorithm.val_render=false`.
- Production requests one GPU, eight CPUs, 64 GB RAM, and 12 hours on `cscc-gpu-p` / `cscc-gpu-qos`.
- No window may cross a demonstration boundary; generated outputs and Slurm logs stay outside Git.

---

### Task 1: Add temporal LIBERO windows

**Files:**
- Modify: `interactive_world_sim/datasets/latent_dynamics/libero90_dataset.py`
- Modify: `tests/test_libero90_dataset.py`

**Interfaces:**
- Consumes: `cfg.horizon`, `cfg.val_horizon`, per-demo HDF5 observations shaped `(L,128,128,3)`, and actions shaped `(L,7)`.
- Produces: training samples shaped `(10,3,128,128)` per view and `(10,7)` actions; validation samples shaped `(200,3,128,128)` per view and `(200,7)` actions.

- [x] **Step 1: Write failing temporal-window tests**

Add a fixture helper that writes frame-distinct values, then add these assertions using small horizons to keep the test fast:

```python
def _write_window_task(path: Path, lengths: tuple[int, ...]) -> None:
    with h5py.File(path, "w") as file:
        data = file.create_group("data")
        for demo_idx, length in enumerate(lengths):
            base = demo_idx * 100
            values = np.arange(base, base + length, dtype=np.uint8)
            images = np.broadcast_to(
                values[:, None, None, None], (length, 128, 128, 3)
            )
            demo = data.create_group(f"demo_{demo_idx}")
            obs = demo.create_group("obs")
            obs.create_dataset("agentview_rgb", data=images)
            obs.create_dataset("eye_in_hand_rgb", data=images + 10)
            demo.create_dataset(
                "actions",
                data=np.repeat(
                    np.arange(base, base + length, dtype=np.float32)[:, None],
                    7,
                    axis=1,
                ),
            )


def _window_cfg(path: Path, horizon: int, val_horizon: int):
    return OmegaConf.create({
        "dataset_dir": str(path), "expected_tasks": 1,
        "val_demos_per_task": 1, "horizon": horizon,
        "val_horizon": val_horizon, "resolution": 128, "action_dim": 7,
        "obs_keys": ["agentview_rgb", "eye_in_hand_rgb"],
    })
```

```python
def test_libero90_dataset_windows_do_not_cross_demos(tmp_path: Path) -> None:
    # demo_0 and demo_1 have length 12; demo_2 is held out.
    _write_window_task(tmp_path / "task.hdf5", (12, 12, 3))
    cfg = _window_cfg(tmp_path, horizon=10, val_horizon=5)
    dataset = Libero90Dataset(cfg)

    assert len(dataset) == 6
    assert dataset[2]["action"][:, 0].tolist() == list(range(2, 12))
    assert dataset[3]["action"][:, 0].tolist() == list(range(100, 110))
    assert dataset[0]["obs"]["agentview_rgb"].shape == (10, 3, 128, 128)
    assert dataset[0]["action"].shape == (10, 7)
```

```python
def test_libero90_validation_pads_with_terminal_frame(tmp_path: Path) -> None:
    _write_window_task(tmp_path / "task.hdf5", (12, 12, 3))
    cfg = _window_cfg(tmp_path, horizon=10, val_horizon=5)
    validation = Libero90Dataset(cfg).get_validation_dataset()
    sample = validation[0]

    assert len(validation) == 1
    assert sample["action"][:, 0].tolist() == [200, 201, 202, 202, 202]
    assert sample["goal"]["agentview_rgb"].equal(
        sample["obs"]["agentview_rgb"][2]
    )
    assert sample["rel_stop_idx"].item() == 2
```

Add a zero-length demo test expecting `ValueError("Demonstrations must contain at least one frame")`, and update the existing Stage 1 validation length expectation from held-out frames to held-out demonstrations.

- [x] **Step 2: Verify the tests fail for missing temporal behavior**

Run:

```bash
/home/youssef.ghallab/miniforge3/envs/iws/bin/python -m pytest tests/test_libero90_dataset.py -q
```

Expected: FAIL because samples still have `T=1`, training length is total frames, and validation is frame-indexed.

- [x] **Step 3: Implement cumulative window indexing and padding**

Store `cfg.horizon` and `cfg.val_horizon`. Keep `_demos` as `(path, demo_name, length)` and build cumulative `_sample_ends`:

```python
sample_end = 0
for path, demo_name, length in selected_demos:
    self._demos.append((path, demo_name, length))
    if split == "training":
        sample_end += max(length - self.horizon + 1, 0)
        self._sample_ends.append(sample_end)
```

For training, map a global index to a demo with `bisect_right`, derive the window start from the previous cumulative end, and slice exactly `horizon` frames. For validation, index one held-out demo directly, slice `min(length, val_horizon)`, and repeat its terminal observation/action until `val_horizon`. Return tensors without adding a singleton time dimension. Set `goal` to the final real observation and `rel_stop_idx` to `real_length - 1`.

Reject `length == 0` in `_validate_demo` with the exact error from Step 1. Preserve lazy read-only handles and `__getstate__` handle clearing.

- [x] **Step 4: Run focused and full wrapper verification**

Run:

```bash
/home/youssef.ghallab/miniforge3/envs/iws/bin/python -m pytest tests/test_libero90_dataset.py -q
/home/youssef.ghallab/miniforge3/envs/iws/bin/python -m pytest tests/test_libero90_dataset.py tests/test_libero90_stage1_job.py -q
git diff --check
```

Expected: all focused tests pass and Git reports no whitespace errors.

- [x] **Step 5: Commit**

```bash
git add -- interactive_world_sim/datasets/latent_dynamics/libero90_dataset.py tests/test_libero90_dataset.py
git commit -m "feat: add temporal LIBERO-90 windows"
```

### Task 2: Add a resumable Stage 2 launcher

**Files:**
- Create: `jobs/train_iws_libero90_stage2.sbatch`
- Create: `tests/test_libero90_stage2_job.py`
- Modify: `docs/superpowers/plans/2026-08-06-libero90-stage2-dynamics.md`

**Interfaces:**
- Consumes: Task 1 temporal samples and the verified Stage 1 checkpoint.
- Produces: a Slurm command for Stage 2 training and optional Stage 2 resume through `RESUME_CKPT`.

- [x] **Step 1: Write failing launcher tests**

Create a focused test that asserts the launcher contains:

```python
assert "#SBATCH --time=12:00:00" in script
assert "dataset.horizon=10 dataset.val_horizon=200" in script
assert "experiment.training.batch_size=$TRAIN_BATCH_SIZE" in script
assert "experiment.validation.limit_batch=1" in script
assert "experiment.validation.batch_size=1" in script
assert "experiment.validation.val_every_n_step=$VAL_EVERY_N_STEP" in script
assert "experiment.training.checkpointing.every_n_train_steps=$CHECKPOINT_EVERY_N_STEPS" in script
assert "algorithm.noise_scheduler.loss_weighting=uniform" in script
assert "algorithm.sampling_strategy=terminal_only" in script
assert "algorithm.val_render=false" in script
assert "algorithm.training_stage=2" in script
assert '"algorithm.load_ae=$STAGE1_CKPT"' in script
```

Add a subprocess test that supplies a missing `STAGE1_CKPT` and asserts the script exits before Conda with `STAGE1_CKPT must be a regular file`. Add source assertions for defaults `MAX_STEPS=1000005`, `TRAIN_BATCH_SIZE=4`, `VAL_EVERY_N_STEP=30000`, `CHECKPOINT_EVERY_N_STEPS=10000`, and optional `RESUME_CKPT` validation.

- [x] **Step 2: Verify the launcher tests fail**

Run:

```bash
/home/youssef.ghallab/miniforge3/envs/iws/bin/python -m pytest tests/test_libero90_stage2_job.py -q
```

Expected: FAIL because the Stage 2 launcher does not exist.

- [x] **Step 3: Implement the Stage 2 launcher**

Copy only the proven environment and validation guards from the Stage 1 launcher. Use these defaults:

```bash
STAGE1_CKPT="${STAGE1_CKPT:-/nfs-stor/youssef.ghallab/Robotics/continual_octo/outputs/real_libero90/stage1_validation_bounded/checkpoints/epoch=0-step=415000.ckpt}"
OUTPUT_DIR="${OUTPUT_DIR:-/nfs-stor/youssef.ghallab/Robotics/continual_octo/outputs/real_libero90/stage2}"
RESUME_CKPT="${RESUME_CKPT:-}"
MAX_STEPS="${MAX_STEPS:-1000005}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-4}"
VAL_EVERY_N_STEP="${VAL_EVERY_N_STEP:-30000}"
CHECKPOINT_EVERY_N_STEPS="${CHECKPOINT_EVERY_N_STEPS:-10000}"
```

Use one GPU, eight CPUs, 64 GB, and 12 hours. Validate the Stage 1 checkpoint, optional resume checkpoint, exactly 90 HDF5 files, Conda activation, and CUDA. Compose the approved Hydra settings verbatim. Append `load=$RESUME_CKPT` only when nonempty.

- [x] **Step 4: Run launcher and regression verification**

Run:

```bash
/home/youssef.ghallab/miniforge3/envs/iws/bin/python -m pytest tests/test_libero90_dataset.py tests/test_libero90_stage1_job.py tests/test_libero90_stage2_job.py -q
bash -n jobs/train_iws_libero90_stage2.sbatch
git diff --check
IWS_ROOT="$PWD" sbatch --test-only jobs/train_iws_libero90_stage2.sbatch
```

Expected: all tests pass, local checks are clean, and Slurm accepts the 12-hour dry run.

- [x] **Step 5: Commit**

```bash
git add -- jobs/train_iws_libero90_stage2.sbatch tests/test_libero90_stage2_job.py docs/superpowers/plans/2026-08-06-libero90-stage2-dynamics.md
git commit -m "feat: launch LIBERO-90 Stage 2 training"
```

### Task 3: Run the Stage 2 GPU smoke gate

**Files:**
- Generated output: `outputs/real_libero90/stage2_smoke`
- Generated logs: `logs/slurm/iws-libero90-s2_<job-id>.{out,err}`

**Interfaces:**
- Consumes: committed Task 1 dataset and Task 2 launcher.
- Produces: evidence that the Stage 1 checkpoint loads and Stage 2 training plus validation execute on one GPU.

- [ ] **Step 1: Submit one smoke job through the production launcher**

Run from the IWS worktree:

```bash
IWS_ROOT="$PWD" \
OUTPUT_DIR=/nfs-stor/youssef.ghallab/Robotics/continual_octo/outputs/real_libero90/stage2_smoke \
MAX_STEPS=2 TRAIN_BATCH_SIZE=1 VAL_EVERY_N_STEP=1 \
CHECKPOINT_EVERY_N_STEPS=1 WANDB_MODE=disabled \
sbatch jobs/train_iws_libero90_stage2.sbatch
```

Expected: exactly one smoke job ID.

- [ ] **Step 2: Verify smoke completion**

Wait for the job to leave the queue. Inspect stdout, stderr, the resolved Hydra config, and smoke checkpoint. Confirm Stage 2, horizon 10, validation horizon 200, two views, action dimension 7, batch 1, max steps 2, bounded validation, rendering disabled, CUDA use, checkpoint load, at least one optimizer step, validation completion, and no traceback/OOM.

- [ ] **Step 3: Record smoke evidence**

Write the job ID, final state, output path, resolved settings, and observed peak memory when available to the task report. Do not commit generated artifacts.

### Task 4: Submit and verify Stage 2 production

**Files:**
- Generated output: `outputs/real_libero90/stage2`
- Generated logs: `logs/slurm/iws-libero90-s2_<job-id>.{out,err}`
- Modify: parent repository `progress.md`
- Modify: parent repository `bugs.md` only if a non-trivial failure occurs

**Interfaces:**
- Consumes: successful Task 3 smoke evidence.
- Produces: one live 12-hour Stage 2 production allocation starting at Stage 2 step zero.

- [ ] **Step 1: Submit exactly one production job**

Run:

```bash
IWS_ROOT="$PWD" \
OUTPUT_DIR=/nfs-stor/youssef.ghallab/Robotics/continual_octo/outputs/real_libero90/stage2 \
sbatch jobs/train_iws_libero90_stage2.sbatch
```

- [ ] **Step 2: Verify startup**

Check `squeue`, Slurm logs, and `.hydra/config.yaml`. Confirm the production values from Global Constraints, CUDA-enabled Lightning startup, and absence of traceback/OOM.

- [ ] **Step 3: Record the launch**

Update parent `progress.md` with the smoke and production job IDs, Stage 1 checkpoint provenance, production output path, and verified startup state. Run `git diff --check`; leave generated artifacts untracked and do not commit parent operational notes unless explicitly requested.
