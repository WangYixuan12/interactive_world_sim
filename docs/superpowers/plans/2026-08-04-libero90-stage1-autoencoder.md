# LIBERO-90 Stage 1 Autoencoder Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train the IWS Stage 1 autoencoder on all LIBERO-90 demonstrations with `agentview_rgb` and `eye_in_hand_rgb` concatenated channel-wise.

**Architecture:** Add one lazy HDF5 dataset that exposes the existing IWS batch contract without converting LIBERO into ALOHA or Zarr. Register it with the existing experiment, then launch the unchanged model through one resumable CIAI Slurm script using a 48-hour budget split into 24-hour allocations.

**Tech Stack:** Python 3.10+, PyTorch, h5py, Hydra/OmegaConf, Lightning, pytest, Bash, Slurm.

## Global Constraints

- Read all 90 files from `/nfs-stor/youssef.ghallab/Robotics/libero_datasets/libero_100/libero_90`.
- Use both `agentview_rgb` and `eye_in_hand_rgb`; existing IWS code must concatenate them into a 6-channel tensor.
- Stage 1 only: no latent dynamics, decoder fine-tuning, replay, or policy training.
- Follow the README settings: horizon 1, validation horizon 1, resolution 128, latent dimension 512, batch size 1, 1,000,005 steps, log every 100 steps, validate every 6,000 steps with validation batch size 10.
- Save checkpoints every 10,000 steps.
- Use one GPU for a 48-hour total budget split into resumable 24-hour Slurm allocations.
- Do not create a copied Zarr dataset or add dependencies.

---

### Task 1: Lazy LIBERO-90 Dataset

**Files:**
- Create: `interactive_world_sim/datasets/latent_dynamics/libero90_dataset.py`
- Create: `configurations/dataset/libero90_dataset.yaml`
- Modify: `interactive_world_sim/datasets/latent_dynamics/__init__.py`
- Modify: `interactive_world_sim/experiments/exp_latent_dyn.py`
- Test: `tests/test_libero90_dataset.py`

**Interfaces:**
- Consumes: LIBERO files shaped as `data/demo_N/obs/{agentview_rgb,eye_in_hand_rgb}` and `data/demo_N/actions`, where `N` is an integer.
- Produces: `Libero90Dataset(cfg, split="training")`, `Libero90Dataset.get_validation_dataset()`, and samples containing `obs`, `goal`, `action`, `is_early_stop`, and `rel_stop_idx`.
- Produces: Hydra dataset name `libero90_dataset` with ordered `obs_keys: [agentview_rgb, eye_in_hand_rgb]`.

- [ ] **Step 1: Write the failing dataset test**

Create two temporary task files, each with three numerically named demonstrations and two frames per demonstration. Configure `expected_tasks=2` and `val_demos_per_task=1` so the production validation rules are exercised without fabricating 90 files.

```python
from pathlib import Path

import h5py
import numpy as np
import torch
from omegaconf import OmegaConf

from interactive_world_sim.datasets.latent_dynamics.libero90_dataset import (
    Libero90Dataset,
)


def _write_task(path: Path, task_value: int) -> None:
    with h5py.File(path, "w") as file:
        data = file.create_group("data")
        for demo_idx in range(3):
            demo = data.create_group(f"demo_{demo_idx}")
            obs = demo.create_group("obs")
            value = task_value + demo_idx
            obs.create_dataset(
                "agentview_rgb",
                data=np.full((2, 128, 128, 3), value, dtype=np.uint8),
            )
            obs.create_dataset(
                "eye_in_hand_rgb",
                data=np.full((2, 128, 128, 3), value + 10, dtype=np.uint8),
            )
            demo.create_dataset(
                "actions", data=np.full((2, 7), value, dtype=np.float32)
            )


def test_libero90_dataset_splits_tasks_and_returns_two_views(tmp_path: Path) -> None:
    _write_task(tmp_path / "task_a.hdf5", 1)
    _write_task(tmp_path / "task_b.hdf5", 20)
    cfg = OmegaConf.create(
        {
            "dataset_dir": str(tmp_path),
            "expected_tasks": 2,
            "val_demos_per_task": 1,
            "resolution": 128,
            "action_dim": 7,
            "obs_keys": ["agentview_rgb", "eye_in_hand_rgb"],
        }
    )

    train = Libero90Dataset(cfg)
    validation = train.get_validation_dataset()
    sample = train[0]
    batched_views = [sample["obs"][key].unsqueeze(0) for key in cfg.obs_keys]

    assert len(train) == 8
    assert len(validation) == 4
    assert sample["obs"]["agentview_rgb"].shape == (1, 3, 128, 128)
    assert sample["obs"]["eye_in_hand_rgb"].shape == (1, 3, 128, 128)
    assert sample["action"].shape == (1, 7)
    assert torch.cat(batched_views, dim=2).shape == (1, 1, 6, 128, 128)
    assert torch.isclose(sample["obs"]["agentview_rgb"][0, 0, 0, 0], torch.tensor(1 / 255))
    assert validation[0]["action"][0, 0].item() == 3
```

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```bash
/home/youssef.ghallab/miniforge3/envs/iws/bin/python -m pytest tests/test_libero90_dataset.py -q
```

Expected: collection fails with `ModuleNotFoundError` naming `libero90_dataset`.

- [ ] **Step 3: Implement the minimal lazy dataset**

Create `Libero90Dataset` as a `BaseImageDataset`. During initialization, sort `*.hdf5`, require `expected_tasks`, sort demonstration names by their numeric suffix, apply the per-task split, and store only `(path, demo_name, length)` plus cumulative frame ends. In `__getitem__`, resolve the frame with `bisect_right`, lazily reuse one read-only HDF5 handle per task file inside each DataLoader worker, return float CHW views scaled by `1 / 255`, and return the matching float32 7-D action with a horizon dimension. Clear the handle dictionary in `__getstate__` so spawned workers never inherit open HDF5 objects, and close cached handles in `__del__`.

Use these exact public signatures: `Libero90Dataset(cfg: DictConfig, split:
str = "training")`, `__len__() -> int`, `__getitem__(idx: int) -> dict[str,
Any]`, `get_validation_dataset() -> Libero90Dataset`, `get_normalizer(mode:
str = "none", **kwargs: dict) -> LinearNormalizer`, `__getstate__() ->
dict[str, Any]`, and `__del__() -> None`.

`get_normalizer` must assign `get_image_range_normalizer()` to both observation keys and `SingleFieldLinearNormalizer.create_identity()` to `action`. Raise `ValueError` for an invalid split, wrong task count, too few demonstrations for the requested validation split, wrong image shape, or wrong action width.

Create `configurations/dataset/libero90_dataset.yaml` with:

```yaml
defaults:
  - base_dataset

dataset_dir: /nfs-stor/youssef.ghallab/Robotics/libero_datasets/libero_100/libero_90
expected_tasks: 90
val_demos_per_task: 5
horizon: 1
val_horizon: 1
resolution: 128
action_dim: 7
obs_keys: [agentview_rgb, eye_in_hand_rgb]
```

Export `Libero90Dataset` from the dataset package and add `libero90_dataset=Libero90Dataset` to `LatentDynExperiment.compatible_datasets`.

- [ ] **Step 4: Run the focused test and verify GREEN**

Run:

```bash
/home/youssef.ghallab/miniforge3/envs/iws/bin/python -m pytest tests/test_libero90_dataset.py -q
```

Expected: `1 passed`.

- [ ] **Step 5: Compose the production Hydra configuration**

Run:

```bash
/home/youssef.ghallab/miniforge3/envs/iws/bin/python main.py --cfg job \
  +name=libero90_stage1 algorithm=latent_world_model \
  experiment=exp_latent_dyn dataset=libero90_dataset \
  algorithm.training_stage=1 algorithm.action_dim=7 \
  "dataset.obs_keys=[agentview_rgb,eye_in_hand_rgb]"
```

Expected: exit 0 and resolved output containing both observation keys, `action_dim: 7`, `training_stage: 1`, and `x_shape` resolving to a 6-channel first dimension.

- [ ] **Step 6: Commit the dataset integration**

```bash
git add tests/test_libero90_dataset.py \
  configurations/dataset/libero90_dataset.yaml \
  interactive_world_sim/datasets/latent_dynamics/libero90_dataset.py \
  interactive_world_sim/datasets/latent_dynamics/__init__.py \
  interactive_world_sim/experiments/exp_latent_dyn.py
git commit -m "feat: add lazy LIBERO-90 dataset"
```

### Task 2: Resumable CIAI Stage 1 Launcher

**Files:**
- Create: `jobs/train_iws_libero90_stage1.sbatch`
- Test: `tests/test_libero90_stage1_job.py`

**Interfaces:**
- Consumes: Hydra dataset `libero90_dataset` from Task 1 and optional environment variables `LIBERO_ROOT`, `OUTPUT_DIR`, `RESUME_CKPT`, `WANDB_MODE`, and `WANDB_ENTITY`.
- Produces: a 24-hour one-GPU Slurm allocation that runs toward a 1,000,005-step target and can resume from `RESUME_CKPT` during the second allocation.

- [ ] **Step 1: Write the failing launcher test**

```python
from pathlib import Path


def test_stage1_job_uses_readme_settings_and_two_views() -> None:
    script = Path("jobs/train_iws_libero90_stage1.sbatch").read_text()

    assert "#SBATCH --time=24:00:00" in script
    assert "#SBATCH --gres=gpu:1" in script
    assert "#SBATCH --partition=cscc-gpu-p" in script
    assert "#SBATCH --qos=cscc-gpu-qos" in script
    assert "dataset=libero90_dataset" in script
    assert 'dataset.obs_keys=[agentview_rgb,eye_in_hand_rgb]' in script
    assert "experiment.training.max_steps=1000005" in script
    assert "experiment.training.batch_size=1" in script
    assert "experiment.validation.val_every_n_step=6000" in script
    assert "experiment.validation.batch_size=10" in script
    assert "experiment.training.checkpointing.every_n_train_steps=10000" in script
    assert "algorithm.latent_dim=512" in script
    assert "algorithm.action_dim=7" in script
    assert "algorithm.training_stage=1" in script
    assert "RESUME_CKPT" in script
```

- [ ] **Step 2: Run the launcher test and verify RED**

Run:

```bash
/home/youssef.ghallab/miniforge3/envs/iws/bin/python -m pytest tests/test_libero90_stage1_job.py -q
```

Expected: failure because `jobs/train_iws_libero90_stage1.sbatch` does not exist.

- [ ] **Step 3: Implement the minimal batch launcher**

Create a strict Bash script with these Slurm resources:

```bash
#!/usr/bin/env bash
#SBATCH --job-name=iws-libero90-s1
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/nfs-stor/youssef.ghallab/Robotics/continual_octo/logs/slurm/iws-libero90-s1_%j.out
#SBATCH --error=/nfs-stor/youssef.ghallab/Robotics/continual_octo/logs/slurm/iws-libero90-s1_%j.err

set -euo pipefail
```

Set the IWS repository, LIBERO root, output directory, and optional checkpoint from environment variables. Require exactly 90 HDF5 files, activate `/home/youssef.ghallab/miniforge3/envs/iws`, and run a Python CUDA assertion before training. Build the Hydra command as a Bash array; append `load="$RESUME_CKPT"` only when the checkpoint exists. Use `wandb.mode=${WANDB_MODE:-online}` and append `wandb.entity="$WANDB_ENTITY"` only when set.

The command must include:

```bash
+name=libero90_stage1 algorithm=latent_world_model
experiment=exp_latent_dyn dataset=libero90_dataset
dataset.horizon=1 dataset.val_horizon=1
dataset.obs_keys=[agentview_rgb,eye_in_hand_rgb]
experiment.training.batch_size=1
experiment.training.max_steps=1000005
experiment.training.log_every_n_steps=100
experiment.validation.limit_batch=1.0
experiment.validation.batch_size=10
experiment.validation.val_every_n_step=6000
experiment.training.checkpointing.every_n_train_steps=10000
algorithm.latent_dim=512 algorithm.action_dim=7
algorithm.training_stage=1
```

- [ ] **Step 4: Run focused launcher verification**

Run:

```bash
/home/youssef.ghallab/miniforge3/envs/iws/bin/python -m pytest tests/test_libero90_stage1_job.py -q
bash -n jobs/train_iws_libero90_stage1.sbatch
sbatch --test-only jobs/train_iws_libero90_stage1.sbatch
```

Expected: one passing test, shell syntax exit 0, and Slurm accepts the 24-hour request without submitting it.

- [ ] **Step 5: Commit the launcher**

```bash
git add tests/test_libero90_stage1_job.py jobs/train_iws_libero90_stage1.sbatch
git commit -m "feat: launch LIBERO-90 Stage 1 training"
```

### Task 3: Final Verification and Initial Submission

**Files:**
- Modify after verified launch: `../progress.md`

**Interfaces:**
- Consumes: verified dataset and launcher from Tasks 1 and 2.
- Produces: one live 24-hour Stage 1 job ID, startup log evidence, and a progress entry. It does not submit the second allocation before a resumable checkpoint exists.

- [ ] **Step 1: Run all focused checks from a clean checkout state**

```bash
/home/youssef.ghallab/miniforge3/envs/iws/bin/python -m pytest \
  tests/test_libero90_dataset.py tests/test_libero90_stage1_job.py -q
bash -n jobs/train_iws_libero90_stage1.sbatch
git diff --check
```

Expected: two passing tests, valid Bash, and no whitespace errors.

- [ ] **Step 2: Submit the first 24-hour allocation**

From the IWS repository root, run:

```bash
submit_output=$(sbatch jobs/train_iws_libero90_stage1.sbatch)
job_id=${submit_output##* }
printf '%s\n' "$submit_output"
```

Expected: output starts with `Submitted batch job` and `job_id` contains its numeric identifier.

- [ ] **Step 3: Verify allocation and startup**

```bash
squeue -j "$job_id" -o '%.18i %.24j %.2t %.10M %.10l %R'
tail -n 80 "/nfs-stor/youssef.ghallab/Robotics/continual_octo/logs/slurm/iws-libero90-s1_${job_id}.out"
tail -n 80 "/nfs-stor/youssef.ghallab/Robotics/continual_octo/logs/slurm/iws-libero90-s1_${job_id}.err"
```

Expected: the job is pending or running; once running, logs show one CUDA device, 90 task files, both view keys, and the Stage 1 trainer starting without a traceback.

- [ ] **Step 4: Record the verified checkpoint in the parent harness**

Append a dated entry to `../progress.md` with the IWS commit, validation commands, Slurm job ID, 24-hour limit, output directory, and note that the second 24-hour allocation must resume from the latest verified checkpoint.

- [ ] **Step 5: Commit the progress record after startup is verified**

Run from the parent repository:

```bash
git add progress.md
git commit -m "docs: record IWS LIBERO-90 Stage 1 launch"
```
