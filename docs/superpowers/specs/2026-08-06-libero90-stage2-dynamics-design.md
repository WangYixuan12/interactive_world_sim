# LIBERO-90 Stage 2 Dynamics Design

## Goal

Train the Interactive World Simulator Stage 2 latent dynamics model on all
LIBERO-90 tasks from the verified Stage 1 checkpoint at optimizer step 415,000.
Preserve the two RGB views and the existing seven-dimensional LIBERO actions.

Stage 2 trains only the dynamics network. The Stage 1 encoder and decoder are
loaded from:

```text
/nfs-stor/youssef.ghallab/Robotics/continual_octo/outputs/real_libero90/stage1_validation_bounded/checkpoints/epoch=0-step=415000.ckpt
```

The checkpoint is a valid Lightning archive with `global_step=415000`. Its
saved configuration matches the required 128-pixel, two-view, six-channel
input, eight-channel latent, latent dimension 512, and action dimension 7.

## Temporal LIBERO Dataset

Extend the existing lazy `Libero90Dataset`; do not convert LIBERO data to Zarr
or add another dataset class.

For training, build index metadata for every full window in each selected
demonstration. With horizon `T`, a demonstration of length `L` contributes
`max(L - T + 1, 0)` windows. Each sample slices observations and actions from
one demonstration only, so no window crosses a terminal boundary. Demos
shorter than the training horizon contribute no samples.

For validation, expose one sample per held-out demonstration. Read up to the
first `val_horizon` frames and right-pad short demonstrations by repeating the
last real observation and action. Reject zero-length demonstrations because
they have no terminal value to repeat. The validation goal is the final real
observation, and `rel_stop_idx` identifies the final real timestep.

The sample contract is:

```text
obs/agentview_rgb:  (T, 3, 128, 128)
obs/eye_in_hand_rgb: (T, 3, 128, 128)
action:              (T, 7)
```

The existing model concatenates the views along the channel dimension to
produce `(B, T, 6, 128, 128)`. HDF5 files remain lazily opened per DataLoader
worker, and worker serialization continues to clear inherited handles.

## Stage 2 Training

Add a dedicated CIAI Slurm launcher rather than overloading the Stage 1 job.
Use the README Stage 2 dynamics settings adapted to LIBERO-90:

- `algorithm.training_stage=2`;
- `algorithm.load_ae` set to the verified Stage 1 checkpoint;
- `algorithm.noise_scheduler.loss_weighting=uniform`;
- `algorithm.sampling_strategy=terminal_only`;
- latent dimension 512 and action dimension 7;
- training horizon 10 and validation horizon 200;
- training batch size 4;
- 1,000,005 target Stage 2 optimizer steps;
- log every 100 steps;
- validate every 30,000 steps;
- save the latest resumable checkpoint every 10,000 steps.

Validation uses one batch of one sequence (`limit_batch=1`, batch size 1) and
disables image rendering. This avoids the unmeasured memory cost and incorrect
six-channel video presentation while retaining the Stage 2 latent dynamics
loss logged by `validation_step`.

The production allocation requests one A100 GPU, eight CPUs, 64 GB host RAM,
and the documented 12-hour maximum on `cscc-gpu-p` with `cscc-gpu-qos`. It
does not set `CUDA_VISIBLE_DEVICES`, use `salloc`, or request exclusivity.

## Launch Gates

Before production, run focused dataset and launcher tests, Bash syntax,
`git diff --check`, and `sbatch --test-only`. The launcher must reject a
missing or invalid Stage 1 checkpoint, a dataset directory without exactly 90
HDF5 files, and a runtime without CUDA.

Then run a short Slurm smoke job using the production code path with one
training batch, horizon 10, one bounded validation batch, and image rendering
disabled. The smoke succeeds only if it loads the Stage 1 encoder/decoder,
constructs the two temporal views and actions with the required shapes,
completes a Stage 2 optimizer step, completes validation, and exits without
CUDA or host-memory failure.

After a successful smoke, submit one 12-hour production job in a fresh output
directory and verify its resolved Hydra configuration and CUDA-enabled
Lightning startup. Later allocations resume from the latest Stage 2 checkpoint
until the run reaches 1,000,005 optimizer steps.

## Verification

Focused dataset tests cover full sliding windows, the final legal training
window, demonstration-boundary isolation, short-validation terminal padding,
goal selection, `rel_stop_idx`, zero-length rejection, and spawned-worker HDF5
handle isolation. Launcher tests cover the exact Stage 2 settings and fail-fast
checkpoint validation.

Generated outputs, logs, checkpoints, W&B files, and smoke artifacts remain
outside Git. Record verified smoke and production job IDs in the parent
repository's operational notes.
