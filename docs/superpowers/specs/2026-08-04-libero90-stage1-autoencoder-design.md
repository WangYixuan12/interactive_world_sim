# LIBERO-90 Stage 1 Autoencoder Design

## Goal

Train the Interactive World Simulator Stage 1 autoencoder on every LIBERO-90
task using both available RGB views. The existing model will concatenate the
two 3-channel views into one 6-channel tensor before encoding.

This work covers only Stage 1 autoencoder training. It does not add latent
dynamics training, decoder fine-tuning, replay, or policy training.

## Data

Read the existing LIBERO-90 HDF5 files directly from
`/nfs-stor/youssef.ghallab/Robotics/libero_datasets/libero_100/libero_90`.
Each file is one task and contains demonstrations under `data/demo_<n>`.

The dataset adapter will:

- require exactly 90 task files;
- read `obs/agentview_rgb` and `obs/eye_in_hand_rgb` as the two views;
- read the matching 7-dimensional `actions` array;
- keep the views as separate 3-channel tensors in the sample so the existing
  IWS training step performs the channel-wise concatenation;
- split each task deterministically, reserving the last five demonstrations
  for validation and using the rest for training;
- index frames with per-demonstration cumulative lengths instead of storing a
  Python tuple for every frame;
- open HDF5 files lazily inside each DataLoader worker and avoid a copied Zarr
  dataset.

Stage 1 uses a horizon of one. Each item therefore returns both views with
shape `(1, 3, 128, 128)` and actions with shape `(1, 7)`. The adapter provides
the normalizers expected by IWS: image range normalization for both views and
an identity action normalizer, because Stage 1 does not condition the decoder
on actions.

## Integration

Add one Hydra dataset configuration for LIBERO-90 and register its dataset
class with the existing latent-dynamics experiment. Do not change the model:
`LatentWorldModel.training_step` already concatenates every configured
observation key along the channel dimension, and `x_shape` already evaluates
to `3 * len(obs_keys)`.

Add one CIAI batch launcher. Its training command will follow the README Stage
1 recipe:

- `algorithm.training_stage=1`;
- latent dimension 512;
- horizon and validation horizon 1;
- resolution 128;
- batch size 1;
- 1,000,005 target optimizer steps;
- log every 100 steps;
- validate every 6,000 steps with batch size 10;
- save a resumable checkpoint every 10,000 steps;
- use one GPU and remain within CIAI's 12-hour job limit.

The launcher will reject a missing dataset, a task count other than 90, or a
runtime without a visible CUDA device before starting training. Output and
Slurm logs remain generated artifacts. A checkpoint override will allow a
later job to resume toward the README's full step target.

## Validation

One focused test will create small temporary LIBERO-style HDF5 files and verify
the deterministic split, tensor shapes, view ordering, pixel scaling, action
shape, and 6-channel model input contract. Validation will also include:

- the focused dataset test in the `iws` environment;
- Hydra configuration composition without starting training;
- `bash -n` for the Slurm launcher;
- `sbatch --test-only` before real submission.

After those gates pass, submit the initial Stage 1 job and verify that Slurm
accepts it and produces startup logs.
