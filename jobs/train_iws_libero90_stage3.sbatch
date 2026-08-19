#!/usr/bin/env bash
#SBATCH --job-name=iws-libero90-s3
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/nfs-stor/youssef.ghallab/Robotics/continual_octo/logs/slurm/iws-libero90-s3_%j.out
#SBATCH --error=/nfs-stor/youssef.ghallab/Robotics/continual_octo/logs/slurm/iws-libero90-s3_%j.err

set -euo pipefail

IWS_ROOT="${IWS_ROOT:-/nfs-stor/youssef.ghallab/Robotics/continual_octo/interactive_world_sim}"
LIBERO_ROOT="${LIBERO_ROOT:-/nfs-stor/youssef.ghallab/Robotics/libero_datasets/libero_100/libero_90}"
STAGE2_CKPT="${STAGE2_CKPT:-/nfs-stor/youssef.ghallab/Robotics/continual_octo/outputs/real_libero90/stage2/checkpoints/epoch=2-step=300000.ckpt}"
OUTPUT_DIR="${OUTPUT_DIR:-/nfs-stor/youssef.ghallab/Robotics/continual_octo/outputs/real_libero90/stage3}"
RESUME_CKPT="${RESUME_CKPT:-}"
MAX_STEPS="${MAX_STEPS:-1000005}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1}"
VAL_EVERY_N_STEP="${VAL_EVERY_N_STEP:-30000}"
CHECKPOINT_EVERY_N_STEPS="${CHECKPOINT_EVERY_N_STEPS:-10000}"
WANDB_ENTITY="${WANDB_ENTITY:-youssef-ghallab-mbzuai}"

if [[ ! -f "$STAGE2_CKPT" ]]; then
  echo "STAGE2_CKPT must be a regular file: $STAGE2_CKPT" >&2
  exit 1
fi
if [[ -n "$RESUME_CKPT" && ! -f "$RESUME_CKPT" ]]; then
  echo "RESUME_CKPT must be a regular file: $RESUME_CKPT" >&2
  exit 1
fi

shopt -s nullglob
files=("$LIBERO_ROOT"/*.hdf5)
if (( ${#files[@]} != 90 )); then
  echo "Expected 90 HDF5 files in $LIBERO_ROOT, found ${#files[@]}" >&2
  exit 1
fi

set +u
source /home/youssef.ghallab/miniforge3/etc/profile.d/conda.sh
conda activate /home/youssef.ghallab/miniforge3/envs/iws
set -u
python -c 'import torch; assert torch.cuda.is_available(), "CUDA is unavailable"'

cd "$IWS_ROOT"
mkdir -p "$OUTPUT_DIR"
command=(
  python main.py
  +name=libero90_stage3 algorithm=latent_world_model
  experiment=exp_latent_dyn dataset=libero90_dataset
  "dataset.dataset_dir=$LIBERO_ROOT"
  dataset.horizon=1 dataset.val_horizon=200
  dataset.obs_keys=[agentview_rgb,eye_in_hand_rgb]
  experiment.training.batch_size=$TRAIN_BATCH_SIZE
  experiment.training.max_steps=$MAX_STEPS
  experiment.training.log_every_n_steps=100
  experiment.validation.limit_batch=1
  experiment.validation.batch_size=1
  experiment.training.data.num_workers=4
  experiment.validation.data.num_workers=4
  experiment.validation.val_every_n_step=$VAL_EVERY_N_STEP
  experiment.training.checkpointing.every_n_train_steps=$CHECKPOINT_EVERY_N_STEPS
  algorithm.latent_dim=512 algorithm.action_dim=7
  algorithm.noise_scheduler.loss_weighting=uniform
  algorithm.sampling_strategy=terminal_only
  algorithm.val_render=false
  algorithm.training_stage=3
  "algorithm.load_ae='$STAGE2_CKPT'"
  "hydra.run.dir=$OUTPUT_DIR"
  "wandb.mode=${WANDB_MODE:-online}"
  "wandb.entity=$WANDB_ENTITY"
)

if [[ -n "$RESUME_CKPT" ]]; then
  export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
  command+=("load='$RESUME_CKPT'")
fi
"${command[@]}"
