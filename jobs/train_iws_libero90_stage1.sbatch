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

IWS_ROOT=/nfs-stor/youssef.ghallab/Robotics/continual_octo/interactive_world_sim
LIBERO_ROOT="${LIBERO_ROOT:-/nfs-stor/youssef.ghallab/Robotics/libero_datasets/libero_100/libero_90}"
OUTPUT_DIR="${OUTPUT_DIR:-/nfs-stor/youssef.ghallab/Robotics/continual_octo/outputs/real_libero90/stage1}"
RESUME_CKPT="${RESUME_CKPT:-}"

shopt -s nullglob
files=("$LIBERO_ROOT"/*.hdf5)
if (( ${#files[@]} != 90 )); then
  echo "Expected 90 HDF5 files in $LIBERO_ROOT, found ${#files[@]}" >&2
  exit 1
fi

source /home/youssef.ghallab/miniforge3/etc/profile.d/conda.sh
conda activate /home/youssef.ghallab/miniforge3/envs/iws
python -c 'import torch; assert torch.cuda.is_available(), "CUDA is unavailable"'

cd "$IWS_ROOT"
mkdir -p "$OUTPUT_DIR"
command=(
  python main.py
  +name=libero90_stage1 algorithm=latent_world_model
  experiment=exp_latent_dyn dataset=libero90_dataset
  "dataset.dataset_dir=$LIBERO_ROOT"
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
  "hydra.run.dir=$OUTPUT_DIR"
  "wandb.mode=${WANDB_MODE:-online}"
)

if [[ -n "$RESUME_CKPT" && -e "$RESUME_CKPT" ]]; then
  command+=("load=$RESUME_CKPT")
fi
if [[ -n "${WANDB_ENTITY:-}" ]]; then
  command+=("wandb.entity=$WANDB_ENTITY")
fi

"${command[@]}"
