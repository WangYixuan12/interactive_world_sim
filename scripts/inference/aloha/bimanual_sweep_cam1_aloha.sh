#!/usr/bin/env bash
# ALOHA teleoperation — Bimanual Sweep (cam1 viewpoint)
# Usage: bash scripts/inference/aloha/bimanual_sweep_cam1_aloha.sh
# Run from repo root.

set -e

python scripts/inference/teleoperate_aloha.py \
  +output_dir='data/wm_demo' \
  +act_horizon=1 \
  +scene=bimanual_sweep_cam_1 \
  "+ckpt_paths=['outputs/bimanual_sweep_cam1/checkpoints/best.ckpt']" \
  dataset=real_aloha_dataset \
  dataset.dataset_dir=data/real_aloha/bimanual_sweep/val \
  "dataset.obs_keys=['camera_1_color']"
