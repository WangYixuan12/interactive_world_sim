#!/usr/bin/env bash
# ALOHA teleoperation — Bimanual Rope (cam1 viewpoint)
# Usage: bash scripts/inference/aloha/bimanual_rope_cam1_aloha.sh
# Run from repo root.

set -e

python scripts/inference/teleoperate_aloha.py \
  +output_dir='data/wm_demo' \
  +act_horizon=1 \
  +scene=bimanual_rope_cam_1 \
  "+ckpt_paths=['outputs/bimanual_rope_cam1/checkpoints/best.ckpt']" \
  dataset=real_aloha_dataset \
  dataset.dataset_dir=data/real_aloha/bimanual_rope/val \
  "dataset.obs_keys=['camera_1_color']"
