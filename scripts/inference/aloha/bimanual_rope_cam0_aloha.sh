#!/usr/bin/env bash
# ALOHA teleoperation — Bimanual Rope (cam0 viewpoint)
# Usage: bash scripts/inference/aloha/bimanual_rope_cam0_aloha.sh
# Run from repo root.

set -e

python scripts/inference/teleoperate_aloha.py \
  +output_dir='data/wm_demo' \
  +act_horizon=1 \
  +scene=bimanual_rope_cam_0 \
  "+ckpt_paths=['outputs/bimanual_rope_cam0/checkpoints/best.ckpt']" \
  dataset=real_aloha_dataset \
  dataset.dataset_dir=data/real_aloha/bimanual_rope/val \
  "dataset.obs_keys=['camera_0_color']"
