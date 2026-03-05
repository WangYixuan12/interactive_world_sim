#!/usr/bin/env bash
# Keyboard teleoperation — Bimanual Rope (cam1 viewpoint)
# Usage: bash scripts/inference/keyboard/bimanual_rope_cam1_kybd.sh
# Run from repo root.

set -e

python scripts/inference/teleoperate_keyboard.py \
  +output_dir='data/wm_demo' \
  +use_joystick=false \
  +use_dataset=false \
  +act_horizon=1 \
  +scene=bimanual_rope_cam_1 \
  "+ckpt_paths=['outputs/bimanual_rope_cam1/checkpoints/best.ckpt']" \
  dataset=real_aloha_dataset \
  dataset.dataset_dir=data/mini/bimanual_rope/val \
  "dataset.obs_keys=['camera_1_color']"
