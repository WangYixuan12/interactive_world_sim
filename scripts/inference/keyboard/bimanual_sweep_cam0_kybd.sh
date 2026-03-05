#!/usr/bin/env bash
# Keyboard teleoperation — Bimanual Sweep (cam0 viewpoint)
# Usage: bash scripts/inference/keyboard/bimanual_sweep_cam0_kybd.sh
# Run from repo root.

set -e

python scripts/inference/teleoperate_keyboard.py \
  +output_dir='data/wm_demo' \
  +use_joystick=false \
  +use_dataset=false \
  +act_horizon=1 \
  +scene=bimanual_sweep_cam_0 \
  "+ckpt_paths=['outputs/bimanual_sweep_cam0/checkpoints/best.ckpt']" \
  dataset=real_aloha_dataset \
  dataset.dataset_dir=data/mini/bimanual_sweep/val \
  "dataset.obs_keys=['camera_0_color']"
