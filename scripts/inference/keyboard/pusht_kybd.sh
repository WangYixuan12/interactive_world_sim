#!/usr/bin/env bash
# Keyboard teleoperation — PushT (cam1)
# Usage: bash scripts/inference/keyboard/pusht_kybd.sh
# Run from repo root.

set -e

python scripts/inference/teleoperate_keyboard.py \
  +output_dir='data/wm_demo' \
  +use_joystick=false \
  +use_dataset=false \
  +act_horizon=1 \
  +scene=real \
  "+ckpt_paths=['outputs/pusht_cam1/checkpoints/best.ckpt']" \
  dataset=real_aloha_dataset \
  dataset.dataset_dir=data/mini/pusht/val \
  "dataset.obs_keys=['camera_1_color']"
