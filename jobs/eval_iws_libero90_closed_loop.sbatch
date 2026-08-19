#!/usr/bin/env bash
#SBATCH --job-name=iws-libero90-eval
#SBATCH --partition=cscc-gpu-p
#SBATCH --qos=cscc-gpu-qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/nfs-stor/youssef.ghallab/Robotics/continual_octo/logs/slurm/iws-libero90-eval_%j.out
#SBATCH --error=/nfs-stor/youssef.ghallab/Robotics/continual_octo/logs/slurm/iws-libero90-eval_%j.err

set -euo pipefail

IWS_ROOT="${IWS_ROOT:-/nfs-stor/youssef.ghallab/Robotics/continual_octo/interactive_world_sim/.worktrees/libero90-stage1}"
DATASET_DIR="${DATASET_DIR:-/nfs-stor/youssef.ghallab/Robotics/libero_datasets/libero_100/libero_90}"
CHECKPOINT="${CHECKPOINT:-/nfs-stor/youssef.ghallab/Robotics/continual_octo/outputs/real_libero90/stage2/checkpoints/epoch=2-step=300000.ckpt}"
OUTPUT_DIR="${OUTPUT_DIR:-/nfs-stor/youssef.ghallab/Robotics/continual_octo/outputs/real_libero90/stage2_closed_loop_eval}"
PHASE="${PHASE:-all}"
TASK_INDICES="${TASK_INDICES:-0 45}"
DEMO_SPLIT="${DEMO_SPLIT:-heldout}"
DEMO_OFFSET="${DEMO_OFFSET:-0}"
HORIZON="${HORIZON:-200}"
SCRIPT="$IWS_ROOT/scripts/evaluation/eval_libero90_closed_loop.py"

[[ -f "$CHECKPOINT" ]] || { echo "Missing checkpoint: $CHECKPOINT" >&2; exit 1; }
[[ -f "$SCRIPT" ]] || { echo "Missing evaluator: $SCRIPT" >&2; exit 1; }
mkdir -p "$OUTPUT_DIR"

set +u
source /home/youssef.ghallab/miniforge3/etc/profile.d/conda.sh
set -u
if [[ "$PHASE" != model ]]; then
  set +u
  conda activate smolvla
  set -u
  export MUJOCO_GL=egl
  read -ra task_indices <<< "$TASK_INDICES"
  python "$SCRIPT" environment \
    --dataset-dir "$DATASET_DIR" \
    --output-data "$OUTPUT_DIR/environment_rollouts.npz" \
    --task-indices "${task_indices[@]}" \
    --demo-split "$DEMO_SPLIT" \
    --demo-offset "$DEMO_OFFSET" \
    --horizon "$HORIZON"
fi

set +u
conda activate /home/youssef.ghallab/miniforge3/envs/iws
set -u
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
cd "$IWS_ROOT"
python "$SCRIPT" model \
  --input-data "$OUTPUT_DIR/environment_rollouts.npz" \
  --checkpoint "$CHECKPOINT" \
  --output-dir "$OUTPUT_DIR"
