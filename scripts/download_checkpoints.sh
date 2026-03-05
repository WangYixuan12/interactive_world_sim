#!/usr/bin/env bash
# Download pretrained checkpoints from Google Drive.
# Usage: bash scripts/download_checkpoints.sh  (run from repo root)
#
# Checkpoints are organized by task and camera:
#   Directory               | Task             | Camera
#   ------------------------|------------------|--------
#   outputs/pusht_cam1/     | pusht            | cam1
#   outputs/single_grasp_cam0/ | single_grasp  | cam0
#   outputs/single_grasp_cam1/ | single_grasp  | cam1
#   outputs/bimanual_sweep_cam0/ | bimanual_sweep | cam0
#   outputs/bimanual_sweep_cam1/ | bimanual_sweep | cam1
#   outputs/bimanual_rope_cam0/  | bimanual_rope  | cam0
#   outputs/bimanual_rope_cam1/  | bimanual_rope  | cam1
#
# Each directory contains:
#   checkpoints/best.ckpt   - pretrained model weights
#   .hydra/config.yaml      - Hydra configuration used at training time

set -e

OUTDIR="outputs"
mkdir -p "$OUTDIR"

# Check for gdown
if ! command -v gdown &> /dev/null; then
    echo "gdown not found. Installing..."
    pip install gdown
fi

echo "Downloading checkpoints to $OUTDIR/ ..."

# Google Drive folder ID: 1wVvQMcGjjyCLH7aeqpeQbDsdcBWximSA
FOLDER_ID="1wVvQMcGjjyCLH7aeqpeQbDsdcBWximSA"

gdown --folder "https://drive.google.com/drive/folders/${FOLDER_ID}" \
      --output "$OUTDIR" \
      --remaining-ok

echo ""
echo "Done. Checkpoints saved under $OUTDIR/:"
find "$OUTDIR" -name "best.ckpt" | sort
