#!/usr/bin/env bash
# Download demonstration data from Google Drive.
# Usage: bash scripts/download_data.sh  (run from repo root)
#
# Data is organized by task:
#   Directory                        | Task
#   ---------------------------------|--------------
#   data/mini/pusht/           | PushT
#   data/mini/single_grasp/    | Single Grasp
#   data/mini/bimanual_sweep/  | Bimanual Sweep
#   data/mini/bimanual_rope/   | Bimanual Rope

set -e

OUTDIR="data/mini"
mkdir -p "$OUTDIR"

# Check for gdown
if ! command -v gdown &> /dev/null; then
    echo "gdown not found. Installing..."
    pip install gdown
fi

echo "Downloading data to $OUTDIR/ ..."

# Google Drive folder ID: 1zTWiVq1SsjuxCBIt6qtlITC9Hp9a4xT1
FOLDER_ID="1zTWiVq1SsjuxCBIt6qtlITC9Hp9a4xT1"

gdown --folder "https://drive.google.com/drive/folders/${FOLDER_ID}" \
      --output "$OUTDIR" \
      --remaining-ok

echo ""
echo "Done. Data saved under $OUTDIR/:"
ls "$OUTDIR"
