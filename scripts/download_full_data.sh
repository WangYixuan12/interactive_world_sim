#!/usr/bin/env bash
# Download full training dataset from Google Drive.
# Usage: bash scripts/download_full_data.sh  (run from repo root)
#
# Data is organized by task:
#   Directory                        | Task
#   ---------------------------------|--------------
#   data/full/pusht/                 | PushT
#   data/full/single_grasp/          | Single Grasp
#   data/full/bimanual_sweep/        | Bimanual Sweep
#   data/full/bimanual_rope/         | Bimanual Rope

set -e

OUTDIR="data/full"
mkdir -p "$OUTDIR"

# Check for gdown
if ! command -v gdown &> /dev/null; then
    echo "gdown not found. Installing..."
    pip install gdown
fi

echo "Downloading full training dataset to $OUTDIR/ ..."

# Google Drive folder ID: 1umtKJI5TmIGxiD3TecjSwLqc9Lik5Gls
FOLDER_ID="1umtKJI5TmIGxiD3TecjSwLqc9Lik5Gls"

gdown --folder "https://drive.google.com/drive/folders/${FOLDER_ID}" \
      --output "$OUTDIR" \
      --remaining-ok

echo ""
echo "Done. Data saved under $OUTDIR/:"
ls "$OUTDIR"
