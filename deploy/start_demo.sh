#!/usr/bin/env bash
# Start the local interactive demo server.
# Usage: bash deploy/start_demo.sh [--port 8000]
#
# Then open http://localhost:8000 in your browser.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

echo "Starting demo server on http://localhost:${PORT}"
echo "Open the URL in your browser, then use WASD / IJKL to control the robot."
echo "Press Ctrl+C to stop."
echo ""

cd "$REPO_ROOT"
uvicorn deploy.server:app --host 0.0.0.0
