#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

TASK="${1:-objectnav}"
MODEL="${2:-spiking}"
CKPT="${3:-}"
SEVERITY="${4:-5}"

if [ -z "$CKPT" ]; then
  echo "Usage: $0 <objectnav|pointnav> <spiking|ann> <checkpoint.pt> [severity]"
  exit 1
fi

bash scripts/eval.sh "$TASK" "$MODEL" "$CKPT"
for name in \
  "Low Lighting" \
  "Motion Blur" \
  "Camera Crack" \
  "Defocus Blur" \
  "Speckle Noise" \
  "Lower FOV" \
  "Spatter"
do
  bash scripts/eval.sh "$TASK" "$MODEL" "$CKPT" "$name" "$SEVERITY"
done
