#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

TASK="${1:-objectnav}"
MODEL="${2:-spiking}"
CKPT="${3:-}"
CORRUPTION="${4:-}"
SEVERITY="${5:-5}"

if [ -z "$CKPT" ]; then
  echo "Usage: $0 <objectnav|pointnav> <spiking|ann> <checkpoint.pt> [corruption] [severity]"
  echo "Corruptions: Low Lighting | Motion Blur | Camera Crack | Defocus Blur | Speckle Noise | Lower FOV | Spatter"
  exit 1
fi

CONFIG="${TASK}_robothor_${MODEL}_ddppo"
OUT="storage/${TASK}-${MODEL}-eval"
TAG="eval_${TASK}_${MODEL}"
EXTRA=()
if [ -n "$CORRUPTION" ]; then
  EXTRA+=(-vc "$CORRUPTION" -vs "$SEVERITY")
  TAG="${TAG}_$(echo "$CORRUPTION" | tr ' ' '_')_s${SEVERITY}"
fi

export PYTHONPATH="${ROOT}${PYTHONPATH:+:$PYTHONPATH}"

python main.py \
  -o "$OUT" \
  -b projects/spikingnav_baselines/experiments \
  "$CONFIG" \
  -c "$CKPT" \
  --eval \
  -s 12345 \
  --extra_tag "$TAG" \
  "${EXTRA[@]}"
