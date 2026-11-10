#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

TASK="${1:-objectnav}"          # objectnav | pointnav
MODEL="${2:-spiking}"           # spiking | ann
MODE="${3:-full}"               # full | smoke
SEED="${4:-12345}"

if [ "$TASK" != "objectnav" ] && [ "$TASK" != "pointnav" ]; then
  echo "Usage: $0 [objectnav|pointnav] [spiking|ann] [full|smoke] [seed]"
  exit 1
fi

CONFIG="${TASK}_robothor_${MODEL}_ddppo"
OUT="storage/${TASK}-${MODEL}-rgb"
TAG="rnav_${TASK}_${MODEL}_${MODE}"

if [ "$MODE" = "smoke" ]; then
  export SPIKINGNAV_NUM_PROCESSES="${SPIKINGNAV_NUM_PROCESSES:-1}"
  export SPIKINGNAV_MAX_STEPS="${SPIKINGNAV_MAX_STEPS:-128}"
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
  export SPIKINGNAV_GPU_IDS="${SPIKINGNAV_GPU_IDS:-0}"
fi
export SPIKINGNAV_VIS_CHUNK="${SPIKINGNAV_VIS_CHUNK:-4}"

export PYTHONPATH="${ROOT}${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
if [ "${SPIKINGNAV_USE_X11:-0}" != "1" ]; then
  unset DISPLAY
fi

bash "${ROOT}/scripts/download_thor.sh"
python "${ROOT}/scripts/write_cuda_vulkan_map.py"

# nvidia-smi can list 6 GPUs while this process can only open 4.
# Keep the Vulkan-backed physical ids, then renumber them for PyTorch.
if [ "$MODE" != "smoke" ] && [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
  CUDA_VISIBLE_DEVICES="$(python - <<'PY'
import json
from pathlib import Path
mapping = json.loads((Path.home() / ".ai2thor" / "cuda-vulkan-mapping.json").read_text())
print(",".join(str(i) for i in sorted(int(k) for k in mapping)))
PY
)"
  export CUDA_VISIBLE_DEVICES
  n_visible="$(awk -F, '{print NF}' <<< "$CUDA_VISIBLE_DEVICES")"
  SPIKINGNAV_GPU_IDS="$(seq -s, 0 $((n_visible - 1)))"
  export SPIKINGNAV_GPU_IDS
  echo "Using CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} (local ids ${SPIKINGNAV_GPU_IDS})"
fi

python main.py \
  -o "$OUT" \
  -b projects/spikingnav_baselines/experiments \
  "$CONFIG" \
  -s "$SEED" \
  --extra_tag "$TAG"
