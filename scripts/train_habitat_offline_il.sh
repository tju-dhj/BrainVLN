#!/usr/bin/env bash
# Offline behavior cloning. Reads storage/il_offline and does not start Habitat.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export SPIKINGNAV_VIS_CHUNK="${SPIKINGNAV_VIS_CHUNK:-2}"
# shellcheck disable=SC1091
source "${ROOT}/scripts/habitat_env.sh"
exec "$PYTHON" "${ROOT}/scripts/train_habitat_offline_il.py" "$@"
