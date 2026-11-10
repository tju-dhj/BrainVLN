#!/usr/bin/env bash
# Evaluate checkpoints as training saves them, then keep the top success rates.
# Point this at a different GPU from the trainer.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export MAGNUM_LOG="${MAGNUM_LOG:-quiet}"
export GLOG_minloglevel="${GLOG_minloglevel:-2}"
# shellcheck disable=SC1091
source "${ROOT}/scripts/habitat_env.sh"

exec "$PYTHON" "${ROOT}/scripts/watch_habitat_eval.py" "$@"
