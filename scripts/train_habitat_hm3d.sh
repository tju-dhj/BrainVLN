#!/usr/bin/env bash
# HM3D ObjectNav with SpikingNav on Habitat 3 (habitat-lab/sim 0.3.x).
# habitat-sim 0.3.1 lives in the dpedvln env on this machine. The spikingnav
# env does not have it. CUDA_VISIBLE_DEVICES must be set before import: this
# process can only open four CUDA devices, and an unset mask crashes torch.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export MAGNUM_LOG="${MAGNUM_LOG:-quiet}"
export GLOG_minloglevel="${GLOG_minloglevel:-2}"
export SPIKINGNAV_VIS_CHUNK="${SPIKINGNAV_VIS_CHUNK:-2}"
# shellcheck disable=SC1091
source "${ROOT}/scripts/habitat_env.sh"

exec "$PYTHON" "${ROOT}/scripts/train_habitat_hm3d.py" "$@"
