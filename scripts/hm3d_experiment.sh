#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source scripts/habitat_env.sh
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export MAGNUM_LOG=quiet
export HABITAT_SIM_LOG=quiet
export GLOG_minloglevel=2
export SPIKINGNAV_VIS_CHUNK="${SPIKINGNAV_VIS_CHUNK:-4}"
case "${1:-}" in
  collect|eval) shift; exec "$PYTHON" -B scripts/collect_hm3d_expert.py "$@" ;;
  train) shift; exec "$PYTHON" -B scripts/train_hm3d_offline_experiment.py "$@" ;;
  *) echo 'Usage: bash scripts/hm3d_experiment.sh {collect|eval|train} ARGS'; exit 2 ;;
esac
