#!/usr/bin/env bash
# DD-PPO: one process per visible GPU. Each process collects its own rollout,
# gradients are averaged, and a finished fraction can preempt slower cards.
# This process can only open four CUDA devices, so list at most four ids.
#
#   CUDA_VISIBLE_DEVICES=0,1 bash scripts/train_habitat_hm3d_dist.sh \
#     --algo ppo --split train --num-envs 8 --preemption-threshold 0.5
#
# --num-envs is the simulator count on each GPU. --total-steps counts
# environment steps across all GPUs. Do not pass --gpu; each rank uses its
# local device.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo "Set CUDA_VISIBLE_DEVICES to the GPUs for this job (at most 4)."
  exit 1
fi

IFS=',' read -ra DEVICES <<< "${CUDA_VISIBLE_DEVICES}"
NPROC="${#DEVICES[@]}"
if (( NPROC < 1 || NPROC > 4 )); then
  echo "CUDA_VISIBLE_DEVICES must list 1 to 4 devices, got ${NPROC}."
  exit 1
fi

export CUDA_VISIBLE_DEVICES
export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export MAGNUM_LOG="${MAGNUM_LOG:-quiet}"
export GLOG_minloglevel="${GLOG_minloglevel:-2}"
export SPIKINGNAV_VIS_CHUNK="${SPIKINGNAV_VIS_CHUNK:-2}"
# shellcheck disable=SC1091
source "${ROOT}/scripts/habitat_env.sh"

exec "$PYTHON" -m torch.distributed.run \
  --nproc_per_node="${NPROC}" \
  --master_port="${MASTER_PORT:-29531}" \
  "${ROOT}/scripts/train_habitat_hm3d.py" "$@"
