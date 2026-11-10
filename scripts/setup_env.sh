#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

ENV_NAME="${ENV_NAME:-spikingnav}"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda is required to create ${ENV_NAME}"
  exit 1
fi

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "conda env ${ENV_NAME} already exists"
else
  conda env create -f environment.yml -n "$ENV_NAME"
fi

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

pip install -e .
pip install 'ai2thor>=2.7.4' 'allenact' 'allenact_plugins[ithor]' 'moviepy==1.0.3'

echo
echo "Environment ${ENV_NAME} is ready. Activate with: conda activate ${ENV_NAME}"
echo "Then run: bash scripts/train.sh objectnav spiking smoke"
