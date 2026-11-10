ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
# Source from the Habitat shell scripts after ROOT is set.
# Prefer the project-local Habitat installation; explicit overrides still win.
source "${ROOT}/scripts/project_runtime.sh"

if [[ -z "${HABITAT_LAB:-}" && -d "${ROOT}/third_party/habitat-lab-v0.3.1/habitat-lab" ]]; then
  HABITAT_LAB="${ROOT}/third_party/habitat-lab-v0.3.1/habitat-lab"
fi
if [[ -z "${HABITAT_PYTHON:-}" && -x "${ROOT}/storage/envs/habitat/bin/python" ]]; then
  HABITAT_PYTHON="${ROOT}/storage/envs/habitat/bin/python"
fi

if [[ -z "${HABITAT_LAB:-}" && -d /amax/daihaojie/DPed-VLN/habitat-lab ]]; then
  HABITAT_LAB=/amax/daihaojie/DPed-VLN/habitat-lab
fi
if [[ -z "${HABITAT_PYTHON:-}" && -x /home/w61/miniconda3/envs/dpedvln/bin/python ]]; then
  HABITAT_PYTHON=/home/w61/miniconda3/envs/dpedvln/bin/python
fi
if [[ -z "${HABITAT_LAB:-}" || ! -d "${HABITAT_LAB}" ]]; then
  echo "Set HABITAT_LAB to a habitat-lab 0.3.x checkout."
  exit 1
fi
if [[ -z "${HABITAT_PYTHON:-}" || ! -x "${HABITAT_PYTHON}" ]]; then
  echo "Set HABITAT_PYTHON to the interpreter that has habitat-sim 0.3.1."
  exit 1
fi
if [[ -d "${ROOT}/storage/runtime_libs/usr/lib/x86_64-linux-gnu" ]]; then
  export LD_LIBRARY_PATH="${ROOT}/storage/runtime_libs/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
export HABITAT_LAB HABITAT_PYTHON
export PYTHONPATH="${ROOT}:${HABITAT_LAB}${PYTHONPATH:+:$PYTHONPATH}"
PYTHON="${HABITAT_PYTHON}"
