# Source this file to keep runtime writes inside this checkout.
SPIKINGNAV_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export TMPDIR="${SPIKINGNAV_ROOT}/storage/tmp"
export PIP_CACHE_DIR="${SPIKINGNAV_ROOT}/storage/runtime_cache/pip"
export TORCH_HOME="${SPIKINGNAV_ROOT}/storage/runtime_cache/torch"
export XDG_CACHE_HOME="${SPIKINGNAV_ROOT}/storage/runtime_cache/xdg"
export XDG_CONFIG_HOME="${SPIKINGNAV_ROOT}/storage/runtime_cache/config"
export XDG_DATA_HOME="${SPIKINGNAV_ROOT}/storage/runtime_cache/data"
export CUDA_CACHE_PATH="${SPIKINGNAV_ROOT}/storage/runtime_cache/cuda"
export MPLCONFIGDIR="${SPIKINGNAV_ROOT}/storage/runtime_cache/matplotlib"
export NUMBA_CACHE_DIR="${SPIKINGNAV_ROOT}/storage/runtime_cache/numba"
export PYTHON_EGG_CACHE="${SPIKINGNAV_ROOT}/storage/runtime_cache/eggs"
export PIP_DISABLE_PIP_VERSION_CHECK=1
for runtime_dir in "$TMPDIR" "$PIP_CACHE_DIR" "$TORCH_HOME" "$XDG_CACHE_HOME" "$XDG_CONFIG_HOME" "$XDG_DATA_HOME" "$CUDA_CACHE_PATH" "$MPLCONFIGDIR" "$NUMBA_CACHE_DIR" "$PYTHON_EGG_CACHE"; do
  case "$(realpath -m "$runtime_dir")" in
    "${SPIKINGNAV_ROOT}"/*) mkdir -p "$runtime_dir" ;;
    *) echo "Runtime directory resolves outside the project: $runtime_dir" >&2; return 1 ;;
  esac
done
unset runtime_dir
