#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p datasets

download_one() {
  local name="$1"
  local suffix="$2"
  local dest="datasets/${name}"
  if [ -d "$dest/train" ] || [ -d "$dest/episodes" ]; then
    echo "Already present: ${dest}"
    return
  fi
  mkdir -p "$dest"
  local url="https://prior-datasets.s3.us-east-2.amazonaws.com/embodied-ai/navigation/${name}${suffix}.tar.gz"
  local archive="/tmp/${name}.tar.gz"
  echo "Downloading ${url}"
  wget -O "$archive" "$url"
  tar -xf "$archive" -C "$dest" --strip-components=1
  rm -f "$archive"
  echo "Saved ${dest}"
}

TARGET="${1:-all}"
case "$TARGET" in
  robothor-pointnav)
    download_one "robothor-pointnav" "-v0"
    ;;
  robothor-objectnav)
    download_one "robothor-objectnav" "-challenge-2021"
    ;;
  all)
    download_one "robothor-pointnav" "-v0"
    download_one "robothor-objectnav" "-challenge-2021"
    ;;
  *)
    echo "Usage: $0 [robothor-pointnav|robothor-objectnav|all]"
    exit 1
    ;;
esac

# RobustNav evaluation episodes live in the official repo.
if [ ! -d third_party/robustnav ]; then
  echo "Cloning allenai/robustnav for robustnav_eval episodes..."
  git clone --depth 1 https://github.com/allenai/robustnav.git third_party/robustnav
fi

for task in pointnav objectnav; do
  src="third_party/robustnav/datasets/robothor-${task}/robustnav_eval"
  dst="datasets/robothor-${task}/robustnav_eval"
  if [ -d "$src" ] && [ ! -e "$dst" ]; then
    mkdir -p "datasets/robothor-${task}"
    ln -s "$(realpath "$src")" "$dst"
    echo "Linked ${dst} -> ${src}"
  fi
done

echo "RoboTHOR datasets are under ${ROOT}/datasets"
