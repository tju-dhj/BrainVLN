#!/usr/bin/env bash
# Download and extract the AI2-THOR CloudRendering build before training.
# AllenAct kills the env worker after 300s, which is not enough for this ~800MB zip.
set -euo pipefail

CACHE="${HOME}/.ai2thor/cache"
RELEASES="${HOME}/.ai2thor/releases"
mkdir -p "$CACHE" "$RELEASES"

eval "$(python - <<'PY'
import ai2thor.build
from ai2thor.platform import CloudRendering
name = ai2thor.build.build_name("CloudRendering", ai2thor.build.COMMIT_ID, False)
url = ai2thor.build.base_url + "builds/" + name + ".zip"
print(f"NAME={name}")
print(f"URL={url}")
PY
)"

DEST="${RELEASES}/${NAME}"
if [[ -x "${DEST}/${NAME}" ]]; then
  echo "AI2-THOR build already installed: ${DEST}/${NAME}"
  exit 0
fi

LOCK="${CACHE}/${NAME}.lock"
exec 9>"$LOCK"
flock 9

if [[ -x "${DEST}/${NAME}" ]]; then
  echo "AI2-THOR build already installed: ${DEST}/${NAME}"
  exit 0
fi

ZIP="${CACHE}/${NAME}.zip"
echo "Downloading ${URL}"
echo "This file is about 800MB. The download resumes if interrupted."
if command -v aria2c >/dev/null 2>&1; then
  aria2c -c -x 8 -s 8 -k 1M --file-allocation=none \
    --summary-interval=30 \
    -d "$CACHE" -o "${NAME}.zip" "$URL"
else
  wget -c -O "$ZIP" "$URL"
fi

rm -rf "$DEST"
mkdir -p "$DEST"
unzip -q "$ZIP" -d "$DEST"
if [[ ! -x "${DEST}/${NAME}" ]]; then
  nested="$(find "$DEST" -mindepth 2 -maxdepth 3 -type f -name "$NAME" | head -1 || true)"
  if [[ -n "$nested" ]]; then
    inner="$(dirname "$nested")"
    shopt -s dotglob
    mv "$inner"/* "$DEST"/
    rmdir "$inner" 2>/dev/null || true
  fi
fi
chmod +x "${DEST}/${NAME}"
echo "Installed ${DEST}/${NAME}"
