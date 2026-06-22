#!/usr/bin/env bash
# One-command bootstrap: fetch Boltz-2 weights + molecule DB and convert the
# checkpoints to native JAX weights. Idempotent (skips files already present).
#
# Usage:
#   bash scripts/setup.sh            # CUDA 13 (default)
#   CUDA=cuda12 bash scripts/setup.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CACHE="$ROOT/.cache/boltz"
BASE="https://huggingface.co/boltz-community/boltz-2/resolve/main"
CUDA="${CUDA:-cuda13}"

mkdir -p "$CACHE"

echo "==> Installing dependencies (uv sync --extra $CUDA --extra torch-bridge --extra dev)"
uv sync --extra "$CUDA" --extra torch-bridge --extra dev

fetch() {  # fetch <url> <dest>
  if [ -f "$2" ]; then echo "    have $(basename "$2")"; else
    echo "==> Downloading $(basename "$2")"; curl -L --fail -o "$2" "$1"; fi
}

fetch "$BASE/boltz2_conf.ckpt" "$CACHE/boltz2_conf.ckpt"
fetch "$BASE/boltz2_aff.ckpt"  "$CACHE/boltz2_aff.ckpt"
if [ -d "$CACHE/mols" ]; then
  echo "    have mols/"
else
  echo "==> Downloading + extracting mols.tar"
  curl -L --fail -o "$CACHE/mols.tar" "$BASE/mols.tar"
  tar -xf "$CACHE/mols.tar" -C "$CACHE" && rm -f "$CACHE/mols.tar"
fi

echo "==> Converting checkpoints to native JAX weights"
uv run --extra torch-bridge python "$ROOT/scripts/export_native_weights.py" \
  --conf-ckpt "$CACHE/boltz2_conf.ckpt" \
  --aff-ckpt "$CACHE/boltz2_aff.ckpt" \
  --out-dir "$ROOT/outputs/native_weights" \
  --features

echo "==> Setup complete. Predict with:"
echo "    uv run python scripts/predict.py --input job.yaml --fmt cif"
