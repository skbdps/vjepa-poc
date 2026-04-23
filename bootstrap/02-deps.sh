#!/usr/bin/env bash
# Phase 02: pinned python deps
# Idempotent — pip install with exact versions is a no-op if already satisfied.
set -euo pipefail

# shellcheck disable=SC1091
source ~/vjepa-env-py311/bin/activate

echo "[02-deps] Installing pinned torch stack + V-JEPA deps..."
# Install torch + torch_xla + torchvision together to prevent version drift.
# timm/einops pulled in the same command to catch resolver conflicts up front.
pip install \
    torch==2.9.0 \
    torchvision==0.24.0 \
    'torch_xla[tpu]==2.9.0' \
    timm==1.0.26 \
    einops==0.8.2

echo "[02-deps] Installed versions:"
pip list | grep -E '^(torch|torch-xla|torchvision|timm|einops)' || true

echo "[02-deps] Smoke test: import torch_xla and get TPU device..."
python -c "
import torch, torch_xla
dev = torch_xla.device()
print(f'torch: {torch.__version__}')
print(f'device: {dev}')
"
echo "[02-deps] Done."
