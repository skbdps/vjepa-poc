#!/usr/bin/env bash
# Phase 04: fetch V-JEPA 2.1 ViT-L weights
# First tries GCS mirror (fast, inside europe-west4); falls back to fbaipublicfiles.
# Weights land in ~/.cache/torch/hub/checkpoints/ where torch.hub.load() expects them.
set -euo pipefail

CACHE=~/.cache/torch/hub/checkpoints
GCS_MIRROR=gs://vjepa-poc-eu/weights
# The exact filename comes from torch.hub — we'll patch this once we know it.
# Placeholder: bootstrap script is informational for now, actual download
# happens the first time via torch.hub.load(pretrained=True). After that,
# we'll have weights mirrored to GCS and this script will fetch from there.

mkdir -p "$CACHE"

echo "[04-weights] NOTE: On first run, weights are downloaded by torch.hub.load() directly."
echo "[04-weights] After that initial download, run:"
echo "[04-weights]   gcloud storage cp <weights_file> $GCS_MIRROR/"
echo "[04-weights] and update this script to fetch from GCS mirror on subsequent recoveries."
echo "[04-weights] (No-op for now.)"
