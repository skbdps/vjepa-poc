#!/usr/bin/env bash
# Phase 03: clone facebookresearch/vjepa2 into ~/vjepa2
# Idempotent — skips if already cloned.
set -euo pipefail

VJEPA=~/vjepa2

if [[ -d "$VJEPA/.git" ]]; then
    echo "[03-clone-vjepa] $VJEPA already exists. Pulling latest..."
    (cd "$VJEPA" && git pull --ff-only)
else
    echo "[03-clone-vjepa] Cloning facebookresearch/vjepa2..."
    git clone https://github.com/facebookresearch/vjepa2.git "$VJEPA"
fi

echo "[03-clone-vjepa] Done. Repo at $VJEPA"
