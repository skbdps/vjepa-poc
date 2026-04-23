#!/usr/bin/env bash
# Phase 03: clone facebookresearch/vjepa2 and patch VJEPA_BASE_URL
# Meta's repo ships with VJEPA_BASE_URL pointing at http://localhost:8300
# (dev leftover). We patch it to the real public URL so pretrained=True works.
# Idempotent — skips clone if already present, reapplies patch each time.
set -euo pipefail

VJEPA=~/vjepa2

if [[ -d "$VJEPA/.git" ]]; then
    echo "[03-clone-vjepa] $VJEPA already exists. Pulling latest..."
    (cd "$VJEPA" && git pull --ff-only)
else
    echo "[03-clone-vjepa] Cloning facebookresearch/vjepa2..."
    git clone https://github.com/facebookresearch/vjepa2.git "$VJEPA"
fi

# Patch VJEPA_BASE_URL — the committed value is localhost:8300 which is a
# dev-environment leftover. Real weights are at dl.fbaipublicfiles.com.
# sed is idempotent: if already patched, the match fails and nothing happens.
BACKBONES="$VJEPA/src/hub/backbones.py"
if grep -q 'VJEPA_BASE_URL = "http://localhost:8300"' "$BACKBONES"; then
    echo "[03-clone-vjepa] Patching VJEPA_BASE_URL localhost -> dl.fbaipublicfiles.com..."
    sed -i 's|VJEPA_BASE_URL = "http://localhost:8300"|VJEPA_BASE_URL = "https://dl.fbaipublicfiles.com/vjepa2"|' "$BACKBONES"
else
    echo "[03-clone-vjepa] VJEPA_BASE_URL already patched or file changed upstream."
fi

# Verify the fix is in place
if grep -q 'VJEPA_BASE_URL = "https://dl.fbaipublicfiles.com/vjepa2"' "$BACKBONES"; then
    echo "[03-clone-vjepa] Done. URL correctly set."
else
    echo "[03-clone-vjepa] WARNING: VJEPA_BASE_URL patch verification failed. Check $BACKBONES manually."
    exit 1
fi
