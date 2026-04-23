#!/usr/bin/env bash
# Phase 01: python venv at ~/vjepa-env-py311
# Idempotent — only creates if missing.
set -euo pipefail

VENV=~/vjepa-env-py311

if [[ -d "$VENV" ]]; then
    echo "[01-venv] Venv already exists at $VENV, skipping creation."
else
    echo "[01-venv] Creating venv at $VENV..."
    python3.11 -m venv "$VENV"
fi

# shellcheck disable=SC1091
source "$VENV/bin/activate"
python -m pip install --quiet --upgrade pip
echo "[01-venv] Done. pip version:"
pip --version
