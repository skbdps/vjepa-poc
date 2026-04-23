#!/usr/bin/env bash
# Phase 00: system packages
# Idempotent — safe to rerun.
set -euo pipefail

echo "[00-apt] Installing system packages..."
sudo apt-get update -q
sudo DEBIAN_FRONTEND=noninteractive apt-get install -yq \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    git

echo "[00-apt] Done."
python3.11 --version
