#!/usr/bin/env bash
# Phase 05: verify TPU + torch_xla + V-JEPA 2.1 all work
set -euo pipefail

# shellcheck disable=SC1091
source ~/vjepa-env-py311/bin/activate
cd ~/vjepa-poc

echo "[05-smoke-test] Running forward_test.py..."
python scripts/forward_test.py
echo "[05-smoke-test] All smoke tests passed."
