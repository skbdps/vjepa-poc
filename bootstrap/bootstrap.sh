#!/usr/bin/env bash
# Master bootstrap script. Runs phases in order.
# Rerunning is safe — each phase is idempotent.
set -euo pipefail

cd "$(dirname "$0")"

for phase in 00-apt.sh 01-venv.sh 02-deps.sh 03-clone-vjepa.sh 04-weights.sh 05-smoke-test.sh; do
    echo "========================================================================"
    echo "RUNNING $phase"
    echo "========================================================================"
    bash "./$phase"
    echo
done

echo "========================================================================"
echo "Bootstrap complete. Environment ready at ~/vjepa-env-py311"
echo "V-JEPA 2.1 source at ~/vjepa2, POC repo at ~/vjepa-poc"
echo "========================================================================"
