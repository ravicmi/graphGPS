#!/usr/bin/env bash
# Run H1 ablation for NCI1 only (3 seeds, early stopping patience=10).
# Usage:
#   bash scripts/run_h1_nci1.sh
#   REPEAT=5 bash scripts/run_h1_nci1.sh

set -euo pipefail

REPEAT="${REPEAT:-3}"
PATIENCE="${PATIENCE:-10}"

cd "$(dirname "$0")/.."

echo "========================================================"
echo "  H1 NCI1-only run"
echo "  Seeds: $REPEAT | Early-stop patience: $PATIENCE epochs"
echo "  Working directory: $(pwd)"
echo "========================================================"

run_exp() {
    local cfg_file="$1"
    local label="$2"
    echo ""
    echo "--- [$label] $cfg_file  (seeds: $REPEAT, patience: $PATIENCE) ---"
    python main.py \
        --cfg "$cfg_file" \
        --repeat "$REPEAT" \
        wandb.use False \
        optim.early_stop_patience "$PATIENCE"
    echo "--- done: $cfg_file ---"
}

run_exp "configs/GPS/nci1-GPS-Ricci-forman.yaml" "C3 NCI1+LapPE+RicciForman"

echo ""
echo "========================================================"
echo "  NCI1 H1 complete."
echo "  Collect: python scripts/collect_h1_results.py --variant forman"
echo "========================================================"
