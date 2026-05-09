#!/usr/bin/env bash
# Run H1 Ollivier-Ricci ablation for MUTAG, ENZYMES, NCI1.
# Usage:
#   bash scripts/run_h1_ollivier.sh
#   REPEAT_SMALL=5 bash scripts/run_h1_ollivier.sh

set -euo pipefail

REPEAT_SMALL="${REPEAT_SMALL:-10}"
REPEAT_NCI1="${REPEAT_NCI1:-3}"
PATIENCE="${PATIENCE:-10}"

cd "$(dirname "$0")/.."

echo "========================================================"
echo "  H1 Ollivier-Ricci Ablation — GraphGPS §2.2 / §2.3"
echo "  Seeds (MUTAG/ENZYMES): $REPEAT_SMALL | NCI1: $REPEAT_NCI1"
echo "  Early-stop patience: $PATIENCE epochs"
echo "  Working directory: $(pwd)"
echo "========================================================"

run_exp() {
    local cfg_file="$1"
    local label="$2"
    local repeat="$3"
    echo ""
    echo "--- [$label] $cfg_file  (seeds: $repeat, patience: $PATIENCE) ---"
    python main.py \
        --cfg "$cfg_file" \
        --repeat "$repeat" \
        wandb.use False \
        optim.early_stop_patience "$PATIENCE"
    echo "--- done: $cfg_file ---"
}

run_exp "configs/GPS/mutag-GPS-Ricci-ollivier.yaml"   "C3 MUTAG+LapPE+RicciOllivier"   "$REPEAT_SMALL"
run_exp "configs/GPS/enzymes-GPS-Ricci-ollivier.yaml" "C3 ENZYMES+LapPE+RicciOllivier" "$REPEAT_SMALL"
run_exp "configs/GPS/nci1-GPS-Ricci-ollivier.yaml"    "C3 NCI1+LapPE+RicciOllivier"    "$REPEAT_NCI1"

echo ""
echo "========================================================"
echo "  All Ollivier H1 experiments complete."
echo "  Collect: python scripts/collect_h1_results.py --variant ollivier"
echo "========================================================"
