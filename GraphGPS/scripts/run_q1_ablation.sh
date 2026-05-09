#!/usr/bin/env bash
# Run Q1 experiments: Baseline (C1, with Pbase) vs. Zero-Embedding (C2, P=∅)
# for each of the 5 study datasets.
#
# After completion, collect results with:
#   python scripts/collect_q1_results.py
#
# Usage:
#   bash scripts/run_q1_ablation.sh                       # sequential, 3 seeds (large), 10 seeds (small)
#   REPEAT=5 bash scripts/run_q1_ablation.sh              # 5 seeds for large datasets
#   REPEAT_SMALL=5 bash scripts/run_q1_ablation.sh        # 5 seeds for small datasets
#   SKIP_SLOW=1 bash scripts/run_q1_ablation.sh           # skip ZINC (2000 epochs) and molhiv
#
# All results are written to results/<config-name>/<seed>/{train,val,test}/stats.json
# The run that finishes last calls agg_runs automatically; the collector reads
# results/<config-name>/agg/test/best.json

set -euo pipefail

REPEAT="${REPEAT:-3}"              # seeds for large datasets (ZINC, molhiv, CIFAR10)
REPEAT_SMALL="${REPEAT_SMALL:-3}"  # seeds for small datasets (MUTAG, ENZYMES, NCI1)
SKIP_SLOW="${SKIP_SLOW:-1}"        # skip ZINC/molhiv/CIFAR10 by default (need GPU)
PATIENCE="${PATIENCE:-10}"         # early stopping patience (0 = disabled)

# Move to GraphGPS root so `python main.py` works
cd "$(dirname "$0")/.."

echo "========================================================"
echo "  Q1 Zero-Embedding Ablation — GraphGPS §2.2"
echo "  Seeds (large datasets): $REPEAT"
echo "  Seeds (small datasets): $REPEAT_SMALL  (MUTAG / ENZYMES / NCI1)"
echo "  Early-stop patience:    $PATIENCE epochs"
echo "  Working directory:    $(pwd)"
echo "========================================================"

run_exp() {
    local cfg_file="$1"
    local label="$2"
    local repeat="${3:-$REPEAT}"   # per-call override; falls back to $REPEAT
    echo ""
    echo "--- [$label] $cfg_file  (seeds: $repeat, patience: $PATIENCE) ---"
    python main.py \
        --cfg "$cfg_file" \
        --repeat "$repeat" \
        wandb.use False \
        optim.early_stop_patience "$PATIENCE"
    echo "--- done: $cfg_file ---"
}

# ── C1: Baseline (with Pbase) ──────────────────────────────────────────────
if [ "$SKIP_SLOW" -eq 0 ]; then
    run_exp "configs/GPS/zinc-GPS+RWSE.yaml"        "C1 ZINC+RWSE"
    run_exp "configs/GPS/ogbg-molhiv-GPS.yaml"      "C1 molhiv+LapPE"
fi
run_exp "configs/GPS/mutag-GPS.yaml"            "C1 MUTAG+LapPE"   "$REPEAT_SMALL"
run_exp "configs/GPS/enzymes-GPS.yaml"          "C1 ENZYMES+LapPE" "$REPEAT_SMALL"
run_exp "configs/GPS/nci1-GPS.yaml"             "C1 NCI1+LapPE"    "$REPEAT_SMALL"
if [ "$SKIP_SLOW" -eq 0 ]; then
    run_exp "configs/GPS/cifar10-GPS.yaml"      "C1 CIFAR10+LapPE"
fi

# ── C2: Zero-Embedding (P=∅) ──────────────────────────────────────────────
if [ "$SKIP_SLOW" -eq 0 ]; then
    run_exp "configs/GPS/zinc-GPS-noPE.yaml"         "C2 ZINC noPE"
    run_exp "configs/GPS/ogbg-molhiv-GPS-noPE.yaml"  "C2 molhiv noPE"
fi
run_exp "configs/GPS/mutag-GPS-noPE.yaml"       "C2 MUTAG noPE"    "$REPEAT_SMALL"
run_exp "configs/GPS/enzymes-GPS-noPE.yaml"     "C2 ENZYMES noPE"  "$REPEAT_SMALL"
run_exp "configs/GPS/nci1-GPS-noPE.yaml"        "C2 NCI1 noPE"     "$REPEAT_SMALL"
if [ "$SKIP_SLOW" -eq 0 ]; then
    run_exp "configs/GPS/cifar10-GPS-noPE.yaml" "C2 CIFAR10 noPE"
fi

echo ""
echo "========================================================"
echo "  All Q1 experiments complete."
echo "  Collect results: python scripts/collect_q1_results.py"
echo "========================================================"
