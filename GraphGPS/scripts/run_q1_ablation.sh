#!/usr/bin/env bash
# Run Q1 experiments: Baseline (C1, with Pbase) vs. Zero-Embedding (C2, P=∅)
# for each of the 5 study datasets.
#
# After completion, collect results with:
#   python scripts/collect_q1_results.py
#
# Usage:
#   bash scripts/run_q1_ablation.sh              # sequential, 3 seeds, CPU/auto GPU
#   REPEAT=5 bash scripts/run_q1_ablation.sh     # 5 seeds per experiment
#   SKIP_SLOW=1 bash scripts/run_q1_ablation.sh  # skip ZINC (2000 epochs) and molhiv
#
# All results are written to results/<config-name>/<seed>/{train,val,test}/stats.json
# The run that finishes last calls agg_runs automatically; the collector reads
# results/<config-name>/agg/test/best.json

set -euo pipefail

REPEAT="${REPEAT:-3}"          # number of independent seeds per config
SKIP_SLOW="${SKIP_SLOW:-0}"    # set to 1 to skip ZINC (2000 ep) and molhiv (slow)

# Move to GraphGPS root so `python main.py` works
cd "$(dirname "$0")/.."

echo "========================================================"
echo "  Q1 Zero-Embedding Ablation — GraphGPS §2.2"
echo "  Seeds per experiment: $REPEAT"
echo "  Working directory:    $(pwd)"
echo "========================================================"

run_exp() {
    local cfg_file="$1"
    local label="$2"
    echo ""
    echo "--- [$label] $cfg_file ---"
    python main.py \
        --cfg "$cfg_file" \
        --repeat "$REPEAT" \
        wandb.use False
    echo "--- done: $cfg_file ---"
}

# ── C1: Baseline (with Pbase) ──────────────────────────────────────────────
if [ "$SKIP_SLOW" -eq 0 ]; then
    run_exp "configs/GPS/zinc-GPS+RWSE.yaml"        "C1 ZINC+RWSE"
    run_exp "configs/GPS/ogbg-molhiv-GPS.yaml"      "C1 molhiv+LapPE"
fi
run_exp "configs/GPS/mutag-GPS.yaml"            "C1 MUTAG+LapPE"
run_exp "configs/GPS/enzymes-GPS.yaml"          "C1 ENZYMES+LapPE"
run_exp "configs/GPS/nci1-GPS.yaml"             "C1 NCI1+LapPE"

# ── C2: Zero-Embedding (P=∅) ──────────────────────────────────────────────
if [ "$SKIP_SLOW" -eq 0 ]; then
    run_exp "configs/GPS/zinc-GPS-noPE.yaml"         "C2 ZINC noPE"
    run_exp "configs/GPS/ogbg-molhiv-GPS-noPE.yaml"  "C2 molhiv noPE"
fi
run_exp "configs/GPS/mutag-GPS-noPE.yaml"       "C2 MUTAG noPE"
run_exp "configs/GPS/enzymes-GPS-noPE.yaml"     "C2 ENZYMES noPE"
run_exp "configs/GPS/nci1-GPS-noPE.yaml"        "C2 NCI1 noPE"

echo ""
echo "========================================================"
echo "  All Q1 experiments complete."
echo "  Collect results: python scripts/collect_q1_results.py"
echo "========================================================"
