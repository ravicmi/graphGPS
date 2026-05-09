#!/usr/bin/env bash
# Run H1 / Q2 experiments: Baseline (C1, with Pbase=LapPE) vs.
# C3 = Pbase + Ricci curvature edge encoding, for each H1 study dataset.
#
# Compared to Q1 (zero-embedding ablation), here we test the *marginal* value
# of a single left-behind SE on top of the standard Pbase:
#
#     Δ(p; D) = Perf_D(fθ(Pbase ∪ {p})) − Perf_D(fθ(Pbase))
#
# Forman is fast (combinatorial). Ollivier is slower (optimal-transport per
# edge); it is staged but commented out by default — flip RUN_OLLIVIER=1 to
# include it.
#
# After completion, collect results with:
#     python scripts/collect_h1_results.py
#
# Usage:
#     bash scripts/run_h1_ablation.sh                     # 10 seeds, Forman only
#     REPEAT_SMALL=5 bash scripts/run_h1_ablation.sh      # 5 seeds
#     RUN_OLLIVIER=1 bash scripts/run_h1_ablation.sh      # also run Ollivier
#     RUN_BASELINE=1 bash scripts/run_h1_ablation.sh      # also (re)run C1 baselines
#
# Results are written to results/<config-name>/<seed>/{train,val,test}/stats.json
# and aggregated to results/<config-name>/agg/test/best.json by agg_runs.

set -euo pipefail

REPEAT_SMALL="${REPEAT_SMALL:-10}"
RUN_OLLIVIER="${RUN_OLLIVIER:-0}"
RUN_BASELINE="${RUN_BASELINE:-0}"

cd "$(dirname "$0")/.."

echo "========================================================"
echo "  H1 Ricci-curvature Ablation — GraphGPS §2.2 / §2.3"
echo "  Seeds: $REPEAT_SMALL  (MUTAG / ENZYMES / NCI1)"
echo "  Ollivier enabled: $RUN_OLLIVIER"
echo "  Re-run C1 baselines: $RUN_BASELINE"
echo "  Working directory:    $(pwd)"
echo "========================================================"

run_exp() {
    local cfg_file="$1"
    local label="$2"
    local repeat="${3:-$REPEAT_SMALL}"
    echo ""
    echo "--- [$label] $cfg_file  (seeds: $repeat) ---"
    python main.py \
        --cfg "$cfg_file" \
        --repeat "$repeat" \
        wandb.use False
    echo "--- done: $cfg_file ---"
}

# Optionally (re)run the C1 baselines so Δ(p; D) has a fresh comparator.
if [ "$RUN_BASELINE" -eq 1 ]; then
    run_exp "configs/GPS/mutag-GPS.yaml"     "C1 MUTAG+LapPE"
    run_exp "configs/GPS/enzymes-GPS.yaml"   "C1 ENZYMES+LapPE"
    run_exp "configs/GPS/nci1-GPS.yaml"      "C1 NCI1+LapPE"
fi

# C3 (Forman): Pbase ∪ {Ricci-Forman}
run_exp "configs/GPS/mutag-GPS-Ricci-forman.yaml"     "C3 MUTAG+LapPE+RicciForman"
run_exp "configs/GPS/enzymes-GPS-Ricci-forman.yaml"   "C3 ENZYMES+LapPE+RicciForman"
run_exp "configs/GPS/nci1-GPS-Ricci-forman.yaml"      "C3 NCI1+LapPE+RicciForman"

# C3 (Ollivier): slower, optimal-transport curvature.
if [ "$RUN_OLLIVIER" -eq 1 ]; then
    run_exp "configs/GPS/mutag-GPS-Ricci-ollivier.yaml"   "C3 MUTAG+LapPE+RicciOllivier"
    run_exp "configs/GPS/enzymes-GPS-Ricci-ollivier.yaml" "C3 ENZYMES+LapPE+RicciOllivier"
    run_exp "configs/GPS/nci1-GPS-Ricci-ollivier.yaml"    "C3 NCI1+LapPE+RicciOllivier"
fi

echo ""
echo "========================================================"
echo "  All H1 experiments complete."
echo "  Collect results: python scripts/collect_h1_results.py"
echo "========================================================"
