#!/usr/bin/env bash
# queneau_d0_grid_b1.sh --- bring the b1 D0 arm up to the same 3x3 grid as S0.
#
# Table 1 pairs S0 rows aggregated over 9 cells (3 languages x 3 scenarios) with D0
# rows built from a single cell (en / security-checkpoint). This fills the other 8,
# and adds Maverick, which had no D0 cells at all: 4 models x 2 regimes x 9 cells.
#
# Provider pins reproduce the June b1 runs exactly (fp8 everywhere, verified against
# provider_routing in the existing 76 cells): Qwen-7B on Together, Qwen-72B and
# Maverick on DeepInfra. Opus 4.8 goes direct and takes no temperature (400 otherwise).
#
# Resumable: a cell whose output file already exists is skipped, so re-running after
# an interruption costs nothing. Usage:
#   bash scripts/queneau_d0_grid_b1.sh anthropic     # the 16 Opus cells
#   bash scripts/queneau_d0_grid_b1.sh openrouter    # the 50 open-weight cells
set -uo pipefail
cd "$(dirname "$0")/.."

ARM="${1:-all}"
PY=../Oulipo-Atlas-2-0/.venv/bin/python
OUT=Experiments/Queneau/queneau-homo/outputs
LANGS="en fr it"
SCENARIOS="grocery-checkout hospital-emergency-room security-checkpoint"

run_model () {                       # $1 model  $2 backend  $3 provider ("" = none)
  local model="$1" backend="$2" provider="$3"
  local slug="${model//\//_}"
  local provarg=""; [ -n "$provider" ] && provarg="--provider $provider"
  for lang in $LANGS; do
    for scen in $SCENARIOS; do
      for regime in blind vis; do
        local visarg=""; [ "$regime" = vis ] && visarg="--source-visible"
        local f="$OUT/homo_D0_${lang}_${scen}_${regime}_b1_${slug}.json"
        if [ -f "$f" ]; then echo "  [have] $(basename "$f")"; continue; fi
        $PY scripts/queneau_run_homo.py --backend "$backend" --model "$model" \
            $provarg --lang "$lang" --degree D0 --scenario "$scen" \
            --n 10 --max-tokens 1024 $visarg --tag b1 \
          || echo "  !! FAILED $(basename "$f")"
      done
    done
  done
}

if [ "$ARM" = anthropic ] || [ "$ARM" = all ]; then
  run_model claude-opus-4-8 anthropic ""
fi
if [ "$ARM" = openrouter ] || [ "$ARM" = all ]; then
  run_model qwen/qwen-2.5-7b-instruct   openrouter Together
  run_model qwen/qwen-2.5-72b-instruct  openrouter DeepInfra
  run_model meta-llama/llama-4-maverick openrouter DeepInfra
fi
echo "=== $ARM arm done: $(ls $OUT/homo_D0_*_b1_*.json | wc -l) D0 b1 cells on disk ==="
