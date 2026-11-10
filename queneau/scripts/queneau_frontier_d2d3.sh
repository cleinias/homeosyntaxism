#!/usr/bin/env bash
# queneau_frontier_d2d3.sh --- put a frontier model on the loose rungs of the Notations ladder.
#
# The gap this fills: every closed-weight cell in the Queneau corpus is at S0 or D0. Opus 4.8
# has 18 S0 cells and 21 D0 cells; it has never been asked for D2 (clause nesting) or D3
# (clause inventory), in any language, in any regime, and neither has any other frontier model.
#
# That matters because it is exactly where the argument turns. Section 4 shows Opus
# reconstructing the ordered tree and gaming it into nonsense -- fidelity 0.85, coherence 0.01,
# the calque corner -- and then argues that loosening to the clause degrees is what buys form
# AND sense. The second half has only ever been shown on open weights. We assert a gradient and
# demonstrate a frontier model failing at one end of it without ever asking it to attempt the
# other.
#
# Three outcomes, all worth having:
#   - it clears both bars      -> the calque corner is a property of OVER-SPECIFICATION, not of
#                                 the model, which sharpens the thesis
#   - it games D2/D3 as well   -> the calque behaviour belongs to the model regardless of how
#                                 loose the target is: a different, stronger claim
#   - it refuses               -> the ethico-political point a third time, after Fable 5 at D0
#                                 and the coherence judge on degenerate text
#
# Prompts are byte-identical to the ones the ACES ladder sent (verified: 18/18), so Opus and
# Qwen answer the same instruction and the cells drop straight into the existing comparisons.
# The pre-flight re-checks that here before spending anything.
#
# 2 degrees x 3 languages x 3 scenarios x N=10 = 180 generations, ~$2.25 all in.
# Source-blind only, matching the ladder, which has no copy regime.
#
#   bash scripts/queneau_frontier_d2d3.sh --dry-run
#   bash scripts/queneau_frontier_d2d3.sh
set -uo pipefail
cd "$(dirname "$0")/.."

PY=../Oulipo-Atlas-2-0/.venv/bin/python
OUT=Experiments/Queneau/queneau-homo/outputs
MODEL="${MODEL:-claude-opus-4-8}"
TAG="${TAG:-b1}"
N="${N:-10}"
DRY=0; [ "${1:-}" = "--dry-run" ] && DRY=1

echo "== prompt parity check (must match the ACES ladder byte for byte) =="
$PY - <<'PROMPTCHECK' || { echo "!! parity check failed — not running"; exit 1; }
import json, glob, importlib.util, sys
spec = importlib.util.spec_from_file_location("qrh", "scripts/queneau_run_homo.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
bad = 0
for deg, tag in (("D2c", "ladderD2c"), ("D2", "ladderD2")):
    for lang in ("en", "fr", "it"):
        s = json.load(open(f"Experiments/Queneau/queneau-homo/spec/specs_{lang}_{deg}.json"))
        for scen in ("security-checkpoint", "hospital-emergency-room", "grocery-checkout"):
            p = m.build_prompt(s, lang, scen, deg, False)
            g = glob.glob(f"Experiments/Queneau/queneau-homo/outputs/homo_{deg}_{lang}_{scen}_*{tag}"
                          f"_Qwen_Qwen2.5-7B-Instruct.json")
            if g and json.load(open(g[0]))["prompt"] != p:
                print(f"   MISMATCH {deg} {lang} {scen}"); bad += 1
print(f"   18 prompts checked, {bad} mismatches")
sys.exit(1 if bad else 0)
PROMPTCHECK

echo
echo "== generating: $MODEL, 2 degrees x 3 langs x 3 scenarios x N=$N =="
for DEG in D2c D2; do
  for LANG in en fr it; do
    for SCEN in security-checkpoint hospital-emergency-room grocery-checkout; do
      F="$OUT/homo_${DEG}_${LANG}_${SCEN}_blind_${TAG}_${MODEL}.json"
      if [ -f "$F" ]; then echo "  [have] $(basename "$F")"; continue; fi
      if [ "$DRY" = 1 ]; then echo "  [would run] $DEG $LANG $SCEN"; continue; fi
      $PY scripts/queneau_run_homo.py --backend anthropic --model "$MODEL" \
          --lang "$LANG" --degree "$DEG" --scenario "$SCEN" \
          --n "$N" --max-tokens 1024 --tag "$TAG" \
        || echo "  !! CELL FAILED $DEG $LANG $SCEN"
    done
  done
done

echo
echo "=========================================================================="
echo "  frontier cells on disk: $(ls $OUT/homo_D2c_*_${TAG}_${MODEL}.json $OUT/homo_D2_*_${TAG}_${MODEL}.json 2>/dev/null | wc -l) of 18"
echo "  next:  $PY scripts/queneau_score_homo.py --glob 'homo_D2*_*_${TAG}_${MODEL}.json'"
echo "  then compare against the Qwen ladder at the same degrees and prompt."
echo "=========================================================================="
