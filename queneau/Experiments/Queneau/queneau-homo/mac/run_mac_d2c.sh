#!/usr/bin/env bash
# run_mac_d2c.sh — fill the one degree the laptop tier never ran: D2c, clause NESTING.
#
# The minimal-computing table (tab:mincompute) has rows S0, D0, D1, D3 and no D2, not because
# anyone decided to leave it out but because the original overnight run covered {S0,D0,D1,D2}
# and D2c did not exist yet. Note the off-by-one: files named D2 are the paper's D3 (clause
# INVENTORY); the paper's D2 (clause NESTING) lives in files named D2c. So this run adds the
# missing row rather than repeating one.
#
# Prompt parity is not assumed, it is checked: gen_ollama.py's D2c entry was ported verbatim
# from scripts/queneau_run_homo.py and verified byte-identical to the prompt the ACES ladder
# actually sent, in all three scenarios. The pre-flight below re-checks it here before spending
# a single token, because a laptop-vs-cluster row is worthless if the two were told different
# things — which is exactly what we found had happened at D0.
#
# EN only, 3 models x 3 scenarios x N=10 = 90 generations. Idempotent: gen_ollama.py skips any
# cell already written, so it is safe to re-run after an interruption.
#
# Run:  ./run_mac_d2c.sh        (self-caffeinates so the Mac will not sleep mid-run)

set -uo pipefail

# Re-exec under caffeinate so the Mac will not sleep mid-run. Hand caffeinate the INTERPRETER
# and the script as an argument, never the script alone: `caffeinate -i "$0"` would execute the
# file directly, which fails with "Operation not permitted" whenever the copy arrived without
# its execute bit or still carries com.apple.quarantine — and it fails even when the user
# correctly invoked `bash run_mac_d2c.sh`, because this line throws that bash away.
if command -v caffeinate >/dev/null 2>&1 && [ -z "${CAFFEINATED:-}" ]; then
  exec env CAFFEINATED=1 caffeinate -i /bin/bash "$0" "$@"
fi

HERE="$(cd "$(dirname "$0")" && pwd)"; cd "$HERE"
ENDPOINT="${ENDPOINT:-http://127.0.0.1:11434}"
N="${N:-10}"
TEMP="${TEMP:-0.9}"
SCENARIOS="security-checkpoint hospital-emergency-room grocery-checkout"

MODELS=(
  "qwen2.5:7b-instruct-q4_K_M|mac-7b-q4"
  "qwen2.5:7b-instruct-q8_0|mac-7b-q8"
  "qwen2.5:14b-instruct-q4_K_M|mac-14b-q4"
)

command -v ollama  >/dev/null 2>&1 || { echo "Install Ollama first: https://ollama.com/download"; exit 1; }
command -v python3 >/dev/null 2>&1 || { echo "Need python3 — run: xcode-select --install"; exit 1; }
[ -f spec/specs_en_D2c.json ] || { echo "missing spec/specs_en_D2c.json — copy the whole mac/ folder over"; exit 1; }

if ! curl -sf "$ENDPOINT/api/tags" >/dev/null 2>&1; then
  echo "starting 'ollama serve' ..."
  (ollama serve >/dev/null 2>&1 &)
  sleep 6
fi
mkdir -p outputs

# --- pre-flight: the prompt this machine will send must be the one the cluster sent -----------
echo "== prompt parity check =="
python3 - <<'PY' || { echo "!! prompt check failed — not running"; exit 1; }
import json, importlib.util
spec = importlib.util.spec_from_file_location("go", "gen_ollama.py")
go = importlib.util.module_from_spec(spec); spec.loader.exec_module(go)
s = json.load(open("spec/specs_en_D2c.json"))
for scen, expect in (("security-checkpoint", 2955),
                     ("hospital-emergency-room", 2923),
                     ("grocery-checkout", 2943)):
    n = len(go.build_prompt(s, "en", scen, "D2c"))
    flag = "ok" if n == expect else "MISMATCH"
    print(f"   {scen:26} {n} chars (cluster sent {expect}) {flag}")
    assert n == expect, f"{scen}: prompt is {n} chars, cluster sent {expect}"
print("   -> identical to the ACES ladder prompt")
PY

echo "== pulling models (skip if already present) =="
PULLED=()
for cfg in "${MODELS[@]}"; do
  M="${cfg%%|*}"
  ok=0
  for att in 1 2 3; do
    echo "  pull $M  (attempt $att)"
    if ollama pull "$M"; then ok=1; break; fi
    sleep 10
  done
  if [ "$ok" = 1 ]; then PULLED+=("$cfg"); else echo "  [warn] could NOT pull $M — skipping"; fi
done
[ "${#PULLED[@]}" -gt 0 ] || { echo "no models pulled — aborting"; exit 1; }

echo "== generating: ${#PULLED[@]} model(s) x 3 scenarios x D2c x N=$N =="
for cfg in "${PULLED[@]}"; do
  IFS='|' read -r M TAG <<< "$cfg"
  echo "-- $M  ($TAG) --"
  for S in $SCENARIOS; do
    python3 gen_ollama.py --endpoint "$ENDPOINT" --model "$M" \
        --lang en --degree D2c --scenario "$S" --n "$N" --temperature "$TEMP" --tag "$TAG" \
      || echo "  [warn] cell $TAG/$S/D2c errored — continuing"
  done
done

echo "======================================================================"
echo "  D2c cells now in $HERE/outputs:"
ls outputs/homo_D2c_* 2>/dev/null | wc -l | sed 's/^/    /'
echo "  expected 9 (3 models x 3 scenarios)"
echo
echo "  Copy them back into the repo and score:"
echo "    Experiments/Queneau/queneau-homo/outputs/"
echo "    <atlas-venv>/bin/python scripts/queneau_score_homo.py --glob 'homo_D2c_*mac-*'"
echo "======================================================================"
