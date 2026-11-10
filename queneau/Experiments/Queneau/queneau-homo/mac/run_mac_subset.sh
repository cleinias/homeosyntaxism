#!/usr/bin/env bash
# run_mac_subset.sh — overnight minimal-computing run on a MacBook Air (Apple Silicon, Ollama).
#
# Validates the "runs on a student laptop" tier of the constrained-homeosyntaxism tool on the
# ACTUAL target stack: Apple-silicon unified memory + Metal + GGUF via Ollama. Scoped to the
# models/quants that fit a 16 GB machine — 7B-Q4_K_M, 7B-Q8_0, 14B-Q4_K_M — so a 24 GB Air is a
# faithful proxy (those three produce identical output on 16 GB and 24 GB; RAM headroom changes
# only fit/speed, never the tokens). Source-blind, protocol- and prompt-identical to the ACES
# runs; only the box + quant differ.
#
# EN subset: 3 models × 3 scenarios × {S0,D0,D1,D2} × N=10 = 360 generations. gen_ollama.py loops
# N (Ollama ignores n>1). Outputs → ./outputs/. Bring them back and score with the usual scorer.
#
# Run:  ./run_mac_subset.sh      (self-caffeinates so the Mac won't sleep mid-run)

set -uo pipefail

# --- keep the Mac awake for the whole overnight run -------------------------------------------
# Hand caffeinate the interpreter, not the script: `caffeinate -i "$0"` executes the file
# directly and dies with "Operation not permitted" whenever the copy lost its execute bit or
# carries com.apple.quarantine — including when the user correctly ran `bash run_mac_subset.sh`,
# since this line discards that bash.
if command -v caffeinate >/dev/null 2>&1 && [ -z "${CAFFEINATED:-}" ]; then
  exec env CAFFEINATED=1 caffeinate -i /bin/bash "$0" "$@"
fi

HERE="$(cd "$(dirname "$0")" && pwd)"; cd "$HERE"
ENDPOINT="${ENDPOINT:-http://127.0.0.1:11434}"
N="${N:-10}"
TEMP="${TEMP:-0.9}"
SCENARIOS="security-checkpoint hospital-emergency-room grocery-checkout"
DEGREES="S0 D0 D1 D2"

# (ollama model tag | output tag) — the <=16 GB tier only
MODELS=(
  "qwen2.5:7b-instruct-q4_K_M|mac-7b-q4"
  "qwen2.5:7b-instruct-q8_0|mac-7b-q8"
  "qwen2.5:14b-instruct-q4_K_M|mac-14b-q4"
)

command -v ollama  >/dev/null 2>&1 || { echo "Install Ollama first: https://ollama.com/download"; exit 1; }
command -v python3 >/dev/null 2>&1 || { echo "Need python3 — run: xcode-select --install"; exit 1; }

# --- make sure the Ollama server is up --------------------------------------------------------
if ! curl -sf "$ENDPOINT/api/tags" >/dev/null 2>&1; then
  echo "starting 'ollama serve' ..."
  (ollama serve >/dev/null 2>&1 &)
  sleep 6
fi
mkdir -p outputs

# --- pre-flight: pull all three up front, with retries, so problems surface NOW, not at 3am ---
echo "== pulling models (~22 GB total: 4.7 + 8.1 + 9.0 GB) =="
PULLED=()
for cfg in "${MODELS[@]}"; do
  M="${cfg%%|*}"
  ok=0
  for att in 1 2 3; do
    echo "  pull $M  (attempt $att)"
    if ollama pull "$M"; then ok=1; break; fi
    sleep 10
  done
  if [ "$ok" = 1 ]; then
    PULLED+=("$cfg")
  else
    echo "  [warn] could NOT pull $M — skipping it (check ollama.com/library/qwen2.5/tags)"
  fi
done
[ "${#PULLED[@]}" -gt 0 ] || { echo "no models pulled — aborting"; exit 1; }

# --- generate (idempotent: gen_ollama.py skips cells already written) -------------------------
echo "== generating: ${#PULLED[@]} model(s) × 3 scenarios × 4 degrees × N=$N =="
START=$(date +%s 2>/dev/null || echo 0)
for cfg in "${PULLED[@]}"; do
  IFS='|' read -r M TAG <<< "$cfg"
  echo "-- $M  ($TAG) --"
  for S in $SCENARIOS; do
    for D in $DEGREES; do
      python3 gen_ollama.py --endpoint "$ENDPOINT" --model "$M" \
          --lang en --degree "$D" --scenario "$S" --n "$N" --temperature "$TEMP" --tag "$TAG" \
        || echo "  [warn] cell $TAG/$S/$D errored — continuing"
    done
  done
done

echo "======================================================================"
echo "  done: $(ls outputs 2>/dev/null | wc -l | tr -d ' ') files in $HERE/outputs"
echo "  copy ./outputs back to the repo and score (same Stanza + Claude judge):"
echo "    <atlas-venv>/bin/python scripts/queneau_score_homo.py --glob 'homo_*mac-*'"
echo "======================================================================"
