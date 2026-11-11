#!/usr/bin/env bash
# run_fable5_1840.sh — Fable 5 on the 1840 Manzoni incipit, for Appendix A of the CHR 2027 paper.
#
# Appendix A says Fable 5 accepted the clause-architecture (D2) specification in all ten runs.
# The run behind that sentence used the pre-1840 text; this re-runs it on the 1840 edition,
# with the same prompt as the Opus 5 runs: English, source-blind, ten generations, Fable 5's
# default (always-on) thinking at effort low, 16k-token budget. Then it scores the run.
#
# Run from the repository root:
#     export ANTHROPIC_API_KEY=sk-ant-...      # or put it in .env (see .env.example)
#     bash scripts/run_fable5_1840.sh
#
# Needs Python 3.12 with `pip install -r requirements.txt` (stanza 1.12.2; the pin matters),
# and internet for the API and, the first time, for Stanza's English model.
# Cost: ten Fable 5 calls (~3k tokens in, up to 16k out each) plus ~20 Sonnet 4.6 judge calls.
set -euo pipefail
cd "$(dirname "$0")/.."

RUN=results/unrestricted_en_1840_fable5_n10.json
SCORED=results/scored_unrestricted_en_1840_fable5_n10.json
PY=${PYTHON:-python3}

if [[ -z "${ANTHROPIC_API_KEY:-}" ]] && ! grep -qs '^ANTHROPIC_API_KEY=.' .env; then
    echo "No ANTHROPIC_API_KEY in the environment or in .env" >&2; exit 1
fi
if [[ -e $RUN ]]; then
    echo "$RUN already exists; move it away to run again" >&2; exit 1
fi
"$PY" -c 'import stanza' 2>/dev/null || { echo "stanza missing: pip install -r requirements.txt" >&2; exit 1; }

echo "== 1/3  generating (Fable 5, 10 candidates)"
"$PY" scripts/run_unrestricted_manzoni.py \
    --constrained results/production_D2_en_pilot-1840_claude-opus-5.json \
    --model claude-fable-5 --n 10 --max-tokens 16000 --effort low \
    --out "$RUN"

echo "== 2/3  refusals"
"$PY" - "$RUN" <<'EOF'
import json, sys
d = json.load(open(sys.argv[1]))
n, ref = d.get("n"), d.get("refused")
print(f"   {n} generations kept, {ref} refused")
print("   -> the Appendix A sentence stands" if not ref else
      "   -> REFUSALS: the Appendix A sentence must change; tell Stefano")
EOF

echo "== 3/3  scoring (Stanza + Sonnet 4.6 judges)"
"$PY" scripts/score_production.py "$RUN"
"$PY" scripts/make_run_review.py --scored "$SCORED" --out results/manzoni_en_1840_fable5_review.md \
    || echo "   (reading copy not written; the scored file is what matters)"

cat <<EOF

Done. Commit and push the run:
    git add $RUN $SCORED
    git commit -m "Fable 5 on the 1840 incipit (Appendix A)"
    git push
The reading copy results/manzoni_en_1840_fable5_review.md is for us; it need not be committed.
EOF
