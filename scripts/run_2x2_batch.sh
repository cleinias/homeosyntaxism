#!/usr/bin/env bash
# run_2x2_batch.sh — ten more texts in each arm of the 2x2 (patterns x embargo), Opus 5, IT, D2,
# so that every arm reaches twenty (SF, 2026-11-27). Same scripts and arguments as the 11-19
# runs, new names; each runner refuses to send unless its final prompt equals the 11-19 arm's
# prompt byte for byte (--expect-prompt-of), and never overwrites a run.
#
#   scripts/run_2x2_batch.sh check   # dry run: prompts and output names checked, nothing sent
#   scripts/run_2x2_batch.sh run     # the four arms in parallel, one log each in logs/
#
# 11-19 arm (pat+emb, pat, emb, none)          -> this batch
#   production_D2_it_effectfree-1840_...        -> production_D2_it_effectfree-1840-b1127_...
#   production_D2_it_effectfree-noemb-1840_...  -> production_D2_it_effectfree-noemb-1840-b1127_...
#   unrestricted_it_1840_opus5_clean_n10        -> unrestricted_it_1840_opus5_clean_n10_b1127
#   unrestricted_it_1840_opus5_noemb_n10        -> unrestricted_it_1840_opus5_noemb_n10_b1127
set -euo pipefail
cd "$(dirname "$0")/.."
mode=${1:-check}
B=b1127
JULY=results/production_D2_it_constr-1840_claude-opus-5.json
COMMON=(--model claude-opus-5 --n 10 --max-tokens 4096 --disable-thinking)

arm_pat_emb() {
  python3 scripts/run_homeo.py --backend anthropic "${COMMON[@]}" --lang it --degree D2 \
    --domain specs/domain_brief_3item.txt --same-prompt-as "$JULY" \
    --prompt-edits specs/effect_free_edits_it_D2.json --tag "effectfree-1840-$B" \
    --expect-prompt-of results/production_D2_it_effectfree-1840_claude-opus-5.json "$@"
}
arm_pat() {
  python3 scripts/run_homeo.py --backend anthropic "${COMMON[@]}" --lang it --degree D2 \
    --domain specs/domain_brief_3item.txt --same-prompt-as "$JULY" \
    --prompt-edits specs/effect_free_noembargo_edits_it_D2.json --tag "effectfree-noemb-1840-$B" \
    --expect-prompt-of results/production_D2_it_effectfree-noemb-1840_claude-opus-5.json "$@"
}
arm_emb() {
  python3 scripts/run_unrestricted_manzoni.py --constrained "$JULY" "${COMMON[@]}" \
    --out "results/unrestricted_it_1840_opus5_clean_n10_$B.json" \
    --expect-prompt-of results/unrestricted_it_1840_opus5_clean_n10.json "$@"
}
arm_none() {
  python3 scripts/run_unrestricted_manzoni.py --constrained "$JULY" "${COMMON[@]}" \
    --prompt-edits specs/noembargo_edits_it_D2.json \
    --out "results/unrestricted_it_1840_opus5_noemb_n10_$B.json" \
    --expect-prompt-of results/unrestricted_it_1840_opus5_noemb_n10.json "$@"
}

case "$mode" in
  check)
    for a in arm_pat_emb arm_pat arm_emb arm_none; do echo "== $a"; $a --dry-run; done ;;
  run)
    # every arm is checked again before anything is sent
    for a in arm_pat_emb arm_pat arm_emb arm_none; do $a --dry-run >/dev/null; done
    # shellcheck disable=SC1090
    source ~/.config/anthropic/env
    mkdir -p logs
    pids=()
    for a in arm_pat_emb arm_pat arm_emb arm_none; do
      $a > "logs/2x2_${B}_${a#arm_}.log" 2>&1 & pids+=($!)
    done
    rc=0; for p in "${pids[@]}"; do wait "$p" || rc=1; done
    tail -n 3 logs/2x2_${B}_*.log
    exit $rc ;;
  *) echo "usage: $0 check|run" >&2; exit 2 ;;
esac
