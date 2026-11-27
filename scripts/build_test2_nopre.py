#!/usr/bin/env python3
"""build_test2_nopre.py — test 2's manifest with the frame's opening line dropped (SF, 2026-11-27).

The refusal check (scripts/build_refusal_diag.py, results/probe_refusal_diag_1127_*) found that Opus 5
answered the revised P7-02 identity prompt, refused three times as sent, once the frame's opening line
("You are assisting with a scholarly experiment in structural literary analysis.") was dropped. SF
asked for all of test 2 to be sent again on Opus 5 in that frame, so that every cell comes from one
frame, with refusals recorded and the remaining tries sent (run_probe_batch.py
--continue-after-refusal).

Every item of specs/probe/loopA_patterns_it_D2.json is copied with that one line, and the blank line
after it, removed; nothing else changes (checked). Writes specs/probe/loopA_patterns_it_D2_nopre.json;
refuses to overwrite.

  python3 scripts/build_test2_nopre.py
  python3 scripts/run_probe_batch.py --manifest specs/probe/loopA_patterns_it_D2_nopre.json \\
      --backend anthropic --model claude-opus-5 --disable-thinking --n 5 --tag loopA-1127 \\
      --continue-after-refusal
"""
import hashlib, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "specs" / "probe" / "loopA_patterns_it_D2.json"
OUT = ROOT / "specs" / "probe" / "loopA_patterns_it_D2_nopre.json"
PRE = "You are assisting with a scholarly experiment in structural literary analysis.\n\n"


def main():
    if OUT.exists():
        sys.exit(f"build_test2_nopre: {OUT.relative_to(ROOT)} exists; refusing to overwrite")
    raw = SRC.read_bytes()
    man = json.loads(raw)
    items = []
    for it in man["items"]:
        p = it["prompt"]
        assert p.startswith(PRE) and p.count(PRE) == 1, it["id"]
        items.append({**it, "prompt": p[len(PRE):]})
        assert PRE + items[-1]["prompt"] == p
    OUT.write_text(json.dumps({
        "built_by": "scripts/build_test2_nopre.py",
        "test": man["test"],
        "what": man["what"] + "; frame without its opening line (" + PRE.strip() + ")",
        "derived_from": {"file": str(SRC.relative_to(ROOT)), "sha1": hashlib.sha1(raw).hexdigest()},
        "sources": man["sources"],
        "items": items}, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)}: {len(items)} prompts")


if __name__ == "__main__":
    main()
