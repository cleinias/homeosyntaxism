#!/usr/bin/env python3
"""build_p2_opening_check.py — is it P2-01's claim or our word "opening" that names the novel?

SF, 2026-11-27. In the test-2 rerun (frame without its opening line), the unchanged P2-01 entry led
Opus 5 to I promessi sposi 5 of 5 ("Quel ramo del lago di Como" a novenario). Only one wording had
been sent, and it asks for the passage's OPENING syntagm, which may point at the novel's first line
by itself. This builds the identity prompt for P2-01 in two variants, everything else as in the
rerun (specs/probe/loopA_patterns_it_D2_nopre.json, item T2-ident-C-P2-01-both):

  one    "The opening syntagm of the passage scans" -> "One syntagm of the passage scans"
  drop   the opening-verse clause dropped: "The passage contains >= 3 total embedded metric units
         (novenario / settenario / endecasillabo) inside descriptive sentences. ..."

Writes specs/probe/p2_opening_check_1127.json; refuses to overwrite.

  python3 scripts/build_p2_opening_check.py
  python3 scripts/run_probe_batch.py --manifest specs/probe/p2_opening_check_1127.json \\
      --backend anthropic --model claude-opus-5 --disable-thinking --n 5 --tag p2check-1127 \\
      --continue-after-refusal
"""
import hashlib, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "specs" / "probe" / "loopA_patterns_it_D2_nopre.json"
OUT = ROOT / "specs" / "probe" / "p2_opening_check_1127.json"
BASE = "T2-ident-C-P2-01-both"
OPENING = ("The opening syntagm of the passage scans as a canonical Italian verse (novenario, "
           "settenario, or endecasillabo); the passage contains")
VARIANTS = {"one": "One syntagm of the passage scans as a canonical Italian verse (novenario, "
                   "settenario, or endecasillabo); the passage contains",
            "drop": "The passage contains"}


def main():
    if OUT.exists():
        sys.exit(f"build_p2_opening_check: {OUT.relative_to(ROOT)} exists; refusing to overwrite")
    raw = SRC.read_bytes()
    base = next(i for i in json.loads(raw)["items"] if i["id"] == BASE)
    assert base["prompt"].count(OPENING) == 1
    items = [{**{k: v for k, v in base.items() if k not in ("id", "prompt")},
              "id": f"P2check-{k}", "base": BASE, "variant": k,
              "prompt": base["prompt"].replace(OPENING, new)} for k, new in VARIANTS.items()]
    OUT.write_text(json.dumps({
        "built_by": "scripts/build_p2_opening_check.py",
        "what": "P2-01 identity prompt without 'opening': one syntagm / clause dropped",
        "derived_from": {"file": str(SRC.relative_to(ROOT)), "sha1": hashlib.sha1(raw).hexdigest(),
                         "item": BASE},
        "items": items}, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)}: {len(items)} prompts")
    for it in items:
        print(f"--- {it['id']}\n" + it["prompt"].split("SPECIFICATION:\n")[1].split("\n\nAnswer")[0])


if __name__ == "__main__":
    main()
