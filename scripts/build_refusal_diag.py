#!/usr/bin/env python3
"""build_refusal_diag.py — which part of the test-2 frame trips Opus 5's refusal (SF, 2026-11-27)?

Opus 5 refused 11 of the 32 test-2 items (stop_reason refusal, category `cyber`), every time and on
retry. This builds a small manifest of variants of two refused prompts, each changing ONE part of the
frame and never the pattern entry, so that a frame that clears the refusal can be used for a full,
comparable rerun of test 2:

  base prompts   T2-ident-C-P7-02-revised   (identity; refused 5 of 5 and on retry)
                 T2-recon-C-P5-01-both      (reconstruction; refused 5 of 5)
  variants       v0 control, the prompt as sent on 11-26
                 v1 no preamble ("You are assisting with a scholarly experiment ...")
                 v2 no "Do not write any new text."
                 v3 frame words: "specification" -> "description", "source" -> "original"
                    (in the frame only; the entry keeps its own words)
                 v4 the label "SPECIFICATION:" -> "PATTERN:"
                 v5 free answer instead of the field form ("Answer in a few sentences ...")

The prompts are cut from specs/probe/loopA_patterns_it_D2.json, so nothing is retyped. Writes
specs/probe/refusal_diag_1127.json; refuses to overwrite.

  python3 scripts/build_refusal_diag.py
  python3 scripts/run_probe_batch.py --manifest specs/probe/refusal_diag_1127.json \\
      --backend anthropic --model claude-opus-5 --disable-thinking --n 1 --tag diag-1127 --only '<id>'
"""
import hashlib, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "specs" / "probe" / "loopA_patterns_it_D2.json"
OUT = ROOT / "specs" / "probe" / "refusal_diag_1127.json"
BASES = ["T2-ident-C-P7-02-revised", "T2-recon-C-P5-01-both"]
PRE = "You are assisting with a scholarly experiment in structural literary analysis.\n\n"
LABEL = "SPECIFICATION:\n"
FORM = "\n\nAnswer in exactly this form, and nothing else:\n"


def split(prompt):
    """frame head (up to and including the label), entry, frame tail (from the answer form on)."""
    assert prompt.count(LABEL) == 1 and prompt.count(FORM) == 1
    head, rest = prompt.split(LABEL)
    entry, tail = rest.split(FORM)
    return head + LABEL, entry, FORM + tail


def once(s, old, new):
    assert s.count(old) == 1, old
    return s.replace(old, new)


def variants(prompt):
    head, entry, tail = split(prompt)
    v = {"v0": head + entry + tail,
         "v1": once(head, PRE, "") + entry + tail,
         "v2": once(head, "Do not write any new text. ", "") + entry + tail,
         "v3": head.replace("specification", "description").replace("source", "original")
               + entry + tail,
         "v4": once(head, LABEL, "PATTERN:\n") + entry + tail,
         "v5": head + entry + "\n\nAnswer in a few sentences, naming the author and the title of the "
               "work if you know them, or saying that you do not.\n"}
    assert v["v0"] == prompt
    for k, p in v.items():
        assert entry in p, k   # the pattern entry is never touched
    return v


def main():
    if OUT.exists():
        sys.exit(f"build_refusal_diag: {OUT.relative_to(ROOT)} exists; refusing to overwrite")
    raw = SRC.read_bytes()
    src = {i["id"]: i for i in json.loads(raw)["items"]}
    items = []
    for b in BASES:
        for k, p in variants(src[b]["prompt"]).items():
            items.append({"id": f"diag-{b}-{k}", "base": b, "variant": k, "prompt": p})
    OUT.write_text(json.dumps({
        "built_by": "scripts/build_refusal_diag.py",
        "what": "refusal diagnosis: variants of two refused test-2 prompts, one frame change each",
        "sources": {"manifest": str(SRC.relative_to(ROOT)), "sha1": hashlib.sha1(raw).hexdigest()},
        "items": items}, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)}: {len(items)} prompts")


if __name__ == "__main__":
    main()
