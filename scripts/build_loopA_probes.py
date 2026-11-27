#!/usr/bin/env python3
"""build_loopA_probes.py — build the prompt manifests for the loop-A probes (2026-11-26).

Two tests from the context-leakage draft (PoeticsToday/S3-4_S4-3_context-leakage_draft_2026-11-26.md,
note 15). Both ask a model to find the source; neither asks it to write.

  Test 2, per-pattern probe (before generation): each pattern entry of the Italian D2 prompt, alone,
  in its July wording and in its revised ("effect-free") wording. Entries the revision left alone
  appear once. Two questions per entry:
    identity        from which work and author was this specification derived?
    reconstruction  what did the source passage contain, who acted in it, which words does the
                    specification echo? (scored by hand against the incipit)
  Test 3, output probe: each of the 60 texts of the six Opus 5 IT D2 arms in Table 2, alone, with
  two questions:
    uncued  does this passage derive from, rewrite or imitate a known literary work?
    cued    this passage transposes a passage of an existing work; which one?

The entries are cut verbatim from the saved prompts of the July and revised runs, and the texts are
read from the run files, so nothing is retyped. Writes specs/probe/loopA_patterns_it_D2.json and
specs/probe/loopA_outputs_it_D2.json; each item carries its prompt verbatim. Refuses to overwrite.

  python scripts/build_loopA_probes.py            # build both manifests
  python scripts/build_loopA_probes.py --show 3   # also print the first 3 prompts of each

Second batch (SF, 2026-11-27: firm up test 3 on the 2x2 dataset). --batch b1127 builds only a test-3
manifest, specs/probe/loopA_outputs_it_D2_b1127.json, from the ten new texts of each of the four 2x2
arms (scripts/run_2x2_batch.sh), with the same two frames, verbatim. The arm keys are those of the
first manifest (stranger = constrained, no embargo) so the two can be pooled; item ids carry the
batch tag.

  python scripts/build_loopA_probes.py --batch b1127
"""
import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
OUTDIR = ROOT / "specs" / "probe"

PROMPTS = {"july": RES / "production_D2_it_constr-1840_claude-opus-5_PROMPT.txt",
           "revised": RES / "production_D2_it_effectfree-1840_claude-opus-5_PROMPT.txt"}

# the six arms of Table 2, in its column order
ARMS = [("july", "production_D2_it_constr-1840_claude-opus-5.json"),
        ("replicate", "production_D2_it_constr-1840-rep_claude-opus-5.json"),
        ("revised", "production_D2_it_effectfree-1840_claude-opus-5.json"),
        ("stranger", "production_D2_it_effectfree-noemb-1840_claude-opus-5.json"),
        ("nopat", "unrestricted_it_1840_opus5_clean_n10.json"),
        ("nopat-noemb", "unrestricted_it_1840_opus5_noemb_n10.json")]

# the four 2x2 arms of the test-3 table, for a later batch (--batch TAG): same keys as above
ARMS_2X2_BATCH = [("revised", "production_D2_it_effectfree-1840-{b}_claude-opus-5.json"),
                  ("stranger", "production_D2_it_effectfree-noemb-1840-{b}_claude-opus-5.json"),
                  ("nopat", "unrestricted_it_1840_opus5_clean_n10_{b}.json"),
                  ("nopat-noemb", "unrestricted_it_1840_opus5_noemb_n10_{b}.json")]

# names that would give the source away if a FRAME carried them (the texts are checked, not refused)
SOURCE_TERMS = ["manzoni", "manzonian", "lecco", "promessi", "como", "spagnol", "spanish",
                "guarnigione", "garrison"]

PREAMBLE = "You are assisting with a scholarly experiment in structural literary analysis.\n\n"

ANSWER_IDENTITY = """Answer in exactly this form, and nothing else:
AUTHOR: <name, or UNKNOWN>
TITLE: <title of the work, or UNKNOWN>
PASSAGE: <which part of the work, or UNKNOWN>
CONFIDENCE: <a number from 0 to 100>
EVIDENCE: <one or two sentences: what in the {what} led you to this answer>
"""

T2_IDENTITY = PREAMBLE + """TASK: Do not write any new text. The specification below was written for a writing \
experiment. It describes one formal pattern, abstracted from one existing literary passage (which is \
not shown) and from a critical reading of that passage; a writer was then asked to reproduce the \
pattern in a new setting. Identify the source: from which literary work, by which author, was this \
specification derived?

SPECIFICATION:
{entry}

""" + ANSWER_IDENTITY.format(what="specification")

T2_RECON = PREAMBLE + """TASK: Do not write any new text. The specification below was written for a writing \
experiment. It describes one formal pattern, abstracted from one existing literary passage (which is \
not shown) and from a critical reading of that passage. Reconstruct, as far as the specification \
allows, what the source passage contained: what it described or narrated, who or what acted in it, \
and which of its words or phrases the specification seems to echo. Where the specification gives you \
no ground for a reconstruction, write UNKNOWN. You need not identify the work; name it only if you \
recognise it.

SPECIFICATION:
{entry}

Answer in exactly this form, and nothing else:
CONTENT: <what the source passage described or narrated, or UNKNOWN>
AGENTS: <who or what acts in the source passage, or UNKNOWN>
ECHOED_WORDS: <words or phrases of the source that the specification seems to echo, or UNKNOWN>
AUTHOR: <name, or UNKNOWN>
TITLE: <title of the work, or UNKNOWN>
CONFIDENCE: <a number from 0 to 100: your confidence in the reconstruction>
"""

T3_UNCUED = PREAMBLE + """TASK: Do not write any new text. Read the passage of Italian prose below. Does it \
derive from, rewrite or imitate a known literary work? If it does, identify that work and its \
author. If you see no such source, answer UNKNOWN.

TEXT:
{text}

""" + ANSWER_IDENTITY.format(what="text")

T3_CUED = PREAMBLE + """TASK: Do not write any new text. The passage of Italian prose below is a \
transposition: it was written by keeping the structure of a passage from an existing literary work \
and replacing that passage's content with a new setting. The source passage is not shown. Identify \
it: from which literary work, by which author, was this passage derived?

TEXT:
{text}

""" + ANSWER_IDENTITY.format(what="text")


def sha1(b):
    return hashlib.sha1(b).hexdigest()


def entries(prompt):
    """The pattern entries of a generation prompt, in order: {id: two-line entry, dedented}."""
    block = re.search(r"\nFORMAL PATTERNS[^\n]*\n[^\n]*\n(.*?)\n\nTARGET DOMAIN", prompt, re.S)
    if not block:
        sys.exit("build_loopA_probes: no FORMAL PATTERNS block")
    lines = block.group(1).split("\n")
    out = {}
    for head, spec in zip(lines[0::2], lines[1::2]):
        m = re.match(r"\s*\[(C-P\d-\d\d)\]", head)
        if not (m and spec.strip().startswith("Specification:")):
            sys.exit(f"build_loopA_probes: unexpected entry layout at {head[:60]!r}")
        out[m.group(1)] = head.strip() + "\n  " + spec.strip()
    return out


def check_frame(template):
    frame = template.replace("{entry}", "").replace("{text}", "").lower()
    hits = [t for t in SOURCE_TERMS if t in frame]
    if hits:
        sys.exit(f"build_loopA_probes: source terms in a probe frame: {hits}")


def output_items(arms, id_tag=""):
    """Test-3 items: every text of each arm's run file, under both frames."""
    items, src = [], {}
    for arm, fn in arms:
        b = (RES / fn).read_bytes()
        gens = json.loads(b)["gens"]
        if len(gens) != 10:
            sys.exit(f"build_loopA_probes: {fn} has {len(gens)} texts, expected 10")
        src[arm] = {"file": f"results/{fn}", "sha1": sha1(b)}
        for k, gen in enumerate(gens):
            i, g = gen["i"], gen["text"]
            if i != k:
                sys.exit(f"build_loopA_probes: {fn} text {k} is numbered {i}")
            hits = [t for t in SOURCE_TERMS if re.search(rf"\b{t}", g, re.I)]
            if hits:
                print(f"  note: {arm} cand {i} contains {hits} (kept: the texts are probed as written)")
            for cue, tpl in (("uncued", T3_UNCUED), ("cued", T3_CUED)):
                items.append({"id": f"T3-{cue}-{arm}-{id_tag}c{i:02d}", "test": 3, "cue": cue,
                              "arm": arm, "cand": i, "text_sha1": sha1(g.encode("utf-8")),
                              "prompt": tpl.format(text=g)})
    return items, src


def write(path, obj):
    if path.exists():
        sys.exit(f"build_loopA_probes: {path.relative_to(ROOT)} exists; refusing to overwrite")
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {path.relative_to(ROOT)}: {len(obj['items'])} prompts")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--show", type=int, default=0)
    ap.add_argument("--batch", default=None, help="test-3 manifest for a later batch of the 2x2 arms")
    a = ap.parse_args()
    for t in (T2_IDENTITY, T2_RECON, T3_UNCUED, T3_CUED):
        check_frame(t)
    if a.batch:
        items, src = output_items([(arm, fn.format(b=a.batch)) for arm, fn in ARMS_2X2_BATCH],
                                  id_tag=f"{a.batch}-")
        print(f"test 3, batch {a.batch}: {len(ARMS_2X2_BATCH)} arms x 10 texts x 2 cues")
        write(OUTDIR / f"loopA_outputs_it_D2_{a.batch}.json", {
            "built_by": f"scripts/build_loopA_probes.py --batch {a.batch}", "test": 3,
            "what": f"output probe, batch {a.batch}: each Opus 5 IT D2 text of the four 2x2 arms "
                    "(scripts/run_2x2_batch.sh), alone; uncued and cued; frames as in "
                    "loopA_outputs_it_D2.json",
            "sources": src, "items": items})
        for it in items[:a.show]:
            print(f"\n===== {it['id']} =====\n{it['prompt']}")
        return

    # ---- test 2: per-pattern probe ----
    raw = {k: p.read_bytes() for k, p in PROMPTS.items()}
    ent = {k: entries(b.decode("utf-8")) for k, b in raw.items()}
    if list(ent["july"]) != list(ent["revised"]):
        sys.exit("build_loopA_probes: the two prompts list different patterns")
    specs = []
    for pid in ent["july"]:
        if ent["july"][pid] == ent["revised"][pid]:
            specs.append((pid, "both", ent["july"][pid]))
        else:
            specs += [(pid, "july", ent["july"][pid]), (pid, "revised", ent["revised"][pid])]
    changed = sum(w == "july" for _, w, _ in specs)
    print(f"test 2: {len(ent['july'])} entries, {changed} reworded -> {len(specs)} distinct specifications")
    items = []
    for q, tpl in (("identity", T2_IDENTITY), ("reconstruction", T2_RECON)):
        for pid, wording, e in specs:
            items.append({"id": f"T2-{q[:5]}-{pid}-{wording}", "test": 2, "question": q,
                          "pattern": pid, "wording": wording, "prompt": tpl.format(entry=e)})
    write(OUTDIR / "loopA_patterns_it_D2.json", {
        "built_by": "scripts/build_loopA_probes.py", "test": 2,
        "what": "per-pattern probe: each pattern entry alone, July and revised wording; "
                "questions identity and reconstruction",
        "sources": {k: {"file": str(p.relative_to(ROOT)), "sha1": sha1(raw[k])} for k, p in PROMPTS.items()},
        "items": items})
    shown = items[:a.show]

    # ---- test 3: output probe ----
    items, src = output_items(ARMS)
    print(f"test 3: {len(ARMS)} arms x 10 texts x 2 cues")
    write(OUTDIR / "loopA_outputs_it_D2.json", {
        "built_by": "scripts/build_loopA_probes.py", "test": 3,
        "what": "output probe: each Opus 5 IT D2 text of Table 2's six arms, alone; uncued and cued",
        "sources": src, "items": items})
    for it in shown + items[:a.show]:
        print(f"\n===== {it['id']} =====\n{it['prompt']}")


if __name__ == "__main__":
    main()
