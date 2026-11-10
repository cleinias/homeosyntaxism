#!/usr/bin/env python3
"""Detect generations that leak the specification into the prose, and price the effect.

WHY

The threefold criterion is form held / sense diverged / coherent, and the third is
delegated to an LLM judge asked whether the passage "describes ONE coherent, sensible
situation". That question has a gap we did not notice until SF hand-labelled a blind
sample and scored several passages 0 for a reason the judge was never asked about: the
model had emitted its own scaffolding -- part-of-speech tags, bracketed period markers --
into the text, or had degenerated into doubling every word.

The judge does not penalise either. A passage carrying 188 POS tags and 188 bracket
markers interleaved through the prose was scored 1.00, fully coherent. Strictly it is
not wrong -- such a passage may still describe one situation -- but we USE coherence as
the axis that certifies output as real prose, and it does not detect a model that has
failed by reciting its instructions.

This is measurement only. It rewrites nothing: it reports the rate, where it falls, and
what each affected cell's coherence would be without the contaminated generations, so
the decision about tables is taken on numbers rather than on impression.

DETECTION, and why these three signals

  POS tags   standalone UPPERCASE tag tokens (NOUN, VERB, ADJ...). Case matters: the
             French `aux` and the English `part` are ordinary words in lower case, and
             matching case-insensitively would flag half the French corpus. Threshold 3,
             because one stray capital is not a leak.
  [n]        bracketed integers, the period markers from the spec block. Effectively
             impossible in generated prose; threshold 3 anyway.
  doubling   immediate repetition of the same word (>2 chars, ignoring punctuation).
             A stutter is legitimate in dialogue, so the threshold is 10 -- this is
             aimed at the degenerate `the the and and then then` failure mode.

A flagged generation is NOT necessarily incoherent, and this does not claim it is. It is
a generation whose surface shows the model failed to separate its instructions from its
output -- which is a different failure from incoherence and is currently invisible.

    python scripts/queneau_contamination_check.py                # report
    python scripts/queneau_contamination_check.py --show 12      # + excerpts, to verify
    python scripts/queneau_contamination_check.py --all-runs     # include unreported runs
"""
import argparse
import collections
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
S = ROOT / "Experiments/Queneau/queneau-homo/outputs/scored"
O = ROOT / "Experiments/Queneau/queneau-homo/outputs"
OUT = ROOT / "Experiments/Queneau/queneau-homo/results/contamination_check.json"

# WHICH CELLS THE PAPER REPORTS
#
# A cell is reported when a figure printed in the paper is computed from it. This used to
# be two substring tuples, and on 2026-08-10 both were wrong: `ladderT1` sat in the
# exclusion list under the comment "controls that appear in no table" -- true when written,
# false the morning Table 1 moved onto that very run -- while `_anchor_`/`_opus48-anchor_`
# were still listed as reported after Table 5's Maverick and Opus rows moved to the hosted
# arm. A substring list cannot say "these three models, in English only", which is what
# Table 4 actually draws from ladderS0, so it could not have been made correct either.
#
# Each rule below names the table it feeds. If a table's provenance changes, this list
# changes with it -- and the run tag alone is never enough to decide.
#
#   deg     the degree segment of the filename (NB: files named D2 are the paper's D3,
#           and the paper's D2 -- clause nesting -- lives in files named D2c)
#   tag     substring identifying the run
#   models  model substrings, or () for "every model in that run"
#   langs   languages, or () for all three
REPORTED = (
    # -- Table 1: the two Qwen rows, both degrees, both regimes (ACES 2030984) ----------
    dict(deg="S0", tag="_ladderT1_", models=(), langs=(), table="T1 Qwen"),
    dict(deg="D0", tag="_ladderT1_", models=(), langs=(), table="T1 Qwen"),
    # -- Table 1's Opus rows, and Table 5's Opus row: closed weights, hosted ------------
    dict(deg="S0", tag="_b1_claude-opus-4-8", models=(), langs=(), table="T1 Opus"),
    dict(deg="D0", tag="_b1_claude-opus-4-8", models=(), langs=(), table="T1/T5 Opus"),
    # -- Table 5's Maverick row: source-blind only -------------------------------------
    dict(deg="D0", tag="_blind_b1_meta-llama_llama-4-maverick", models=(), langs=(),
         table="T5 Maverick"),
    # -- Table 3: the degree ladder, all seven Qwen, three languages --------------------
    dict(deg="D0", tag="_blind_ladderD0soft_", models=(), langs=(), table="T3/T5 D0"),
    dict(deg="D1", tag="ladderD1_", models=(), langs=(), table="T3 D1"),
    dict(deg="D2c", tag="_blind_ladderD2c_", models=(), langs=(), table="T3 D2 nesting"),
    dict(deg="D2", tag="ladderD2_", models=(), langs=(), table="T3 D3 inventory"),
    # -- Table 4: the laptop tiers, English only ---------------------------------------
    *(dict(deg=d, tag="_mac-", models=(), langs=("en",), table="T4 laptop")
      for d in ("S0", "D0", "D1", "D2c", "D2")),
    # -- Table 4's S0 cluster reference: only the three models that column compares -----
    dict(deg="S0", tag="_blind_ladderS0_",
         models=("Qwen2.5-7B-", "Qwen2.5-14B-", "Qwen2.5-72B-"), langs=("en",),
         table="T4 S0 cluster"),
)
# Superseded or never-reported runs, kept on disk. `_ladder_` is the strict July D0 prompt;
# bf16ctrl is the 32B precision control; the anchors were Table 5's rows before it moved to
# the hosted arm; the b1 Qwen grid was Table 1's before it moved to the cluster.
SUPERSEDED = ("_ladder_", "bf16ctrl", "_anchor_", "-anchor_", "_b1_qwen_")

# PENDING -- verified, on disk, feeding no printed figure YET.
#
# A third state was needed on 2026-08-15. `maverick09` is neither reported (the submitted
# Table 5 ships the hosted Maverick row) nor superseded (it supersedes nothing -- it is the
# successor, and the run it replaces is the discarded 0.7 one). Filing it under either
# label would put a false statement in the file whose entire design principle is that each
# rule names what it feeds. Filing it under SUPERSEDED would have been the cheaper lie and
# exactly the 2026-08-10 defect in a new costume.
#
# Pending cells are EXCLUDED from the default corpus, so no printed census moves; they are
# listed under their own heading so they stay visible, and `--include-pending` prices them.
PENDING = (
    # -- ACES 2032162, tag maverick09: the corrected 0.9 re-run of the 0.7 job caught on
    #    2026-08-10. Nine D0 cells, three languages, gate 9/9 byte-identical to the
    #    softened family. Candidate to replace Table 5's hosted Maverick row in the
    #    camera-ready and in PT. To promote, move this dict into REPORTED (the printed
    #    census in Appendix B changes when you do, which is the point of the ceremony).
    dict(deg="D0", tag="_blind_maverick09_", models=(), langs=(),
         table="T5 Maverick (0.9 cluster, camera-ready candidate)"),
)

POS = re.compile(r"(?<![\w'])(NOUN|VERB|ADJ|ADV|DET|ADP|PRON|PROPN|AUX|CCONJ|SCONJ|"
                 r"PART|NUM|INTJ|PUNCT|SYM)(?![\w'])")
BRACK = re.compile(r"\[\d+\]")
MIN_POS, MIN_BRACK, MIN_DBL = 3, 3, 10


def doubling(text):
    w = [x.lower().strip(".,;:!?«»\"'()—–-") for x in text.split()]
    return sum(1 for a, b in zip(w, w[1:]) if a == b and len(a) > 2)


def features(text):
    p, b, d = len(POS.findall(text)), len(BRACK.findall(text)), doubling(text)
    return {"pos": p, "brack": b, "dbl": d,
            "flagged": p >= MIN_POS or b >= MIN_BRACK or d >= MIN_DBL}


def _claimed_by(rules, name):
    """Which rule in `rules` claims this cell, or None. A cell claimed twice is fine --
    the generator yields it once -- but a cell claimed by nothing is simply not reported."""
    for r in rules:
        if f"_homo_{r['deg']}_" not in name:
            continue
        if r["tag"] not in name:
            continue
        if r["langs"] and not any(f"_homo_{r['deg']}_{l}_" in name for l in r["langs"]):
            continue
        if r["models"] and not any(m in name for m in r["models"]):
            continue
        return r["table"]
    return None


def reported_by(name):
    return _claimed_by(REPORTED, name)


def pending_by(name):
    return _claimed_by(PENDING, name)


def cells(all_runs, include_pending=False):
    for f in sorted(S.glob("scored_*.json")):
        n = f.name
        if not all_runs and reported_by(n) is None and not (
                include_pending and pending_by(n)):
            continue
        raw_p = O / n[len("scored_"):]
        if not raw_p.exists():
            continue
        yield n[12:-5], json.loads(f.read_text(encoding="utf-8")), json.loads(raw_p.read_text(encoding="utf-8"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--show", type=int, default=0, help="print N flagged excerpts to eyeball")
    ap.add_argument("--all-runs", action="store_true")
    ap.add_argument("--include-pending", action="store_true",
                    help="also price the PENDING runs (they feed no printed figure)")
    a = ap.parse_args()

    if not a.all_runs:
        claimed = collections.Counter()
        pending = collections.Counter()
        unclaimed = []
        for f in sorted(S.glob("scored_homo_*.json")):
            t = reported_by(f.name)
            if t:
                claimed[t] += 1
                continue
            p = pending_by(f.name)
            if p:
                pending[p] += 1
            else:
                unclaimed.append(f.name)
        print("  reported cells, by the table that draws on them:")
        for t, n in claimed.most_common():
            print(f"    {t:20} {n:4}")
        print(f"    {'TOTAL':20} {sum(claimed.values()):4}"
              f"   ({len(unclaimed) + sum(pending.values())} cells on disk feed no table)")
        if pending:
            print("  pending -- verified, feeding no printed figure yet"
                  f"{' (priced below)' if a.include_pending else ''}:")
            for t, n in pending.most_common():
                print(f"    {t:52} {n:4}")
        stray = [n for n in unclaimed if not any(k in n for k in SUPERSEDED)]
        if stray:
            print(f"    WARNING: {len(stray)} unclaimed cells are not on the superseded "
                  f"list -- check whether a table needs them:")
            for n in stray[:8]:
                print(f"      {n}")
        print()

    rows, per_cell = [], {}
    for cell, sc, raw in cells(a.all_runs, a.include_pending):
        coh = {r["i"]: r["coherence"] for r in sc["rows"]}
        clean_c, dirty_c = [], []
        for g in raw["gens"]:
            if g["i"] not in coh:
                continue                       # refused or unparseable: not ours to judge
            t = " ".join(g["text"].split())
            f = features(t)
            rows.append({"cell": cell, "i": g["i"], "degree": raw.get("degree"),
                         "model": raw.get("model"), "lang": raw.get("lang"),
                         "coherence": coh[g["i"]], "words": len(t.split()), **f,
                         "excerpt": t[:160]})
            (dirty_c if f["flagged"] else clean_c).append(coh[g["i"]])
        if dirty_c:
            per_cell[cell] = {"n_flagged": len(dirty_c), "n_clean": len(clean_c),
                              "coh_all": (sum(dirty_c) + sum(clean_c)) / (len(dirty_c) + len(clean_c)),
                              "coh_clean_only": (sum(clean_c) / len(clean_c)) if clean_c else None,
                              "coh_flagged": sum(dirty_c) / len(dirty_c),
                              "degree": next(r["degree"] for r in rows if r["cell"] == cell),
                              "model": next(r["model"] for r in rows if r["cell"] == cell)}

    flagged = [r for r in rows if r["flagged"]]
    print(f"  {len(flagged)} flagged of {len(rows)} scored generations "
          f"({100*len(flagged)/max(len(rows),1):.1f}%) in {len(per_cell)} cells\n")

    fc = [r["coherence"] for r in flagged if r["coherence"] is not None]
    cc = [r["coherence"] for r in rows if not r["flagged"] and r["coherence"] is not None]
    print(f"  coherence as the judge scored them:")
    print(f"    flagged  mean {sum(fc)/len(fc):.2f}   {sum(1 for v in fc if v>=0.5)}/{len(fc)} rated 'coherent' (>=0.50)")
    print(f"    clean    mean {sum(cc)/len(cc):.2f}")

    for label, keyf in (("degree", lambda r: r["degree"]),
                        ("model", lambda r: (r["model"] or "?").split("/")[-1]),
                        ("language", lambda r: r["lang"])):
        c = collections.Counter(keyf(r) for r in flagged)
        tot = collections.Counter(keyf(r) for r in rows)
        print(f"\n  by {label}:")
        for k, v in c.most_common(8):
            print(f"    {str(k):32} {v:4} of {tot[k]:4}  ({100*v/tot[k]:.1f}%)")

    print(f"\n  cells whose coherence moves most if flagged generations are excluded:")
    movers = sorted((x for x in per_cell.values() if x["coh_clean_only"] is not None),
                    key=lambda x: -abs(x["coh_clean_only"] - x["coh_all"]))
    print(f"    {'degree':7} {'model':26} {'as scored':>10} {'clean only':>11} {'delta':>7} {'n':>7}")
    for x in movers[:12]:
        print(f"    {x['degree']:7} {(x['model'] or '?').split('/')[-1][:26]:26} "
              f"{x['coh_all']:10.2f} {x['coh_clean_only']:11.2f} "
              f"{x['coh_clean_only']-x['coh_all']:+7.2f} {x['n_flagged']:3}/{x['n_flagged']+x['n_clean']:<3}")

    if a.show:
        print(f"\n  === {a.show} flagged excerpts, to check the detector is not over-firing ===")
        for r in sorted(flagged, key=lambda r: -(r["pos"] + r["brack"] + r["dbl"]))[:a.show]:
            print(f"\n    POS={r['pos']} [n]={r['brack']} dbl={r['dbl']}  coherence={r['coherence']}"
                  f"  {r['degree']} {(r['model'] or '?').split('/')[-1]}")
            print(f"      {r['excerpt']}…")

    OUT.write_text(json.dumps(
        {"thresholds": {"pos": MIN_POS, "brack": MIN_BRACK, "doubling": MIN_DBL},
         "n_scored": len(rows), "n_flagged": len(flagged),
         "cells": per_cell,
         "flagged": [{k: v for k, v in r.items() if k != "excerpt"} for r in flagged]},
        indent=1), encoding="utf-8")
    print(f"\n  wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
