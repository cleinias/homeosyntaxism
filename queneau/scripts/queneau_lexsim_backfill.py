#!/usr/bin/env python3
"""Compute lex-sim (content-lemma Jaccard) for the Queneau ladder, which never had it.

WHY

Section 3 defines a criterion whose second demand is "diverge in sense", measured by two
values: lex-sim (content-word overlap, must be LOW) and sem-diverg (the frame judge, must
be HIGH). The production scorer computes both. The LADDER scorer -- which produces every
number in Tables 1-5 -- computes neither, in 792 cells. So the paper's main experiment
asserts the second demand rather than measuring it.

lex-sim is the half that costs nothing: content-lemma Jaccard between source and
generation, straight off a Stanza parse, no judge and no API. This backfills it.

    lex-sim = |content lemmas(src) & content lemmas(gen)| / |union|      LOW = new lexis

matching Experiments/Manzoni/homeosyntaxism/scripts/score_production.py exactly: CONTENT = NOUN, VERB, ADJ,
ADV, PROPN, and the same TM.parse pipeline.

WRITES A SIDE FILE, NOT THE SCORED CELLS. The scored files back a paper that is verified
at 237 assertions and about to be resubmitted; nothing here should touch them until the
figure is wanted. Output goes to results/ladder_lexsim.json keyed by cell and generation.

    python scripts/queneau_lexsim_backfill.py --limit 20     # sample, to time it
    python scripts/queneau_lexsim_backfill.py                # the reported corpus
    python scripts/queneau_lexsim_backfill.py --all-runs     # every cell on disk
"""
import argparse
import contextlib
import importlib.util
import io
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
OUT = ROOT / "Experiments/Queneau/queneau-homo/results/ladder_lexsim.json"
CONTENT = {"NOUN", "VERB", "ADJ", "ADV", "PROPN"}


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    m = importlib.util.module_from_spec(spec)
    with contextlib.redirect_stdout(io.StringIO()):
        spec.loader.exec_module(m)
    return m


CC = _load("cc", "scripts/queneau_contamination_check.py")
import tree_metric as TM  # noqa: E402

S, O = CC.S, CC.O


def lemmas(toks):
    return {t["lemma"] for t in toks if t.get("upos") in CONTENT}


def jaccard(a, b):
    if not a and not b:
        return None
    return round(len(a & b) / len(a | b), 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all-runs", action="store_true",
                    help="every cell on disk, not just the ones the paper reports")
    ap.add_argument("--limit", type=int, default=0, help="stop after N cells (timing)")
    a = ap.parse_args()

    cells = []
    for f in sorted(S.glob("scored_homo_*.json")):
        if not a.all_runs and CC.reported_by(f.name) is None:
            continue
        raw = O / f.name[len("scored_"):]
        if raw.exists():
            cells.append((f, raw))
    if a.limit:
        cells = cells[:a.limit]
    print(f"  {len(cells)} cells to process\n")

    src_cache = {}          # (lang, sha of source) -> lemma set; the source is one text
    out, t0, ngen = {}, time.time(), 0
    for i, (sf, rf) in enumerate(cells, 1):
        raw = json.loads(rf.read_text(encoding="utf-8"))
        lang = raw.get("lang") or sf.name.split("_")[3]
        src = raw.get("source") or ""
        key = (lang, hash(src))
        if key not in src_cache:
            src_cache[key] = lemmas(TM.parse(src, lang=lang)) if src else set()
        sl = src_cache[key]
        rows = {}
        for g in raw.get("gens", []):
            try:
                gl = lemmas(TM.parse(g["text"], lang=lang))
            except Exception as e:
                rows[str(g["i"])] = None
                print(f"    parse failed {sf.name} i={g['i']}: {type(e).__name__}")
                continue
            rows[str(g["i"])] = jaccard(sl, gl)
            ngen += 1
        out[sf.name] = {"lang": lang, "table": CC.reported_by(sf.name), "lex_sim": rows}
        if i % 20 == 0 or i == len(cells):
            el = time.time() - t0
            print(f"    {i}/{len(cells)} cells · {ngen} generations · {el:.0f}s "
                  f"({el/max(ngen,1):.2f}s/gen)")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")
    vals = [v for c in out.values() for v in c["lex_sim"].values() if v is not None]
    print(f"\n  wrote {OUT.relative_to(ROOT)}")
    print(f"  {len(out)} cells, {len(vals)} generations scored")
    if vals:
        vals_sorted = sorted(vals)
        print(f"  lex-sim  min {vals_sorted[0]:.3f}  median "
              f"{vals_sorted[len(vals_sorted)//2]:.3f}  max {vals_sorted[-1]:.3f}  "
              f"mean {sum(vals)/len(vals):.3f}")


if __name__ == "__main__":
    main()
