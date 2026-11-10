#!/usr/bin/env python3
"""queneau_judge_agreement.py --- do the primary and fallback coherence judges agree?

Sonnet 4.6 (the judge of record) refuses a small number of these passages outright, so
three D0 cells were rescued with Opus 4.8 (queneau_fallback_judge.py). That leaves two
cells whose coherence is a MIXTURE of the two judges. The mixture is only defensible if
the judges score the same texts alike; if Opus runs systematically high or low, those
cells are biased relative to their neighbours and the fix is worse than the hole.

So: take texts Sonnet already scored, re-judge them with Opus, and compare. The sample is
stratified deterministically across the coherence range and across languages rather than
drawn at random -- disagreement is much likelier at the ambiguous middle than at the ends,
and a random draw from a corpus this skewed would mostly sample the ends and miss it.

    python scripts/queneau_judge_agreement.py --glob 'homo_D0_*ladderD0soft*' --n 24
"""
import argparse
import json
import os
import sys
from pathlib import Path
from statistics import mean

FALLBACK = os.environ.get("QUENEAU_FALLBACK_JUDGE", "claude-opus-4-8")
os.environ["SEM_JUDGE_MODEL"] = FALLBACK          # read at import time by sem_metrics

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import sem_metrics as SM   # noqa: E402

OUT = ROOT / "Experiments" / "Queneau" / "queneau-homo" / "outputs"
SCORED = OUT / "scored"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="homo_D0_*ladderD0soft*")
    ap.add_argument("--n", type=int, default=24)
    a = ap.parse_args()

    # collect every row the PRIMARY judge scored, with its text
    pool = []
    for raw in sorted(OUT.glob(a.glob + ".json")):
        sc = SCORED / f"scored_{raw.stem}.json"
        if not sc.exists():
            continue
        d = json.loads(raw.read_text(encoding="utf-8"))
        s = json.loads(sc.read_text(encoding="utf-8"))
        texts = {g["i"]: g["text"].strip() for g in d["gens"]}
        for r in s["rows"]:
            if r.get("judge_model") or r.get("coherence") is None:
                continue                      # skip rows the fallback already judged
            pool.append({"lang": d["lang"], "sonnet": r["coherence"], "text": texts[r["i"]]})
    if not pool:
        print("no primary-judged rows found"); return

    # deterministic stratified pick: per language, evenly spaced along the coherence range
    sample = []
    for lang in ("en", "fr", "it"):
        rows = sorted([p for p in pool if p["lang"] == lang], key=lambda p: (p["sonnet"], p["text"]))
        if not rows:
            continue
        k = max(1, a.n // 3)
        step = max(1, len(rows) // k)
        sample += rows[::step][:k]
    print(f"pool {len(pool)} primary-judged rows; sampling {len(sample)}; fallback = {FALLBACK}\n")

    print(f"{'lang':5} {'sonnet':>7} {'opus':>7} {'delta':>7}   text")
    rec = []
    for p in sample:
        try:
            o = SM.coh_judge(p["text"], lang=p["lang"])
        except SM.JudgeError as e:
            print(f"{p['lang']:5} {p['sonnet']:7.2f}  REFUSED        {p['text'][:44]}"); continue
        if o is None:
            continue
        rec.append((p["lang"], p["sonnet"], o))
        print(f"{p['lang']:5} {p['sonnet']:7.2f} {o:7.2f} {o-p['sonnet']:+7.2f}   {p['text'][:44]}", flush=True)

    if not rec:
        print("\nnothing comparable"); return
    d = [o - s for _, s, o in rec]
    sv = [s for _, s, _ in rec]; ov = [o for _, _, o in rec]
    ms, mo = mean(sv), mean(ov)
    cov = sum((x-ms)*(y-mo) for x, y in zip(sv, ov))
    var = (sum((x-ms)**2 for x in sv) * sum((y-mo)**2 for y in ov)) ** 0.5
    cross = sum(1 for _, s, o in rec if (s >= 0.5) != (o >= 0.5))
    print(f"\nn={len(rec)}   sonnet mean {ms:.3f}   opus mean {mo:.3f}")
    print(f"mean delta (opus - sonnet) {mean(d):+.3f}   mean |delta| {mean(abs(x) for x in d):.3f}")
    print(f"pearson r {cov/var if var else float('nan'):.3f}")
    print(f"disagree across the 0.50 pass bar: {cross}/{len(rec)}")


if __name__ == "__main__":
    main()
