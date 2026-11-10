#!/usr/bin/env python3
"""queneau_summarize.py --- consolidate Phase-0 results into clean per-degree summaries.

Structure comes from the per-cell scored files (outputs/scored/); coherence comes from the
judge cache (results/coh_cache.json, the source of truth) matched to each generation. Writes
results/summary_{degree}.json (per model: s0_pos, d2_clause, held_rate, coherence, cells, n)
and prints the ladder. Supersedes the ambiguous shared ladder_summary.json / coherence_summary.json.

  ../Oulipo-Atlas-2-0/.venv/bin/python scripts/queneau_summarize.py
"""
import glob
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "Experiments" / "Queneau" / "queneau-homo" / "outputs"
RES = ROOT / "Experiments" / "Queneau" / "queneau-homo" / "results"
CACHE = json.loads((RES / "coh_cache.json").read_text(encoding="utf-8"))


def sz(m):
    if "Maverick" in m:
        return 17.0  # 17B ACTIVE params (400B MoE) — placed by effective size
    mm = re.search(r"-(\d+\.?\d*)B[-.]", m)
    return float(mm.group(1)) if mm else 999.0


def coh_of(text, lang):
    return CACHE.get(hashlib.sha1((lang + " " + text.strip()).encode()).hexdigest())


def main():
    # degree -> model -> metric lists
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for outf in sorted(OUT.glob("homo_*.json")):
        d = json.loads(outf.read_text(encoding="utf-8"))
        deg, model, lang = d["degree"], d["model"], d["lang"]
        sc = OUT / "scored" / f"scored_{outf.stem}.json"
        if sc.exists():
            agg = json.loads(sc.read_text(encoding="utf-8"))["agg"]
            for k in ("s0_pos", "d2_clause", "held_rate"):
                data[deg][model][k].append(agg[k])
        cohs = [c for c in (coh_of(g["text"], lang) for g in d["gens"]) if c is not None]
        if cohs:
            data[deg][model]["coherence"].append(mean(cohs))
        data[deg][model]["cells"].append(1)

    for deg in sorted(data):
        ladder = []
        for m, mm in data[deg].items():
            row = {"model": m, "cells": len(mm["cells"])}
            for k in ("s0_pos", "d2_clause", "held_rate", "coherence"):
                row[k] = round(mean(mm[k]), 3) if mm[k] else None
            ladder.append(row)
        ladder.sort(key=lambda r: sz(r["model"]))
        (RES / f"summary_{deg}.json").write_text(
            json.dumps({"degree": deg, "ladder": ladder}, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"\n=== summary_{deg}.json ===")
        print(f"{'model':40}{'s0_pos':>8}{'d2_clause':>10}{'held':>7}{'coher':>7}{'cells':>7}")
        for r in ladder:
            f = lambda k: f"{r[k]:.3f}" if r[k] is not None else "  -  "  # noqa: E731
            print(f"{r['model'].split('/')[-1]:40}{f('s0_pos'):>8}{f('d2_clause'):>10}{f('held_rate'):>7}{f('coherence'):>7}{r['cells']:>7}")


if __name__ == "__main__":
    main()
