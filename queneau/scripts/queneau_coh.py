#!/usr/bin/env python3
"""queneau_coh.py --- judge-only coherence pass (no Stanza re-parse).

Reads the generation JSONs and scores each gen's coherence with the LLM judge
(sem_metrics; claude-sonnet-4-6), aggregating per model into the coherence-vs-size
ladder. Reuses the same coh_cache.json the scorer uses. Fast: ~1 tiny API call per gen.

  ../Oulipo-Atlas-2-0/.venv/bin/python scripts/queneau_coh.py --glob 'homo_D0_*.json'
"""
import argparse
import os
import glob
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import sem_metrics as SM  # noqa: E402

OUT = ROOT / "Experiments" / "Queneau" / "queneau-homo" / "outputs"
RES = ROOT / "Experiments" / "Queneau" / "queneau-homo" / "results"
# The judge cache is scoped by BACKEND, not just by text. Direct-API Sonnet 4.6 and
# TAMU's Bedrock Sonnet 4.6 are the same model but not the same instrument -- TAMU has
# extended thinking on and rejects temperature, and a 40-item agreement check on
# 2026-08-27 found exact agreement on only 29/40, a mean |diff| of 0.0875, and a
# systematic +0.0825 upward shift, including one 0.00 -> 1.00 reversal. Mixing the two
# in one file would bias coherence upward for whichever cells happen to be judged later
# and leave no way to tell which era a number came from.
_BACKEND = "tamu" if os.environ.get("SEM_JUDGE_BACKEND", "anthropic") == "tamu" else "anthropic"
CACHE = RES / ("coh_cache.json" if _BACKEND == "anthropic"
               else f"coh_cache_{_BACKEND}.json")

FR = ("Voici un texte français : «{t}»\n\nLe texte décrit-il UNE situation cohérente et sensée, "
      "ou est-il grammaticalement correct mais incohérent (mots ou propositions qui ne composent "
      "pas une seule situation sensée) ? Répondez UNIQUEMENT par un nombre de 0 à 1 : "
      "0 = incohérent, 1 = pleinement cohérent. Le nombre seulement.")


def coh(text, lang, cache):
    key = hashlib.sha1((lang + " " + text).encode()).hexdigest()
    if key in cache:
        return cache[key]
    val = None
    for _ in range(2):                    # one retry; tolerate empty/blocked judge responses
        try:
            val = SM._num(SM.anthropic(FR.format(t=text))) if lang == "fr" else SM.coh_judge(text, lang=lang)
            break
        except Exception:
            val = None
    if val is not None:
        cache[key] = val
    return val


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="homo_D0_*.json")
    a = ap.parse_args()
    RES.mkdir(parents=True, exist_ok=True)
    cache = json.loads(CACHE.read_text(encoding="utf-8")) if CACHE.exists() else {}
    by = defaultdict(list)
    cells = []
    for f in sorted(OUT.glob(a.glob)):
        d = json.loads(Path(f).read_text(encoding="utf-8"))
        m, lang = d["model"], d["lang"]
        vals = [v for v in (coh(g["text"].strip(), lang, cache) for g in d["gens"]) if v is not None]
        if not vals:
            print(f"  {m.split('/')[-1]:24} {lang} {d['scenario']:22} coh=NA (all judge calls failed)", flush=True)
            continue
        by[m].extend(vals)
        cells.append({"model": m, "lang": lang, "scenario": d["scenario"],
                      "coherence": round(mean(vals), 3), "n": len(vals)})
        print(f"  {m.split('/')[-1]:24} {lang} {d['scenario']:22} coh={mean(vals):.2f} n={len(vals)}", flush=True)
        CACHE.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
    o = {'0.5B': 0, '1.5B': 1, '3B': 2, '7B': 3, '14B': 4, '32B': 5, '72B': 6}
    sz = lambda m: next((v for k, v in o.items() if k in m), 9)  # noqa: E731
    ladder = [{"model": m, "coherence": round(mean(v), 3), "n": len(v)} for m, v in by.items()]
    ladder.sort(key=lambda r: sz(r["model"]))
    (RES / "coherence_summary.json").write_text(
        json.dumps({"ladder": ladder, "cells": cells}, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n=== coherence ladder (mean over model's gens) ===")
    for r in ladder:
        print(f"{r['model'].split('/')[-1]:24} coh={r['coherence']:.3f}  n={r['n']}")
    print(f"\nwrote results/coherence_summary.json ({len(ladder)} models)")


if __name__ == "__main__":
    main()
