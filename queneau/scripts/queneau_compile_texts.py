#!/usr/bin/env python3
"""queneau_compile_texts.py --- collect ALL generated texts (D0/D1/D2) for qualitative reading.

Organized language -> scenario -> model ladder, and under each model its degrees (D0, D1, D2),
each with all N generations annotated by s0_pos (structure) and coherence. Source-blind runs only
(the ACES D0/D1/D2 experiments). Writes one big markdown file for printing.

  ../Oulipo-Atlas-2-0/.venv/bin/python scripts/queneau_compile_texts.py [--out PATH]
"""
import argparse
import glob
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parent.parent
QH = ROOT / "Experiments" / "Queneau" / "queneau-homo"
OUT, SCORED, SPEC = QH / "outputs", QH / "outputs" / "scored", QH / "spec"
CACHE = json.loads((QH / "results" / "coh_cache.json").read_text(encoding="utf-8")) \
    if (QH / "results" / "coh_cache.json").exists() else {}

LANG_NAME = {"en": "English", "fr": "French (Queneau original)", "it": "Italian (Eco)"}
SCENARIOS = ["security-checkpoint", "hospital-emergency-room", "grocery-checkout"]
DEGREE_DESC = {
    "D0": "ordered dependency tree — full surface + relations (source-blind)",
    "D1": "order-free dependency triples (source-blind)",
    "D2": "clause architecture (source-blind)",
}


def coh(text, lang):
    return CACHE.get(hashlib.sha1((lang + " " + text.strip()).encode()).hexdigest())


def sz(m):
    if "Maverick" in m:
        return 17.0
    if "opus" in m:
        return 900.0
    if "fable" in m:
        return 950.0
    mm = re.search(r"-(\d+\.?\d*)B[-.]", m)
    return float(mm.group(1)) if mm else 990.0


def name(m):
    if "Maverick" in m:
        return "Llama-4-Maverick (17B active)"
    if "opus-4-8" in m:
        return "Claude Opus 4.8"
    if "fable-5" in m:
        return "Claude Fable 5"
    return m.replace("Qwen/", "").replace("-Instruct", "")


def fnum(x):
    return f"{x:.2f}" if isinstance(x, (int, float)) else "—"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(QH / "ALL_TEXTS_D0_D1_D2.md"))
    a = ap.parse_args()

    # idx[lang][scenario][model][degree] = {"gens":[...], "srows":{i:s0}, }
    idx = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    present_degrees, ncells = set(), 0
    for f in sorted(glob.glob(str(OUT / "homo_*.json"))):
        d = json.loads(Path(f).read_text(encoding="utf-8"))
        deg, lang, scen, m = d["degree"], d["lang"], d["scenario"], d["model"]
        srows = {}
        sc = SCORED / ("scored_" + Path(f).stem + ".json")
        if sc.exists():
            srows = {r["i"]: r.get("s0_pos") for r in json.loads(sc.read_text(encoding="utf-8"))["rows"]}
        idx[lang][scen][m][deg] = {"gens": d["gens"], "s0": srows}
        present_degrees.add(deg); ncells += 1

    L = []
    L.append("# Queneau *Notations* — all generated texts (D0 / D1 / D2)\n")
    L.append(f"*Source-blind homosyntactic/homeosyntactic generations, for qualitative reading. "
             f"Organized by language → scenario → model ladder; under each model, its degrees. "
             f"Degrees present: {', '.join(sorted(present_degrees))}. Each generation is annotated "
             f"`(s0 = POS-surface similarity to source; coh = coherence judge 0–1)`.*\n")
    L.append("Degrees: " + "; ".join(f"**{k}** = {v}" for k, v in DEGREE_DESC.items()
                                     if k in present_degrees) + ".\n")

    for lang in ("en", "fr", "it"):
        if lang not in idx:
            continue
        L.append(f"\n---\n\n# {LANG_NAME[lang]}\n")
        src = json.loads((SPEC / f"specs_{lang}_D0.json").read_text(encoding="utf-8"))["source"]
        L.append(f"**Source (Queneau *Notations*):** {src}\n")
        for scen in SCENARIOS:
            if scen not in idx[lang]:
                continue
            L.append(f"\n## {LANG_NAME[lang]} · {scen}\n")
            for m in sorted(idx[lang][scen], key=sz):
                L.append(f"\n### {name(m)}\n")
                for deg in ("D0", "D1", "D2"):
                    cell = idx[lang][scen][m].get(deg)
                    if not cell:
                        continue
                    gens, s0 = cell["gens"], cell["s0"]
                    cohs = [coh(g["text"], lang) for g in gens]
                    ms0 = mean([v for v in s0.values() if v is not None]) if s0 else None
                    mco = mean([c for c in cohs if c is not None]) if any(c is not None for c in cohs) else None
                    L.append(f"\n#### {deg} — {DEGREE_DESC[deg]}  ·  mean s0 {fnum(ms0)}, "
                             f"mean coh {fnum(mco)}, N={len(gens)}\n")
                    for g in gens:
                        c = coh(g["text"], lang)
                        L.append(f"{g['i']+1}. *(s0 {fnum(s0.get(g['i']))}, coh {fnum(c)})* "
                                 + " ".join(g["text"].split()) + "\n")

    text = "\n".join(L) + "\n"
    Path(a.out).write_text(text, encoding="utf-8")
    print(f"wrote {a.out}")
    print(f"  cells: {ncells}  |  degrees: {sorted(present_degrees)}")
    print(f"  ~{len(text.split()):,} words, {len(text)//1024} KB")


if __name__ == "__main__":
    main()
