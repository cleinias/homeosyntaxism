#!/usr/bin/env python3
"""Does judge disagreement reach the part of the gradient the argument uses?

THE QUESTION

Every earlier judge comparison drew its passages from the nine cells where Sonnet
refused something -- the most degenerate slice of the corpus, chosen adversarially. On
that slice Mistral runs 0.25 above Sonnet in English and 0.44 in Italian, answers only
0 or 1, and three frontier judges split the refused ten by LANGUAGE rather than quality.

That measures disagreement where the text is word-salad. It says nothing about the region
the paper's positive claim occupies: the clause degrees, where coherence runs 0.87-0.99
and where the argument asserts form can be held AND sense kept. We have no measurement
there, so we cannot say whether the instrument is unreliable in general or only on
degenerate input -- and the appendix must not imply the first having tested only the
second.

DESIGN

A stratified sample across the whole ladder, not just the top, so the comparison has a
gradient of its own: S0 and D0, where coherence is low and disagreement is known, against
D2 and D3, where the argument lives. Same three languages and the same prompts the judge
of record was given. Passages are drawn deterministically -- sorted by a hash of cell and
index -- so the sample reproduces without a random seed.

Contaminated generations are skipped: they are excluded from every figure in the paper,
so including them would measure agreement on text neither judge is asked about.

WHAT WOULD SETTLE IT

  agreement good at D2/D3     -> disagreement is confined to degenerate text; the figures
                                 the argument rests on are judge-robust
  mediocre but ordering holds -> both judges still rank the clause degrees far above the
                                 surface, which is the contrast the paper actually uses
  poor at D2/D3 too           -> a real problem, better found now than by a reviewer

Does NOT write results/coh_cache.json: different judge, and that cache has no judge in
its key.

    python scripts/queneau_judge_agreement_top.py --dry-run
    python scripts/queneau_judge_agreement_top.py --per-degree 15
"""
import argparse
import hashlib
import json
import re
import statistics
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
S = ROOT / "Experiments/Queneau/queneau-homo/outputs/scored"
O = ROOT / "Experiments/Queneau/queneau-homo/outputs"
OUT = ROOT / "Experiments/Queneau/queneau-homo/results/judge_agreement_top.json"
MODEL = "mistral-large-2512"
SCEN = ("grocery-checkout", "security-checkpoint", "hospital-emergency-room")

RUNGS = [("S0", "S0", "blind_ladderS0"), ("D0", "D0", "blind_ladderD0soft"),
         ("D2", "D2c", "blind_ladderD2c"), ("D3", "D2", "ladderD2")]
MODELS = ["Qwen_Qwen2.5-7B-Instruct", "Qwen_Qwen2.5-32B-Instruct", "Qwen_Qwen2.5-72B-Instruct"]

POS = re.compile(r"(?<![\w'])(NOUN|VERB|ADJ|ADV|DET|ADP|PRON|PROPN|AUX|CCONJ|SCONJ|"
                 r"PART|NUM|INTJ|PUNCT|SYM)(?![\w'])")
BRACK = re.compile(r"\[\d+\]")


def doubling(t):
    w = [x.lower().strip(".,;:!?«»\"'()—–-") for x in t.split()]
    return sum(1 for a, b in zip(w, w[1:]) if a == b and len(a) > 2)


def contaminated(t):
    return len(POS.findall(t)) >= 3 or len(BRACK.findall(t)) >= 3 or doubling(t) >= 10


def env(k):
    for line in (ROOT / ".env").read_text().splitlines():
        if "=" in line and not line.strip().startswith("#"):
            a, b = line.split("=", 1)
            if a.strip() == k:
                return b.strip().strip('"').strip("'")
    raise SystemExit(f"{k} not in .env")


def prompt_for(text, lang):
    if lang == "en":
        return (f"Here is an English text: «{text}»\n\n"
                "Does the text describe ONE coherent, sensible situation, or is it "
                "grammatical but incoherent (words or clauses that do not compose a single "
                "sensible situation)? Answer ONLY with a number from 0 to 1: "
                "0 = incoherent/nonsensical, 1 = fully coherent. The number only.")
    if lang == "fr":
        return (f"Voici un texte français : «{text}»\n\n"
                "Le texte décrit-il UNE situation cohérente et sensée, ou est-il "
                "grammaticalement correct mais incohérent (mots ou propositions "
                "qui ne composent pas une seule situation sensée) ? Répondez UNIQUEMENT par un "
                "nombre de 0 à 1 : 0 = incohérent, 1 = pleinement cohérent. Le nombre seulement.")
    return (f"Ecco una frase italiana: «{text}»\n\n"
            "La frase descrive UNA situazione coerente e sensata, oppure è "
            "grammaticalmente corretta ma incoerente (parole o proposizioni che non "
            "compongono un'unica situazione sensata)? Rispondi SOLO con un numero da 0 "
            "a 1: 0 = incoerente/insensata, 1 = pienamente coerente. Solo il numero.")


def num(s):
    m = re.search(r"(?<![\d.])(0(?:\.\d+)?|1(?:\.0+)?)", s or "")
    return max(0.0, min(1.0, float(m.group(1)))) if m else None


def ask(text, lang, key, attempts=5):
    body = {"model": MODEL, "temperature": 0, "max_tokens": 12,
            "messages": [{"role": "user", "content": prompt_for(text, lang)}]}
    for k in range(attempts):
        try:
            req = urllib.request.Request(
                "https://api.mistral.ai/v1/chat/completions", data=json.dumps(body).encode(),
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                method="POST")
            r = json.loads(urllib.request.urlopen(req, timeout=90).read())
            return num(r["choices"][0]["message"]["content"])
        except urllib.error.HTTPError as e:
            if e.code not in (408, 429, 500, 502, 503, 504):
                return None
        except Exception:                                        # noqa: BLE001
            pass
        if k < attempts - 1:
            time.sleep(min(3 * 2 ** k, 30))
    return None


def collect(per_degree):
    picked = []
    for label, deg, tag in RUNGS:
        pool = []
        for m in MODELS:
            for l in ("en", "fr", "it"):
                for s in SCEN:
                    f = S / f"scored_homo_{deg}_{l}_{s}_{tag}_{m}.json"
                    if not f.exists():
                        f = Path(str(f).replace("_blind_", "_"))
                    if not f.exists():
                        continue
                    d = json.loads(f.read_text(encoding="utf-8"))
                    raw = json.loads((O / f.name[len("scored_"):]).read_text(encoding="utf-8"))
                    txt = {g["i"]: " ".join(g["text"].split()) for g in raw["gens"]}
                    for r in d["rows"]:
                        t = txt.get(r["i"], "")
                        if not t or contaminated(t) or r["coherence"] is None:
                            continue
                        pool.append({"rung": label, "cell": f.name[12:-5], "i": r["i"],
                                     "lang": l, "model": m.split("_")[-1],
                                     "sonnet": r["coherence"], "text": t})
        pool.sort(key=lambda x: hashlib.sha1((x["cell"] + str(x["i"])).encode()).hexdigest())
        picked += pool[:per_degree]
    return picked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-degree", type=int, default=15)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    items = collect(a.per_degree)
    print(f"  {len(items)} passages, {a.per_degree} per rung, contaminated excluded")
    for label, _, _ in RUNGS:
        v = [x for x in items if x["rung"] == label]
        if v:
            print(f"    {label:3} n={len(v):3}  Sonnet mean "
                  f"{statistics.mean(x['sonnet'] for x in v):.2f}   langs {sorted({x['lang'] for x in v})}")
    if a.dry_run:
        print("  --dry-run: nothing sent")
        return
    key = env("MISTRAL_API_KEY")
    for k, x in enumerate(items, 1):
        x["mistral"] = ask(x["text"], x["lang"], key)
        del x["text"]
        if k % 15 == 0:
            print(f"    [{k}/{len(items)}]", flush=True)
    OUT.write_text(json.dumps({"judge": MODEL, "items": items}, indent=1), encoding="utf-8")

    print("\n  === agreement by rung ===")
    print(f"  {'rung':5} {'n':>3} {'Sonnet':>7} {'Mistral':>8} {'delta':>7} "
          f"{'mean|d|':>8} {'r':>6} {'cross 0.5':>11}")
    for label, _, _ in RUNGS:
        v = [x for x in items if x["rung"] == label and x["mistral"] is not None]
        if not v:
            continue
        s = [x["sonnet"] for x in v]; m = [x["mistral"] for x in v]
        ms, mm = statistics.mean(s), statistics.mean(m)
        try:
            cov = sum((p - mm) * (q - ms) for p, q in zip(m, s)) / len(v)
            r = cov / (statistics.pstdev(m) * statistics.pstdev(s))
        except Exception:                                        # noqa: BLE001
            r = float("nan")
        cross = sum(1 for p, q in zip(m, s) if (p >= .5) != (q >= .5))
        print(f"  {label:5} {len(v):3} {ms:7.2f} {mm:8.2f} {mm-ms:+7.2f} "
              f"{statistics.mean(abs(p-q) for p, q in zip(m, s)):8.2f} {r:6.2f} {cross:7}/{len(v):<4}")

    print("\n  ordering check — does each judge rank the clause degrees above the surface?")
    for who in ("sonnet", "mistral"):
        vals = {}
        for lab, _, _ in RUNGS:
            v = [x[who] for x in items if x["rung"] == lab and x.get(who) is not None]
            if v:
                vals[lab] = statistics.mean(v)
        ok = vals.get("D2", 0) > vals.get("S0", 1) and vals.get("D3", 0) > vals.get("S0", 1)
        print(f"    {who:8} " + "  ".join(f"{k}:{v:.2f}" for k, v in vals.items())
              + f"   clause degrees above surface: {'YES' if ok else 'NO'}")
    print(f"\n  wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
