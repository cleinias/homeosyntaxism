#!/usr/bin/env python3
"""Score the judge-refused passages, and their cell-mates, with a second-family judge.

THE OBJECTION THIS ANSWERS

A reviewer may reasonably ask why the coherence judge is a closed, safety-aligned model
at all: Sonnet 4.6 declines ~0.19% of passages, the declines fall where coherence is
lowest, and an open or more permissive judge (Mistral, or an abliterated model) would
not have refused. Two claims in the appendix currently rest on argument rather than
measurement:

  1. that the refused passages are the DEGENERATE ones, so the affected cells report
     coherence as an upper bound;
  2. that the refusal is a property of the classifier rather than of the text.

This run measures both. It scores all 90 generations of the nine affected cells with
mistral-large-2512 -- the ten Sonnet refused, and the eighty it scored normally -- using
the SAME prompts, so the refused group has a control group judged by the same instrument.

WHY MISTRAL DIRECT, NOT OPENROUTER

OpenRouter routes to whichever provider is cheapest at whatever quantization it happens
to serve; we learned today that this matters (the 7B differs measurably between fp8 on
Together and bf16 on our nodes). For a judge in a robustness check, an unrecorded
serving precision is exactly what a reviewer would go after. The vendor endpoint gives
one documented model version and nothing to explain. The id is pinned (mistral-large-2512),
not the floating `-latest` alias.

WHY NOT QWEN-72B, though it is open and we already run it: it is one of the SUBJECTS.
Having a subject's own family judge its output invites a second objection to fix the
first.

DELIBERATELY DOES NOT WRITE results/coh_cache.json. That cache is keyed on
sha1(lang + text) with no judge model in the key, so writing Mistral's scores into it
would let a later Sonnet pass silently reuse them -- the same trap that made the Opus
fallback dangerous.

REPORT IT EITHER WAY. If Mistral scores the refused passages HIGH, that is an awkward
result -- it would mean either that Mistral judges coherence poorly or that our reading
of those passages is wrong -- and it goes in the appendix regardless. Running this and
publishing only the convenient outcome would be worse than not running it.

    python scripts/queneau_openweights_judge_check.py --dry-run
    python scripts/queneau_openweights_judge_check.py
"""
import argparse
import json
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
S = ROOT / "Experiments/Queneau/queneau-homo/outputs/scored"
O = ROOT / "Experiments/Queneau/queneau-homo/outputs"
OUT = ROOT / "Experiments/Queneau/queneau-homo/results/openweights_judge_check.json"
MODEL = "mistral-large-2512"
REPORTED = ("_b1_", "_blind_ladderD0soft_", "_blind_ladderS0_", "ladderD1_",
            "_blind_ladderD2c_", "ladderD2_", "_mac-", "_anchor_", "_opus48-anchor_")


def env(key):
    for line in (ROOT / ".env").read_text().splitlines():
        if line.strip().startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        if k.strip() == key:
            return v.strip().strip('"').strip("'")
    raise SystemExit(f"{key} not in .env")


def prompt_for(text, lang):
    """Byte-identical to the prompts sem_metrics/queneau_score_homo send Sonnet, so the
    two judges are answering the same question and the comparison means something."""
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


def ask(text, lang, key, attempts=4):
    body = {"model": MODEL, "temperature": 0, "max_tokens": 12,
            "messages": [{"role": "user", "content": prompt_for(text, lang)}]}
    last = None
    for k in range(attempts):
        try:
            req = urllib.request.Request(
                "https://api.mistral.ai/v1/chat/completions",
                data=json.dumps(body).encode(),
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                method="POST")
            r = json.loads(urllib.request.urlopen(req, timeout=90).read())
            ch = r["choices"][0]
            txt = ch["message"]["content"]
            # a refusal here is the interesting outcome, not an error: record it
            return {"score": num(txt), "raw": (txt or "")[:200],
                    "finish": ch.get("finish_reason")}
        except urllib.error.HTTPError as e:
            last = f"HTTP {e.code}"
            if e.code not in (408, 429, 500, 502, 503, 504):
                return {"score": None, "raw": last, "finish": "http_error"}
        except Exception as e:                                   # noqa: BLE001
            last = repr(e)
        if k < attempts - 1:
            time.sleep(min(2 * 2 ** k, 20))
    return {"score": None, "raw": str(last), "finish": "failed"}


def collect():
    """The nine cells Sonnet refused something in, with every generation labelled."""
    items = []
    for f in sorted(S.glob("scored_*.json")):
        n = f.name
        if "ladderT1" in n or "bf16ctrl" in n or "_ladder_" in n:
            continue                                    # not reported / superseded
        if not any(k in n for k in REPORTED):
            continue
        d = json.loads(f.read_text(encoding="utf-8"))
        if not d.get("n_refused"):
            continue
        raw = json.loads((O / n[len("scored_"):]).read_text(encoding="utf-8"))
        sonnet = {r["i"]: r["coherence"] for r in d["rows"]}
        for g in raw["gens"]:
            items.append({"cell": n[12:-5], "i": g["i"], "lang": raw["lang"],
                          "refused_by_sonnet": g["i"] in d.get("refused_i", []),
                          "sonnet": sonnet.get(g["i"]),
                          "words": len(g["text"].split()),
                          "text": " ".join(g["text"].split())})
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    items = collect()
    ref = [x for x in items if x["refused_by_sonnet"]]
    ctl = [x for x in items if not x["refused_by_sonnet"]]
    print(f"  {len(items)} generations from {len({x['cell'] for x in items})} cells")
    print(f"    {len(ref)} refused by Sonnet, {len(ctl)} scored normally (the control)")
    if a.dry_run:
        print("  --dry-run: nothing sent")
        return
    key = env("MISTRAL_API_KEY")
    for k, x in enumerate(items, 1):
        r = ask(x["text"], x["lang"], key)
        x.update({"mistral": r["score"], "mistral_raw": r["raw"], "finish": r["finish"]})
        del x["text"]                                   # keep the record small; texts are on disk
        flag = " <-- REFUSED BY SONNET" if x["refused_by_sonnet"] else ""
        if k % 10 == 0 or x["refused_by_sonnet"]:
            print(f"    [{k:3}/{len(items)}] {x['cell'][:44]:46} i={x['i']:<2} "
                  f"mistral={x['mistral']} sonnet={x['sonnet']}{flag}", flush=True)
    OUT.write_text(json.dumps({"judge": MODEL, "endpoint": "api.mistral.ai",
                               "n": len(items), "items": items}, indent=1), encoding="utf-8")

    def mean(v):
        v = [x for x in v if x is not None]
        return sum(v) / len(v) if v else float("nan")
    mr = [x["mistral"] for x in ref]
    mc = [x["mistral"] for x in ctl]
    print(f"\n  === {MODEL} on the nine affected cells ===")
    print(f"    refused by Sonnet (n={len(ref)}) : mistral {mean(mr):.2f}"
          f"   [{len([x for x in mr if x is None])} unscored by Mistral too]")
    print(f"    scored by Sonnet  (n={len(ctl)}) : mistral {mean(mc):.2f}"
          f"   sonnet {mean([x['sonnet'] for x in ctl]):.2f}")
    print(f"\n    gap (control - refused), Mistral's own scale: {mean(mc)-mean(mr):+.2f}")
    print(f"    Mistral vs Sonnet on the 80 both scored:      "
          f"{mean(mc)-mean([x['sonnet'] for x in ctl]):+.2f}")
    print(f"\n  wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
