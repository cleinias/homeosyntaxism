#!/usr/bin/env python3
"""Do OTHER safety-aligned models refuse the same ten passages?

A DIFFERENT QUESTION FROM THE MISTRAL RUN

queneau_openweights_judge_check.py asked whether the passages Sonnet declined are
actually incoherent, and used a permissive open model to ask it. That model refused
nothing, which tells us the refusal is not forced by the text -- but a permissive
model declining to refuse says nothing about aligned ones.

This asks the other half: is declining these passages a property of Anthropic's
classifier, or of safety-aligned models generally? Two other aligned families, same ten
passages, same prompts.

  - all three refuse  -> a general property of aligned classifiers meeting degenerate
                         text, which is a considerably more interesting finding than a
                         vendor quirk, and belongs in the appendix as such
  - only Sonnet       -> specific to one classifier; the appendix should say so plainly
                         rather than generalising from a single vendor

WHAT COUNTS AS A REFUSAL is deliberately liberal here: an API-level refusal, a content
filter, an empty completion, or a reply that declines in prose instead of returning a
number. All are recorded verbatim, because the interesting cases are the ones that do
not arrive as a tidy error code. A transport failure (429, timeout) is NOT a refusal and
is retried, then reported separately -- conflating the two is how the July scoring lost
four generations.

Does not touch results/coh_cache.json: these are different judges, and that cache has no
judge in its key.

    python scripts/queneau_aligned_judge_refusal_check.py --dry-run
    python scripts/queneau_aligned_judge_refusal_check.py
"""
import argparse
import json
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
KEY = ROOT / "Docs and Research notes/CHR2027 paper/judge_refusal_handlabel_KEY.json"
S = ROOT / "Experiments/Queneau/queneau-homo/outputs/scored"
O = ROOT / "Experiments/Queneau/queneau-homo/outputs"
OUT = ROOT / "Experiments/Queneau/queneau-homo/results/aligned_judge_refusal_check.json"
MODELS = ["openai/gpt-5", "google/gemini-2.5-pro"]

DECLINE = re.compile(r"\b(I can't|I cannot|I'm sorry|I am sorry|I won'?t|unable to (?:help|assist|"
                     r"comply)|can'?t help|not able to|against my|violat|policy|refus)\b", re.I)


def env(k):
    for line in (ROOT / ".env").read_text().splitlines():
        if "=" in line and not line.strip().startswith("#"):
            a, b = line.split("=", 1)
            if a.strip() == k:
                return b.strip().strip('"').strip("'")
    raise SystemExit(f"{k} not in .env")


def prompt_for(text, lang):
    """Byte-identical to what Sonnet was sent, so a refusal here is comparable."""
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


def ask(model, text, lang, key, attempts=4):
    # 2000, not 16: GPT-5 is a reasoning model and its reasoning tokens are charged
    # against the SAME budget -- at 16 it burned ~448 of them thinking and emitted an
    # empty completion every time, which looks exactly like a refusal and is not one.
    # Whatever a judge needs to think, the answer is still one token.
    body = {"model": model, "temperature": 0, "max_tokens": 2000,
            "messages": [{"role": "user", "content": prompt_for(text, lang)}]}
    last = None
    for k in range(attempts):
        try:
            req = urllib.request.Request(
                "https://openrouter.ai/api/v1/chat/completions",
                data=json.dumps(body).encode(),
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json",
                         "HTTP-Referer": "https://codeberg.org/Cleinias/homeosyntaxism",
                         "X-Title": "homeosyntaxism judge-refusal check"},
                method="POST")
            r = json.loads(urllib.request.urlopen(req, timeout=120).read())
            if "error" in r and not r.get("choices"):
                msg = json.dumps(r["error"])[:300]
                return {"outcome": "refused_api", "score": None, "raw": msg}
            ch = r["choices"][0]
            txt = (ch.get("message") or {}).get("content") or ""
            fin = ch.get("finish_reason")
            if fin in ("content_filter", "safety"):
                return {"outcome": "refused_filter", "score": None, "raw": f"{fin}: {txt[:200]}"}
            v = num(txt)
            if v is None:
                # no number came back: either a prose decline or an empty completion
                kind = "refused_prose" if DECLINE.search(txt) else ("empty" if not txt.strip()
                                                                    else "unparseable")
                return {"outcome": kind, "score": None, "raw": txt[:300]}
            return {"outcome": "scored", "score": v, "raw": txt[:120]}
        except urllib.error.HTTPError as e:
            detail = e.read()[:300].decode("utf-8", "replace")
            last = f"HTTP {e.code}: {detail}"
            if e.code in (400, 403, 451):          # policy-ish rejections arrive as 4xx
                return {"outcome": "refused_http", "score": None, "raw": last}
            if e.code not in (408, 429, 500, 502, 503, 504):
                return {"outcome": "error", "score": None, "raw": last}
        except Exception as e:                                          # noqa: BLE001
            last = repr(e)
        if k < attempts - 1:
            time.sleep(min(3 * 2 ** k, 30))
    return {"outcome": "transport_failed", "score": None, "raw": str(last)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    keyrows = [x for x in json.loads(KEY.read_text(encoding="utf-8")) if x["refused"]]
    items = []
    for k in keyrows:
        raw = json.loads((O / f"homo_{k['cell']}.json").read_text(encoding="utf-8"))
        txt = next(" ".join(g["text"].split()) for g in raw["gens"] if g["i"] == k["i"])
        items.append({**k, "text": txt})
    print(f"  {len(items)} passages Sonnet refused; judges: {', '.join(MODELS)}")
    if a.dry_run:
        for x in items:
            print(f"    {x['cell'][:52]:54} i={x['i']:<2} {x['lang']} {len(x['text'].split()):4}w")
        return
    okey = env("OPENROUTER_API_KEY")
    res = []
    for x in items:
        row = {k: x[k] for k in ("cell", "i", "lang", "sonnet", "mistral")}
        row["words"] = len(x["text"].split())
        for m in MODELS:
            r = ask(m, x["text"], x["lang"], okey)
            row[m] = r
            print(f"    {x['cell'][:40]:42} i={x['i']:<2} {m:22} {r['outcome']:16} "
                  f"{'' if r['score'] is None else r['score']}", flush=True)
        res.append(row)
    OUT.write_text(json.dumps({"models": MODELS, "n": len(res), "items": res}, indent=1),
                   encoding="utf-8")
    print(f"\n  === did other aligned models refuse the ten? ===")
    for m in MODELS:
        outs = [r[m]["outcome"] for r in res]
        refused = [o for o in outs if o.startswith("refused")]
        scored = [r[m]["score"] for r in res if r[m]["score"] is not None]
        print(f"    {m:24} refused {len(refused):2}/10   scored {len(scored):2}/10"
              + (f"   mean {sum(scored)/len(scored):.2f}" if scored else "")
              + (f"   other: {[o for o in outs if not o.startswith('refused') and o!='scored']}"
                 if any(o != 'scored' and not o.startswith('refused') for o in outs) else ""))
    print(f"    {'claude-sonnet-4-6':24} refused 10/10   (by construction: this is that set)")
    print(f"\n  wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
