#!/usr/bin/env python3
"""sem_metrics.py — frame-divergence and coherence metrics (build-and-compare).

The threefold criterion's axes 2 (semantic divergence) and 3 (coherence) were
unmeasured: lemma_jacc catches only LEXICAL calque, and a sentence-embedding
(mpnet) FAILED to separate frame-mirror calque from genuine divergence. Here we
build two approaches per axis and VALIDATE them on a hand-labeled set
(reproducibility/graded-corpus/metric_validation.tsv) before trusting any:

  FRAME-DIVERGENCE
    frame_struct : predicate-argument overlap = F1 of (head_lemma, deprel, lemma)
                   content triples (Stanza). Lemma-level -> expected to catch a
                   LEXICAL calque but MISS frame-mirroring with new words.
    frame_judge  : LLM-judge (Claude) -- "same scene, just relexicalized?" 0..1.
    [frame_mpnet : sentence-embedding cosine -- the approach that failed; --mpnet]

  COHERENCE
    coh_judge    : LLM-judge (Claude) -- "one coherent situation?" 0..1.
    [coh_struct  : embedding/NLI cohesion -- next pass.]

For frame we report SIMILARITY (high = calque, low = divergent); divergence = 1 - sim.
Validation criterion: a good frame metric gives calques HIGH sim and divergent
candidates LOW sim (big gap); a good coherence metric gives coherent HIGH, incoherent LOW.

  python scripts/sem_metrics.py            # struct + judges (light)
  python scripts/sem_metrics.py --mpnet    # also the embedding frame metric
Run with the Atlas venv python (Stanza; sentence-transformers only with --mpnet).
"""
import argparse
import json
import os
import re
import sys
import urllib.request
from collections import Counter
from pathlib import Path

# judge model: override with SEM_JUDGE_MODEL=claude-opus-4-8 (etc.)
JUDGE_MODEL = os.environ.get("SEM_JUDGE_MODEL", "claude-sonnet-4-6")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import tree_metric as TM  # noqa: E402  (Stanza parse)

ENV = ROOT / ".env"
VAL = ROOT / "reproducibility" / "graded-corpus" / "metric_validation.tsv"
CONTENT = {"NOUN", "VERB", "ADJ", "ADV", "PROPN"}


def env(k):
    # the process environment wins (e.g. `source ~/.config/anthropic/env`), so a
    # rotated key does not have to be copied into the repo's .env
    if os.environ.get(k):
        return os.environ[k]
    for line in ENV.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith(k + "="):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    return None


# ---------- judge routing: TAMU Chat AI for the models it serves --------------
# TAMU Chat AI is a LiteLLM proxy onto AWS Bedrock. Use it for every Claude model
# it carries; the direct Anthropic API stays for what it does not (Opus 5) and as
# an override. Four quirks of that endpoint, each of which costs a debugging hour:
#
#   1. Cloudflare 403s the default Python-urllib User-Agent with "error code:
#      1010". Any ordinary UA gets through. This looks like an auth failure and
#      is not.
#   2. `temperature` is rejected outright -- extended thinking is on server-side.
#      The direct API calls set temperature=0.0; TAMU cannot, so the judge is NOT
#      byte-identical across the two backends even at the same model version.
#   3. `max_tokens` must exceed the server's thinking budget. The direct call uses
#      12 (it wants one number); 200 fails here, 2048 works.
#   4. Responses stream by default and errors arrive inside the SSE body rather
#      than as an HTTP status, so a 400 reads as malformed output. Always
#      "stream": false.
#
# Model ids carry a `protected.` prefix and contain spaces.
TAMU_BASE = os.environ.get("TAMU_AI_BASE", "https://chat-api.tamu.ai/openai")
TAMU_MODELS = {
    "claude-sonnet-4-6": "protected.Claude Sonnet 4.6",
    "claude-sonnet-4-5": "protected.Claude Sonnet 4.5",
    "claude-opus-4-8":   "protected.Claude Opus 4.8",
    "claude-opus-4-7":   "protected.Claude Opus 4.7",
    "claude-opus-4-6":   "protected.Claude Opus 4.6",
    "claude-haiku-4-5":  "protected.Claude-Haiku-4.5",
}
# auto = TAMU when it carries the model and a key exists; else direct API.
# DECIDED 2026-08-27: the DIRECT Anthropic API is the judge of record. TAMU is
# opt-in via SEM_JUDGE_BACKEND=tamu, for exploratory work only.
#
# Why, despite TAMU being free and carrying the same model version: it is not the
# same instrument. A 40-item agreement check against the cached direct-API values
# found exact agreement on 29/40, mean |diff| 0.0875, and a SYSTEMATIC +0.0825
# upward shift -- larger than the ~0.05 noise floor the paper already treats as
# indistinguishable. One case reversed outright, 0.00 -> 1.00. That reversal is
# the dangerous one: the gate argument in the paper's §4 rests on high-form,
# zero-coherence candidates, and a judge that scores those 1.00 dissolves the
# evidence. TAMU runs Bedrock with extended thinking on and rejects temperature,
# which the direct calls set to 0.0.
JUDGE_BACKEND = os.environ.get("SEM_JUDGE_BACKEND", "anthropic")   # anthropic | tamu
TAMU_MIN_TOKENS = 2048


def _tamu_key():
    return os.environ.get("CHAT_AI_TAMU_API_KEY") or env("CHAT_AI_TAMU_API_KEY")


def _use_tamu(model):
    """Opt-in only. Never infer TAMU from the mere presence of a key: the judge of
    record is the direct API, and silently switching instruments would bias the
    coherence axis upward for whichever cells happened to be judged later."""
    if JUDGE_BACKEND != "tamu":
        return False
    if model not in TAMU_MODELS:
        raise JudgeError(f"SEM_JUDGE_BACKEND=tamu but TAMU does not carry {model}")
    return True


def _tamu_call(prompt, model, max_tokens):
    body = {"model": TAMU_MODELS[model],
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max(max_tokens, TAMU_MIN_TOKENS),
            "stream": False}          # never omit: errors hide in the SSE body
    req = urllib.request.Request(
        TAMU_BASE + "/chat/completions", data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {_tamu_key()}",
                 "content-type": "application/json",
                 "User-Agent": "manzoni-judge/1.0"},   # not Python-urllib: Cloudflare 1010
        method="POST")
    r = json.loads(urllib.request.urlopen(req, timeout=120).read())
    return r["choices"][0]["message"]["content"]


def anthropic(prompt, max_tokens=12):
    if _use_tamu(JUDGE_MODEL):
        return _tamu_call(prompt, JUDGE_MODEL, max_tokens)
    body = {"model": JUDGE_MODEL, "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}]}
    if "opus-4-8" not in JUDGE_MODEL:   # opus-4-8 rejects the temperature param
        body["temperature"] = 0.0
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages", data=json.dumps(body).encode(),
        headers={"x-api-key": env("ANTHROPIC_API_KEY"), "anthropic-version": "2023-06-01",
                 "content-type": "application/json"}, method="POST")
    r = json.loads(urllib.request.urlopen(req, timeout=60).read())
    return r["content"][0]["text"]


def _num(s):
    m = re.search(r"(?<![\d.])(0(?:\.\d+)?|1(?:\.0+)?)", s)
    return max(0.0, min(1.0, float(m.group(1)))) if m else None


# ---------- frame: structural (predicate-argument, lemma-level) ----------
def frame_struct_sim(src_toks, gen_toks):
    def triples(toks):
        id2 = {t["i"]: t for t in toks}
        c = Counter()
        for t in toks:
            if t["upos"] == "PUNCT":
                continue
            h = id2[t["head"]]["lemma"] if t["head"] >= 0 else "ROOT"
            c[(h, t["deprel"], t["lemma"])] += 1
        return c
    a, b = triples(src_toks), triples(gen_toks)
    overlap = sum((a & b).values())
    sa, sb = sum(a.values()), sum(b.values())
    if not sa or not sb:
        return 0.0
    p, r = overlap / sb, overlap / sa
    return round(2 * p * r / (p + r), 3) if (p + r) else 0.0


# ---------- frame: LLM-judge ----------
def frame_judge_sim(src, gen, lang="it"):
    if lang == "en":
        p = (f"Here are two English texts.\nTEXT A (source): «{src}»\n"
             f"TEXT B (generated): «{gen}»\n\n"
             "Does B describe the SAME situation or scene as A, only in different words "
             "(a semantic calque), or a GENUINELY DIFFERENT situation? "
             "Answer ONLY with a number from 0 to 1: 0 = an entirely different scene, "
             "1 = the exact same scene merely reworded. The number only.")
    else:
        p = (f"Ecco due frasi italiane.\nFRASE A (origine): «{src}»\n"
             f"FRASE B (generata): «{gen}»\n\n"
             "B descrive la STESSA situazione o scena di A, soltanto con parole diverse "
             "(un calco semantico), oppure una situazione GENUINAMENTE DIVERSA? "
             "Rispondi SOLO con un numero da 0 a 1: 0 = scena del tutto diversa, "
             "1 = stessa identica scena solo riformulata. Solo il numero.")
    return _num(anthropic(p))


# ---------- coherence: LLM-judge ----------
def coh_judge(text, lang="it"):
    if lang == "en":
        p = (f"Here is an English text: «{text}»\n\n"
             "Does the text describe ONE coherent, sensible situation, or is it "
             "grammatical but incoherent (words or clauses that do not compose a single "
             "sensible situation)? Answer ONLY with a number from 0 to 1: "
             "0 = incoherent/nonsensical, 1 = fully coherent. The number only.")
    else:
        p = (f"Ecco una frase italiana: «{text}»\n\n"
             "La frase descrive UNA situazione coerente e sensata, oppure è "
             "grammaticalmente corretta ma incoerente (parole o proposizioni che non "
             "compongono un'unica situazione sensata)? Rispondi SOLO con un numero da 0 "
             "a 1: 0 = incoerente/insensata, 1 = pienamente coerente. Solo il numero.")
    return _num(anthropic(p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mpnet", action="store_true", help="also run the sentence-embedding frame metric")
    a = ap.parse_args()

    emb = None
    if a.mpnet:
        from sentence_transformers import SentenceTransformer, util
        emb = (SentenceTransformer("paraphrase-multilingual-mpnet-base-v2"), util)

    rows = []
    for line in VAL.read_text(encoding="utf-8").splitlines():
        if line.startswith("#") or line.startswith("id\t") or not line.strip():
            continue
        f = line.split("\t")
        rows.append(dict(id=f[0], frame=f[1], coh=f[2], src=f[3], cand=f[4]))

    src_cache = {}
    print(f"{'id':3}{'frame':>10}{'coh':>5}  {'f_struct':>8}{'f_judge':>8}"
          + (f"{'f_mpnet':>8}" if emb else "") + f"{'coh_jdg':>8}")
    for r in rows:
        if r["src"] not in src_cache:
            src_cache[r["src"]] = TM.parse(r["src"])
        s, g = src_cache[r["src"]], TM.parse(r["cand"])
        r["f_struct"] = frame_struct_sim(s, g)
        r["f_judge"] = frame_judge_sim(r["src"], r["cand"])
        r["coh_jdg"] = coh_judge(r["cand"])
        if emb:
            model, util = emb
            v = model.encode([r["src"], r["cand"]])
            r["f_mpnet"] = round(float(util.cos_sim(v[0], v[1])), 3)
        fj = r["f_judge"] if r["f_judge"] is not None else -1.0
        cj = r["coh_jdg"] if r["coh_jdg"] is not None else -1.0
        print(f"{r['id']:3}{r['frame']:>10}{r['coh']:>5}  {r['f_struct']:>8.3f}"
              f"{fj:>8.3f}" + (f"{r.get('f_mpnet',-1):>8.3f}" if emb else "")
              + f"{cj:>8.3f}", flush=True)

    def mean(rs, k):
        v = [x[k] for x in rs if x.get(k) is not None]
        return sum(v) / len(v) if v else float("nan")

    cal = [r for r in rows if r["frame"] == "calque"]
    div = [r for r in rows if r["frame"] == "divergent"]
    yes = [r for r in rows if r["coh"] == "yes"]
    no = [r for r in rows if r["coh"] == "no"]
    fkeys = ["f_struct", "f_judge"] + (["f_mpnet"] if emb else [])
    print("\n=== FRAME similarity: want calque HIGH, divergent LOW (big gap = good metric) ===")
    print(f"{'metric':10}{'calque':>9}{'divergent':>11}{'gap':>8}")
    for k in fkeys:
        c, d = mean(cal, k), mean(div, k)
        print(f"{k:10}{c:>9.3f}{d:>11.3f}{c-d:>8.3f}")
    print("\n=== COHERENCE: want coherent HIGH, incoherent LOW ===")
    print(f"{'metric':10}{'coh=yes':>9}{'coh=no':>9}{'gap':>8}")
    print(f"{'coh_jdg':10}{mean(yes,'coh_jdg'):>9.3f}{mean(no,'coh_jdg'):>9.3f}"
          f"{mean(yes,'coh_jdg')-mean(no,'coh_jdg'):>8.3f}")

    tag = JUDGE_MODEL.replace("/", "_")
    fn = ROOT / "reproducibility" / "graded-corpus" / f"metric_validation_scored_{tag}.json"
    fn.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\njudge model: {JUDGE_MODEL}\nwrote {fn.name}")


if __name__ == "__main__":
    main()
