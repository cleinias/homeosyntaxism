#!/usr/bin/env python3
"""run_production.py — Phase D production generation: whole-passage D2
(homeosyntactic) deformation of the Manzoni incipit into the border/migration
domain, oversample-and-select.

D2 = hold the CLAUSE ARCHITECTURE of each period (clause count + subordination
schema, order-free), drop the surface, set the whole passage in a NEW domain
given as a BRIEF (not a lexeme correspondence table — the table is the
calque-inducer we are avoiding). We oversample N candidates per run; scoring +
selection happen off-cluster (scripts/score_production.py) because the frame /
coherence judges need the Anthropic API and compute nodes are offline.

Backends:
  vllm-http  : ACES self-hosted (Qwen3-235B-A22B-Instruct-2507-FP8). Uses the
               OpenAI-compatible /v1/chat/completions with n-sampling (one call,
               N candidates).
  anthropic  : the Opus-4.8 comparison run (loops N times; omits the temperature
               param, which opus-4-8 rejects).

  # ACES (inside the sbatch, vllm already serving on :8000):
  python scripts/run_production.py --backend vllm-http \
      --endpoint http://127.0.0.1:8000 \
      --model Qwen/Qwen3-235B-A22B-Instruct-2507-FP8 \
      --periods all --n 16 --temperature 0.9 --tag qwen3-235b-aces

  # Opus 4.8 comparison (first two periods, off-cluster):
  python scripts/run_production.py --backend anthropic \
      --model claude-opus-4-8 --periods 1-2 --n 4 --tag opus48-cmp
"""
import argparse
import json
import os
import pathlib
import hashlib
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results"
ENV = ROOT / ".env"

# --- source: the Manzoni geographical incipit (Promessi Sposi 1840, cap. I) ---
SOURCE_REF = {"it": "texts/incipit_it.txt", "en": "texts/incipit_en.txt"}

# The Italian source is READ, not retyped.  Until 2026-07-30 this was a string
# literal pasted in at the Phase D launch, and it was not the 1840 text the project
# declares and the critic's workbook analyses: it read "verso l'estremità da un'ampia
# montagna ... in un grande specchio rovesciati coll'immagine ripetuti" where the
# quarantana reads "o piuttosto smarrito in un gruppo, in un andirivieni di montagne
# ... e che l'acqua riflette capovolti, co' paesetti posti sulle rive", and "sul
# declivio" for "sulla costa".  Periods 8 and 9 parsed differently in consequence.
# The old literal is kept at texts/incipit_it_as_parsed.txt so that runs recorded
# against it stay resolvable and identifiable; nothing new should use it.
INCIPIT = (ROOT / SOURCE_REF["it"]).read_text(encoding="utf-8").strip()

# --- target domain: border/migration, central Mediterranean (DECIDED 2026-06-10).
# A BRIEF describing the scene-world, NOT a lexeme correspondence table. ---
DOMAIN_BRIEF = (
    "il paesaggio di confine e di migrazione del Mediterraneo centrale: il "
    "Canale di Sicilia fra le coste del Nordafrica e l’isola di Lampedusa, con "
    "le sue acque divise in fasce giuridiche (acque internazionali, zone SAR di "
    "ricerca e soccorso, acque territoriali), le rotte e le imbarcazioni dei "
    "migranti, il faro, gli scogli e le insenature dell’isola, i mezzi e gli "
    "uomini del soccorso e del controllo (Guardia Costiera, Guardia di Finanza), "
    "il centro di prima accoglienza, i pescatori dell’isola."
)


def periods(text):
    return re.split(r"(?<=[.])\s+(?=[A-Z])", text.strip())


def build_prompt(src):
    return (
        "Ecco un brano descrittivo italiano (l’incipit dei Promessi Sposi di "
        f"Manzoni):\n«{src}»\n\n"
        "Il brano è composto da più periodi. Osserva l’ARCHITETTURA SINTATTICA "
        "di OGNI periodo: quante proposizioni contiene, quali sono principali e "
        "quali subordinate, e come sono collegate (relative, causali, temporali, "
        "completive, concessive…).\n\n"
        "Scrivi UN NUOVO brano descrittivo italiano che:\n"
        "1. abbia lo STESSO NUMERO DI PERIODI e, in ciascun periodo, la STESSA "
        "architettura sintattica del periodo corrispondente dell’originale (lo "
        "stesso numero di proposizioni e lo stesso schema di subordinazione); "
        "sei libero di cambiare l’ordine delle parole;\n"
        "2. sia ambientato INTERAMENTE nel dominio seguente, come UN’UNICA "
        f"descrizione geografica coerente e continua:\n{DOMAIN_BRIEF}\n"
        "3. NON riutilizzi il tema, le entità né il lessico dell’originale "
        "(niente lago, Como, Adda, monti, Lecco, Resegone, vendemmia, ecc.): "
        "inventa contenuto e lessico del tutto nuovi, propri del dominio "
        "indicato.\n\n"
        "Rispondi soltanto con il brano, senza spiegazioni, titoli né virgolette."
    )


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


def _post(url, body, headers):
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers=headers, method="POST")
    for attempt in range(6):
        try:
            return json.loads(urllib.request.urlopen(req, timeout=600).read())
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 529) and attempt < 5:
                time.sleep(min(3 * 2 ** attempt, 40)); continue
            raise
        except Exception:
            if attempt < 5:
                time.sleep(min(3 * 2 ** attempt, 40)); continue
            raise


def gen_vllm(endpoint, model, prompt, n, temperature, max_tokens):
    r = _post(endpoint.rstrip("/") + "/v1/chat/completions",
              {"model": model, "messages": [{"role": "user", "content": prompt}],
               "n": n, "temperature": temperature, "top_p": 0.95,
               "max_tokens": max_tokens},
              {"content-type": "application/json"})
    return [c["message"]["content"].strip() for c in r["choices"]]




def source_sha1(text):
    return hashlib.sha1(text.strip().encode("utf-8")).hexdigest()


def resolve_source(obj, root=ROOT):
    """The source text a run or a spec was built from.

    Runs written before 2026-07-30 carry the passage inline, which duplicated a
    copyrighted translation into every English result file.  Later ones carry
    `source_ref` + `source_sha1` and the text is read back from the file and
    verified, so a source that changes under us fails loudly instead of silently
    rescoring against a different edition.
    """
    if obj.get("source"):
        return obj["source"]
    ref, want = obj.get("source_ref"), obj.get("source_sha1")
    if not ref:
        raise SystemExit("resolve_source: no `source` and no `source_ref` in this file")
    text = (pathlib.Path(root) / ref).read_text(encoding="utf-8").strip()
    got = source_sha1(text)
    if want and got != want:
        raise SystemExit(f"resolve_source: {ref} hashes to {got[:12]} but the file "
                         f"records {want[:12]} — the source has changed since the run")
    return text


NO_TEMPERATURE = ("opus-4-8", "opus-5", "fable-5")


def gen_anthropic(model, prompt, n, temperature, max_tokens, extra_body=None):
    out = []
    for _ in range(n):
        body = {"model": model, "max_tokens": max_tokens,
                "messages": [{"role": "user", "content": prompt}]}
        # Newer Anthropic models reject the parameter outright: Opus 5 answers
        # "`temperature` is deprecated for this model" with a 400.  Sampling on those
        # comes from the model's own default, so an n>1 run still varies.
        if not any(k in model for k in NO_TEMPERATURE):
            body["temperature"] = temperature
        body.update(extra_body or {})
        r = _post("https://api.anthropic.com/v1/messages", body,
                  {"x-api-key": env("ANTHROPIC_API_KEY"),
                   "anthropic-version": "2023-06-01",
                   "content-type": "application/json"})
        out.append(r["content"][0]["text"].strip())
    return out


def gen_openrouter(model, prompt, n, temperature, max_tokens, provider):
    """OpenRouter, provider-pinned to fp8 for cross-model precision parity.
    Loops n single-sample calls (n>1 is unreliable across providers). Returns
    (texts, meta); meta records the resolved provider for the reproducibility note.
    Reasoning models (e.g. deepseek-v4-pro) put the trace in message.reasoning and
    the answer in message.content — take content; give generous max_tokens."""
    out, used = [], None
    for _ in range(n):
        body = {"model": model, "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature, "top_p": 0.95, "max_tokens": max_tokens}
        # Pin the provider and the precision only when a provider was asked for.
        # Sending {"order": [None]} — which is what happened when --provider was
        # omitted — is a 400, and the pin exists for cross-model precision parity,
        # which does not apply when the caller has expressed no preference.
        if provider:
            body["provider"] = {"order": [provider], "quantizations": ["fp8"],
                                "allow_fallbacks": False}
        r = _post("https://openrouter.ai/api/v1/chat/completions", body,
                  {"Authorization": "Bearer " + env("OPENROUTER_API_KEY"),
                   "content-type": "application/json"})
        used = r.get("provider", used)
        txt = (r["choices"][0]["message"].get("content") or "").strip()
        out.append(txt if txt else "[ERR empty content (reasoning ate budget?)]")
    return out, {"requested_provider": provider, "resolved_provider": used, "quant": "fp8"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", required=True, choices=["vllm-http", "anthropic"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--endpoint", default="http://127.0.0.1:8000")
    ap.add_argument("--periods", default="all",
                    help="'all' or a 1-based range like '1-2' / '1'")
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()

    ps = periods(INCIPIT)
    if a.periods == "all":
        src = INCIPIT.strip()
        sel = f"all ({len(ps)})"
    else:
        m = re.match(r"(\d+)(?:-(\d+))?$", a.periods)
        i, j = int(m.group(1)), int(m.group(2) or m.group(1))
        src = " ".join(ps[i - 1:j])
        sel = f"{i}-{j}"
    prompt = build_prompt(src)
    print(f"backend={a.backend} model={a.model} periods={sel} "
          f"n={a.n} t={a.temperature} src_words={len(src.split())}", flush=True)

    t0 = time.time()
    if a.backend == "vllm-http":
        gens = gen_vllm(a.endpoint, a.model, prompt, a.n, a.temperature, a.max_tokens)
    else:
        gens = gen_anthropic(a.model, prompt, a.n, a.temperature, a.max_tokens)
    dt = round(time.time() - t0, 1)

    OUT.mkdir(parents=True, exist_ok=True)
    slug = a.model.replace("/", "_")
    fn = OUT / f"production_{a.tag}_{slug}.json"
    fn.write_text(json.dumps({
        "model": a.model, "backend": a.backend, "periods": sel,
        "n": a.n, "temperature": a.temperature, "max_tokens": a.max_tokens,
        "source": src, "domain_brief": DOMAIN_BRIEF, "prompt": prompt,
        "elapsed_s": dt,
        "gens": [{"i": k, "text": g} for k, g in enumerate(gens)],
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {fn.name}: {len(gens)} candidates in {dt}s", flush=True)
    for k, g in enumerate(gens[:2]):
        print(f"\n--- candidate {k} ({len(g.split())} words) ---\n{g[:600]}", flush=True)


if __name__ == "__main__":
    main()
