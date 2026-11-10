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
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "Experiments" / "Manzoni" / "graded-corpus-production"
ENV = ROOT / ".env"

# --- source: the Manzoni geographical incipit (Promessi Sposi 1840, cap. I) ---
# WRONG TEXT — do not use.  This literal is NOT the 1840 quarantana the project
# declares: it reads "verso l'estremità da un'ampia montagna ... in un grande specchio
# rovesciati coll'immagine ripetuti" and "sul declivio", where the 1840 text reads
# "o piuttosto smarrito in un gruppo, in un andirivieni di montagne ... e che l'acqua
# riflette capovolti, co' paesetti posti sulle rive" and "sulla costa".  Periods 8 and
# 9 parse differently in consequence (11 clauses/depth 6 against 14/depth 5, and depth
# 5 against 4).  Every Manzoni run before 2026-07-30 was generated and scored against
# it.  The live pipeline now reads Experiments/Manzoni/homeosyntaxism/texts/incipit_it.txt; this copy is
# kept only so those runs stay identifiable, and is read from the same file it was
# archived to.
INCIPIT = (ROOT / "Experiments/Manzoni/homeosyntaxism/texts/incipit_it_as_parsed.txt").read_text(
    encoding="utf-8").strip()

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


def gen_anthropic(model, prompt, n, temperature, max_tokens, stats=None, refusal_retries=4):
    """n samples, one call each.

    A `refusal` stop_reason (HTTP 200, empty/partial content) is a *stochastic*
    classifier decision rather than a stable property of the prompt: the refusal
    record for this corpus notes an Opus call that refused once and passed on retry,
    and the same prompt yields wildly different refusal rates across cells. Raising
    on the first one therefore threw away a whole cell over a coin flip — retry it,
    and record how often it happened so the rate stays visible in the run file.
    Refused calls produce no output and are not billed."""
    out = []
    refused = attempts = 0
    for _ in range(n):
        for k in range(refusal_retries + 1):
            body = {"model": model, "max_tokens": max_tokens,
                    "messages": [{"role": "user", "content": prompt}]}
            if not any(m in model for m in ("opus-4-8", "opus-4-7", "fable-5", "mythos-5")):
                body["temperature"] = temperature   # sampling params removed on opus-4.7+/fable-5 (400)
            r = _post("https://api.anthropic.com/v1/messages", body,
                      {"x-api-key": env("ANTHROPIC_API_KEY"),
                       "anthropic-version": "2023-06-01",
                       "content-type": "application/json"})
            attempts += 1
            if r.get("stop_reason") == "refusal":
                refused += 1
                if k < refusal_retries:
                    time.sleep(min(2 * 2 ** k, 20)); continue
                break                               # give this sample up, keep the cell
            text = next(b["text"] for b in r["content"] if b.get("type") == "text")
            out.append(text.strip())
            break
    if stats is not None:
        stats["refused"], stats["attempts"] = refused, attempts
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
                "temperature": temperature, "top_p": 0.95, "max_tokens": max_tokens,
                "provider": {"order": [provider], "quantizations": ["fp8"],
                             "allow_fallbacks": False}}
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
