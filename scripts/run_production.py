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
OUT = ROOT / "results"
ENV = ROOT / ".env"

# --- source: the Manzoni geographical incipit (Promessi Sposi 1840, cap. I) ---
INCIPIT = (
    "Quel ramo del lago di Como, che volge a mezzogiorno, tra due catene non "
    "interrotte di monti, tutto a seni e a golfi, a seconda dello sporgere e "
    "del rientrare di quelli, vien, quasi a un tratto, a ristringersi, e a "
    "prender corso e figura di fiume, tra un promontorio a destra, e un’ampia "
    "costiera dall’altra parte; e il ponte, che ivi congiunge le due rive, par "
    "che renda ancor più sensibile all’occhio questa trasformazione, e segni il "
    "punto in cui il lago cessa, e l’Adda rincomincia, per ripigliar poi nome "
    "di lago dove le rive, allontanandosi di nuovo, lascian l’acqua distendersi "
    "e rallentarsi in nuovi golfi e in nuovi seni. "
    "La costiera, formata dal deposito di tre grossi torrenti, scende appoggiata "
    "a due monti contigui, l’uno detto di san Martino, l’altro, con voce "
    "lombarda, il Resegone, dai molti suoi cocuzzoli in fila, che in vero lo "
    "fanno somigliare a una sega: talché non è chi, al primo vederlo, purché "
    "sia di fronte, come per esempio di su le mura di Milano che guardano a "
    "settentrione, non lo discerna tosto, a un tal contrassegno, in quella lunga "
    "e vasta giogaia, dagli altri monti di nome più oscuro e di forma più "
    "comune. "
    "Per un buon pezzo, la costa sale con un pendìo lento e continuo; poi si "
    "rompe in poggi e in valloncelli, in erte e in ispianate, secondo "
    "l’ossatura de’ due monti, e il lavoro dell’acque. "
    "Il lembo estremo, tagliato dalle foci de’ torrenti, è quasi tutto ghiaia e "
    "ciottoloni; il resto, campi e vigne, sparse di terre, di ville, di casali; "
    "in qualche parte boschi, che si prolungano su per la montagna. "
    "Lecco, la principale di quelle terre, e che dà nome al territorio, giace "
    "poco discosto dal ponte, alla riva del lago, anzi viene in parte a trovarsi "
    "nel lago stesso, quando questo ingrossa: un gran borgo al giorno d’oggi, e "
    "che s’incammina a diventar città. "
    "Ai tempi in cui accaddero i fatti che prendiamo a raccontare, quel borgo, "
    "già considerabile, era anche un castello, e aveva perciò l’onore "
    "d’alloggiare un comandante, e il vantaggio di possedere una stabile "
    "guarnigione di soldati spagnoli, che insegnavan la modestia alle fanciulle "
    "e alle donne del paese, accarezzavan di tempo in tempo le spalle a qualche "
    "marito, a qualche padre; e, sul finir dell’estate, non mancavan mai di "
    "spandersi nelle vigne, per diradar l’uve, e alleggerire a’ contadini le "
    "fatiche della vendemmia. "
    "Dall’una all’altra di quelle terre, dall’alture alla riva, da un poggio "
    "all’altro, correvano, e corrono tuttavia, strade e stradette, più o men "
    "ripide, o piane; ogni tanto affondate, sepolte tra due muri, donde, "
    "alzando lo sguardo, non iscoprite che un pezzo di cielo e qualche vetta di "
    "monte; ogni tanto elevate su terrapieni aperti: e da qui la vista spazia "
    "per prospetti più o meno estesi, ma ricchi sempre e sempre qualcosa nuovi, "
    "secondo che i diversi punti piglian più o meno della vasta scena "
    "circostante, e secondo che questa o quella parte campeggia o si scorcia, "
    "spunta o sparisce a vicenda. "
    "Dove un pezzo, dove un altro, dove una lunga distesa di quel vasto e "
    "variato specchio dell’acqua; di qua lago, chiuso all’estremità o "
    "verso l’estremità da un’ampia montagna, e di mano in mano allargato tra "
    "altri monti che si spiegano, a uno a uno, allo sguardo, e in un grande "
    "specchio rovesciati coll’immagine ripetuti; di là braccio di fiume, poi "
    "lago, poi fiume ancora, che va a perdersi in lucido serpeggiamento pur "
    "tra’ monti che l’accompagnano, degradando via via, e perdendosi quasi "
    "anch’essi nell’orizzonte. "
    "Il luogo stesso da dove contemplate que’ vari spettacoli, vi fa spettacolo "
    "da ogni parte: il monte di cui passeggiate le falde, vi svolge, al di "
    "sopra e d’intorno, le sue cime e le sue balze, distinte, rilevate, mutabili "
    "quasi a ogni passo, aprendosi e contornandosi in gioghi ciò che v’era "
    "sembrato prima un sol giogo, e comparendo in vetta ciò che poco innanzi "
    "vi si rappresentava sul declivio: e l’ameno, il domestico di quelle falde "
    "tempera gradevolmente il selvaggio, e orna vie più il magnifico "
    "dell’altre vedute."
)

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
