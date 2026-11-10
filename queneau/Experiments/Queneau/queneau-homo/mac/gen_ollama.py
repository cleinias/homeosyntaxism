#!/usr/bin/env python3
"""gen_ollama.py — standalone Mac/Ollama generator for the minimal-computing run.

Self-contained: pure Python stdlib, no repo imports. Talks to a local Ollama server's
OpenAI-compatible endpoint. Ollama ignores n>1, so this LOOPS n single-completion requests
(temperature sampling gives n distinct candidates) — unlike the vLLM harness, which sends n in
one call. The prompt logic below is copied VERBATIM from scripts/queneau_run_homo.py, and the
output JSON schema matches it exactly, so the files drop straight into the existing scorer:
    queneau_score_homo.py --glob 'homo_*mac-*'
"""
import argparse
import json
import re
import time
import urllib.request
import urllib.error
from pathlib import Path

HERE = Path(__file__).resolve().parent
SPEC_DIR = HERE / "spec"
OUT = HERE / "outputs"

# ------------------------------------------------------------------ prompt logic (verbatim) --
LANG_NAME = {"fr": "French", "it": "Italian", "en": "English"}
MODERN = {"fr": "Write modern, natural French.",
          "it": "Write modern, natural Italian.",
          "en": "Write modern, natural English."}

# Functional scenario briefs: name the domain's elements by function, no correspondence table
# (a table would force a calque). The model lexicalizes on its own.
SCENARIOS = {
    "security-checkpoint": [
        "an airport/border security checkpoint at a busy hour",
        "the queue and the barriers; the divestment of belongings into trays",
        "the scanners, metal detectors, and the body-search gesture",
        "an officer and a traveller singled out of the line; documents and a wanded pat-down",
        "the register of control: instructions, signage, the impersonal courtesy of authority",
    ],
    "hospital-emergency-room": [
        "a hospital emergency room at a busy hour",
        "the triage desk, the waiting chairs, the gurneys along the corridor",
        "nurses and a doctor moving between bays; monitors, drips, a curtain drawn",
        "a patient in distress and a companion; a complaint about the wait",
        "the register of clinical urgency: intercom pages, wristbands, forms",
    ],
    "grocery-checkout": [
        "a supermarket checkout line at a busy hour",
        "the conveyor belt, the barcode scanner, the bagging area, the card reader",
        "the cashier and a shopper; a cart, a divider bar, a receipt",
        "another shopper crowding from behind; a small friction over space in the queue",
        "the register of retail routine: beeps, the total announced, loyalty-card prompts",
    ],
}


def spec_block(specs):
    return "\n".join(p["spec"] for p in specs["periods"])


def brief_block(scenario):
    return "\n".join("  - " + s for s in SCENARIOS[scenario])


DEGREE_PROMPT = {
    "S0": {
        "kind": "ISOSYNTACTIC",
        "form_target": ("Reproduce, period by period, the ORDERED PART-OF-SPEECH SEQUENCE below: at "
            "each position emit a word of exactly the part of speech shown, in exactly that order "
            "(each token reads '[position]POS'). Same tags, same order --- only the words are yours. "
            "Nothing about dependency or clause structure is specified: only the bare linear POS string."),
        "req1": ("SURFACE PRESERVED --- the ordered part-of-speech sequence above reproduced period by "
            "period, position for position, in your own words."),
    },
    "D0": {
        "kind": "HOMOSYNTACTIC",
        "form_target": ("Reproduce, period by period, the ORDERED DEPENDENCY TREE below: at each "
            "position emit a word of the part of speech shown, in the same order, AND make its "
            "syntactic head and relation match the arc shown (each token reads '[position]POS "
            "relation->[head position]'). Same tags, same order, same arcs --- only the words are yours."),
        "req1": ("STRUCTURE PRESERVED --- the ordered dependency tree above reproduced period by "
            "period, position for position, in your own words."),
    },
    "D1": {
        "kind": "HOMEOSYNTACTIC",
        "form_target": ("Reproduce, period by period, the DEPENDENCY TRIPLES below --- the same "
            "multiset of (head part-of-speech -> relation -> dependent part-of-speech) triples, in "
            "ANY word order you like. Which words depend on which, and how, is fixed; the linear "
            "order and the words themselves are entirely yours."),
        "req1": ("DEPENDENCY STRUCTURE PRESERVED --- the same set of head->relation->dependent "
            "triples per period, in any order and words."),
    },
    "D2": {
        "kind": "HOMEOSYNTACTIC",
        "form_target": ("Reproduce, period by period, the CLAUSE ARCHITECTURE below --- the same "
            "inventory and nesting of clauses (main / relative / adverbial / complement / coordinate, "
            "to the stated depth). The WORD ORDER and the words themselves are entirely yours; only "
            "the clause skeleton is fixed."),
        "req1": ("CLAUSE ARCHITECTURE PRESERVED --- the same clause inventory and nesting depth per "
            "period, in any word order and words you choose."),
    },
    # D2c = the NESTING degree (the paper's D2; the entry above, keyed "D2", is the paper's D3
    # inventory rung -- the file names are off by one from the column labels, a leftover of the
    # 2026-07-30 relabel). Ported verbatim from scripts/queneau_run_homo.py so the laptop and the
    # cluster answer a byte-identical instruction; verified against the stored ladderD2c prompt.
    "D2c": {
        "kind": "HOMEOSYNTACTIC",
        "form_target": ("Reproduce, period by period, the CLAUSE-NESTING SCHEMA below --- an indented "
            "tree in which each line is a clause (main / relative / adverbial / complement / "
            "coordinate) and indentation marks subordination (a clause indented under another is "
            "embedded in it). Hold the SAME nesting: the same clauses, each embedded under the same "
            "kind of parent clause, to the same depth. The WORD ORDER and the words themselves are "
            "entirely yours; only the clause-embedding tree is fixed."),
        "req1": ("CLAUSE NESTING PRESERVED --- the same clause-embedding tree per period (which "
            "clause embeds in which, to the same depth), in any word order and words you choose."),
    },
}


def build_prompt(specs, lang, scenario, degree):
    ln = LANG_NAME[lang]
    D = DEGREE_PROMPT[degree]
    return f"""You are assisting with a scholarly experiment in structural literary analysis.

TASK: Write an ORIGINAL short prose passage in {ln}, set in the target scenario below. It must be
a {D['kind']} transposition of a hidden source: hold the source's syntactic form at the level
specified below while inventing entirely new words and moving the content into the target
scenario. It is NOT a paraphrase or a translation.

FORM TARGET (hold this). {D['form_target']}
Reproduce it for all {specs['n_periods']} periods, in this order:
{spec_block(specs)}

TARGET SCENARIO --- render the content here (in your own words):
{brief_block(scenario)}

REQUIREMENTS --- both hold together:
1. {D['req1']}
2. COHERENT --- one continuous, fully sensible passage in the target scenario; new lexicon,
   no trace of any source content.

{MODERN[lang]} Produce the {specs['n_periods']} periods in order as continuous prose --- no
numbering, no headings. Output ONLY the passage: no preamble, no self-audit."""

# ------------------------------------------------------------------ Ollama backend + driver --
def _chat_once(endpoint, model, prompt, temperature, top_p, max_tokens, timeout=900):
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature, "top_p": top_p,
        "max_tokens": max_tokens, "stream": False,
    }).encode("utf-8")
    req = urllib.request.Request(endpoint.rstrip("/") + "/v1/chat/completions",
                                 data=body, headers={"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.loads(r.read().decode("utf-8"))
    return d["choices"][0]["message"]["content"].strip()


def gen_loop(endpoint, model, prompt, n, temperature, top_p, max_tokens):
    gens = []
    for k in range(n):
        for attempt in range(3):
            try:
                gens.append(_chat_once(endpoint, model, prompt, temperature, top_p, max_tokens))
                break
            except Exception as e:  # noqa: BLE001
                if attempt == 2:
                    print(f"    [warn] candidate {k} failed after 3 tries: {e}", flush=True)
                else:
                    time.sleep(3 * (attempt + 1))
    return gens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", default="http://127.0.0.1:11434")
    ap.add_argument("--model", required=True)
    ap.add_argument("--lang", default="en", choices=["fr", "it", "en"])
    ap.add_argument("--degree", default="D0", choices=["S0", "D0", "D1", "D2", "D2c"])
    ap.add_argument("--scenario", required=True, choices=sorted(SCENARIOS))
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()

    specs = json.loads((SPEC_DIR / f"specs_{a.lang}_{a.degree}.json").read_text(encoding="utf-8"))
    prompt = build_prompt(specs, a.lang, a.scenario, a.degree)
    OUT.mkdir(parents=True, exist_ok=True)
    slug = re.sub(r"[^A-Za-z0-9._-]", "_", a.model)
    label = f"{a.degree}_{a.lang}_{a.scenario}"
    fn = OUT / f"homo_{label}_{a.tag}_{slug}.json"
    if fn.exists():
        print(f"  [skip] {fn.name} already done", flush=True)
        return
    print(f"  [{label}] {a.model}  n={a.n} (looped)", flush=True)
    t0 = time.time()
    gens = gen_loop(a.endpoint, a.model, prompt, a.n, a.temperature, a.top_p, a.max_tokens)
    dt = round(time.time() - t0, 1)
    fn.write_text(json.dumps({
        "model": a.model, "backend": "ollama-openai-loop", "degree": a.degree, "lang": a.lang,
        "scenario": a.scenario, "n": a.n, "temperature": a.temperature, "top_p": a.top_p,
        "max_tokens": a.max_tokens, "provider_routing": None,
        "source": specs["source"], "prompt": prompt, "elapsed_s": dt,
        "gens": [{"i": k, "text": g} for k, g in enumerate(gens)],
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"  [{label}] wrote {fn.name}: {len(gens)}/{a.n} in {dt}s", flush=True)


if __name__ == "__main__":
    main()
