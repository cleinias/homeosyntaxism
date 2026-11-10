#!/usr/bin/env python3
"""queneau_run_homo.py --- Phase-0 HOMOsyntactic generator (Queneau *Notations*, D0).

Holds the FULL ordered dependency tree (D0) IDENTICALLY --- same POS, same order, same arcs,
period by period --- and only moves the content into a target scenario. This is the
homosyntactic (iso-/homo-syntaxisme) case, not the homeo case: here high structural overlap
is the GOAL, and the question ("minimal sizes") is at what model scale coherence survives the
identical-structure constraint. Structure-only: no critical-claim patterns.

Reads Experiments/Queneau/queneau-homo/spec/specs_{lang}_D0.json (queneau_build_specs.py) and one
scenario brief; reuses run_production's backends (vllm-http / openrouter / anthropic).

  python scripts/queneau_run_homo.py --backend vllm-http --model Qwen/Qwen2.5-7B-Instruct \
      --lang fr --scenario security-checkpoint --n 10 --tag qwen7b
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import run_production as RP  # noqa: E402

SPEC_DIR = ROOT / "Experiments" / "Queneau" / "queneau-homo" / "spec"
OUT = ROOT / "Experiments" / "Queneau" / "queneau-homo" / "outputs"

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


def build_prompt(specs, lang, scenario, degree, source_visible=False):
    ln = LANG_NAME[lang]
    D = DEGREE_PROMPT[degree]
    origin = "the source shown below" if source_visible else "a hidden source"
    src_block = ""
    if source_visible:
        src_block = (f"\nSOURCE TEXT --- this is the exemplar whose syntactic form you must hold. Do NOT\n"
            f"reuse its words, its content, or its setting; take only its syntactic form:\n{specs['source']}\n")
    return f"""You are assisting with a scholarly experiment in structural literary analysis.

TASK: Write an ORIGINAL short prose passage in {ln}, set in the target scenario below. It must be
a {D['kind']} transposition of {origin}: hold the source's syntactic form at the level
specified below while inventing entirely new words and moving the content into the target
scenario. It is NOT a paraphrase or a translation.
{src_block}
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", required=True, choices=["vllm-http", "anthropic", "openrouter"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--endpoint", default="http://127.0.0.1:8000")
    ap.add_argument("--provider", default=None)
    ap.add_argument("--lang", required=True, choices=["fr", "it", "en"])
    ap.add_argument("--degree", default="D0", choices=["S0", "D0", "D1", "D2", "D2c"])
    ap.add_argument("--scenario", required=True, choices=sorted(SCENARIOS))
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--source-visible", action="store_true",
                    help="copy-ful regime: show the actual source text in the prompt "
                         "(blindness is the only variable changed vs the default source-blind run)")
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()

    specs = json.loads((SPEC_DIR / f"specs_{a.lang}_{a.degree}.json").read_text(encoding="utf-8"))
    prompt = build_prompt(specs, a.lang, a.scenario, a.degree, a.source_visible)
    OUT.mkdir(parents=True, exist_ok=True)
    slug = a.model.replace("/", "_")
    regime = "vis" if a.source_visible else "blind"
    label = f"{a.degree}_{a.lang}_{a.scenario}"
    print(f"[{label}] backend={a.backend} model={a.model} n={a.n} prompt_chars={len(prompt)}", flush=True)
    t0 = time.time()
    stats = {}
    if a.backend == "vllm-http":
        gens = RP.gen_vllm(a.endpoint, a.model, prompt, a.n, a.temperature, a.max_tokens); ormeta = None
    elif a.backend == "openrouter":
        gens, ormeta = RP.gen_openrouter(a.model, prompt, a.n, a.temperature, a.max_tokens, a.provider)
    else:
        gens = RP.gen_anthropic(a.model, prompt, a.n, a.temperature, a.max_tokens, stats=stats)
        ormeta = None
    dt = round(time.time() - t0, 1)
    fn = OUT / f"homo_{label}_{regime}_{a.tag}_{slug}.json"
    fn.write_text(json.dumps({
        "model": a.model, "backend": a.backend, "degree": a.degree, "lang": a.lang, "scenario": a.scenario,
        "source_visible": a.source_visible, "regime": regime,
        "n": a.n, "temperature": a.temperature, "max_tokens": a.max_tokens, "provider_routing": ormeta,
        "source": specs["source"], "prompt": prompt, "elapsed_s": dt,
        **stats,                        # refused / attempts, when the backend reports them
        "gens": [{"i": k, "text": g} for k, g in enumerate(gens)],
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    short = "" if len(gens) == a.n else f"  !! SHORT: {len(gens)}/{a.n}"
    ref = f" (refused {stats['refused']} of {stats['attempts']} attempts)" if stats else ""
    print(f"[{label}] wrote {fn.name}: {len(gens)} candidates in {dt}s{ref}{short}", flush=True)
    if gens:
        print(f"  e.g. ({len(gens[0].split())} w): {gens[0][:200]}", flush=True)


if __name__ == "__main__":
    main()
