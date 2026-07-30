#!/usr/bin/env python3
"""run_homeo.py — generalized D_n constrained-homeomorphism generator (supersedes run_d2.py).

Generates a source-blind transposition that holds the form at a chosen degree D_n + the
patterns + the threefold criterion. The degree's "form target" is the only thing that
changes across rungs; patterns / brief / embargo / divergence / coherence are constant.

The rungs form a PARTIAL ORDER (not a chain): D0 is the top, S and D1 are independent
projections of it (S forgets the arcs, D1 forgets the order) and are incomparable.
  D0  hold the full ordered dependency tree (POS + order + relations — the top)
  S   hold the exact POS sequence (the surface — the calque/Oulipo end; forgets the arcs)
  D1  hold the dependency structure, order-free (forgets the order)
  D2  hold the clause NESTING — which clause embeds in which     [the operative rung]
  D3  hold the clause INVENTORY — which types, how many (forget the nesting)
  D4  hold only the hypotaxis profile (clause count + depth — the loosest)

The D2/D3 split replaces the old single "clause architecture" rung with its two
grains: what was D2 grain c is now D2, D2 grain a is now D3, and the old D3 is D4.
Runs recorded before 2026-07-30 carry `grain` instead of `degree`; map c->D2, a->D3.

  python scripts/run_homeo.py --backend anthropic --model claude-opus-4-8 \
      --lang it --degree D0 --n 6 --tag opus48
  python scripts/run_homeo.py --backend openrouter --model qwen/qwen3-235b-a22b-2507 \
      --provider deepinfra --lang it --degree D2 --n 6 --tag qwen-or

Reads spec/specs_{lang}_{degree}.json (from build_specs.py); reuses run_production backends.
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
# the brief swap and the source-blindness assertion live with the unrestricted runner;
# reused here so a constrained run cannot silently ship the old ten-element brief
import run_unrestricted_manzoni as RUM  # noqa: E402
import run_production as RP  # noqa: E402

SPEC_DIR = ROOT / "spec"
LANG_NAME = {"it": "Italian", "en": "English"}
MODERN = {"it": "Write MODERN literary Italian (no apocope, no archaic elisions).",
          "en": "Write MODERN literary English."}

# per-degree: (name, the instruction telling the model what form to hold)
DEGREE = {
    "S": ("the exact part-of-speech sequence (S, the surface)",
          "Reproduce, period by period, the PART-OF-SPEECH SEQUENCE below exactly --- the same tags "
          "in the same order. Only the words are yours."),
    "D0": ("the full ordered dependency tree (D0 --- POS + order + relations, the top of the lattice)",
           "Reproduce, period by period, the ORDERED DEPENDENCY TREE below: at each position emit a "
           "word of the part of speech shown, in the same order, AND make its syntactic head and "
           "relation match the arc shown (each token reads '[position]POS relation->[head position]'). "
           "Same tags, same order, same arcs. Only the words are yours."),
    "D1": ("the dependency structure (D1, order-free)",
           "Reproduce, period by period, the DEPENDENCY RELATIONS below (head POS -> relation -> "
           "dependent POS) --- the same set of relations, but in any word order you like."),
    "D2": ("the clause nesting (D2, the operative rung)",
           "Reproduce, period by period, the CLAUSE-NESTING SCHEMA below --- which clause embeds in "
           "which (each node is a clause type, indentation is subordination); word order within it "
           "is entirely free."),
    "D3": ("the clause inventory (D3)",
           "Reproduce, period by period, the CLAUSE INVENTORY below --- which clause types and how "
           "many of each; how they nest, and the word order, are yours."),
    "D4": ("the hypotaxis profile (D4, the loosest rung)",
           "Match, period by period, only the CLAUSE COUNT and MAXIMUM SUBORDINATION DEPTH below; "
           "you are otherwise free in structure, order, and words."),
}


def spec_block(degree, specs):
    return "\n".join(p["spec"] for p in specs["periods"])


def build_prompt(degree, specs, lang):
    ln = LANG_NAME[lang]
    name, instr = DEGREE[degree]
    return f"""You are assisting with a scholarly experiment in structural literary analysis.

TASK: Write an ORIGINAL descriptive opening (incipit) in {ln} for a narrative set in the target
domain below. It must be a CONSTRAINED HOMEOSYNTACTIC transposition: hold {name}, AND instantiate
every formal pattern listed --- while inventing entirely new words, images, and content. It is NOT
a paraphrase, translation, or reconstruction of any source.

FORM TARGET. {instr}
Reproduce it for all {specs['n_periods']} periods, in this order:
{spec_block(degree, specs)}

FORMAL PATTERNS --- the critical claims this experiment tests. Each MUST be instantiated with NEW
material (e.g. the antiphrasis pattern requires a DIFFERENT antiphrasis on target-domain material):
{specs['constraint_block']}

TARGET DOMAIN --- {specs['domain']} (in your own words):
{specs['brief']}

LEXICAL EMBARGO (absolute): none of these source content words may appear, in any inflected form:
{", ".join(specs['embargo'])}

REQUIREMENTS --- all three hold together (the constrained homeomorphism):
1. FORM PRESERVED at {name} (as specified above), in your own words.
2. DIVERGENT --- a genuinely different scene; new lexicon and content; no calque; satisfy each pattern with fresh material.
3. COHERENT --- one continuous, fully sensible geographical description.

{MODERN[lang]} Produce the {specs['n_periods']} periods in order as continuous prose --- no numbering,
no headings. Output ONLY the passage: no preamble, no self-audit."""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["vllm-http", "anthropic", "openrouter"])
    ap.add_argument("--model")
    ap.add_argument("--endpoint", default="http://127.0.0.1:8000")
    ap.add_argument("--provider", default=None)
    ap.add_argument("--lang", default="it", choices=["it", "en"])
    ap.add_argument("--degree", default="D2", choices=["S", "D0", "D1", "D2", "D3", "D4"],
                    help="D2 = clause nesting (operative), D3 = clause inventory, "
                         "D4 = depth profile. Supersedes the old D2 --grains a,c: "
                         "grain c is now D2, grain a is now D3, the old D3 is D4")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--tag")
    ap.add_argument("--domain", metavar="FILE",
                    help="replacement TARGET DOMAIN block (e.g. specs/domain_brief_3item.txt); "
                         "same swap run_unrestricted_manzoni.py does, so a constrained run can "
                         "use the clean brief")
    ap.add_argument("--print-prompt", action="store_true",
                    help="assemble the prompt, write it to results/prompt_<label>_<lang>.json "
                         "and stop -- so scripts/audit_leakage.py can be run on the prompt "
                         "BEFORE a generation budget is spent on it")
    a = ap.parse_args()

    if not a.print_prompt and not (a.backend and a.model and a.tag):
        ap.error("--backend, --model and --tag are required unless --print-prompt")

    specs = json.loads((SPEC_DIR / f"specs_{a.lang}_{a.degree}.json").read_text(encoding="utf-8"))
    RP.OUT.mkdir(parents=True, exist_ok=True)
    slug = (a.model or "").replace("/", "_")
    for _ in [None]:
        prompt = build_prompt(a.degree, specs, a.lang)
        if a.domain:
            before = len(prompt)
            prompt = RUM.replace_domain(prompt, Path(a.domain).read_text(encoding="utf-8"))
            print(f"[domain] {before} -> {len(prompt)} chars; brief from {a.domain}", flush=True)
        RUM.assert_source_blind(prompt)
        label = a.degree
        if a.print_prompt:
            RP.OUT.mkdir(parents=True, exist_ok=True)
            fn = RP.OUT / f"prompt_{label}_{a.lang}.json"
            fn.write_text(json.dumps({"degree": a.degree, "lang": a.lang,
                                      "prompt": prompt}, ensure_ascii=False, indent=1),
                          encoding="utf-8")
            print(f"[{label}/{a.lang}] prompt_chars={len(prompt)} -> {fn.name}")
            continue
        print(f"[{label}/{a.lang}] backend={a.backend} model={a.model} n={a.n} prompt_chars={len(prompt)}", flush=True)
        t0 = time.time()
        if a.backend == "vllm-http":
            gens = RP.gen_vllm(a.endpoint, a.model, prompt, a.n, a.temperature, a.max_tokens)
            ormeta = None
        elif a.backend == "openrouter":
            gens, ormeta = RP.gen_openrouter(a.model, prompt, a.n, a.temperature, a.max_tokens, a.provider)
        else:
            gens = RP.gen_anthropic(a.model, prompt, a.n, a.temperature, a.max_tokens); ormeta = None
        dt = round(time.time() - t0, 1)
        fn = RP.OUT / f"production_{label}_{a.tag}_{slug}.json"
        fn.write_text(json.dumps({
            "model": a.model, "backend": a.backend, "degree": a.degree, "lang": a.lang,
            "n": a.n, "temperature": a.temperature, "max_tokens": a.max_tokens, "provider_routing": ormeta,
            "source": specs["source"], "prompt": prompt, "elapsed_s": dt,
            "gens": [{"i": k, "text": g} for k, g in enumerate(gens)],
        }, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"[{label}/{a.lang}] wrote {fn.name}: {len(gens)} candidates in {dt}s", flush=True)
        if gens:
            print(f"  e.g. ({len(gens[0].split())} w): {gens[0][:200]}", flush=True)


if __name__ == "__main__":
    main()
