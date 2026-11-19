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
Writes results/production_{degree}_{lang}_{tag}_{model}.json and, beside it, the prompt
verbatim as ..._PROMPT.txt; the run file also keeps the request body exactly as sent and
each response's metadata (stop reason, usage, block types). An existing run is never
overwritten.
"""
import argparse
import json
import sys
import time
from datetime import datetime, timezone
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

LEXICAL EMBARGO (absolute): these are the source's names and its marked words. None may
appear, in any inflected form:
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
    ap.add_argument("--max-tokens", type=int, default=2048,
                    help="on models whose thinking is on by default (Opus 5+) this caps "
                         "thinking and text together, so give headroom")
    ap.add_argument("--disable-thinking", action="store_true",
                    help="thinking={type: disabled} — Opus 5 otherwise spends the budget "
                         "on the thinking block and returns no passage")
    ap.add_argument("--effort", choices=["low", "medium", "high", "xhigh", "max"])
    ap.add_argument("--tag")
    ap.add_argument("--domain", metavar="FILE",
                    help="replacement TARGET DOMAIN block (e.g. specs/domain_brief_3item.txt); "
                         "same swap run_unrestricted_manzoni.py does, so a constrained run can "
                         "use the clean brief")
    ap.add_argument("--print-prompt", action="store_true",
                    help="assemble the prompt, write it to results/prompt_<label>_<lang>.json "
                         "and stop -- so scripts/audit_leakage.py can be run on the prompt "
                         "BEFORE a generation budget is spent on it")
    ap.add_argument("--same-prompt-as", metavar="RUNFILE",
                    help="a rerun names the run it repeats: the assembled prompt must equal "
                         "that run file's `prompt` byte for byte, or nothing is sent. With "
                         "--print-prompt it is a pre-flight check that needs no model")
    ap.add_argument("--prompt-edits", metavar="FILE",
                    help="JSON {edits: [[old, new], ...]} applied to the assembled prompt, each "
                         "old string matching exactly once: a variant arm of an existing run "
                         "(e.g. specs/effect_free_edits_it_D2.json). With --same-prompt-as, the "
                         "check runs on the prompt BEFORE the edits")
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
        if a.same_prompt_as:
            ref = json.loads(Path(a.same_prompt_as).read_text(encoding="utf-8"))["prompt"]
            if prompt != ref:
                k = next((i for i, (x, y) in enumerate(zip(prompt, ref)) if x != y),
                         min(len(prompt), len(ref)))
                sys.exit(f"run_homeo: [{label}/{a.lang}] prompt differs from "
                         f"{Path(a.same_prompt_as).name} at char {k} ({len(prompt)} vs "
                         f"{len(ref)} chars): here {prompt[max(0, k - 30):k + 30]!r}, "
                         f"there {ref[max(0, k - 30):k + 30]!r}")
            print(f"[{label}/{a.lang}] prompt identical to {Path(a.same_prompt_as).name}",
                  flush=True)
        edits = None
        if a.prompt_edits:
            edits = json.loads(Path(a.prompt_edits).read_text(encoding="utf-8"))["edits"]
            for old, new in edits:
                k = prompt.count(old)
                if k != 1:
                    sys.exit(f"run_homeo: prompt edit matches {k} times, not once: {old[:70]!r}")
                prompt = prompt.replace(old, new)
            RUM.assert_source_blind(prompt)
            print(f"[{label}/{a.lang}] {len(edits)} prompt edits from {a.prompt_edits}: "
                  f"{len(prompt)} chars", flush=True)
        if a.print_prompt:
            RP.OUT.mkdir(parents=True, exist_ok=True)
            fn = RP.OUT / f"prompt_{label}_{a.lang}.json"
            fn.write_text(json.dumps({"degree": a.degree, "lang": a.lang,
                                      "prompt": prompt}, ensure_ascii=False, indent=1),
                          encoding="utf-8")
            print(f"[{label}/{a.lang}] prompt_chars={len(prompt)} -> {fn.name}")
            continue
        # language in the name, not smuggled into the tag: an EN run used to be told
        # apart only by a "-en" suffix on --tag, so forgetting it silently overwrote
        # the IT run of the same degree AND slipped past the copyright ignore rule
        fn = RP.OUT / f"production_{label}_{a.lang}_{a.tag}_{slug}.json"
        # The prompt is saved verbatim beside the run, before any call is made, so a
        # run that dies half-way still leaves the prompt it was given.  A run file is
        # never overwritten: rerunning a tag is refused before anything is spent.
        pfn = fn.with_name(fn.stem + "_PROMPT.txt")
        for f in (fn, pfn):
            if f.exists():
                sys.exit(f"run_homeo: {f.name} exists; refusing to overwrite a run. "
                         f"Use a new --tag.")
        pfn.write_text(prompt, encoding="utf-8")
        print(f"[{label}/{a.lang}] backend={a.backend} model={a.model} n={a.n} prompt_chars={len(prompt)}"
              f" -> {pfn.name}", flush=True)
        # relative to the repo, so a public run file carries no home directory
        def repo_ref(f):
            return f and (str(Path(f).resolve().relative_to(ROOT))
                          if Path(f).resolve().is_relative_to(ROOT) else f)
        domain_ref = repo_ref(a.domain)
        RP.RECORD = []
        started = datetime.now(timezone.utc).isoformat(timespec="seconds")
        t0 = time.time()
        # A run that fails part-way is still written, with what was sent and what came
        # back: the first Opus 5 call at default thinking (2026-11-19) died on a
        # thinking-only answer and took the record of its request with it.
        gens, ormeta, failure = [], None, None
        try:
            if a.backend == "vllm-http":
                gens = RP.gen_vllm(a.endpoint, a.model, prompt, a.n, a.temperature, a.max_tokens)
            elif a.backend == "openrouter":
                gens, ormeta = RP.gen_openrouter(a.model, prompt, a.n, a.temperature,
                                                 a.max_tokens, a.provider)
            else:
                extra = {}
                if a.disable_thinking:
                    extra["thinking"] = {"type": "disabled"}
                if a.effort:
                    extra["output_config"] = {"effort": a.effort}
                gens = RP.gen_anthropic(a.model, prompt, a.n, a.temperature, a.max_tokens,
                                        extra_body=extra)
        except RP.PartialRun as e:
            gens, failure = e.out, str(e)
        except (SystemExit, Exception) as e:
            failure = str(e) or repr(e)
        dt = round(time.time() - t0, 1)
        # What was sent, from the requests themselves.  The bodies of an n-call run
        # are identical (same prompt, same parameters), so one copy is kept unless
        # they differ.  `temperature` used to be written as asked even when the model
        # refuses the parameter and it was never sent (Opus 5): it is now the value in
        # the request, None when none was sent.
        sent = RP.RECORD
        RP.RECORD = None
        bodies = [s["body"] for s in sent]
        same = all(b == bodies[0] for b in bodies)
        fn.write_text(json.dumps({
            "model": a.model, "backend": a.backend, "degree": a.degree, "lang": a.lang,
            "n": a.n, "tag": a.tag, "started_utc": started,
            "temperature": bodies[0].get("temperature") if bodies else None,
            "temperature_asked": a.temperature, "max_tokens": a.max_tokens,
            # --disable-thinking acts on the Anthropic backend only
            "thinking": (("disabled" if a.disable_thinking else "default")
                         if a.backend == "anthropic" else None),
            "effort": a.effort, "provider_routing": ormeta,
            "endpoint": a.endpoint if a.backend == "vllm-http" else None,
            "domain_file": domain_ref,
            "source_ref": specs.get("source_ref"), "source_sha1": specs.get("source_sha1"),
            "prompt": prompt, "prompt_file": pfn.name,
            "same_prompt_as": Path(a.same_prompt_as).name if a.same_prompt_as else None,
            "prompt_edits": ({"file": repo_ref(a.prompt_edits), "edits": edits}
                             if edits else None),
            "elapsed_s": dt, "failure": failure,
            "request" if same else "requests": bodies[0] if same and bodies else bodies,
            "responses": [s["response"] for s in sent],
            "gens": [{"i": k, "text": g} for k, g in enumerate(gens)],
        }, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"[{label}/{a.lang}] wrote {fn.name}: {len(gens)} candidates in {dt}s", flush=True)
        if failure:
            sys.exit(f"run_homeo: FAILED, recorded in {fn.name}: {failure}")
        if gens:
            print(f"  e.g. ({len(gens[0].split())} w): {gens[0][:200]}", flush=True)


if __name__ == "__main__":
    main()
