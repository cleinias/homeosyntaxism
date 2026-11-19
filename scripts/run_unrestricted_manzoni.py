#!/usr/bin/env python3
"""run_unrestricted_manzoni.py — UNRESTRICTED homeosyntactic transposition of the
Manzoni incipit, for the short paper's Appendix A.

The short paper's instrument is source-blind degree-holding + the threefold criterion, with NO
critical-claim patterns (those belong to the full paper's constrained homeomorphism). This script
derives the unrestricted prompt from a recorded constrained-production prompt by removing the
FORMAL PATTERNS block and its two in-text references, keeping only: the D2 clause-inventory target
the target-domain brief, the lexical embargo, and the threefold requirement. Source-blind.
The degree is inherited from the constrained run it is derived from (D2 = clause nesting,
D3 = clause inventory; pre-2026-07-30 runs say grain c and grain a for the same two).

It reads the constrained prompt from a production JSON at runtime, so the script itself embeds no
copyrighted source material (the EN/Colquhoun embargo list stays in the gitignored input file).

Beside --out it writes the prompt verbatim as <out>_PROMPT.txt, before any call is made; the run
file also keeps the request body exactly as sent and each response's metadata (stop reason, usage,
block types, refusals included). An existing run is never overwritten.

  python scripts/run_unrestricted_manzoni.py \
      --constrained results/production_D2_it_pilot-1840_claude-opus-5.json \
      --model claude-opus-4-8 --n 10 --out results/production_d2a_unrestricted_it_opus48.json
"""
import argparse, hashlib, json, re, sys, time
from datetime import datetime, timezone
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import run_production as RP  # shared API helpers (_post, env)


def unrestrict(prompt: str) -> str:
    """Constrained-homeomorphism prompt -> unrestricted homeo prompt (patterns stripped)."""
    p = prompt
    # 1. remove the whole FORMAL PATTERNS block (from its header up to TARGET DOMAIN)
    p = re.sub(r'FORMAL PATTERNS\b.*?(?=TARGET DOMAIN\b)', '', p, flags=re.DOTALL)
    # 2. TASK sentence: drop the "AND instantiate every formal pattern" clause + de-constrain the label
    p = p.replace('CONSTRAINED HOMEOSYNTACTIC transposition', 'HOMEOSYNTACTIC transposition')
    p = re.sub(r'period by period,\s*AND instantiate\s*every formal pattern listed below\s*—\s*while',
               'period by period — while', p, flags=re.DOTALL)
    # run_homeo's wording since the D_n refactor: "hold <degree>, AND instantiate
    # every formal pattern listed --- while inventing ..."
    p = re.sub(r',\s*AND instantiate\s*\n?every formal pattern listed\s*---\s*while',
               ' --- while', p, flags=re.DOTALL)
    # 3. REQUIREMENTS: drop the "satisfy every pattern" clause + rename the label
    p = re.sub(r';\s*satisfy every pattern with fresh material,\s*not by copying a phrasing\.',
               '.', p, flags=re.DOTALL)
    p = p.replace('this is the constrained homeomorphism', 'the threefold criterion')
    # run_homeo's wording since the D_n refactor, which the two lines above never matched:
    # every no-pattern prompt until 2026-11-19 still said "satisfy each pattern with fresh
    # material" and "(the constrained homeomorphism)" while listing no pattern
    p = re.sub(r';\s*satisfy each pattern with fresh material\.', '.', p)
    p = p.replace('(the constrained homeomorphism)', '(the threefold criterion)')
    p = re.sub(r'\n{3,}', '\n\n', p).strip() + "\n"
    # verify no pattern residue: not a pattern block, and not a word about patterns
    assert 'C-P' not in p and 'FORMAL PATTERNS' not in p and 'pattern' not in p.lower() \
        and 'constrained' not in p.lower(), "pattern references not fully removed"
    return p


def replace_domain(prompt: str, block: str) -> str:
    """Swap the TARGET DOMAIN brief for a leaner one (everything else untouched).

    The original brief listed ten elements, several of which pre-supplied content the
    experiment is supposed to elicit — including a fisherman at his nets, who then
    appeared in 10/10 generations and imported a mid-text individual agent the source
    never has.  This lets a run specify its own brief.
    """
    new, k = re.subn(r'TARGET DOMAIN\b.*?(?=LEXICAL EMBARGO\b)',
                     block.strip() + "\n\n", prompt, flags=re.DOTALL)
    assert k == 1, f"expected exactly one TARGET DOMAIN block, replaced {k}"
    return new


def replace_discourse(prompt: str, spec: dict) -> str:
    """Change the DISCOURSE TYPE the prompt asks for (the CHR reviewer-2 control).

    The production prompt asks for a "descriptive opening (incipit) ... for a narrative"
    and requires "one continuous, fully sensible geographical description", so a domain
    swap alone would still ask for a description.  `spec` gives the replacement for the
    TASK's text-type phrase ("what") and for requirement 3 ("coherent").  Form target,
    embargo and requirements 1-2 are untouched.
    """
    new, k1 = re.subn(r'(TASK: Write an ORIGINAL )descriptive opening \(incipit\)( in \w+ )'
                      r'for a narrative set in the target\s*domain below\.',
                      lambda m: m.group(1) + spec["what"] + m.group(2).rstrip() +
                      ", set in the target domain below.", prompt)
    new, k2 = re.subn(r'3\. COHERENT --- .*?\n',
                      lambda m: "3. COHERENT --- " + spec["coherent"] + "\n", new)
    assert k1 == 1 and k2 == 1, f"discourse swap failed: task {k1}, coherent {k2}"
    assert "geographical" not in new and "descriptive opening" not in new
    return new


# Terms that would identify the source. The LEXICAL EMBARGO legitimately contains
# source vocabulary -- that is what it is for -- so it is excluded from the check.
SOURCE_TERMS = ["manzoni", "manzonian", "lecco", "promessi", "como",
                "spagnol", "spanish", "guarnigione", "garrison"]


def assert_source_blind(prompt: str) -> None:
    embargo = re.search(r'LEXICAL EMBARGO\b.*?(?=\n\n[A-Z])', prompt, flags=re.DOTALL)
    rest = prompt.replace(embargo.group(0), "") if embargo else prompt
    hits = [t for t in SOURCE_TERMS if t in rest.lower()]
    assert not hits, f"source-identifying terms outside the embargo: {hits}"


def gen_opus(model, prompt, n, max_tokens=2048, extra_body=None):
    """Resilient Anthropic loop: skip the occasional refusal, keep good candidates."""
    out, refused, att, cap = [], 0, 0, max(n * 3, 24)
    while len(out) < n and att < cap:
        att += 1
        body = {"model": model, "max_tokens": max_tokens,
                "messages": [{"role": "user", "content": prompt}]}
        body.update(extra_body or {})
        try:
            r = RP._post("https://api.anthropic.com/v1/messages", body,
                         {"x-api-key": RP.env("ANTHROPIC_API_KEY"),
                          "anthropic-version": "2023-06-01", "content-type": "application/json"})
        except Exception as e:
            print(f"  post err: {str(e)[:70]}", flush=True); time.sleep(2); continue
        if r.get("stop_reason") == "refusal":
            refused += 1; continue
        try:
            out.append(next(b["text"] for b in r["content"] if b.get("type") == "text").strip())
        except (StopIteration, KeyError):
            # no text block came back — say why instead of skipping silently
            # (e.g. stop_reason=max_tokens with the whole budget spent on thinking)
            print(f"  no text block: stop_reason={r.get('stop_reason')} "
                  f"usage={r.get('usage', {}).get('output_tokens')}", flush=True)
            continue
    return out, refused, att


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--constrained", required=True, help="recorded constrained-production JSON")
    ap.add_argument("--backend", default="anthropic",
                    choices=["anthropic", "vllm-http"],
                    help="anthropic = frontier reference; vllm-http = self-hosted "
                         "open weights (ACES), served by the sbatch on :8000")
    ap.add_argument("--endpoint", default="http://127.0.0.1:8000",
                    help="vllm-http only")
    ap.add_argument("--temperature", type=float, default=0.9,
                    help="vllm-http only; the Anthropic path omits temperature")
    ap.add_argument("--model", default="claude-opus-4-8")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--max-tokens", type=int, default=2048,
                    help="response budget; on models with default-on thinking "
                         "(Opus 5+) this caps thinking+text together, so give headroom")
    ap.add_argument("--disable-thinking", action="store_true",
                    help="pass thinking={type: disabled} (Opus 5: legal at effort "
                         "high or below; closest analogue to the 4.8 no-thinking regime; "
                         "rejected by Fable 5, whose thinking is always on)")
    ap.add_argument("--effort", choices=["low", "medium", "high", "xhigh", "max"],
                    help="output_config effort level (default: model default)")
    ap.add_argument("--domain", help="file holding a replacement TARGET DOMAIN block")
    ap.add_argument("--discourse", help="JSON {what, coherent}: change the requested "
                    "discourse type (see replace_discourse)")
    ap.add_argument("--prompt-edits", metavar="FILE",
                    help="JSON {edits: [[old, new], ...]} applied after the other swaps, each old "
                         "string matching exactly once (e.g. specs/noembargo_edits_it_D2.json)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    d = json.loads(Path(a.constrained).read_text(encoding="utf-8"))
    # Inherit the degree of the constrained source.  Runs recorded before the
    # 2026-07-30 relabelling carry `grain` instead: c was the nesting schema (now D2),
    # a was the clause inventory (now D3).
    LEGACY = {"c": "D2", "a": "D3"}
    degree = d.get("degree") or LEGACY.get(d.get("grain", "a"), "D3")
    prompt = unrestrict(d["prompt"])
    if a.domain:
        before = len(prompt)
        prompt = replace_domain(prompt, Path(a.domain).read_text(encoding="utf-8"))
        print(f"[domain]     {before} -> {len(prompt)} chars; brief from {a.domain}", flush=True)
    if a.discourse:
        prompt = replace_discourse(prompt, json.loads(Path(a.discourse).read_text(encoding="utf-8")))
        print(f"[discourse]  text type from {a.discourse}", flush=True)
    edits = None
    if a.prompt_edits:
        edits = json.loads(Path(a.prompt_edits).read_text(encoding="utf-8"))["edits"]
        for old, new in edits:
            k = prompt.count(old)
            if k != 1:
                sys.exit(f"run_unrestricted_manzoni: prompt edit matches {k} times, not once: "
                         f"{old[:70]!r}")
            prompt = prompt.replace(old, new)
        print(f"[edits]      {len(edits)} from {a.prompt_edits}: {len(prompt)} chars", flush=True)
    assert_source_blind(prompt)
    # The prompt is saved verbatim beside the run before any call is made, and a run
    # file is never overwritten: the 1840 Italian no-pattern run lost its prompt when
    # a scorer wrote over it, and the prompt had to be rebuilt.
    out = Path(a.out)
    pfn = out.with_name(out.stem + "_PROMPT.txt")
    for f in (out, pfn):
        if f.exists():
            sys.exit(f"run_unrestricted_manzoni: {f} exists; refusing to overwrite a run")
    pfn.write_text(prompt, encoding="utf-8")
    print(f"[unrestrict] {len(d['prompt'])} -> {len(prompt)} chars; source-blind {degree}, no patterns"
          f" -> {pfn.name}", flush=True)
    RP.RECORD = []
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    t0 = time.time()
    extra = {}
    if a.disable_thinking:
        extra["thinking"] = {"type": "disabled"}
    if a.effort:
        extra["output_config"] = {"effort": a.effort}
    if a.backend == "vllm-http":
        # one request, n completions; no refusal channel on the OpenAI-compatible API
        gens = RP.gen_vllm(a.endpoint, a.model, prompt, a.n, a.temperature, a.max_tokens)
        refused, att = 0, 1
    else:
        gens, refused, att = gen_opus(a.model, prompt, a.n, max_tokens=a.max_tokens,
                                      extra_body=extra)
    # what was sent, from the requests themselves (see run_production.RECORD); one
    # copy of the body when the calls are identical, as they are in an n-call run
    sent = RP.RECORD
    RP.RECORD = None
    bodies = [x["body"] for x in sent]
    same = all(b == bodies[0] for b in bodies)
    out.write_text(json.dumps({
        "model": a.model, "backend": a.backend, "degree": degree,
        # recorded because the scorer used to default a missing lang to Italian and
        # silently parsed an English run with the Italian model
        "lang": "en" if (d.get("source_ref") or "").endswith("_en.txt") else "it",
        "regime": "unrestricted-homeo-source-blind", "derived_from": str(a.constrained),
        # so the constrained prompt this one was cut from can be matched by hash
        "derived_from_prompt_sha1": hashlib.sha1(d["prompt"].encode("utf-8")).hexdigest(),
        "domain_brief": a.domain, "discourse": a.discourse,
        "prompt_edits": {"file": a.prompt_edits, "edits": edits} if edits else None,
        "n": len(gens), "refused": refused, "attempts": att, "started_utc": started,
        # the value in the request; None when none was sent (the Anthropic path)
        "temperature": bodies[0].get("temperature") if bodies else None,
        "max_tokens": a.max_tokens,
        "thinking": (("disabled" if a.disable_thinking else "default")
                     if a.backend == "anthropic" else None),
        "effort": a.effort, "prompt": prompt, "prompt_file": pfn.name,
        "source_ref": d.get("source_ref"), "source_sha1": d.get("source_sha1"),
        "source": d.get("source"),   # only present when derived from a pre-2026-07-30 run
        "elapsed_s": round(time.time() - t0, 1),
        "request" if same else "requests": bodies[0] if same and bodies else bodies,
        # one per call that got an answer, refusals and empty answers included, so
        # these do not line up one-to-one with gens
        "responses": [x["response"] for x in sent],
        "gens": [{"i": k, "text": g} for k, g in enumerate(gens)],
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[done] wrote {a.out}: {len(gens)} gens (refused {refused}, {att} attempts)", flush=True)


if __name__ == "__main__":
    main()
