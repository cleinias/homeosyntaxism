#!/usr/bin/env python3
"""run_unrestricted_manzoni.py — UNRESTRICTED homeosyntactic D2 (grain a) transposition of the
Manzoni incipit, for the short paper's Appendix A.

The short paper's instrument is source-blind degree-holding + the threefold criterion, with NO
critical-claim patterns (those belong to the full paper's constrained homeomorphism). This script
derives the unrestricted prompt from a recorded constrained-production prompt by removing the
FORMAL PATTERNS block and its two in-text references, keeping only: the D2 clause-inventory target
(grain a), the target-domain brief, the lexical embargo, and the threefold requirement. Source-blind.

It reads the constrained prompt from a production JSON at runtime, so the script itself embeds no
copyrighted source material (the EN/Colquhoun embargo list stays in the gitignored input file).

  python scripts/run_unrestricted_manzoni.py \
      --constrained results/production_d2a_opus48_claude-opus-4-8.json \
      --model claude-opus-4-8 --n 10 --out results/production_d2a_unrestricted_it_opus48.json
"""
import argparse, json, re, sys, time
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
    # 3. REQUIREMENTS: drop the "satisfy every pattern" clause + rename the label
    p = re.sub(r';\s*satisfy every pattern with fresh material,\s*not by copying a phrasing\.',
               '.', p, flags=re.DOTALL)
    p = p.replace('this is the constrained homeomorphism', 'the threefold criterion')
    p = re.sub(r'\n{3,}', '\n\n', p).strip() + "\n"
    # verify no pattern residue
    assert 'C-P' not in p and 'formal pattern' not in p.lower() and 'FORMAL PATTERNS' not in p, \
        "pattern references not fully removed"
    return p


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
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    d = json.loads(Path(a.constrained).read_text(encoding="utf-8"))
    grain = d.get("grain", "a")  # inherit the grain of the constrained source (a, c, ...)
    prompt = unrestrict(d["prompt"])
    print(f"[unrestrict] {len(d['prompt'])} -> {len(prompt)} chars; source-blind grain-{grain}, no patterns",
          flush=True)
    t0 = time.time()
    extra = {}
    if a.disable_thinking:
        extra["thinking"] = {"type": "disabled"}
    if a.effort:
        extra["output_config"] = {"effort": a.effort}
    gens, refused, att = gen_opus(a.model, prompt, a.n, max_tokens=a.max_tokens,
                                  extra_body=extra)
    Path(a.out).write_text(json.dumps({
        "model": a.model, "backend": "anthropic", "grain": grain,
        "regime": "unrestricted-homeo-source-blind", "derived_from": str(a.constrained),
        "n": len(gens), "refused": refused, "attempts": att, "temperature": None,
        "max_tokens": a.max_tokens,
        "thinking": "disabled" if a.disable_thinking else "default",
        "effort": a.effort, "source": d.get("source"), "prompt": prompt,
        "elapsed_s": round(time.time() - t0, 1),
        "gens": [{"i": k, "text": g} for k, g in enumerate(gens)],
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[done] wrote {a.out}: {len(gens)} gens (refused {refused}, {att} attempts)", flush=True)


if __name__ == "__main__":
    main()
