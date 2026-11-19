#!/usr/bin/env python3
"""run_probe.py — the recognition probe: can a model name the source from our prompt alone?

Sends a probe prompt verbatim (specs/probe/recognition_<variant>_it_D2.txt: one of our
generation prompts with the task replaced by "identify the work and author this material was
derived from" and the closing instructions by a fixed answer form), n times, and records the
answers with what was sent and what came back. A reply counts as recognition when AUTHOR names
Manzoni or TITLE names I promessi sposi (or a translation of the title).

  python scripts/run_probe.py --variant A --backend anthropic --model claude-opus-5 \\
      --disable-thinking --tag probe-1840
  python scripts/run_probe.py --variant D --backend openrouter --model qwen/qwen3-235b-a22b-2507 \\
      --provider deepinfra --temperature 0.9 --tag probe-1840

Writes results/probe_<variant>_<tag>_<model>.json and, beside it, the prompt as _PROMPT.txt.
Variants and their rationale: Docs and Research notes/PoeticsToday/Recognition-probe-prompts.md.
"""
import argparse
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import run_production as RP  # noqa: E402
import run_unrestricted_manzoni as RUM  # noqa: E402

FIELDS = ("AUTHOR", "TITLE", "PASSAGE", "CONFIDENCE", "EVIDENCE")


def parse(text):
    out = {}
    for f in FIELDS:
        # tolerate markdown bold on either side of the colon ("**AUTHOR:** X", "**AUTHOR**: X")
        m = re.search(rf"^\s*\**{f}\**\s*:\s*\**\s*(.*?)\s*\**\s*$", text, flags=re.M | re.I)
        out[f.lower()] = m.group(1).strip() if m else None
    c = re.search(r"\d+", out["confidence"] or "")
    out["confidence"] = int(c.group(0)) if c else None
    out["recognised"] = bool(re.search(r"manzoni", out["author"] or "", re.I)
                             or re.search(r"promessi|betrothed|fianc", out["title"] or "", re.I))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=["A", "B", "C", "D", "E"])
    ap.add_argument("--backend", required=True, choices=["anthropic", "openrouter"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--provider", default=None)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--temperature", type=float, default=0.9,
                    help="sent to OpenRouter; the Anthropic path omits it for Opus 5")
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--disable-thinking", action="store_true")
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()

    src = ROOT / "specs" / "probe" / f"recognition_{a.variant}_it_D2.txt"
    prompt = src.read_text(encoding="utf-8")
    RUM.assert_source_blind(prompt)
    fn = RP.OUT / f"probe_{a.variant}_{a.tag}_{a.model.replace('/', '_')}.json"
    pfn = fn.with_name(fn.stem + "_PROMPT.txt")
    for f in (fn, pfn):
        if f.exists():
            sys.exit(f"run_probe: {f.name} exists; refusing to overwrite a run")
    pfn.write_text(prompt, encoding="utf-8")
    print(f"[probe {a.variant}] {a.backend} {a.model} n={a.n} prompt_chars={len(prompt)} -> {pfn.name}",
          flush=True)

    RP.RECORD = []
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    t0 = time.time()
    gens, ormeta, failure = [], None, None
    try:
        if a.backend == "openrouter":
            gens, ormeta = RP.gen_openrouter(a.model, prompt, a.n, a.temperature, a.max_tokens,
                                             a.provider)
        else:
            extra = {"thinking": {"type": "disabled"}} if a.disable_thinking else {}
            gens = RP.gen_anthropic(a.model, prompt, a.n, a.temperature, a.max_tokens,
                                    extra_body=extra)
    except RP.PartialRun as e:
        gens, failure = e.out, str(e)
    except (SystemExit, Exception) as e:
        failure = str(e) or repr(e)
    dt = round(time.time() - t0, 1)
    sent, RP.RECORD = RP.RECORD, None
    bodies = [s["body"] for s in sent]
    same = all(b == bodies[0] for b in bodies)
    answers = [{"i": k, "text": g, **parse(g)} for k, g in enumerate(gens)]
    fn.write_text(json.dumps({
        "probe": "recognition", "variant": a.variant, "prompt_source": str(src.relative_to(ROOT)),
        "model": a.model, "backend": a.backend, "n": a.n, "tag": a.tag, "started_utc": started,
        "temperature": bodies[0].get("temperature") if bodies else None,
        "max_tokens": a.max_tokens,
        "thinking": (("disabled" if a.disable_thinking else "default")
                     if a.backend == "anthropic" else None),
        "provider_routing": ormeta, "prompt": prompt, "prompt_file": pfn.name,
        "elapsed_s": dt, "failure": failure,
        "recognised": sum(x["recognised"] for x in answers),
        "request" if same else "requests": bodies[0] if same and bodies else bodies,
        "responses": [s["response"] for s in sent],
        "answers": answers,
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    rec = sum(x["recognised"] for x in answers)
    print(f"[probe {a.variant}] wrote {fn.name}: recognised {rec}/{len(answers)} in {dt}s", flush=True)
    for x in answers[:3]:
        print(f"  {x['author']} | {x['title']} | conf {x['confidence']}", flush=True)
    if failure:
        sys.exit(f"run_probe: FAILED, recorded in {fn.name}: {failure}")


if __name__ == "__main__":
    main()
