#!/usr/bin/env python3
"""run_probe_batch.py — send every prompt of a probe manifest, n times, and record what came back.

The manifests are built by scripts/build_loopA_probes.py (specs/probe/loopA_*_it_D2.json); each item
carries its prompt verbatim. This script sends them as they are, through one backend, and writes one
results file per (manifest, tag, model):

  results/probe_<manifest stem>_<tag>_<model slug>.json   every item: prompt id, request body as
                                                          sent, response metadata, answers parsed
  results/probe_<manifest stem>_<tag>_<model slug>_PROMPTS.jsonl   the prompts as sent, one per line

The results file is rewritten after every item, so a job killed at its walltime keeps what it had
finished, and a rerun with the same tag resumes: items already answered are skipped, but a failed or
short item is sent again whole and its record replaced (its partial answers and its refusal with it).
To try failed items again without touching the first attempt, use --retry K: the items that failed
or came back short in the main file are sent once more and written to <stem>_retryK.json, and an item
already in that file is never sent again (a further attempt takes K+1). A reply counts as recognition when AUTHOR names Manzoni or TITLE names
I promessi sposi (or a translation of the title), the rule of scripts/run_probe.py.

  # ACES, inside the sbatch (vllm serving on :8000):
  python scripts/run_probe_batch.py --manifest specs/probe/loopA_patterns_it_D2.json \\
      --backend vllm-http --endpoint http://127.0.0.1:8000 \\
      --model Qwen/Qwen3-235B-A22B-Instruct-2507-FP8 --n 5 --tag loopA-1126
  # Opus 5 (needs SF's approval of the cost first):
  python scripts/run_probe_batch.py --manifest specs/probe/loopA_outputs_it_D2.json \\
      --backend anthropic --model claude-opus-5 --disable-thinking --n 1 --tag loopA-1126
  # what would be sent, and nothing else:
  python scripts/run_probe_batch.py --manifest ... --model ... --n 1 --tag x --dry-run
"""
import argparse
import hashlib
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import run_production as RP  # noqa: E402  (_post, RECORD, the three generators)

FIELDS = ("AUTHOR", "TITLE", "PASSAGE", "CONFIDENCE", "EVIDENCE",
          "CONTENT", "AGENTS", "ECHOED_WORDS")


def parse(text):
    """Every answer field the prompt asked for; recognition as in run_probe.py."""
    out = {}
    for f in FIELDS:
        m = re.search(rf"^\s*\**{f}\**\s*:\s*\**\s*(.*?)\s*\**\s*$", text, flags=re.M | re.I)
        if m:
            out[f.lower()] = m.group(1).strip()
    c = re.search(r"\d+", out.get("confidence") or "")
    out["confidence"] = int(c.group(0)) if c else None
    out["recognised"] = bool(re.search(r"manzoni", out.get("author") or "", re.I)
                             or re.search(r"promessi|betrothed|fianc", out.get("title") or "", re.I))
    return out


def sha1(s):
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def dump(path, obj):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--backend", choices=["vllm-http", "anthropic", "openrouter"])
    ap.add_argument("--endpoint", default="http://127.0.0.1:8000")
    ap.add_argument("--model", required=True)
    ap.add_argument("--provider", default=None, help="OpenRouter provider pin")
    ap.add_argument("--n", type=int, required=True, help="answers per prompt")
    ap.add_argument("--temperature", type=float, default=0.9,
                    help="sent to vLLM and OpenRouter; the Anthropic path omits it for Opus 5")
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--disable-thinking", action="store_true")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--only", default=None, help="regex on item ids: send only these")
    ap.add_argument("--retry", type=int, default=None, metavar="K",
                    help="resend the main file's failed/short items into <stem>_retryK.json")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    mpath = (ROOT / a.manifest) if not Path(a.manifest).is_absolute() else Path(a.manifest)
    mraw = mpath.read_text(encoding="utf-8")
    man = json.loads(mraw)
    items = [it for it in man["items"] if not a.only or re.search(a.only, it["id"])]
    stem = f"probe_{mpath.stem}_{a.tag}_{a.model.replace('/', '_')}"
    fn, pfn = RP.OUT / f"{stem}.json", RP.OUT / f"{stem}_PROMPTS.jsonl"
    if a.retry is not None:  # the main file is only read; the prompts are already in its PROMPTS file
        if not fn.exists():
            sys.exit(f"run_probe_batch: --retry needs {fn.name}")
        base = json.loads(fn.read_text(encoding="utf-8"))
        ok = {i for i, v in base["items"].items() if not v["failure"] and len(v["answers"]) == a.n}
        items = [it for it in items if it["id"] in base["items"] and it["id"] not in ok]
        fn = RP.OUT / f"{stem}_retry{a.retry}.json"
    print(f"[probe-batch] {mpath.name} ({len(items)} prompts) {a.backend} {a.model} n={a.n} -> {fn.name}",
          flush=True)
    if a.dry_run:
        for it in items[:2]:
            print(f"\n===== {it['id']} =====\n{it['prompt']}")
        print(f"\n[dry run] {len(items)} prompts x n={a.n}; nothing sent")
        return
    if not a.backend:
        sys.exit("run_probe_batch: --backend is required unless --dry-run")

    # what defines the run; the endpoint (a port on whichever node served the model) does not
    params = {"backend": a.backend, "model": a.model,
              "provider": a.provider, "n": a.n, "temperature_arg": a.temperature,
              "max_tokens": a.max_tokens,
              "thinking": (("disabled" if a.disable_thinking else "default")
                           if a.backend == "anthropic" else None)}
    if fn.exists():
        res = json.loads(fn.read_text(encoding="utf-8"))
        if res["manifest_sha1"] != sha1(mraw) or res["params"] != params:
            sys.exit(f"run_probe_batch: {fn.name} exists with a different manifest or parameters; "
                     "refusing to mix runs (use another --tag)")
        print(f"[probe-batch] resuming: {len(res['items'])} items on file", flush=True)
    else:
        res = {"probe": "loopA", "manifest": str(mpath.relative_to(ROOT)), "manifest_sha1": sha1(mraw),
               "manifest_sources": man.get("sources"), "tag": a.tag, "params": params,
               "started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "prompt_file": pfn.name, "endpoints": [], "items": {}}
    if a.backend == "vllm-http" and a.endpoint not in res["endpoints"]:
        res["endpoints"].append(a.endpoint)
    # a failed or short item is sent again on resume; a complete one never is
    done = {i for i, v in res["items"].items() if not v["failure"] and len(v["answers"]) == a.n}
    if a.retry is not None:  # one attempt per item per retry file: nothing in it is replaced
        done = set(res["items"])
        res.setdefault("retry_of", stem + ".json")
    logged = set()
    if pfn.exists():
        logged = {json.loads(line)["id"] for line in pfn.read_text(encoding="utf-8").splitlines() if line}
    with open(pfn, "a", encoding="utf-8") as pf:
        for it in items:
            if it["id"] not in logged:
                pf.write(json.dumps({"id": it["id"], "prompt": it["prompt"]}, ensure_ascii=False) + "\n")

    t_all, fails_in_a_row = time.time(), 0
    for k, it in enumerate(items):
        if it["id"] in done:
            continue
        RP.RECORD = []
        t0, gens, failure = time.time(), [], None
        try:
            if a.backend == "vllm-http":
                gens = RP.gen_vllm(a.endpoint, a.model, it["prompt"], a.n, a.temperature, a.max_tokens)
            elif a.backend == "openrouter":
                gens, _ = RP.gen_openrouter(a.model, it["prompt"], a.n, a.temperature, a.max_tokens,
                                            a.provider)
            else:
                extra = {"thinking": {"type": "disabled"}} if a.disable_thinking else {}
                gens = RP.gen_anthropic(a.model, it["prompt"], a.n, a.temperature, a.max_tokens,
                                        extra_body=extra)
        except Exception as e:  # keep what was paid for, record why it stopped
            gens = getattr(e, "out", gens)
            failure = str(e) or repr(e)
        sent, RP.RECORD = RP.RECORD, None
        answers = [{"j": j, "text": g, **parse(g)} for j, g in enumerate(gens)]
        meta = {key: v for key, v in it.items() if key != "prompt"}
        res["items"][it["id"]] = {
            **meta, "prompt_sha1": sha1(it["prompt"]), "elapsed_s": round(time.time() - t0, 1),
            "failure": failure, "recognised": sum(x["recognised"] for x in answers),
            "requests": [s["body"] for s in sent], "responses": [s["response"] for s in sent],
            "answers": answers}
        res["finished_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        dump(fn, res)
        print(f"  [{k + 1}/{len(items)}] {it['id']}: {len(answers)} answers, recognised "
              f"{res['items'][it['id']]['recognised']}{'  FAILED: ' + failure[:80] if failure else ''}",
              flush=True)
        fails_in_a_row = fails_in_a_row + 1 if failure else 0
        if fails_in_a_row >= 3:
            sys.exit(f"run_probe_batch: three failures in a row, stopping; {fn.name} keeps what was done")
    n_fail = sum(1 for v in res["items"].values() if v["failure"])
    print(f"[probe-batch] wrote {fn.name}: {len(res['items'])} items, {n_fail} failed, "
          f"{round(time.time() - t_all)}s", flush=True)


if __name__ == "__main__":
    main()
