#!/usr/bin/env python3
"""queneau_fallback_judge.py --- rescue cells the primary coherence judge refuses to score.

Claude Sonnet 4.6, the judge of record, declines a small fraction of these passages
outright: HTTP 200 with stop_reason "refusal", category "bio", and it does so
deterministically -- five retries on the identical text change nothing, because a
refusal is a property of the text and not of the network. Under the reliability
invariant added on 2026-08-08 the whole cell is then refused rather than written short,
which is right (loud beats lossy) but leaves a hole.

This re-judges only those cells on a fallback model, as Anthropic's own refusal payload
suggests, and records WHICH model judged each row so the mixture is auditable rather
than invisible.

Two deliberate choices:

  * It does NOT touch results/coh_cache.json. That cache is keyed on sha1(lang + text)
    with no judge model in the key, so writing Opus judgements into it would let a later
    Sonnet run silently reuse them and quietly make the judge non-uniform everywhere.
    The fallback judgements live only in the scored cells they belong to.

  * Rows carry `judge_model`, and the cell carries `judge_fallback`, so a reader of the
    scored file can tell which instrument produced which number without consulting a
    journal entry.

    python scripts/queneau_fallback_judge.py --glob 'homo_D0_*ladderD0soft*32B*'
    python scripts/queneau_fallback_judge.py --list        # show refused cells only
"""
import argparse
import json
import os
import sys
from pathlib import Path
from statistics import mean

FALLBACK = os.environ.get("QUENEAU_FALLBACK_JUDGE", "claude-opus-4-8")
os.environ["SEM_JUDGE_MODEL"] = FALLBACK          # read at import time by sem_metrics

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import tree_metric as TM   # noqa: E402
import sem_metrics as SM   # noqa: E402

SPEC = ROOT / "Experiments" / "Queneau" / "queneau-homo" / "spec"
OUT = ROOT / "Experiments" / "Queneau" / "queneau-homo" / "outputs"
SCORED = OUT / "scored"

_src = {}


def src_toks(lang, degree):
    key = (lang, degree)
    if key not in _src:
        spec = json.loads((SPEC / f"specs_{lang}_{degree}.json").read_text(encoding="utf-8"))
        _src[key] = TM.parse(spec["source"], lang=lang)
    return _src[key]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="homo_*.json")
    ap.add_argument("--struct-bar", type=float, default=0.60)
    ap.add_argument("--coh-bar", type=float, default=0.50)
    ap.add_argument("--list", action="store_true", help="only report which cells are unscored")
    a = ap.parse_args()

    todo = [p for p in sorted(OUT.glob(a.glob))
            if not (SCORED / f"scored_{p.stem}.json").exists()]
    if not todo:
        print(f"no unscored cells match {a.glob} — nothing to rescue"); return
    print(f"{len(todo)} unscored cell(s); fallback judge = {FALLBACK}\n")
    if a.list:
        for p in todo: print(f"  {p.name}")
        return

    for p in todo:
        d = json.loads(p.read_text(encoding="utf-8"))
        lang, scen, model, degree = d["lang"], d["scenario"], d["model"], d["degree"]
        s = src_toks(lang, degree)
        rows, refused = [], 0
        for g in d["gens"]:
            text = g["text"].strip()
            try:
                m = TM.compliance(s, TM.parse(text, lang=lang))
            except Exception as e:
                print(f"    [skip gen i={g['i']}: unparseable — {e}]", flush=True); continue
            try:
                m["coherence"] = SM.coh_judge(text, lang=lang)   # already returns a 0..1 float
                m["judge_model"] = FALLBACK
            except SM.JudgeError as e:
                refused += 1
                print(f"    [gen i={g['i']}: FALLBACK ALSO REFUSED — {e}]", flush=True)
                continue
            if m["coherence"] is None:
                refused += 1
                print(f"    [gen i={g['i']}: fallback gave no parseable score]", flush=True); continue
            m["words"] = len(text.split())
            m["held"] = m["s0_pos"] >= a.struct_bar
            m["pass"] = bool(m["held"] and m["coherence"] >= a.coh_bar)
            rows.append({"i": g["i"], **m})

        if not rows:
            print(f"  !! {p.name}: fallback rescued nothing ({refused} refusals)"); continue
        agg = {k: round(mean(r[k] for r in rows), 3)
               for k in ("s0_pos", "d1_triple", "d2_clause", "words")}
        agg["depth_gap"] = round(mean(r["depth_gap"] for r in rows), 2)
        agg["coherence"] = round(mean(r["coherence"] for r in rows), 3)
        agg["held_rate"] = round(mean(r["held"] for r in rows), 3)
        agg["pass_rate"] = round(mean(r["pass"] for r in rows), 3)
        SCORED.mkdir(parents=True, exist_ok=True)
        (SCORED / f"scored_{p.stem}.json").write_text(json.dumps({
            "model": model, "lang": lang, "scenario": scen, "n": len(rows),
            "judge_fallback": {"model": FALLBACK,
                               "reason": "primary judge (claude-sonnet-4-6) refused this cell",
                               "rows_rescued": len(rows), "rows_still_refused": refused},
            "agg": agg, "rows": rows}, ensure_ascii=False, indent=1), encoding="utf-8")
        short = "" if len(rows) == len(d["gens"]) else f"  !! {len(rows)}/{len(d['gens'])}"
        print(f"  {model:34} {lang} {scen:24} coh={agg['coherence']:.2f} "
              f"s0_pos={agg['s0_pos']:.2f} rescued={len(rows)}{short}", flush=True)


if __name__ == "__main__":
    main()
