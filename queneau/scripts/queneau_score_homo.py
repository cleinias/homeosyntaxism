#!/usr/bin/env python3
"""queneau_score_homo.py --- score the Phase-0 HOMOsyntactic runs (Ch. 3, "Minimal sizes").

Two axes, and the point is their tension:
  STRUCTURE HELD  --- s0_pos (POS-sequence edit similarity to the source), plus d1_triple /
                      d2_clause / depth_gap. HIGH s0_pos = the identical-structure constraint
                      was met (unlike the D2 line, here high overlap is SUCCESS).
  COHERENT        --- an LLM-judge 0..1 (does the passage read as one sensible situation in
                      the target scenario). fr/it/en.

The "minimal size" finding lives in the gap: a model can hold the tree yet be incoherent
(structure bought by grammatical padding). S* = the smallest rung that clears BOTH bars across
all 9 (lang × scenario) cells.

Reads Experiments/Queneau/queneau-homo/outputs/homo_*.json, writes per-cell scored files and a
per-model ladder summary. CPU parse (Stanza) + Anthropic judge; runs locally, not on the cluster.
  ../Oulipo-Atlas-2-0/.venv/bin/python scripts/queneau_score_homo.py
  ... --glob 'homo_*qwen*' --struct-bar 0.60 --coh-bar 0.50 --no-coherence

RELIABILITY INVARIANT (added 2026-08-08, after an audit found silently thinned cells).
A scored cell must contain one row per generation in its source file, or not be written
at all. The failure this guards against is subtle: a transient judge failure used to drop
a single generation, the cell was written anyway with a smaller `n`, and nothing
downstream ever compared `n` to the source. A thinned cell is indistinguishable from a
complete one, so its mean silently rests on a biased subsample. Three defences, all here:
the judge retries (sem_metrics.anthropic), infrastructure failures propagate instead of
being swallowed per-row (score_cell), and a short cell raises rather than writing
(--allow-partial overrides, deliberately explicit). See the 2026-08-08 journal entry.
"""
import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import tree_metric as TM   # noqa: E402
import sem_metrics as SM   # noqa: E402

SPEC_DIR = ROOT / "Experiments" / "Queneau" / "queneau-homo" / "spec"
OUT = ROOT / "Experiments" / "Queneau" / "queneau-homo" / "outputs"
SCORED = OUT / "scored"
RESULTS = ROOT / "Experiments" / "Queneau" / "queneau-homo" / "results"
CACHE = RESULTS / "coh_cache.json"

COH_PROMPT = {
    "fr": ("Voici un texte français : «{t}»\n\nLe texte décrit-il UNE situation cohérente et "
           "sensée, ou est-il grammaticalement correct mais incohérent (mots ou propositions "
           "qui ne composent pas une seule situation sensée) ? Répondez UNIQUEMENT par un "
           "nombre de 0 à 1 : 0 = incohérent, 1 = pleinement cohérent. Le nombre seulement."),
}


def coherence(text, lang, cache):
    key = hashlib.sha1((lang + " " + text).encode()).hexdigest()
    if cache.get(key) is not None:      # never trust a cached None: a null was a FAILED
        return cache[key]               # judge call, and caching it made the failure permanent
    if lang == "fr":
        val = SM._num(SM.anthropic(COH_PROMPT["fr"].format(t=text)))
    else:                       # it / en handled by the shared judge
        val = SM.coh_judge(text, lang=lang)
    if val is None:             # judge answered, but not with a parseable 0..1
        raise SM.JudgeError(f"unparseable judge score for a {lang} text")
    cache[key] = val
    return val


_src_cache = {}


def src_toks(lang):
    if lang not in _src_cache:
        src = json.loads((SPEC_DIR / f"specs_{lang}_D0.json").read_text(encoding="utf-8"))["source"]
        _src_cache[lang] = TM.parse(src, lang=lang)
    return _src_cache[lang]


# Which fidelity metric each degree actually asks the model to hold. `held` used to be
# s0_pos at EVERY degree, which is only right for S0 and D0; at the looser degrees it
# reported whether the generation reproduced the ordered part-of-speech surface -- which
# is not what those degrees ask for, and which a good D2 or D3 generation deliberately
# does not do. So held_rate and pass_rate in every D1/D2c/D2 scored file were numbers
# with no meaning, sitting in the JSON waiting to be read as if they had one.
#
# Nothing in the CHR paper ever read them: its only two held-rate columns are S0/D0
# (tab:blind-vs-copy-data) and D0 (tab:homosyntaxism), and Table 2's one loose-degree
# "Form: Y" was assigned by hand on clause-inv. But the companion paper is ABOUT the
# loose degrees, so this is exactly where the mistake would have been made next.
#
# NB the file-label off-by-one from the 2026-07-30 relabel: the runner's "D2c" is the
# paper's D2 (clause nesting) and the runner's "D2" is the paper's D3 (clause inventory).
HELD_METRIC = {"S0": "s0_pos",      # bare ordered POS surface
               "D0": "s0_pos",      # ordered tree -- fidelity still read off the surface
               "D1": "d1_triple",   # order-free dependency triples
               "D2c": "clause_nest",  # paper's D2: clause nesting
               "D2": "d2_clause"}   # paper's D3: clause inventory


def score_cell(path, cache, do_coh, struct_bar, coh_bar, allow_partial=False):
    d = json.loads(path.read_text(encoding="utf-8"))
    lang, scenario, model = d["lang"], d["scenario"], d["model"]
    degree = d.get("degree")
    if degree not in HELD_METRIC:
        raise RuntimeError(f"{path.name}: unknown degree {degree!r}; refusing to guess "
                           f"which metric 'held' should test")
    held_metric = HELD_METRIC[degree]
    s = src_toks(lang)
    rows, skipped, refused = [], [], []
    for g in d["gens"]:
        text = g["text"].strip()
        try:
            gt = TM.parse(text, lang=lang)
            m = TM.compliance(s, gt)                    # s0_pos, d1_*, d2_clause, depth_gap
            m["coherence"] = coherence(text, lang, cache) if do_coh else None
            m["words"] = len(text.split())
            # one fixed bar, applied to whichever invariant THIS degree asked for --
            # which is what the paper's definition says ("it holds its level when its
            # form-fidelity clears a fixed bar"), and what the code did not do
            m["held"] = m[held_metric] >= struct_bar
            m["pass"] = bool(m["held"] and (m["coherence"] is None or m["coherence"] >= coh_bar))
            rows.append({"i": g["i"], **m})
        # THREE FAILURE CLASSES, THREE HANDLINGS. The distinction that matters is not
        # how the call failed but whether RE-RUNNING IT WOULD RECOVER THE ROW.
        #
        #   parse failure   -- property of the DATA. Stanza cannot parse a degenerate
        #                      generation and never will. Skip the row; a rerun changes
        #                      nothing. Gated by --allow-partial so it stays visible.
        #   judge refusal   -- property of the DATA (measured 2026-08-08: deterministic,
        #                      five identical attempts, five identical refusals). Skip
        #                      the row for the same reason, but record it SEPARATELY:
        #                      unlike a parse failure it biases coherence upward, since
        #                      the refused passages are the degenerate ones.
        #   judge error     -- property of the INFRASTRUCTURE. The same text succeeds on
        #                      the next attempt, so skipping removes a row for a reason
        #                      uncorrelated with content and biases the mean over a
        #                      subsample nobody chose. Kill the cell; the rerun is free.
        #
        # July's single `except Exception` collapsed all three and cost four generations.
        # August's fix collapsed the first two into the third and cost five whole cells
        # (58/63 on the S0 ladder pass) to protect nine good rows apiece from one
        # unjudgeable passage. Refusals are data, not outages.
        except SM.JudgeRefusal as e:                    # data: unjudgeable, and always will be
            refused.append(g["i"])
            print(f"    [REFUSED gen i={g['i']} in {path.name}: {e}]", flush=True)
        except SM.JudgeError:
            raise                                       # infrastructure: kill the cell, loudly
        except Exception as e:                          # bad data: one gen must not sink the cell
            skipped.append(g["i"])
            print(f"    [skip gen i={g['i']} in {path.name}: {e}]", flush=True)
    if not rows:
        raise RuntimeError("no scorable generations in cell")
    # A cell whose row count is quietly below its generation count reads downstream as
    # a complete cell: `n` is the only trace and nothing checked it. Refuse to write one.
    if skipped and not allow_partial:
        raise RuntimeError(f"only {len(rows)}/{len(d['gens'])} generations scored "
                           f"(unparseable: i={skipped}); rerun, or --allow-partial to accept")
    agg = {k: round(mean(r[k] for r in rows), 3)
           for k in ("s0_pos", "d1_triple", "d2_clause", "words")}
    agg["depth_gap"] = round(mean(r["depth_gap"] for r in rows), 2)
    if do_coh:
        agg["coherence"] = round(mean(r["coherence"] for r in rows), 3)
    agg["held_rate"] = round(mean(r["held"] for r in rows), 3)
    agg["pass_rate"] = round(mean(r["pass"] for r in rows), 3)
    # Refusals are NOT gated by --allow-partial: no rerun recovers them, so refusing to
    # write would mean never writing this cell at all. They are instead recorded in the
    # cell itself, so a reader of the scored file can see that its coherence is an upper
    # bound without having to consult a log that may not exist any more.
    meta = {}
    if refused:
        meta = {"n_refused": len(refused), "refused_i": refused,
                "coherence_note": (f"{len(refused)} of {len(d['gens'])} generations were "
                                   "refused by the judge and are excluded; refusals fall on "
                                   "degenerate passages, so this cell's coherence is an "
                                   "UPPER BOUND")}
        print(f"    [{path.name}: {len(rows)}/{len(d['gens'])} scored, "
              f"{len(refused)} refused -- coherence is an upper bound]", flush=True)
    SCORED.mkdir(parents=True, exist_ok=True)
    (SCORED / f"scored_{path.stem}.json").write_text(
        json.dumps({"model": model, "lang": lang, "scenario": scenario,
                    "n": len(rows), "degree": degree, "held_metric": held_metric,
                    "struct_bar": struct_bar, **meta, "agg": agg, "rows": rows},
                   ensure_ascii=False, indent=1),
        encoding="utf-8")
    return {"model": model, "lang": lang, "scenario": scenario, "n": len(rows),
            **({"n_refused": len(refused)} if refused else {}), **agg}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="homo_*.json")
    ap.add_argument("--struct-bar", type=float, default=0.60, help="s0_pos threshold for 'held'")
    ap.add_argument("--coh-bar", type=float, default=0.50, help="coherence threshold for 'pass'")
    ap.add_argument("--no-coherence", action="store_true", help="skip the LLM judge (structure only)")
    ap.add_argument("--skip-existing", action="store_true",
                    help="resume: reuse already-written scored_*.json instead of re-scoring")
    ap.add_argument("--allow-partial", action="store_true",
                    help="write a cell even when some generations failed to parse "
                         "(default: refuse, so a thinned cell can never pass as complete)")
    a = ap.parse_args()
    do_coh = not a.no_coherence

    RESULTS.mkdir(parents=True, exist_ok=True)
    cache = json.loads(CACHE.read_text(encoding="utf-8")) if CACHE.exists() else {}
    files = sorted(OUT.glob(a.glob))
    if not files:
        print(f"no files match {OUT}/{a.glob}"); return
    cells, failed = [], []
    for k, p in enumerate(files):
        sp = SCORED / f"scored_{p.stem}.json"
        if a.skip_existing and sp.exists():
            d = json.loads(sp.read_text(encoding="utf-8"))
            cells.append({"model": d["model"], "lang": d["lang"], "scenario": d["scenario"],
                          "n": d["n"], **d["agg"]})
            print(f"  [cached] {sp.name}", flush=True); continue
        try:
            cells.append(score_cell(p, cache, do_coh, a.struct_bar, a.coh_bar, a.allow_partial))
            c = cells[-1]
            line = f"  {c['model']:42} {c['lang']} {c['scenario']:22} s0_pos={c['s0_pos']:.2f}"
            line += f" coh={c.get('coherence', float('nan')):.2f}" if do_coh else ""
            line += f" pass={c['pass_rate']:.2f}"
            print(line, flush=True)
        except Exception as e:
            failed.append((p.name, str(e)))
            print(f"  !! {p.name}: {e}", flush=True)
        # flush the judge cache as we go: a long unattended pass should not lose
        # every paid-for judgement because the process was killed at cell 60 of 66
        if do_coh and k % 10 == 9:
            CACHE.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
    if do_coh:   # never rewrite the shared judge cache on a structure-only pass (avoids clobber races)
        CACHE.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")

    if failed:   # loud, at the end, where an unattended run's operator will actually see it
        print(f"\n!!! {len(failed)} CELL(S) NOT WRITTEN — rerun with --skip-existing:")
        for name, err in failed:
            print(f"    {name}\n        {err}")
    short = [c for c in cells if c["n"] < 10]
    if short:
        print(f"\n!!! {len(short)} cell(s) hold fewer than 10 rows:")
        for c in short:
            print(f"    n={c['n']:2}  {c['model']} {c['lang']} {c['scenario']}")

    # per-model ladder summary: mean across the model's cells (the "minimal sizes" curve)
    bymodel = defaultdict(list)
    for c in cells:
        bymodel[c["model"]].append(c)
    ladder = []
    for model, cs in bymodel.items():
        row = {"model": model, "cells": len(cs),
               "s0_pos": round(mean(c["s0_pos"] for c in cs), 3),
               "d2_clause": round(mean(c["d2_clause"] for c in cs), 3),
               "pass_rate": round(mean(c["pass_rate"] for c in cs), 3)}
        if do_coh:
            row["coherence"] = round(mean(c["coherence"] for c in cs), 3)
        ladder.append(row)
    ladder.sort(key=lambda r: r["model"])
    (RESULTS / "ladder_summary.json").write_text(
        json.dumps({"struct_bar": a.struct_bar, "coh_bar": a.coh_bar, "ladder": ladder},
                   ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n=== ladder summary (per model, mean over its cells) ===")
    hdr = "model".ljust(44) + "s0_pos  d2_cl  " + ("coher  " if do_coh else "") + "pass"
    print(hdr)
    for r in ladder:
        line = f"{r['model']:44}{r['s0_pos']:.3f}  {r['d2_clause']:.3f}  "
        line += f"{r['coherence']:.3f}  " if do_coh else ""
        print(line + f"{r['pass_rate']:.3f}")
    print(f"\nwrote results/ladder_summary.json ({len(ladder)} models)")


if __name__ == "__main__":
    main()
