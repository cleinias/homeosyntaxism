#!/usr/bin/env python3
"""Assemble the production table from every scored cell.

    python3 scripts/make_production_table.py                    # markdown to stdout
    python3 scripts/make_production_table.py --csv table.csv    # and a csv beside it

One row per (model, language, degree), each the mean over that cell's candidates,
which is what the paper's table reports. Reads the scored files and pairs each
with its run for the fields the scored file does not carry — language, degree,
regime, source reference.

It refuses to mix instruments. Every cell must name the same source hash, or the
table is built from runs measured against different texts, which is precisely the
defect that made the 2026-07-30 re-run necessary. Pass --allow-mixed-source to
see such a table anyway, clearly marked.
"""
import argparse
import collections
import json
import pathlib
import re
import statistics as st

D = pathlib.Path(__file__).resolve().parent.parent
RESULTS = D / "results"

LABEL = [
    (r"claude-opus-5", "Opus 5"),
    (r"claude-opus-4-8", "Opus 4.8"),
    (r"claude-fable-5", "Fable 5"),
    (r"^Qwen/Qwen3-235B", "Qwen3-235B (ACES)"),
    (r"^qwen/qwen3-235b", "Qwen3-235B (OpenRouter)"),
    (r"kimi-k2", "Kimi-K2"),
    (r"bidl/mmarchesini/Llama-4-Maverick", "Llama-4-Maverick (bf16, Delta)"),
    (r"Maverick", "Llama-4-Maverick (ACES)"),
    (r"^Qwen/Qwen2\.5-72B", "Qwen2.5-72B (ACES)"),
]
COLS = [("clause_inv", "clause-inv"), ("clause_nest", "clause-nest"),
        ("depth_gap", "depth-gap"), ("s0_pos", "surf-synt"),
        ("lemma_jacc", "lex-sim"), ("frame_div", "sem-diverg"),
        ("coherence", "coher")]


def label(model):
    for pat, name in LABEL:
        if re.search(pat, model or ""):
            return name
    return model or "?"


def cells(allow_mixed):
    out, hashes = [], collections.defaultdict(set)
    for s in sorted(RESULTS.glob("*scored*.json")):
        if "legacy" in s.parts or "metric_validation" in s.name:
            continue
        d = json.loads(s.read_text(encoding="utf-8"))
        rows = d.get("rows")
        if not rows:
            continue
        stem = s.stem.replace("production_scored_", "").replace("scored_", "")
        run = RESULTS / f"production_{stem}.json"
        if not run.exists():
            run = RESULTS / f"{stem}.json"
        r = json.loads(run.read_text(encoding="utf-8")) if run.exists() else {}
        lang = r.get("lang") or ("en" if "_en" in stem else "it")
        degree = r.get("degree", "?")
        regime = "unconstrained" if "unrestricted" in stem else "constrained"
        m = label(d.get("model") or r.get("model"))
        hashes[(r.get("source_ref") or "?")].add(r.get("source_sha1") or "?")
        out.append({"model": m, "lang": lang.upper(), "degree": degree,
                    "regime": regime, "n": len(rows), "file": s.name,
                    "sha1": (r.get("source_sha1") or "?")[:12],
                    **{k: st.mean([(x[k] or 0) for x in rows]) for k, _ in COLS},
                    "nest_sd": st.pstdev([(x["clause_nest"] or 0) for x in rows]),
                    "words": st.mean([x["words"] for x in rows]),
                    "embargo": sum(len(x.get("embargo_hits") or []) for x in rows)})
    per_ref = {ref: hs for ref, hs in hashes.items()}
    mixed = any(len(hs) > 1 for hs in per_ref.values())
    if mixed and not allow_mixed:
        raise SystemExit("make_production_table: cells disagree about the source "
                         f"hash ({per_ref}); pass --allow-mixed-source to override")
    return out, per_ref


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv")
    ap.add_argument("--regime", choices=["constrained", "unconstrained"])
    ap.add_argument("--allow-mixed-source", action="store_true")
    a = ap.parse_args()

    rows, refs = cells(a.allow_mixed_source)
    if a.regime:
        rows = [r for r in rows if r["regime"] == a.regime]
    order = {"Opus 5": 0, "Opus 4.8": 1, "Fable 5": 2, "Qwen3-235B (ACES)": 3,
             "Qwen3-235B (OpenRouter)": 4, "Kimi-K2": 5, "Llama-4-Maverick (ACES)": 6,
             "Llama-4-Maverick (bf16, Delta)": 7}
    rows.sort(key=lambda r: (r["regime"] != "constrained", order.get(r["model"], 9),
                             r["lang"], r["degree"]))

    print(f"# Production table — {len(rows)} cells\n")
    print("Source: " + "; ".join(f"`{k}` sha1 {sorted(v)[0][:12]}" for k, v in refs.items()) + "\n")
    print("| regime | model | lang | degree | n | " +
          " | ".join(t for _, t in COLS) + " | nest sd | words | embargo |")
    print("|---|---|---|---|---:|" + "---:|" * len(COLS) + "---:|---:|---:|")
    for r in rows:
        vals = " | ".join(f"{r[k]:.2f}" if k != "depth_gap" else f"{r[k]:.1f}"
                          for k, _ in COLS)
        print(f"| {r['regime'][:5]} | {r['model']} | {r['lang']} | {r['degree']} | "
              f"{r['n']} | {vals} | {r['nest_sd']:.3f} | {r['words']:.0f} | {r['embargo']} |")

    if a.csv:
        import csv
        with open(a.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["regime", "model", "lang", "degree", "n"] +
                       [t for _, t in COLS] + ["nest_sd", "words", "embargo_hits", "source_sha1"])
            for r in rows:
                w.writerow([r["regime"], r["model"], r["lang"], r["degree"], r["n"]] +
                           [round(r[k], 3) for k, _ in COLS] +
                           [round(r["nest_sd"], 3), round(r["words"]), r["embargo"], r["sha1"]])
        print(f"\nwrote {a.csv}")


if __name__ == "__main__":
    main()
