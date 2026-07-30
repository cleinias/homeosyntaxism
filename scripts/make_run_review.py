#!/usr/bin/env python3
"""Reading copy of a scored run: summary table first, then every candidate in full.

    python3 scripts/make_run_review.py --scored results/scored_unrestricted_it_1840_opus5_n10.json \
        --out "../Docs and Research notes/CHR2027 paper/manzoni_it_1840_opus5.md"

Written for the close-reading pass that picks which generation a paper quotes, so
it leads with the numbers that decide the shortlist and then gives the texts
whole, in rank order, with each candidate's own scores at its head.  Takes the
scored file (which carries the texts) and, when it can find it, the run file
beside it, for the provenance line: model, degree, source reference and hash.

Markdown, because these get read on screen and pasted into chat; pandoc will make
a PDF of it if one is wanted.
"""
import argparse
import json
import pathlib
import re
import statistics as st

METRICS = [("clause_inv", "inventory", "the F1 overlap of the clause-type multiset (D3)"),
           ("clause_nest", "nesting", "the F1 overlap of typed clause-attachment edges (D2)"),
           ("depth_gap", "depth gap", "distance from the source's subordination depth"),
           ("coherence", "coherence", "judge: one coherent situation, 0–1"),
           ("frame_div", "frame div.", "judge: distance in sense from the source, 0–1"),
           ("s0_pos", "surface", "part-of-speech sequence similarity — deliberately low"),
           ("lemma_jacc", "lexicon", "lemma overlap with the source — deliberately low")]


def periods(text):
    return len(re.split(r"(?<=[.])\s+(?=[A-Z])", text.strip()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scored", required=True)
    ap.add_argument("--run", help="the run file, for provenance (default: guessed)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--title")
    a = ap.parse_args()

    sp = pathlib.Path(a.scored)
    d = json.loads(sp.read_text(encoding="utf-8"))
    rows = sorted(d["rows"], key=lambda r: -r["score"])

    runp = pathlib.Path(a.run) if a.run else sp.with_name(sp.stem.replace("scored_", "") + ".json")
    run = json.loads(runp.read_text(encoding="utf-8")) if runp.exists() else {}
    src = d.get("source") or ""
    src_words = len(src.split()) if isinstance(src, str) else 0

    L = []
    L.append(f"# {a.title or sp.stem}\n")
    L.append(f"*Generated from `{sp.name}`"
             + (f" and `{runp.name}`" if run else "") + ".*\n")

    L.append("## Provenance\n")
    L.append(f"- **Model** {d.get('model') or run.get('model', '?')}"
             + (f" · backend {run['backend']}" if run.get("backend") else ""))
    if run.get("degree"):
        L.append(f"- **Degree** {run['degree']}"
                 + (f" · regime `{run['regime']}`" if run.get("regime") else ""))
    if run.get("source_ref"):
        L.append(f"- **Source** `{run['source_ref']}` · sha1 `{(run.get('source_sha1') or '')[:12]}`"
                 + (f" · {src_words} words" if src_words else ""))
    elif src_words:
        L.append(f"- **Source** {src_words} words (inline in the scored file)")
    L.append(f"- **Candidates** {len(rows)}"
             + (f" · refusals {run['refused']}" if run.get("refused") is not None else ""))
    if run.get("derived_from"):
        L.append(f"- **Prompt derived from** `{pathlib.Path(run['derived_from']).name}`")
    if run.get("note"):
        L.append(f"- **Note** {run['note']}")
    L.append("")

    L.append("## Summary\n")
    # short headers: the full words collide when this goes through pandoc to PDF
    L.append("| cand | words | per. | inv | nest | depth | coher | "
             "f-div | surf | lex | score | embargo |")
    L.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    for r in rows:
        L.append(f"| **{r['i'] + 1}** | {r['words']} | {periods(r['text'])} | "
                 f"{r['clause_inv']:.2f} | {r['clause_nest']:.2f} | {r['depth_gap']:.1f} | "
                 f"{(r['coherence'] or 0):.2f} | {(r['frame_div'] or 0):.2f} | "
                 f"{r['s0_pos']:.2f} | {r['lemma_jacc']:.3f} | {r['score']:.3f} | "
                 f"{', '.join(r['embargo_hits']) or '—'} |")

    def agg(k, f="{:.3f}"):
        v = [r[k] or 0 for r in rows]
        return (f.format(st.mean(v)), f.format(min(v)), f.format(max(v)),
                f.format(st.pstdev(v)))
    L.append("\n| | mean | min | max | sd |")
    L.append("|---|---:|---:|---:|---:|")
    for key, label, _ in METRICS:
        m, lo, hi, sd = agg(key, "{:.2f}" if key == "depth_gap" else "{:.3f}")
        L.append(f"| {label} | {m} | {lo} | {hi} | {sd} |")
    w = [r["words"] for r in rows]
    L.append(f"| words | {st.mean(w):.0f} | {min(w)} | {max(w)} | {st.pstdev(w):.0f} |")
    L.append("\nWhat the columns mean: " + "; ".join(f"**{lab}** — {why}" for _, lab, why in METRICS)
             + ". `score` is the mean of the five criterion axes, with surface and lexicon "
             "counted inverted, so it rewards holding the form while freeing the words.\n")

    L.append("---\n\n## The candidates, in rank order\n")
    for n, r in enumerate(rows, 1):
        L.append(f"### {n}. Candidate {r['i'] + 1} — inventory {r['clause_inv']:.2f} · "
                 f"nesting {r['clause_nest']:.2f} · coherence {(r['coherence'] or 0):.2f} · "
                 f"score {r['score']:.3f}\n")
        L.append(f"*{r['words']} words in {periods(r['text'])} periods · depth gap "
                 f"{r['depth_gap']:.1f} · frame divergence {(r['frame_div'] or 0):.2f} · "
                 f"embargo: {', '.join(r['embargo_hits']) or 'clean'}*\n")
        for para in r["text"].split("\n"):
            if para.strip():
                L.append(para.strip() + "\n")
        L.append("")

    out = pathlib.Path(a.out)
    out.write_text("\n".join(L), encoding="utf-8")
    print(f"wrote {out} ({len(rows)} candidates, {sum(len(x.split()) for x in L)} words)")


if __name__ == "__main__":
    main()
