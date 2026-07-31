#!/usr/bin/env python3
"""Where the points are lost: the source's syntax beside a generation's, period by period.

    python3 scripts/make_syntax_diff.py --scored results/production_scored_D2_it_constr-1840_*.json \\
        --cand 1 --level D3 --out ../"Docs and Research notes"/PoeticsToday/diff_it_D3.tex

The F1 scores say how much of the form is held and never where it goes. This lays
the two analyses side by side, one period per block, at either level:

  --level D3   the clause-type INVENTORY: which types, how many.  The diff is a
               multiset difference, so a period reads "source has two adverbial
               clauses, the generation one" — the missing one is the lost point.
  --level D2   the clause NESTING: the indented tree, and beneath it the typed
               attachment edges (parent-type -> child-type) that the metric
               actually counts, with the ones present in only one side marked.

Per-period F1 is printed for both levels, so the periods that cost the most are
visible before reading a line of the trees.  Note the headline score is computed
over the whole text at once; the per-period figures are a decomposition of it and
will not average to it exactly.

Emits LaTeX (landscape, two columns) for pdflatex.  Run with the Stanza venv.
"""
import argparse
import json
import pathlib
import re
import sys
from collections import Counter

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import run_production as RP      # noqa: E402
import tree_metric as TM         # noqa: E402
import build_specs as BS         # noqa: E402

ESC = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
       "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}",
       "^": r"\textasciicircum{}"}


def esc(s):
    return "".join(ESC.get(c, c) for c in str(s))


def periods(text):
    return [p.strip() for p in re.split(r"(?<=[.])\s+(?=[A-Z])", text.strip()) if p.strip()]


def f1(a, b):
    overlap = sum((a & b).values())
    sa, sb = sum(a.values()), sum(b.values())
    if not sa or not sb:
        return 0.0
    p, r = overlap / sb, overlap / sa
    return 2 * p * r / (p + r) if (p + r) else 0.0


def label(deprel):
    return BS.LABEL.get(deprel, deprel)


def inventory(tok):
    return Counter(label(t["deprel"]) for t in tok if BS.is_clause_head(t))


def edges(tok):
    return Counter((label(p), label(c)) for (p, c), n in TM.clause_edges(tok).items()
                   for _ in range(n))


def tree(tok):
    return BS.emit_D2_nesting(tok).split("\n")


def inventory_block(src, gen):
    """two columns of counted clause types, marked where they differ"""
    only_s, only_g = src - gen, gen - src
    rows = []
    for lab in sorted(set(src) | set(gen)):
        s, g = src.get(lab, 0), gen.get(lab, 0)
        mark = "" if s == g else (r"\;$\leftarrow$ missing " + str(s - g) if s > g
                                  else r"\;$\rightarrow$ extra " + str(g - s))
        rows.append((f"{s} {lab}" if s else "--", f"{g} {lab}{mark}" if g else
                     r"\textit{absent}" + mark))
    return rows, sum(only_s.values()), sum(only_g.values())


def nesting_block(src, gen):
    """the trees side by side, then the typed edges present in only one of them"""
    st, gt = tree(src), tree(gen)
    n = max(len(st), len(gt))
    rows = [((st[i] if i < len(st) else ""), (gt[i] if i < len(gt) else "")) for i in range(n)]
    es, eg = edges(src), edges(gen)
    lost = [f"{p} $\\to$ {c}" for (p, c), k in (es - eg).items() for _ in range(k)]
    added = [f"{p} $\\to$ {c}" for (p, c), k in (eg - es).items() for _ in range(k)]
    return rows, lost, added


HEAD = r"""\documentclass[10pt,a4paper,landscape]{article}
\usepackage[utf8]{inputenc}\usepackage[T1]{fontenc}
\usepackage[margin=1.5cm,landscape]{geometry}
\usepackage{longtable,array,parskip,xcolor}
\setlength{\emergencystretch}{3em}
\newcommand{\shead}[1]{\par\medskip\noindent\textbf{\large #1}\par\smallskip}
\begin{document}
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scored", required=True)
    ap.add_argument("--cand", type=int, required=True, help="candidate number as printed (1-based)")
    ap.add_argument("--level", choices=["D2", "D3"], required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    sp = pathlib.Path(a.scored)
    d = json.loads(sp.read_text(encoding="utf-8"))
    row = next(r for r in d["rows"] if r["i"] == a.cand - 1)
    runp = sp.with_name(sp.stem.replace("production_scored_", "production_") + ".json")
    run = json.loads(runp.read_text(encoding="utf-8")) if runp.exists() else {}
    lang = run.get("lang", "it")
    src_text = RP.resolve_source(run if run else d)

    sp_, gp_ = periods(src_text), periods(row["text"])
    print(f"source {len(sp_)} periods, generation {len(gp_)}")
    stoks = [TM.parse(p, lang=lang) for p in sp_]
    gtoks = [TM.parse(p, lang=lang) for p in gp_]

    L = [HEAD]
    what = "clause inventory (D3)" if a.level == "D3" else "clause nesting (D2)"
    cell = sp.stem.replace("production_scored_", "").replace("_claude-opus-5", " · Opus 5")
    L.append(r"\section*{" + esc(f"{what} — Manzoni's incipit beside {cell}, candidate {a.cand}") + "}")
    L.append(esc(f"Language {lang}. Whole-text scores for this candidate: inventory "
                 f"{row['clause_inv']:.3f}, nesting {row['clause_nest']:.3f}, coherence "
                 f"{row['coherence']}. Source {len(sp_)} periods, generation {len(gp_)}. "
                 f"Per-period figures below decompose the whole-text score; they do not average "
                 f"to it.") + r"\par")

    for k in range(max(len(sp_), len(gp_))):
        s = stoks[k] if k < len(stoks) else []
        g = gtoks[k] if k < len(gtoks) else []
        if a.level == "D3":
            score = f1(inventory(s), inventory(g)) if s and g else 0.0
        else:
            score = f1(edges(s), edges(g)) if s and g else 0.0
        sw = len(sp_[k].split()) if k < len(sp_) else 0
        gw = len(gp_[k].split()) if k < len(gp_) else 0
        L.append(r"\shead{" + f"Period {k + 1} \\hfill F1 {score:.3f} \\quad "
                 f"({sw} words $\\to$ {gw})" + "}")
        L.append(r"\begin{longtable}{@{}p{0.47\textwidth}@{\hspace{2em}}p{0.47\textwidth}@{}}")
        L.append(r"\textbf{source} & \textbf{generation}\\[2pt]")
        if a.level == "D3":
            rows, miss, extra = inventory_block(inventory(s), inventory(g))
            for x, y in rows:
                L.append(f"{esc(x)} & {y}" + r"\\")
            L.append(r"\multicolumn{2}{@{}l@{}}{\small " +
                     esc(f"{miss} clause(s) of the source unmatched, {extra} added") + r"}\\")
        else:
            rows, lost, added = nesting_block(s, g)
            for x, y in rows:
                L.append(r"\texttt{" + esc(x).replace(" ", "~") + r"} & \texttt{" +
                         esc(y).replace(" ", "~") + r"}\\")
            L.append(r"\multicolumn{2}{@{}p{0.96\textwidth}@{}}{\small \textbf{edges lost:} " +
                     (", ".join(lost) if lost else "none") + r"}\\")
            L.append(r"\multicolumn{2}{@{}p{0.96\textwidth}@{}}{\small \textbf{edges added:} " +
                     (", ".join(added) if added else "none") + r"}\\")
        L.append(r"\end{longtable}")

    L.append(r"\end{document}")
    out = pathlib.Path(a.out)
    out.write_text("\n".join(L), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
