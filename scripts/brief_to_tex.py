#!/usr/bin/env python3
"""brief_to_tex.py — render a target-domain brief (spec/brief_<lang>.json) as a
self-standing LaTeX document, for slides / reserve / the paper.

  python scripts/brief_to_tex.py --lang it > brief_it.tex
  lualatex brief_it.tex          # unicode-native engine (the brief has — ' ° etc.)

The brief is the source-blind scene-world the generator transposes into; see
docs/Target_Domain_Brief.md for what each element is and how to author your own.
"""
import argparse
import json
import sys
from pathlib import Path

SPEC = Path(__file__).resolve().parent.parent / "spec"
SPECIALS = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
            "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
LANG_NAME = {"it": "Italian", "en": "English"}


def esc(s):
    return "".join(SPECIALS.get(c, c) for c in s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", default="it", choices=["it", "en"])
    ap.add_argument("--out", default=None, help="write here instead of stdout")
    a = ap.parse_args()

    bdata = json.loads((SPEC / f"brief_{a.lang}.json").read_text(encoding="utf-8"))
    items = "\n".join(rf"\item {esc(e)}" for e in bdata["elements"])
    tex = rf"""\documentclass[11pt]{{article}}
\usepackage[margin=1in]{{geometry}}
\usepackage{{parskip}}
\usepackage{{enumitem}}
\setlength{{\emergencystretch}}{{3em}}
\begin{{document}}
\begin{{center}}\Large\textbf{{Target-domain brief}}\\[2pt]
\normalsize {esc(bdata['domain'])} \;({LANG_NAME[a.lang]})\end{{center}}

\noindent The source-blind scene-world for the constrained homeosyntactic transposition:
the elements the generator must deploy, in its own words, while holding the syntactic form.

\begin{{itemize}}[itemsep=4pt]
{items}
\end{{itemize}}
\end{{document}}
"""
    if a.out:
        Path(a.out).write_text(tex, encoding="utf-8")
        print(f"wrote {a.out}", file=sys.stderr)
    else:
        sys.stdout.write(tex)


if __name__ == "__main__":
    main()
