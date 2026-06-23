#!/usr/bin/env python3
"""overlap_metrics.py — E3: distance-from-source metrics (anti-calque audit).

Measures how much a generated text copies the source excerpt:
  - lexical_overlap: fraction of the generated text's content-word TYPES that
    also occur in the source (type-level; embargo-relevant)
  - ngram4_overlap: fraction of the generated text's 4-grams (token-level)
    that appear verbatim in the source (calque detector)
  - longest_common: longest run of consecutive shared tokens

Usage:
    python scripts/overlap_metrics.py                # all packages in outputs/ + results.json
    python scripts/overlap_metrics.py <generated.txt> [...]
"""

import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SOURCE_FILE = PROJECT_ROOT / "Texts" / "Manzoni" / "Promessi_Sposi_1840_Capitolo_I.txt"
OUT_ROOT = PROJECT_ROOT / "reproducibility" / "manzoni-mapping" / "outputs"
EXCERPT_END_MARKER = "orna vie più il magnifico"

# Minimal Italian function-word list (content words = everything else, len > 2)
STOP = set("""il lo la i gli le un uno una del dello della dei degli delle al allo alla
ai agli alle dal dallo dalla dai dagli dalle nel nello nella nei negli nelle sul sullo
sulla sui sugli sulle di a da in con su per tra fra e ed o od ma se che chi cui non
più meno molto poco tanto quasi anche ancora già mai sempre poi qui qua là lì dove
come quando perché però dunque quindi mentre essere stato sono è era erano sia fosse
avere ha hanno aveva avevano questo questa questi queste quello quella quelli quelle
quel suo sua suoi sue loro nostro vostro mio tuo si vi ci ne lo li costa via degli
d'un d'una all dell nell sull quell tal tale tali ogni alcun alcuni alcune
""".split())


def tokens(text: str):
    return [t.lower() for t in re.findall(r"[a-zàèéìòóù]+(?:'[a-zàèéìòóù]+)?", text.lower())]


def content_words(toks):
    return {t for t in toks if t not in STOP and len(t) > 2}


def ngrams(toks, n=4):
    return {tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)}


def longest_common_run(a, b):
    bset = set(b)
    best = cur = 0
    bjoined = "\x00".join(b)
    # check runs of a present as contiguous subsequence in b
    for i in range(len(a)):
        if a[i] not in bset:
            continue
        j = i
        while j < len(a) and "\x00".join(a[i:j + 1]) in bjoined:
            j += 1
        best = max(best, j - i)
    return best


def measure(gen_text: str, src_text: str):
    gt, st = tokens(gen_text), tokens(src_text)
    gc, sc = content_words(gt), content_words(st)
    g4, s4 = ngrams(gt), ngrams(st)
    return {
        "lexical_overlap": round(len(gc & sc) / max(len(gc), 1), 3),
        "ngram4_overlap": round(len(g4 & s4) / max(len(g4), 1), 3),
        "longest_common_tokens": longest_common_run(gt, st),
        "gen_content_types": len(gc),
    }


def main():
    src = SOURCE_FILE.read_text(encoding="utf-8")
    src = src[: src.find(".", src.find(EXCERPT_END_MARKER)) + 1]

    targets = []
    if len(sys.argv) > 1:
        targets = [(Path(p).parent.name or Path(p).name, Path(p).read_text(encoding="utf-8"))
                   for p in sys.argv[1:]]
    else:
        for d in sorted(OUT_ROOT.iterdir()):
            f = d / "generated.txt"
            if f.exists():
                targets.append((d.name.split("--")[1][:34], f.read_text(encoding="utf-8")))
        rj = PROJECT_ROOT / "notebooks" / "results.json"
        if rj.exists():
            targets.append(("claude-sonnet-4-6 (notebook)",
                            json.loads(rj.read_text())["generated_text"]))

    print(f"{'text':<36}{'lex_ovl':>9}{'4gram_ovl':>11}{'longest_run':>13}{'types':>7}")
    for name, text in targets:
        # strip self-audit sections if present
        body = re.split(r"#+\s*(RAPPORTO|SELF-AUDIT|Self-Audit|AUDIT)", text)[0]
        m = measure(body, src)
        print(f"{name:<36}{m['lexical_overlap']:>9}{m['ngram4_overlap']:>11}"
              f"{m['longest_common_tokens']:>13}{m['gen_content_types']:>7}")


if __name__ == "__main__":
    main()
