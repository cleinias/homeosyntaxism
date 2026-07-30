#!/usr/bin/env python3
"""Audit a recorded constrained-production prompt for hint leakage.

    python3 scripts/audit_leakage.py results/production_d2c_opus48_claude-opus-4-8.json
    python3 scripts/audit_leakage.py <prompt.json> --brief specs/domain_brief_3item.txt

The constrained prompt has four blocks that are supposed to do different jobs:
FORMAL PATTERNS states the critic's formal claims, TARGET DOMAIN names the world
to write about, LEXICAL EMBARGO forbids the source's content words, and
REQUIREMENTS states the threefold criterion.  Two things can go wrong between
them, and both weaken what a successful generation is evidence *of*:

  LEAKAGE   a pattern's lexical realization is handed over by the domain block
            (or, worse, by the pattern's own specification), so instantiating
            the pattern shows the model can take a hint, not that the form
            generates the content.

  CONFLICT  a pattern or the domain block asks for a word the embargo forbids,
            so the two constraints cannot both be satisfied and the model must
            silently pick one.

Both are reported per pattern, mechanically, from the prompt as it was actually
sent.  With --brief the same audit runs against a replacement domain brief, so a
candidate brief can be checked before a re-run rather than after.
"""
import argparse
import json
import pathlib
import re
import sys
import unicodedata

HEADERS = ["FORMAL PATTERNS", "TARGET DOMAIN", "LEXICAL EMBARGO", "REQUIREMENTS"]

# Function words in both prompt languages, plus the specification metalanguage
# ("clause", "verifiable", ...) that every pattern shares by construction and
# that would otherwise swamp the overlap it is the point of this script to find.
STOP = set("""
a acute adherence adnominal against all also an and any are as at
be because been below both but by
can canonical checking clause clauses comment comparing complement composed
contains context continuous coordinated coordination correction
da dal dalla degli dei del della delle di
each early e ed en every
finite for form from full function functions
give given governing
had has have his how
il in include including inside instantiated into is it its
la le li lo
made main marked marks may more must
nel nella no non nor not
of on one only or order other over own
par per
performing period periods placed present presupposes
reads recurs relative requires
same sentence sentences shape share shared should so span specification such
sustained
text than that the their them then there these they this those three
to two
un una uno
verbs verifiable verse via
was what when which while whose with within word words
you your
""".split())

MIN_LEN = 4          # shorter tokens carry no domain content
STEM = 5             # prefix length for the inflection-tolerant embargo match


def fold(s):
    """lowercase, accents stripped -- so 'perché' and 'perche' are one token."""
    return "".join(c for c in unicodedata.normalize("NFD", s.lower())
                   if unicodedata.category(c) != "Mn")


def toks(s):
    return re.findall(r"[a-z']{2,}", fold(s))


def content(s):
    return {t for t in toks(s) if len(t) >= MIN_LEN and t not in STOP}


def blocks(prompt):
    """header -> block text, by locating the headers in order."""
    at = {}
    for h in HEADERS:
        i = prompt.find(h)
        if i < 0:
            sys.exit(f"audit_leakage: block {h!r} not found in the prompt")
        at[h] = i
    out, order = {}, sorted(at.items(), key=lambda kv: kv[1])
    for (h, i), nxt in zip(order, order[1:] + [(None, len(prompt))]):
        out[h] = prompt[i:nxt[1]]
    return out


def patterns(block):
    """[(id, label, text)] -- one entry per [C-Pn-mm] bullet."""
    hits = list(re.finditer(r"^\s*\[(C-P\d+-\d+)\]\s*(.*)$", block, re.M))
    if not hits:
        sys.exit("audit_leakage: no [C-Pn-mm] pattern bullets found")
    out = []
    for m, nxt in zip(hits, hits[1:] + [None]):
        end = nxt.start() if nxt else len(block)
        out.append((m.group(1), m.group(2).strip(), block[m.start():end]))
    return out


def embargo_terms(block):
    body = block.split("Use different words.", 1)[-1]
    return {fold(w.strip()) for w in body.replace("\n", " ").split(",")
            if w.strip() and " " not in w.strip()}


def phrases(a, b, n=3):
    """shared word n-grams, longest first -- verbatim reuse between blocks."""
    def grams(s):
        w = toks(s)
        return {" ".join(w[i:i + k]): k
                for k in range(n, min(9, len(w)) + 1)
                for i in range(len(w) - k + 1)}
    ga, gb = grams(a), grams(b)
    shared = sorted(set(ga) & set(gb), key=lambda g: (-ga[g], g))
    # drop any n-gram contained in a longer shared one
    keep = []
    for g in shared:
        if not any(g in longer for longer in keep):
            keep.append(g)
    return keep


def clash(words, emb):
    """embargo hits: (word, 'exact'|'stem:<prefix>')."""
    stems = {e[:STEM] for e in emb if len(e) >= STEM}
    out = []
    for w in sorted(words):
        if w in emb:
            out.append((w, "exact"))
        elif len(w) >= STEM and w[:STEM] in stems:
            out.append((w, f"stem:{w[:STEM]}-"))
    return out


def report(prompt, brief_path=None):
    b = blocks(prompt)
    dom = b["TARGET DOMAIN"]
    label = "TARGET DOMAIN (as sent)"
    if brief_path:
        dom = pathlib.Path(brief_path).read_text(encoding="utf-8")
        label = f"TARGET DOMAIN (from {brief_path})"
    emb = embargo_terms(b["LEXICAL EMBARGO"])
    dom_words = content(dom)
    pats = patterns(b["FORMAL PATTERNS"])

    print(f"# Leakage audit\n\n{label}: {len(dom_words)} content words; "
          f"embargo: {len(emb)} forms; patterns: {len(pats)}\n")

    print("## Per pattern\n")
    print("| pattern | leaked from domain | verbatim phrases | embargo conflicts |")
    print("|---|---|---|---|")
    for pid, lab, text in pats:
        spec = content(text)
        leak = sorted(spec & dom_words)
        ph = [p for p in phrases(text, dom) if len(p.split()) >= 3]
        cl = clash(spec, emb)
        print(f"| **{pid}** {lab.split(' from ')[-1][:40]} "
              f"| {', '.join(leak) or '—'} "
              f"| {'; '.join(f'“{p}”' for p in ph[:3]) or '—'} "
              f"| {', '.join(f'{w} ({how})' for w, how in cl) or '—'} |")

    print("\n## The domain block against the embargo\n")
    cl = clash(dom_words, emb)
    print(", ".join(f"`{w}` ({how})" for w, how in cl) or "no conflicts")

    print("\n## Totals\n")
    leaking = [pid for pid, _, t in pats if content(t) & dom_words]
    conflicting = [pid for pid, _, t in pats if clash(content(t), emb)]
    print(f"- patterns whose lexis overlaps the domain block: "
          f"**{len(leaking)}/{len(pats)}** ({', '.join(leaking) or 'none'})")
    print(f"- patterns asking for embargoed words: "
          f"**{len(conflicting)}/{len(pats)}** ({', '.join(conflicting) or 'none'})")
    print(f"- domain-block words under embargo: **{len(cl)}**")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("prompt_json", help="a recorded production run (uses its 'prompt')")
    ap.add_argument("--brief", help="audit against this domain brief instead")
    a = ap.parse_args()
    d = json.loads(pathlib.Path(a.prompt_json).read_text(encoding="utf-8"))
    if "prompt" not in d:
        sys.exit("audit_leakage: no 'prompt' key in that run file")
    report(d["prompt"], a.brief)
