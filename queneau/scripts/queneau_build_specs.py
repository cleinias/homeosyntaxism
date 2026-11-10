#!/usr/bin/env python3
"""queneau_build_specs.py --- D0 spec extractor for the Phase-0 homosyntactic re-run.

Locks Queneau's *Notations* at D0 (the full ordered dependency tree: POS + linear order +
relations --- the top of the homeomorphism lattice) in three languages (fr/it/en), so the
Phase-0 tests can be re-grounded on the category-theoretic form-object instead of the ad-hoc
rubric. Homosyntactic = hold the WHOLE tree identically; only the words (and the target
domain) change. Structure-only: no critical-claim pattern block, no lexical embargo, no brief
(the scenario is injected at generation time by queneau_run_homo.py).

Emits Experiments/Queneau/queneau-homo/spec/specs_{lang}_D0.json in the schema run_homeo expects:
  {lang, degree, source, n_periods, periods:[{k, words, spec}]}

One Stanza parse per language; Stanza does the sentence ("period") segmentation.
Run with the Atlas venv python (Stanza 1.12.x):
  ../Oulipo-Atlas-2-0/.venv/bin/python scripts/queneau_build_specs.py
"""
import argparse
import json
import re
import sys
from pathlib import Path

import stanza

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tree_metric as TM  # noqa: E402  (clause_bag + _depth — same defs the scorer uses)

ROOT = Path(__file__).resolve().parent.parent

# human labels for the based clause deprels (D2 inventory readout)
LABEL = {"root": "main clause", "acl": "relative/adnominal clause",
         "advcl": "adverbial clause", "ccomp": "finite complement clause",
         "xcomp": "open complement (infinitive)", "csubj": "clausal subject",
         "conj": "coordinated clause", "parataxis": "parataxis"}
SRC = {
    "fr": ROOT / "Experiments/Queneau/queneau_prototype" / "Notations_FR.md",
    "it": ROOT / "Experiments/Queneau/queneau_prototype" / "Notazioni_IT.md",
    "en": ROOT / "Experiments/Queneau/queneau_prototype" / "Notation_EN.md",
}
OUT = ROOT / "Experiments" / "Queneau" / "queneau-homo" / "spec"


def extract_source(md_path):
    """Pull the source-text block out of the prototype markdown: the lines after a
    '## Source Text' header, up to the next '---' or '##' rule. Drops the IT file's
    trailing 'English Version (for reference)' section."""
    lines = md_path.read_text(encoding="utf-8").splitlines()
    out, collecting = [], False
    for ln in lines:
        s = ln.strip()
        if s.lower().startswith("## source text"):
            collecting = True
            continue
        if collecting:
            if s == "---" or s.startswith("## "):
                break
            out.append(ln)
    text = " ".join(x.strip() for x in out if x.strip())
    return re.sub(r"\s+", " ", text).strip()


def emit_D0(words):
    """The full ordered dependency tree: at each (non-punct) position, POS and the arc
    relation->[head position]. Identical formatting to build_specs.emit_D0."""
    nonp = [w for w in words if w.upos != "PUNCT"]
    pos = {w.id: k for k, w in enumerate(nonp, 1)}
    return "; ".join(f"[{k}]{w.upos} {w.deprel}->[{pos.get(w.head, 0)}]"
                     for k, w in enumerate(nonp, 1))


def tm_toks(sent):
    """Convert a Stanza sentence to tree_metric's tok format (0-based i, based deprel,
    head -1 for root) so TM.clause_bag / TM._depth apply directly."""
    out = []
    for w in sent.words:
        h = int(w.head)
        out.append({"i": w.id - 1, "upos": w.upos, "lemma": (w.lemma or w.text).lower(),
                    "deprel": (w.deprel or "").split(":")[0].lower(),
                    "head": (-1 if h == 0 else h - 1)})
    return out


def emit_D1(sent):
    """D1 (order-free dependency structure): the multiset of (head-POS -> relation -> dep-POS)
    triples. Uses the SAME triple bag the scorer's d1_triple metric uses (TM.triple_bag)."""
    c = TM.triple_bag(tm_toks(sent))
    return "; ".join(f"{n}x {h}-{r}->{d}" for (h, r, d), n in c.most_common())


def emit_D2(sent):
    """D2 (clause architecture): the per-period clause-type inventory + max nesting depth.
    Uses the SAME clause detection the scorer's d2_clause metric uses (TM.clause_bag)."""
    toks = tm_toks(sent)
    cnt = TM.clause_bag(toks)
    inv = "; ".join(f"{n} {LABEL.get(dep, dep)}{'s' if n > 1 and not LABEL.get(dep, dep).endswith('clause') else ''}"
                    for dep, n in cnt.most_common())
    return f"{inv}; subordination nested up to depth {TM._depth(toks)}."


CLAUSE_HEAD = {"root", "csubj", "ccomp", "xcomp", "advcl", "acl", "parataxis"}


def is_clause_head(t):
    return t["deprel"] in CLAUSE_HEAD or (t["deprel"] == "conj" and t["upos"] == "VERB")


def emit_D2_schema(toks):
    """D2 grain c (clause-nesting SCHEMA): the indented clause tree --- each line a clause type,
    indentation = subordination (a clause indented under another is embedded in it). Same clause
    detection as emit_D2; this is the richer grain-c target (which clause embeds in which) vs
    grain a's counts + max depth."""
    id2 = {t["i"]: t for t in toks}
    heads = [t for t in toks if is_clause_head(t)]
    hids = {t["i"] for t in heads}
    children = {t["i"]: [] for t in heads}
    root = next((t for t in heads if t["deprel"] == "root"), None)
    for t in heads:
        if t is root:
            continue
        p = t["head"]
        while p in id2 and p not in hids:
            p = id2[p]["head"]
        (children[p] if p in children else children.get(root["i"] if root else -1, [])).append(t["i"])
    for k in children:
        children[k].sort()
    out = []

    def walk(i, d):
        out.append("  " * (d + 1) + "- " + LABEL.get(id2[i]["deprel"], id2[i]["deprel"]))
        for ch in children.get(i, []):
            walk(ch, d + 1)
    if root:
        walk(root["i"], 0)
    return "\n".join(out)


def build(lang, degree):
    source = extract_source(SRC[lang])
    stanza.download(lang, processors="tokenize,pos,lemma,depparse", verbose=False)
    nlp = stanza.Pipeline(lang, processors="tokenize,pos,lemma,depparse",
                          verbose=False, download_method=None)
    doc = nlp(source)
    periods = []
    for k, sent in enumerate(doc.sentences, 1):
        nwords = len([w for w in sent.words if w.upos != "PUNCT"])
        if degree == "D0":
            spec_str = (f"Period {k} (~{nwords} words) ordered dependency tree "
                        f"([pos] relation->[head position]): " + emit_D0(sent.words))
        elif degree == "D1":
            spec_str = (f"Period {k} (~{nwords} words) dependency triples (order-free): "
                        + emit_D1(sent))
        elif degree == "D2c":
            spec_str = (f"Period {k} (about {nwords} words) clause-nesting schema:\n"
                        + emit_D2_schema(tm_toks(sent)))
        else:  # D2 (grain a: clause inventory + max depth)
            spec_str = f"Period {k} (~{nwords} words) clause architecture: " + emit_D2(sent)
        periods.append({"k": k, "words": nwords, "spec": spec_str})
    spec = {"lang": lang, "degree": degree, "source": source,
            "n_periods": len(periods), "periods": periods}
    OUT.mkdir(parents=True, exist_ok=True)
    fn = OUT / f"specs_{lang}_{degree}.json"
    fn.write_text(json.dumps(spec, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote spec/{fn.name}: lang={lang} degree={degree}, {len(periods)} periods, "
          f"{sum(p['words'] for p in periods)} content-positions")
    return spec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--degree", default="D0", choices=["D0", "D1", "D2", "D2c"])
    ap.add_argument("--langs", nargs="*", default=["fr", "it", "en"])
    a = ap.parse_args()
    for lang in a.langs:
        spec = build(lang, a.degree)
        print(f"  period 1: {spec['periods'][0]['spec'][:150]}...")


if __name__ == "__main__":
    main()
