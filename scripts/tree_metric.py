#!/usr/bin/env python3
"""tree_metric.py — multi-degree syntactic compliance between a source and a
generated sentence, computed from spaCy parses of BOTH (same parser on both
sides, so parser bias cancels — gold parses are for corpus grading, not this).

Degrees (coarser = more abstract invariant, more freedom in the fiber):
  D0  pos_edit   : 1 - normalized POS-sequence edit distance (verbatim morphology,
                   order-SENSITIVE). The strict surface invariant.
  D1  deprel_f1  : F1 overlap of dependency-RELATION multisets (order-free, lexis-free)
  D1  triple_f1  : F1 overlap of (head_UPOS, deprel, dep_UPOS) triples — finer:
                   which category governs which via which relation
  D2  clause_f1  : F1 overlap of CLAUSE-type multiset (root/ccomp/advcl/acl/…)
  +   depth_gap  : |hypotaxis_depth(src) - depth(gen)|  (lower = closer)

Multiset F1 is order-free and uses POS+relation labels (not words), so a
reordered, relexicalised, off-topic sentence with the same syntax scores HIGH —
which is exactly the homeosyntax target the flat POS string (D0) forbids.

NB: a calque ALSO scores high here (it preserves structure). Distinguishing a
calque from genuine divergence needs the companion SEMANTIC distance (sentence
embedding, computed in the scorer) — high D1 *and* high semantic distance = good.
"""
from collections import Counter
import stanza

CLAUSE_BASE = {"root", "csubj", "ccomp", "xcomp", "advcl", "acl", "parataxis"}
_NLP = None


def base(d):
    return d.split(":", 1)[0]


def nlp(lang="it"):
    """Per-language Stanza UD pipeline, cached. English uses the same UD deprel
    set (acl/advcl/ccomp/xcomp/conj/...) as Italian, so all downstream clause-
    architecture metrics work unchanged across languages."""
    global _NLP
    if not isinstance(_NLP, dict):
        _NLP = {}
    if lang not in _NLP:
        _NLP[lang] = stanza.Pipeline(lang, processors="tokenize,pos,lemma,depparse",
                                     use_gpu=False, verbose=False)
    return _NLP[lang]


def parse(text, lang="it"):
    """Stanza UPOS ~0.98 (spaCy-sm mistagged ADJ as NOUN). Multi-sentence text is
    flattened with global indices; each sentence root -> -1."""
    toks, off = [], 0
    for s in nlp(lang)(text).sentences:
        for w in s.words:
            h = int(w.head)
            toks.append({"i": int(w.id) - 1 + off, "upos": w.upos,
                         "lemma": (w.lemma or w.text).lower(),
                         "head": (-1 if h == 0 else h - 1 + off),
                         "deprel": base((w.deprel or "").lower())})
        off += len(s.words)
    return toks


def _content(toks):
    return [t for t in toks if t["upos"] != "PUNCT"]


def _lev(a, b):
    if not a and not b:
        return 0
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def s0_pos_edit(s, g):
    a = [t["upos"] for t in _content(s)]
    b = [t["upos"] for t in _content(g)]
    if not a and not b:
        return 1.0
    return 1.0 - _lev(a, b) / max(len(a), len(b), 1)


def deprel_bag(toks):
    return Counter(t["deprel"] for t in _content(toks))


def triple_bag(toks):
    id2 = {t["i"]: t for t in toks}
    c = Counter()
    for t in _content(toks):
        h = id2[t["head"]]["upos"] if t["head"] >= 0 else "ROOT"
        c[(h, t["deprel"], t["upos"])] += 1
    return c


def clause_bag(toks):
    return Counter(t["deprel"] for t in toks
                   if t["deprel"] in CLAUSE_BASE
                   or (t["deprel"] == "conj" and t["upos"] in ("VERB", "AUX")))


def _isch(t):
    return (t["deprel"] in CLAUSE_BASE
            or (t["deprel"] == "conj" and t["upos"] in ("VERB", "AUX")))


def clause_edges(toks):
    """Multiset of typed clause-attachment edges (parent-clause-deprel -> child-clause-deprel):
    each clause head is linked to its nearest clause-head ancestor (the root clause -> ROOT).
    This is the D2 metric --- it scores the clause NESTING (which clause embeds in which),
    the structure the D3 inventory ignores."""
    id2 = {t["i"]: t for t in toks}
    hids = {t["i"] for t in toks if _isch(t)}
    c = Counter()
    for t in toks:
        if not _isch(t):
            continue
        if t["deprel"] == "root":
            c[("ROOT", "root")] += 1
            continue
        p = t["head"]
        while p >= 0 and p not in hids:
            p = id2[p]["head"]
        parent = id2[p]["deprel"] if p in hids else "ROOT"
        c[(parent, t["deprel"])] += 1
    return c


def _f1(a, b):
    overlap = sum((a & b).values())
    sa, sb = sum(a.values()), sum(b.values())
    if not sa or not sb:
        return 0.0
    p, r = overlap / sb, overlap / sa
    return 2 * p * r / (p + r) if (p + r) else 0.0


def _depth(toks):
    id2 = {t["i"]: t for t in toks}
    is_cl = {t["i"]: (t["deprel"] in CLAUSE_BASE
                      or (t["deprel"] == "conj" and t["upos"] in ("VERB", "AUX")))
             for t in toks}
    best = 0
    for t in toks:
        if not is_cl[t["i"]]:
            continue
        d, cur = 0, t
        while cur["head"] >= 0:
            cur = id2[cur["head"]]
            if is_cl.get(cur["i"]):
                d += 1
        best = max(best, d)
    return best


def compliance(src_toks, gen_toks):
    return {"s0_pos": round(s0_pos_edit(src_toks, gen_toks), 3),
            "d1_deprel": round(_f1(deprel_bag(src_toks), deprel_bag(gen_toks)), 3),
            "d1_triple": round(_f1(triple_bag(src_toks), triple_bag(gen_toks)), 3),
            "clause_inv": round(_f1(clause_bag(src_toks), clause_bag(gen_toks)), 3),
            "clause_nest": round(_f1(clause_edges(src_toks), clause_edges(gen_toks)), 3),
            "d2_clause": round(_f1(clause_bag(src_toks), clause_bag(gen_toks)), 3),  # legacy name
            "depth_gap": abs(_depth(src_toks) - _depth(gen_toks))}


def _selftest():
    SRC = "I passeggeri aspettavano il treno che era in ritardo perché nevicava."
    cases = {
        "same-architecture, new topic":
            "Gli studenti temevano l'esame che sembrava difficile perché incombeva.",
        "calque (same order, word-sub)":
            "I bambini guardavano il film che era in TV perché pioveva.",
        "reordered, same architecture":
            "Perché nevicava, il treno che era in ritardo lo aspettavano i passeggeri.",
        "simple (different structure)":
            "Marco mangia una mela rossa.",
    }
    s = parse(SRC)
    print(f"SOURCE: {SRC}\n")
    print(f"{'case':34}{'s0_pos':>7}{'d1_deprel':>10}{'d1_triple':>10}"
          f"{'d2_clause':>10}{'depth_gap':>10}")
    for label, txt in cases.items():
        c = compliance(s, parse(txt))
        print(f"{label:34}{c['s0_pos']:>7}{c['d1_deprel']:>10}{c['d1_triple']:>10}"
              f"{c['d2_clause']:>10}{c['depth_gap']:>10}")
    print("\nExpect: structural cases (same-architecture / calque / reordered) score "
          "HIGH on d1/d2; the simple sentence scores LOW. Order-free metrics "
          "(d1/d2) stay high for the reordered case while s0_pos drops.")


if __name__ == "__main__":
    _selftest()
