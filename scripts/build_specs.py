#!/usr/bin/env python3
"""build_specs.py — generalized D_n spec extractor (supersedes build_d2_specs.py).

Emits a source-blind generation spec at any degree of the homeomorphism lattice. The degree
structure is a PARTIAL ORDER, not a chain: the full tree D0 is the top, and S and D1 are two
INDEPENDENT forgetful projections of it (S drops the arcs, D1 drops the order), so S and D1 are
incomparable — that independence keeps the degree axis from collapsing into a single "amount of
form" scale (and keeps it orthogonal to the constraint-load axis).
  D0  the full ordered dependency tree : POS + linear order + relations (the top).
  S   surface : the ordered POS sequence (forget the arcs — the tree's leaves; the calque end).
  D1            : the order-free dependency structure (forget the order; head-POS —rel→ dep-POS).
  D2            : clause NESTING — which clause embeds in which (the operative rung).
  D3            : clause INVENTORY — which clause types, how many (forget the nesting).
  D4            : the hypotaxis profile — clause count + max depth only (the loosest rung).
Plus the rubric (constraints), the functional brief, and the source-word embargo (all
degree-independent). One parse; every rung is a projection of it (tree_metric metrics
mirror these: s0_pos↔S, d1_triple↔D1, clause_nest↔D2, clause_inv↔D3, depth↔D4).

NOTE ON THE RELABELLING (2026-07-30). This ladder used to run S/D0/D1/D2/D3 with D2
carrying two "grains", a = inventory and c = nesting schema, and D3 = the depth
profile. The CHR 2027 paper separated the two invariants D2 conflated, so nesting
and inventory are now degrees in their own right and everything past the old D2
shifts up by one. The translation for pre-existing runs and result files:
  grain c  ->  D2 (nesting)     grain a  ->  D3 (inventory)     old D3  ->  D4
Result files written before this date keep their `d2a`/`d2c` names and their
`grain` field; readers map them with the table above.

  python scripts/build_specs.py --lang it --degree D2   # → spec/specs_it_D2.json (clause nesting)
  python scripts/build_specs.py --lang en --degree S    # → spec/specs_en_S.json
Run with the Atlas venv python (Stanza).
"""
import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import tree_metric as TM           # noqa: E402
import run_production as RP         # noqa: E402 (Italian INCIPIT, periods)
from overlap_metrics import tokens, content_words  # noqa: E402

SPEC = ROOT / "spec"
EN_SOURCE = ROOT / "texts" / "incipit_en.txt"  # bring your own (the Colquhoun source is in copyright)
WHITELIST = {"it": {"mare", "acqua", "acque"}, "en": {"sea", "water", "waters"}}
REDACTIONS = [
    (" (novenario preferred, mirroring 'Quel ramo del lago di Como')",
     " (novenario, settenario, or endecasillabo)"),
    (", on the model of 'il lago cessa, e l'Adda rincomincia'", ""),
]
CONTENT = {"NOUN", "VERB", "ADJ", "ADV", "PROPN"}
CLAUSE_DEPREL = {"root", "csubj", "ccomp", "xcomp", "advcl", "acl", "acl:relcl", "parataxis"}
LABEL = {"root": "main clause", "acl": "relative/adnominal clause", "acl:relcl": "relative clause",
         "advcl": "adverbial clause", "ccomp": "finite complement clause",
         "xcomp": "open complement (infinitive)", "csubj": "clausal subject",
         "conj": "coordinated clause", "parataxis": "parataxis"}


def is_clause_head(t):
    return t["deprel"] == "root" or t["deprel"] in CLAUSE_DEPREL or (t["deprel"] == "conj" and t["upos"] == "VERB")


# ---------- per-degree spec emitters (source-blind) ----------
def emit_S(tok):
    """S (surface): the ordered part-of-speech sequence — forget the arcs (the tree's leaves)."""
    return " ".join(t["upos"] for t in tok if t["upos"] != "PUNCT")


def emit_D0(tok):
    """D0 (the top): the full ordered dependency tree — at each position, the POS and the arc
    (relation -> head position). Holds POS AND order AND relations; S forgets the arcs and D1
    forgets the order, so each is an independent projection of this."""
    nonp = [t for t in tok if t["upos"] != "PUNCT"]
    pos = {t["i"]: k for k, t in enumerate(nonp, 1)}
    return "; ".join(f"[{k}]{t['upos']} {t['deprel']}->[{pos.get(t['head'], 0)}]"
                     for k, t in enumerate(nonp, 1))


def emit_D1(tok):
    """Order-free dependency structure: the multiset of (head-POS —relation→ dep-POS) triples."""
    id2 = {t["i"]: t for t in tok}
    c = Counter()
    for t in tok:
        if t["upos"] == "PUNCT":
            continue
        h = id2[t["head"]]["upos"] if (t["head"] >= 0 and t["head"] in id2) else "ROOT"
        c[(h, t["deprel"], t["upos"])] += 1
    return "; ".join(f"{n}x {h}-{r}->{d}" for (h, r, d), n in c.most_common())


def emit_D3_inventory(tok):
    heads = [t for t in tok if is_clause_head(t)]
    cnt = Counter(LABEL.get("acl:relcl" if t["deprel"] == "acl:relcl" else t["deprel"], t["deprel"]) for t in heads)
    inv = "; ".join(f"{n} {lab}{'s' if n > 1 and not lab.endswith('clause') else ''}" for lab, n in cnt.most_common())
    return f"{inv}; subordination nested up to depth {TM._depth(tok)}."


def emit_D2_nesting(tok):
    id2 = {t["i"]: t for t in tok}
    heads = [t for t in tok if is_clause_head(t)]
    hids = {t["i"] for t in heads}
    children = {t["i"]: [] for t in heads}
    root = None
    for t in heads:
        if t["deprel"] == "root":
            root = t; continue
        p = t["head"]
        while p in id2 and p not in hids:
            p = id2[p]["head"]
        (children[p] if p in children else children.get(root["i"], [])).append(t["i"])
    for k in children:
        children[k].sort()
    lines = []

    def lab(t):
        return LABEL.get("acl:relcl" if t["deprel"] == "acl:relcl" else t["deprel"], t["deprel"])

    def walk(i, d):
        lines.append("  " * (d + 1) + "- " + lab(id2[i]))
        for ch in children.get(i, []):
            walk(ch, d + 1)
    if root:
        walk(root["i"], 0)
    return "\n".join(lines)


def emit_D4_profile(tok):
    return f"{len([t for t in tok if is_clause_head(t)])} clauses total; max subordination depth {TM._depth(tok)}."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", default="it", choices=["it", "en"])
    ap.add_argument("--embargo", default="auto", choices=["auto", "filtered", "full"],
                    help="auto (default) uses spec/embargo_<lang>_filtered.json when it "
                         "exists — the source's names and markers — and the unfiltered "
                         "content-word list otherwise; full forces the old behaviour")
    ap.add_argument("--degree", default="D2", choices=["S", "D0", "D1", "D2", "D3", "D4"],
                    help="D2 = clause nesting, D3 = clause inventory, D4 = depth profile "
                         "(the old D2-grain-c / D2-grain-a / D3 respectively)")
    a = ap.parse_args()
    lang, deg = a.lang, a.degree

    if lang == "it":
        source = RP.INCIPIT.strip(); cfile, redact = "constraints.json", True
    else:
        if not EN_SOURCE.exists():
            sys.exit(f"error: English source {EN_SOURCE.relative_to(ROOT)} not found --- the revised "
                     f"Colquhoun translation is in copyright and is not shipped; place your own copy "
                     f"there to run the English track (see the README).")
        source = EN_SOURCE.read_text(encoding="utf-8").strip()
        cfile, redact = "constraints_en.json", False

    # The transposition domain --- the source-blind functional brief --- is authored in
    # spec/brief_<lang>.json (a "domain" line + one "element" per scene-world item). It is
    # language-specific, so a mismatch is a hard error (checked up front, before the parse;
    # see docs/Target_Domain_Brief.md).
    bf = SPEC / f"brief_{lang}.json"
    if not bf.exists():
        sys.exit(f"error: brief file {bf.name} not found --- author a {lang}-language target-domain "
                 f"brief (see docs/Target_Domain_Brief.md).")
    bdata = json.loads(bf.read_text(encoding="utf-8"))
    if bdata.get("lang") != lang:
        sys.exit(f"error: language mismatch --- {bf.name} declares lang={bdata.get('lang')!r} but "
                 f"--lang {lang} was requested.")
    domain = bdata["domain"]
    brief = "\n".join("  - " + e for e in bdata["elements"])

    periods = re.split(r"(?<=[.])\s+(?=[A-Z])", source)
    out_periods, content_lemmas = [], set()
    for k, p in enumerate(periods, 1):
        tok = TM.parse(p, lang=lang)
        content_lemmas |= {t["lemma"] for t in tok if t["upos"] in CONTENT}
        block = {"k": k, "words": len(p.split())}
        if deg == "S":
            block["spec"] = f"Period {k} (~{len(p.split())} words) POS sequence: " + emit_S(tok)
        elif deg == "D0":
            block["spec"] = (f"Period {k} (~{len(p.split())} words) ordered dependency tree "
                             f"([pos] relation->[head position]): " + emit_D0(tok))
        elif deg == "D1":
            block["spec"] = f"Period {k} (~{len(p.split())} words) dependency triples (any order): " + emit_D1(tok)
        elif deg == "D2":
            block["spec"] = f"Period {k} (about {len(p.split())} words):\n" + emit_D2_nesting(tok)
        elif deg == "D3":
            block["spec"] = f"Period {k} (about {len(p.split())} words): " + emit_D3_inventory(tok)
        else:  # D4
            block["spec"] = f"Period {k} (~{len(p.split())} words): " + emit_D4_profile(tok)
        out_periods.append(block)

    constraints = json.loads((SPEC / cfile).read_text(encoding="utf-8"))["constraints"]
    if redact:
        for c in constraints:
            for old, new in REDACTIONS:
                c["specification"] = c["specification"].replace(old, new)
    constraint_block = "\n".join(
        f"  [{c['id']}] ({c['priority']}) from {c['pattern_id']}: {c['property']}\n    Specification: {c['specification']}"
        for c in constraints)

    # The embargo: by default the filtered list — the source's names and markers —
    # rather than every content word it contains.  build_embargo.py explains why the
    # unfiltered version forbade ordinary Italian and could not be obeyed; --embargo
    # full restores it for comparison.
    full = (sorted(w for w in content_words(tokens(source)) if w not in WHITELIST["it"]) if lang == "it"
            else sorted(w for w in content_lemmas if w not in WHITELIST["en"]))
    filtered_path = SPEC / f"embargo_{lang}_filtered.json"
    if a.embargo == "full" or (a.embargo == "auto" and not filtered_path.exists()):
        embargo, emb_src = full, f"full ({len(full)} content words)"
    else:
        f = json.loads(filtered_path.read_text(encoding="utf-8"))
        embargo = sorted(f["embargo"])
        missing = [w for w in embargo if w not in full]
        if missing:
            sys.exit(f"build_specs: {filtered_path.name} lists words absent from the "
                     f"source's content words ({missing[:5]}) — rebuild it")
        if f.get("decide"):
            print(f"  note: {len(f['decide'])} words still unruled in "
                  f"{filtered_path.name}; they are NOT embargoed")
        emb_src = (f"filtered ({len(embargo)} of {len(full)}; thresholds "
                   f"{f['thresholds']['zipf_free']}/{f['thresholds']['zipf_mark']})")

    spec = {"lang": lang, "degree": deg,
            "source_ref": RP.SOURCE_REF[lang], "source_sha1": RP.source_sha1(source),
            "n_periods": len(periods),
            "periods": out_periods, "embargo": embargo, "embargo_source": emb_src,
            "constraint_block": constraint_block, "brief": brief, "domain": domain}
    out = SPEC / f"specs_{lang}_{deg}.json"
    out.write_text(json.dumps(spec, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote spec/{out.name}: lang={lang} degree={deg}, {len(periods)} periods, "
          f"embargo {emb_src}, {len(constraints)} constraints")
    print("\n=== degree spec, period 1 ===")
    print(out_periods[0]["spec"])


if __name__ == "__main__":
    main()
