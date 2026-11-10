#!/usr/bin/env python3
"""End-to-end verification of the CHR2027 paper: every table cell, every prose figure,
every census count, recomputed from the run records.

WHY THIS EXISTS

Three sweeps of the paper were done by hand, and each found things the previous one had
not, because each asked a slightly different question. The failures were never arithmetic:

  - a figure was correct for the run it came from, but the sentence cited another table
    (0.98 vs 0.94, both real measurements of the same configuration in different runs)
  - a census was computed over a corpus definition that a table move had invalidated,
    with nothing failing (534 cells / 5,289 generations for a corpus of 423 / 4,208)
  - a claim was falsified by data ARRIVING, with no printed number changing at all
    (see scripts/queneau_claims_check.py, which covers that class)

So the checks here are deliberately not "does 0.34 equal 0.34". Each one names the run
files a printed figure must come from, recomputes it, and fails if the paper disagrees --
which means a table whose provenance moves breaks this script loudly instead of quietly.

    python scripts/chr_verify.py            # summary
    python scripts/chr_verify.py -v         # every check, passing ones included

NOTE ON DEGREE NAMES. Files named D2 hold the paper's D3 (clause inventory); the paper's
D2 (clause nesting) is in files named D2c. This is load-bearing and easy to get backwards.
"""
import argparse
import collections
import importlib.util
import json
import re
import statistics as st
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOC = ROOT / "Docs and Research notes/CHR2027 paper/Holding-the-Shape-CHR2027-TRIMMED.lyx"
S = ROOT / "Experiments/Queneau/queneau-homo/outputs/scored"
O = ROOT / "Experiments/Queneau/queneau-homo/outputs"
MZ = ROOT / "Experiments/Manzoni/homeosyntaxism/results"
SCEN = ("grocery-checkout", "security-checkpoint", "hospital-emergency-room")
L3 = ("en", "fr", "it")

# degree -> the fidelity metric that degree is held at
HELD = {"S0": "s0_pos", "D0": "s0_pos", "D1": "d1_triple",
        "D2c": "clause_nest", "D2": "d2_clause"}

POS = re.compile(r"(?<![\w'])(NOUN|VERB|ADJ|ADV|DET|ADP|PRON|PROPN|AUX|CCONJ|SCONJ|"
                 r"PART|NUM|INTJ|PUNCT|SYM)(?![\w'])")
BRACK = re.compile(r"\[\d+\]")

RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok), detail))
    return ok


def eq(name, printed, computed, fmt="{:.2f}"):
    c = fmt.format(computed)
    return check(name, printed == c, f"paper {printed!r}  data {c!r}")


# ---------------------------------------------------------------- corpus helpers
def doubling(t):
    w = [x.lower().strip(".,;:!?«»\"'()—–-") for x in t.split()]
    return sum(1 for a, b in zip(w, w[1:]) if a == b and len(a) > 2)


def contaminated(t):
    return len(POS.findall(t)) >= 3 or len(BRACK.findall(t)) >= 3 or doubling(t) >= 10


_cache = {}


def cell_rows(deg, lang, scen, tag, model):
    """Uncontaminated scored rows of one cell, resolving the ladder's inconsistent
    regime segment (English D1 is `_ladderD1_`, French and Italian `_blind_ladderD1_`)."""
    key = (deg, lang, scen, tag, model)
    if key in _cache:
        return _cache[key]
    rows = []
    for v in (tag, "blind_" + tag, tag.replace("blind_", "")):
        f = S / f"scored_homo_{deg}_{lang}_{scen}_{v}_{model}.json"
        if f.exists():
            d = json.loads(f.read_text(encoding="utf-8"))
            raw = json.loads((O / f.name[len("scored_"):]).read_text(encoding="utf-8"))
            txt = {g["i"]: " ".join(g["text"].split()) for g in raw["gens"]}
            rows = [r for r in d["rows"] if not contaminated(txt.get(r["i"], ""))]
            break
    _cache[key] = rows
    return rows


def pool(deg, tag, model, langs=L3):
    rows = []
    for l in langs:
        for s in SCEN:
            rows += cell_rows(deg, l, s, tag, model)
    return rows


def agg(rows, deg):
    m = lambda k: sum(r[k] for r in rows) / len(rows)
    return {"surf": m("s0_pos"), "inv": m("d2_clause"), "coh": m("coherence"),
            "held": m("held"), "fid": m(HELD[deg]), "n": len(rows)}


# ---------------------------------------------------------------- LyX reading
L = DOC.read_text(encoding="utf-8").split("\n")


def in_note():
    out = [False] * len(L)
    depth, nd = 0, None
    for i, l in enumerate(L):
        if l.startswith("\\begin_inset"):
            depth += 1
            if l.startswith("\\begin_inset Note") and nd is None:
                nd = depth
        elif l == "\\end_inset":
            if nd is not None and depth == nd:
                nd = None
            depth -= 1
        out[i] = nd is not None
    return out


NOTE = in_note()


def flat(a, b, skip_notes=True):
    o = []
    for i in range(a, min(b, len(L))):
        if skip_notes and NOTE[i]:
            continue
        s = L[i]
        if s.startswith("\\begin_inset Formula"):
            o.append(re.sub(r"^\\begin_inset Formula\s*", "", s).strip("$ ")
                     .replace("_{", "").replace("}", ""))
        elif s.startswith(("\\", "<")) or not s.strip():
            continue
        elif s.startswith(("name ", "LatexCommand", "reference ", "plural ", "caps ",
                           "noprefix ", "nolink ", "tuple ", "status ", "key ", "after ",
                           "literal ", "target ", "filename ", "lyxscale ", "width ")):
            continue
        else:
            o.append(s.strip())
    return re.sub(r"\s+", " ", " ".join(o))


BODY = flat(0, len(L))


def table_rows(label, marker):
    i = next(k for k, l in enumerate(L) if l.startswith(f'name "{label}"'))
    s = max(k for k in range(i) if L[k].startswith("\\begin_inset Float table"))
    tb = next(k for k in range(s, i) if L[k].startswith("<lyxtabular"))
    te = max(k for k in range(s, i) if L[k].startswith("</lyxtabular>"))
    out = []
    for a, b in zip([k for k in range(tb, te) if L[k].startswith("<row")],
                    [k for k in range(tb, te) if L[k].startswith("</row>")]):
        v = []
        for k in range(a, b + 1):
            if L[k].startswith("<cell "):
                j = next((m for m in range(k, b + 1) if L[m] == marker), None)
                v.append(None if j is None or L[j + 1].startswith("\\begin_inset")
                         else L[j + 1])
        lvl = ("S0" if any(L[k].endswith("Formula $S_{0}$") for k in range(a, b))
               else "D0" if any(L[k].endswith("Formula $D_{0}$") for k in range(a, b))
               else "")
        out.append((lvl, v))
    return out


QW = {s: f"Qwen_Qwen2.5-{s}-Instruct" for s in
      ("0.5B", "1.5B", "3B", "7B", "14B", "32B", "72B")}


# ================================================================ A. TABLE 1
def table1():
    rows = table_rows("tab:blind-vs-copy-data", "\\lang english")
    for lvl, v in rows[1:]:
        model, regime = v[0], v[2]
        tag_reg = "blind" if regime == "blind" else "vis"
        if model in ("Qwen2.5-7B", "Qwen2.5-72B"):
            tag, m = f"{tag_reg}_ladderT1", QW[model.split("-")[1]]
        elif model == "Opus 4.8":
            tag, m = f"{tag_reg}_b1", "claude-opus-4-8"
        else:
            check(f"T1 {lvl} {model} {regime}: unknown model", False, "no rule")
            continue
        a = agg(pool(lvl, tag, m), lvl)
        base = f"T1 {lvl} {model} {regime}"
        eq(f"{base} surf-synt", v[3], a["surf"])
        eq(f"{base} clause-inv", v[4], a["inv"])
        eq(f"{base} coher", v[5], a["coh"])
        eq(f"{base} held", v[6], a["held"])
        eq(f"{base} n", v[7], a["n"], "{:d}")


# ================================================================ B. TABLE 3
def table3():
    rows = table_rows("tab:gradient", "\\size small")
    cols = [("D0", "blind_ladderD0soft"), ("D1", "ladderD1"),
            ("D2c", "blind_ladderD2c"), ("D2", "ladderD2")]
    for _, v in rows[1:]:
        size = v[0].split("-")[1]
        if size not in QW:
            continue
        cohs = [x.strip() for x in (v[5] or "").split("/")]
        for idx, (deg, tag) in enumerate(cols):
            a = agg(pool(deg, tag, QW[size]), deg)
            eq(f"T3 {v[0]} {deg} fidelity", v[1 + idx], a["fid"])
            if len(cohs) == 4:
                eq(f"T3 {v[0]} {deg} coherence", cohs[idx], a["coh"])


# ================================================================ C. TABLE 4
def table4():
    MAC = {"7B q4": "mac-7b-q4_qwen2.5_7b-instruct-q4_K_M",
           "7B q8": "mac-7b-q8_qwen2.5_7b-instruct-q8_0",
           "14B q4": "mac-14b-q4_qwen2.5_14b-instruct-q4_K_M"}
    CLUSTER = {"7B BF16": ("7B", {"S0": "blind_ladderS0"}),
               "14B BF16": ("14B", {"S0": "blind_ladderS0"}),
               "72B fp8": ("72B", {"S0": "blind_ladderS0"})}
    LADDER = {"D0": "blind_ladderD0soft", "D1": "ladderD1",
              "D2c": "blind_ladderD2c", "D2": "ladderD2"}
    printed = {}
    i = next(k for k, l in enumerate(L) if l.startswith('name "tab:mincompute"'))
    s = max(k for k in range(i) if L[k].startswith("\\begin_inset Float table"))
    tb = next(k for k in range(s, i) if L[k].startswith("<lyxtabular"))
    te = max(k for k in range(s, i) if L[k].startswith("</lyxtabular>"))
    grid = []
    for a, b in zip([k for k in range(tb, te) if L[k].startswith("<row")],
                    [k for k in range(tb, te) if L[k].startswith("</row>")]):
        cells = []
        for k in range(a, b + 1):
            if L[k].startswith("<cell "):
                end = next(m for m in range(k, b + 1) if L[m].startswith("</cell>"))
                txt = [L[m].strip() for m in range(k, end)
                       if not L[m].startswith(("\\", "<")) and L[m].strip()]
                math = [re.sub(r"^\\begin_inset Formula\s*", "", L[m]).strip("$ ")
                        .replace("_{", "").replace("}", "")
                        for m in range(k, end) if L[m].startswith("\\begin_inset Formula")]
                cells.append(" ".join(math + txt))
        grid.append(cells)
    header = grid[0]
    for row in grid[1:]:
        deg_lbl = row[0].strip()
        deg = {"S0": "S0", "D0": "D0", "D1": "D1", "D2": "D2c", "D3": "D2"}.get(deg_lbl)
        if deg is None:
            continue
        for col, val in zip(header[1:], row[1:]):
            col = col.strip()
            key = col.split(" (")[0]
            if key in MAC:
                tag, model = MAC[key].split("_", 1)[0], MAC[key].split("_", 1)[1]
            elif key in CLUSTER:
                size, s0tag = CLUSTER[key]
                tag = s0tag.get(deg, LADDER.get(deg))
                model = QW[size]
            else:
                continue
            rows = pool(deg, tag, model, langs=("en",))
            if not rows:
                check(f"T4 {deg_lbl} {key}", False, "no run files resolved")
                continue
            a = agg(rows, deg)
            eq(f"T4 {deg_lbl} {key}", val.strip(),
               0, "{:s}".format(f"{a['fid']:.2f}/{a['coh']:.2f}"))


# ================================================================ D. TABLE 5
def table5():
    rows = table_rows("tab:homosyntaxism", "\\size small")
    for _, v in rows[1:]:
        lbl = v[0]
        if lbl.startswith("Qwen2.5-"):
            tag, m = "blind_ladderD0soft", QW[lbl.split("-")[1]]
        elif lbl.startswith("Llama-4-Maverick"):
            tag, m = "blind_b1", "meta-llama_llama-4-maverick"
        elif lbl.startswith("Opus"):
            tag, m = "blind_b1", "claude-opus-4-8"
        else:
            continue                                   # Fable 5: "refused", no data
        a = agg(pool("D0", tag, m), "D0")
        eq(f"T5 {lbl} surf-synt", v[1], a["surf"])
        eq(f"T5 {lbl} clause-inv", v[2], a["inv"])
        eq(f"T5 {lbl} held-rate", v[3], a["held"])
        eq(f"T5 {lbl} coher", v[4], a["coh"])


# ================================================================ E. CROSS-TABLE
def cross_table():
    """The same configuration must not print two values in two tables."""
    a = agg(pool("D0", "blind_b1", "claude-opus-4-8"), "D0")
    t1 = {(lvl, v[0], v[2]): v for lvl, v in table_rows("tab:blind-vs-copy-data",
                                                        "\\lang english")[1:]}
    t5 = {v[0]: v for _, v in table_rows("tab:homosyntaxism", "\\size small")[1:]}
    r1, r5 = t1[("D0", "Opus 4.8", "blind")], t5["Opus 4.8"]
    check("Opus D0 blind agrees between Table 1 and Table 5",
          (r1[3], r1[4], r1[5], r1[6]) == (r5[1], r5[2], r5[4], r5[3]),
          f"T1 {r1[3]}/{r1[4]}/{r1[5]}/{r1[6]}  T5 {r5[1]}/{r5[2]}/{r5[4]}/{r5[3]}")
    # the scale-ladder sentence must quote Table 5, not Table 1's second run
    m = re.search(r"Qwen-72B,? coherence ([\d.]+)", BODY)
    check("scale-ladder coherence quotes Table 5's 72B value",
          bool(m) and m.group(1) == t5["Qwen2.5-72B"][4],
          f"prose {m.group(1) if m else '?'}  Table 5 {t5['Qwen2.5-72B'][4]}")


# ================================================================ F. PROSE
def prose():
    t1 = {(lvl, v[0], v[2]): v for lvl, v in table_rows("tab:blind-vs-copy-data",
                                                        "\\lang english")[1:]}
    # "none of N surface generations ... two of M one degree down" over Table 1's population
    m = re.search(r"none of (\d+) surface generations clears both bars;"
                  r" one degree down,? (two) of (\d+)", BODY)
    if not check("body: the both-bars denominators are present", bool(m)):
        return
    s0 = sum(agg(pool("S0", tag, mod), "S0")["n"]
             for tag, mod in (("blind_ladderT1", QW["7B"]), ("blind_ladderT1", QW["72B"]),
                              ("blind_b1", "claude-opus-4-8")))
    d0 = sum(agg(pool("D0", tag, mod), "D0")["n"]
             for tag, mod in (("blind_ladderT1", QW["7B"]), ("blind_ladderT1", QW["72B"]),
                              ("blind_b1", "claude-opus-4-8")))
    eq("body: S0 blind population", m.group(1), s0, "{:d}")
    eq("body: D0 blind population", m.group(3), d0, "{:d}")
    # how many actually clear both bars
    def both(deg):
        n = 0
        for tag, mod in (("blind_ladderT1", QW["7B"]), ("blind_ladderT1", QW["72B"]),
                         ("blind_b1", "claude-opus-4-8")):
            for r in pool(deg, tag, mod):
                if r["held"] and r["coherence"] >= 0.5:
                    n += 1
        return n
    check("body: 0 clear both bars at S0", both("S0") == 0, f"{both('S0')} do")
    check("body: 2 clear both bars at D0", both("D0") == 2, f"{both('D0')} do")
    # Opus refusals in the source-visible D0 cells
    # reworded 2026-11-10: "refused 59 of its 140 attempts ... 81 of 90 in all"
    m = re.search(r"refused (\d+) of its (\d+) attempts.{0,200}?(\d+) of (\d+) in all", BODY)
    if check("body: Opus refusal sentence present", bool(m)):
        def nref(j):
            # generation refusals live under `refused` here (an int); the judge-refusal
            # files use `n_refused` or a list of indices. Accept all three shapes.
            v = j.get("n_refused")
            if v is None:
                v = j.get("refused")
            return len(v) if isinstance(v, list) else (v or 0)
        ref = sum(nref(json.loads((O / f"homo_D0_{l}_{s}_vis_b1_claude-opus-4-8.json")
                                  .read_text(encoding="utf-8")))
                  for l in L3 for s in SCEN)
        att = sum(json.loads((O / f"homo_D0_{l}_{s}_vis_b1_claude-opus-4-8.json")
                             .read_text(encoding="utf-8")).get("attempts", 10)
                  for l in L3 for s in SCEN)
        got = len(pool("D0", "vis_b1", "claude-opus-4-8"))
        eq("body: Opus generation refusals", m.group(1), ref, "{:d}")
        eq("body: Opus generation attempts", m.group(2), att, "{:d}")
        eq("body: Opus surviving generations", m.group(3), got, "{:d}")
    # "144-fold"
    check("body: 144-fold range", "144-fold" in BODY, "72 / 0.5 = 144")


# ================================================================ G. MANZONI
def manzoni():
    R = {(w, l): MZ / f"scored_unrestricted_{l}_1840_{n}_n10.json"
         for w, n in (("opus5", "opus5"),
                      ("qwen", "Qwen3-235B-A22B-Instruct-2507-FP8"))
         for l in ("it", "en")}
    d = {}
    for k, p in R.items():
        if not p.exists():
            check(f"Manzoni {k} record", False, f"missing {p.name}")
            return
        d[k] = json.loads(p.read_text(encoding="utf-8"))["rows"]
    col = lambda rows, k: [r[k] for r in rows]
    for lang, im, ib, nm, nb in (("it", "0.90", "0.94", "0.75", "0.82"),
                                 ("en", "0.86", "0.90", "0.71", "0.76")):
        r = d[("opus5", lang)]
        eq(f"Manzoni Opus5 {lang} inv mean", im, st.mean(col(r, "clause_inv")))
        eq(f"Manzoni Opus5 {lang} inv best", ib, max(col(r, "clause_inv")))
        eq(f"Manzoni Opus5 {lang} nest mean", nm, st.mean(col(r, "clause_nest")))
        eq(f"Manzoni Opus5 {lang} nest best", nb, max(col(r, "clause_nest")))
    for lang, inv, nest in (("it", "0.73", "0.58"), ("en", "0.70", "0.54")):
        r = d[("qwen", lang)]
        eq(f"Manzoni Qwen3 {lang} inv mean", inv, st.mean(col(r, "clause_inv")))
        eq(f"Manzoni Qwen3 {lang} nest mean", nest, st.mean(col(r, "clause_nest")))
    allc = [c for l in ("it", "en") for c in col(d[("opus5", l)], "coherence")]
    check("Manzoni: coherence 1.00 in all twenty", all(c == 1.0 for c in allc),
          f"{len(allc)} generations, min {min(allc)}")
    pairs = [(r["clause_inv"], r["clause_nest"]) for l in ("it", "en")
             for r in d[("opus5", l)]]
    check("Manzoni: inventory exceeds nesting in all twenty",
          all(a > b for a, b in pairs), f"{sum(1 for a,b in pairs if a>b)}/{len(pairs)}")
    for who, lo, hi in (("opus5", 0.02, 0.04), ("qwen", 0.05, 0.07)):
        sds = [st.stdev(col(d[(who, l)], k)) for l in ("it", "en")
               for k in ("clause_inv", "clause_nest")]
        check(f"Manzoni {who}: sd within {lo}-{hi}",
              round(min(sds), 2) >= lo - 0.005 and round(max(sds), 2) <= hi + 0.005,
              f"{min(sds):.3f}-{max(sds):.3f}")
    # the two printed specimens, matched by their full opening
    def norm(t):
        t = unicodedata.normalize("NFKD", t).replace("’", "'")
        return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", " ", t.lower())).strip()
    for lang, opening, want in (
            ("it", "Il canale che separa i due continenti si stringe qui in un braccio",
             ("0.93", "0.82", "0.41", "0.10", "0.90", "1.00")),
            ("en", "The channel that divides Africa from Europe narrows here into a shallow "
             "trough of water, and where the seabed lifts toward the last shelf",
             ("0.90", "0.76", "0.38", "0.11", "1.00", "1.00"))):
        raw = json.loads((MZ / f"unrestricted_{lang}_1840_opus5_n10.json")
                         .read_text(encoding="utf-8"))
        key = norm(opening)
        hit = [g["i"] if "i" in g else i
               for i, g in enumerate(raw["gens"]) if key in norm(g["text"])]
        if not check(f"Manzoni {lang} specimen found", len(hit) == 1,
                     f"{len(hit)} matches"):
            continue
        r = next(x for x in d[("opus5", lang)] if x["i"] == hit[0])
        got = (f"{r['clause_inv']:.2f}", f"{r['clause_nest']:.2f}", f"{r['s0_pos']:.2f}",
               f"{r['lemma_jacc']:.2f}", f"{r['frame_div']:.2f}", f"{r['coherence']:.2f}")
        check(f"Manzoni {lang} specimen header", got == want, f"paper {want} data {got}")
    # source texts still hash to what the runs declared (sha1 over the STRIPPED text)
    import hashlib
    for f in sorted(MZ.glob("unrestricted_*_1840_*.json")):
        j = json.loads(f.read_text(encoding="utf-8"))
        ref, sha = j.get("source_ref"), j.get("source_sha1")
        if not ref:
            continue
        p = ROOT / "Experiments" / "Manzoni" / "homeosyntaxism" / ref
        got = hashlib.sha1(p.read_text(encoding="utf-8").strip().encode()).hexdigest()
        check(f"Manzoni source {ref} hash", got == sha, f"{got[:12]} vs {sha[:12]}")
        break_after = True


# ================================================================ H. CENSUS
def census():
    spec = importlib.util.spec_from_file_location(
        "cc", ROOT / "scripts/queneau_contamination_check.py")
    cc = importlib.util.module_from_spec(spec)
    import io, contextlib
    with contextlib.redirect_stdout(io.StringIO()):
        spec.loader.exec_module(cc)
    files = [f for f in sorted(S.glob("scored_homo_*.json"))
             if cc.reported_by(f.name) and (O / f.name[len("scored_"):]).exists()]
    tot = flag = coh = att = refg = refc = 0
    by_deg = collections.Counter(); by_deg_t = collections.Counter()
    by_cond = collections.Counter(); by_cond_t = collections.Counter()
    for f in files:
        d = json.loads(f.read_text(encoding="utf-8"))
        raw = json.loads((O / f.name[len("scored_"):]).read_text(encoding="utf-8"))
        txt = {g["i"]: " ".join(g["text"].split()) for g in raw["gens"]}
        deg = f.name.split("_")[2]
        n = d.get("n_refused") or len(d.get("refused", []) or [])
        att += len(d["rows"]) + n
        if n:
            refc += 1; refg += n
        cond = "7B" if re.search(r"(Qwen2\.5-7B-|7b-instruct)", f.name) else "other"
        for r in d["rows"]:
            tot += 1; by_deg_t[deg] += 1; by_cond_t[cond] += 1
            if contaminated(txt.get(r["i"], "")):
                flag += 1; by_deg[deg] += 1; by_cond[cond] += 1
                if r["coherence"] >= 0.5:
                    coh += 1
    eq("census: reported cells", re.search(r"touching \d+ of (\d+) cells", BODY).group(1),
       len(files), "{:d}")
    m = re.search(r"(\d+) of ([\d,]+) generations \(([\d.]+)%\) carry leaked", BODY)
    eq("census: flagged", m.group(1), flag, "{:d}")
    eq("census: reported generations", m.group(2), tot, "{:,d}")
    eq("census: flagged share", m.group(3), 100 * flag / tot, "{:.1f}")
    eq("census: rated coherent",
       re.search(r"judge rated (\d+) of them coherent", BODY).group(1), coh, "{:d}")
    r = re.search(r"It is ([\d.]+)% of surface-level generations,? ([\d.]+)% at the "
                  r"ordered-tree degree,? ([\d.]+)% one degree below", BODY)
    eq("census: S0 rate", r.group(1), 100 * by_deg["S0"] / by_deg_t["S0"], "{:.1f}")
    eq("census: D0 rate", r.group(2), 100 * by_deg["D0"] / by_deg_t["D0"], "{:.1f}")
    eq("census: D1 rate", r.group(3), 100 * by_deg["D1"] / by_deg_t["D1"], "{:.1f}")
    check("census: D2/D3 absent", by_deg["D2c"] == 0 and by_deg["D2"] == 0,
          f"D2c {by_deg['D2c']} D2 {by_deg['D2']}")
    check("census: 'almost entirely' the 7B",
          by_cond["7B"] / flag >= 0.85, f"{by_cond['7B']}/{flag}")
    rm = re.search(r"rate is (\d+) generations in ([\d,]+) attempts,? ([\d.]+)%,? "
                   r"touching (\d+) of", BODY)
    eq("census: refusals", rm.group(1), refg, "{:d}")
    eq("census: attempts", rm.group(2), att, "{:,d}")
    eq("census: refusal rate", rm.group(3), 100 * refg / att, "{:.2f}")
    eq("census: cells with a refusal", rm.group(4), refc, "{:d}")
    # the showcase passage must exist in the reported corpus
    ms = re.search(r"One passage carrying (\d+) tags and (\d+) markers", BODY)
    want = (int(ms.group(1)), int(ms.group(2)))
    found = False
    for f in files:
        raw = json.loads((O / f.name[len("scored_"):]).read_text(encoding="utf-8"))
        d = {r["i"]: r for r in json.loads(f.read_text(encoding="utf-8"))["rows"]}
        for g in raw["gens"]:
            t = " ".join(g["text"].split())
            if (len(POS.findall(t)), len(BRACK.findall(t))) == want:
                if g["i"] in d and d[g["i"]]["coherence"] == 1.0:
                    found = True
    check("census: showcase passage is in the reported corpus and scored 1.00", found,
          f"{want[0]} tags / {want[1]} markers")


# ================================================================ I. DOCUMENT
def document():
    bi = sum(1 for l in L if l.startswith("\\begin_inset"))
    ei = sum(1 for l in L if l == "\\end_inset")
    bl = sum(1 for l in L if l.startswith("\\begin_layout"))
    el = sum(1 for l in L if l == "\\end_layout")
    check("document: insets balance", bi == ei, f"{bi}/{ei}")
    check("document: layouts balance", bl == el, f"{bl}/{el}")
    labels = {l[6:-1] for l in L if l.startswith('name "') and l[6:-1].startswith(
        ("sec:", "subsec:", "tab:", "fig:", "app:"))}
    refs = {L[i + 1][11:-1] for i, l in enumerate(L)
            if l.startswith("LatexCommand ref") and L[i + 1].startswith("reference ")}
    check("document: no dangling cross-references", refs <= labels, f"{refs - labels}")
    spec = importlib.util.spec_from_file_location(
        "mk", DOC.parent / "make-anon.py")
    mk = importlib.util.module_from_spec(spec)
    import io, contextlib
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            spec.loader.exec_module(mk)
    except SystemExit:
        pass
    w = mk.words(DOC)
    check("document: body within 6000 words", w <= 6000, f"{w}/6000")
    t = DOC.read_text(encoding="utf-8")
    check("document: exactly two repository links",
          t.count(mk.REPO_HREF) == mk.REPO_LINKS,
          f"{t.count(mk.REPO_HREF)} vs {mk.REPO_LINKS}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()
    for name, fn in (("Table 1", table1), ("Table 3", table3), ("Table 4", table4),
                     ("Table 5", table5), ("cross-table", cross_table),
                     ("body prose", prose), ("Manzoni", manzoni),
                     ("census", census), ("document", document)):
        before = len(RESULTS)
        try:
            fn()
        except Exception as e:
            check(f"{name}: CHECK CRASHED", False, f"{type(e).__name__}: {e}")
        bad = [r for r in RESULTS[before:] if not r[1]]
        print(f"  {name:12} {len(RESULTS)-before:4} checks, "
              f"{len(bad)} failed" + ("" if not bad else "   <-- "))
        for n, _, d in bad:
            print(f"      FAIL {n}: {d}")
        if a.verbose:
            for n, ok, d in RESULTS[before:]:
                if ok:
                    print(f"      ok   {n}: {d}")
    nbad = sum(1 for _, ok, _ in RESULTS if not ok)
    print(f"\n  ==> {len(RESULTS)} checks, {nbad} failed")
    return 1 if nbad else 0


if __name__ == "__main__":
    raise SystemExit(main())
