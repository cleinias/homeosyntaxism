#!/usr/bin/env python3
"""table1_numbers.py — the numbers §4.3 and Table 1 rest on, recomputed from the cells
`make_production_table.py --table1` selects (plus the Opus replicate and clean no-pattern
runs, for the pooled comparison that is still undecided).

    python scripts/table1_numbers.py depth    # Stanza re-parse -> results/table1_depth.json
    python scripts/table1_numbers.py report   # markdown to stdout

`depth` re-parses every text to recover the two clause depths the scorer reduced to
depth_gap, so depth-sim (min/max, as in the root tree_metric) can replace depth-gap in
print. It checks each |ds - dg| against the depth_gap already recorded and reports any
mismatch. No API calls; it writes only results/table1_depth.json, never a run or scored
file. Run both stages with the Stanza venv python (scipy for the tests).

Tests are Welch's t on text-level values (n = 10 per cell, 20 when pooled), two-sided,
with a 95% interval on the difference.
"""
import json
import pathlib
import re
import statistics as st
import subprocess
import sys

D = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(D / "scripts"))
import make_production_table as T   # noqa: E402

RESULTS = D / "results"
DEPTH = RESULTS / "table1_depth.json"
# the runs outside Table 1 that the pooled Opus comparison needs, and the July fp8
# Maverick cells the bf16 run was made against (same 2048 cap)
EXTRA = ["production_scored_D2_{l}_constr-1840-rep_claude-opus-5.json",
         "scored_unrestricted_{l}_1840_opus5_clean_n10.json"]
FP8_2048 = "production_scored_{d}_{l}_maverick-1840_RedHatAI_Llama-4-Maverick-17B-128E-Instruct-FP8.json"


def run_of(s):
    stem = s.stem.replace("production_scored_", "").replace("scored_", "")
    for p in (RESULTS / f"production_{stem}.json", RESULTS / f"{stem}.json"):
        if p.exists():
            return p
    raise SystemExit(f"table1_numbers: no run file for {s.name}")


def load(s):
    sd = json.loads(s.read_text(encoding="utf-8"))
    rp = run_of(s)
    r = json.loads(rp.read_text(encoding="utf-8"))
    stem = s.stem
    lang = r.get("lang") or ("en" if "_en_" in stem else "it")
    return {"file": s.name, "run": rp, "meta": r, "rows": sd["rows"],
            "model": T.label(sd.get("model") or r.get("model")),
            "lang": lang.upper(), "degree": r.get("degree", "?"),
            "regime": "none" if "unrestricted" in stem else "patterns"}


def table1_cells():
    paths = [p for p in T.table1(sorted(RESULTS.glob("*scored*.json")))
             if "metric_validation" not in p.name]
    return [load(p) for p in paths]


def superseded_cells():
    """The lower-cap runs Table 1 replaced (2048 originals; Qwen3 IT D3 at 4096 too)."""
    every = sorted(RESULTS.glob("*scored*.json"))
    out = {}
    for p in T.table1(every):
        if not T.MT.search(p.name):
            continue
        k = T.MT.sub("", p.name)
        for q in every:
            if q != p and T.MT.sub("", q.name) == k:
                out[q.name] = load(q)
    return out


def extra_cells():
    out = {}
    for l in ("en", "it"):
        for pat in EXTRA:
            out[pat.format(l=l)] = load(RESULTS / pat.format(l=l))
        for d in ("D2", "D3"):
            out[FP8_2048.format(d=d, l=l)] = load(RESULTS / FP8_2048.format(d=d, l=l))
    return out


# ---------------------------------------------------------------- depth stage
def bag_json(c):
    return {(k if isinstance(k, str) else ">".join(k)): v for k, v in c.items()}


def depth_stage():
    import run_production as RP
    import tree_metric as TM
    cells = (table1_cells() + list(extra_cells().values())
             + list(superseded_cells().values()))
    cache = json.loads(DEPTH.read_text(encoding="utf-8")) if DEPTH.exists() else {}
    srcs = {}
    for c in cells:
        if c["file"] in cache:
            continue
        lang = c["lang"].lower()
        src = RP.resolve_source(c["meta"])
        if (lang, src) not in srcs:
            st_ = TM.parse(src, lang=lang)
            srcs[(lang, src)] = (st_, {"ds": TM._depth(st_), "inv": bag_json(TM.clause_bag(st_)),
                                       "nest": bag_json(TM.clause_edges(st_))})
        stok, sinfo = srcs[(lang, src)]
        per = {}
        for row in c["rows"]:
            gtok = TM.parse(row["text"], lang=lang)
            dg = TM._depth(gtok)
            inv = round(TM._f1(TM.clause_bag(stok), TM.clause_bag(gtok)), 3)
            nest = round(TM._f1(TM.clause_edges(stok), TM.clause_edges(gtok)), 3)
            per[str(row["i"])] = {
                "dg": dg, "inv": bag_json(TM.clause_bag(gtok)),
                "nest": bag_json(TM.clause_edges(gtok)),
                "gap_matches": abs(sinfo["ds"] - dg) == row["depth_gap"],
                "inv_matches": abs(inv - row["clause_inv"]) < 0.0015,
                "nest_matches": abs(nest - row["clause_nest"]) < 0.0015}
        cache[c["file"]] = {"source": sinfo, "texts": per}
        bad = {k: sum(1 for v in per.values() if not v[k + "_matches"]) for k in ("gap", "inv", "nest")}
        print(f"{c['file']}: source depth {sinfo['ds']}, {len(per)} texts, mismatches {bad}",
              flush=True)
        DEPTH.write_text(json.dumps(cache, indent=1), encoding="utf-8")
    print(f"wrote {DEPTH.name} ({len(cache)} cells)")


# ---------------------------------------------------------------- report stage
def dsim(ds, dg):
    return 1.0 if max(ds, dg) == 0 else min(ds, dg) / max(ds, dg)


def overlap(a, b):
    """Shared share of two count bags taken as proportions: sum of min(p_i, q_i).
    The raw-count F1 the scorer reports caps at 2*min(na,nb)/(na+nb) on totals alone,
    so a text with exactly the source's mix but twice its clauses scores 0.67; this
    is the same comparison with both bags scaled to one, and length drops out."""
    na, nb = sum(a.values()), sum(b.values())
    if not na or not nb:
        return 0.0
    return sum(min(a[k] / na, b.get(k, 0) / nb) for k in a)


def attach_depth(cells, cache):
    for c in cells:
        ent = cache.get(c["file"])
        if ent is None:
            raise SystemExit(f"table1_numbers: {c['file']} not in {DEPTH.name}; run `depth` first")
        src = ent["source"]
        ns = sum(src["inv"].values())
        for row in c["rows"]:
            v = ent["texts"][str(row["i"])]
            ng = sum(v["inv"].values())
            row["depth_sim"] = dsim(src["ds"], v["dg"])
            row["clauses"] = ng
            row["clause_ratio"] = ng / ns
            row["f1_ceiling"] = 2 * min(ns, ng) / (ns + ng) if ns + ng else 0.0
            row["inv_prop"] = overlap(src["inv"], v["inv"])
            row["nest_prop"] = overlap(src["nest"], v["nest"])
            row["_ok"] = v["gap_matches"] and v["inv_matches"] and v["nest_matches"]


def vals(cells, k):
    if isinstance(cells, dict):
        cells = [cells]
    return [r[k] for c in cells for r in c["rows"] if r.get(k) is not None]


def m(cells, k):
    v = vals(cells, k)
    return st.mean(v) if v else float("nan")


def welch(a, b):
    """mean(a) - mean(b), two-sided Welch p, 95% CI."""
    from scipy import stats
    t = stats.ttest_ind(a, b, equal_var=False)
    va, vb = st.variance(a) / len(a), st.variance(b) / len(b)
    se = (va + vb) ** 0.5
    df = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1)) if se else 1
    h = stats.t.ppf(0.975, df) * se
    d = st.mean(a) - st.mean(b)
    return d, t.pvalue, d - h, d + h


def fmt_test(a, b):
    if st.pvariance(a) == 0 and st.pvariance(b) == 0:
        return f"{st.mean(a) - st.mean(b):+.3f} (no spread)"
    d, p, lo, hi = welch(a, b)
    return f"{d:+.3f} (p {p:.2f}; {lo:+.3f} to {hi:+.3f})"


END_OK = re.compile(r"[.!?…»\"”’)\]]$")


def cut_counts(c):
    """(texts the API reports as stopped at the cap, texts that end mid-sentence)."""
    r = c["meta"]
    n_len = None
    resp = r.get("responses")
    if isinstance(resp, list) and resp and isinstance(resp[0], dict):
        if "finish_reasons" in resp[0]:
            n_len = sum(f == "length" for x in resp for f in x["finish_reasons"])
        elif "choices" in resp[0]:
            n_len = sum(ch.get("finish_reason") == "length" for x in resp for ch in x["choices"])
        elif "stop_reason" in resp[0]:
            n_len = sum(x.get("stop_reason") == "max_tokens" for x in resp)
    mid = sum(1 for g in r.get("gens", [])
              if g.get("text") and not END_OK.search(g["text"].rstrip().rstrip("*_ ").rstrip()))
    return n_len, mid


def run_date(c):
    if c["meta"].get("started_utc"):
        return c["meta"]["started_utc"][:10]
    rel = c["run"].relative_to(D)
    out = subprocess.run(["git", "-C", str(D), "log", "--diff-filter=A", "--format=%ad",
                          "--date=short", "--", str(rel)], capture_output=True, text=True)
    lines = out.stdout.split()
    return (lines[-1] + " (commit)") if lines else "?"


def temp_sent(c):
    r = c["meta"]
    if r.get("backend") == "anthropic":
        return "not sent"
    return f"{r.get('temperature')}"


ORDER = ["Opus 5", "Qwen3-235B (ACES)", "Qwen3-235B (OpenRouter)", "Kimi-K2",
         "Llama-4-Maverick (ACES)", "Llama-4-Maverick (bf16, Delta)", "Qwen2.5-72B (ACES)"]


def key(c):
    return (c["regime"] != "patterns", ORDER.index(c["model"]) if c["model"] in ORDER else 9,
            c["lang"], c["degree"])


def find(cells, model, lang, degree=None, regime="patterns"):
    hit = [c for c in cells if c["model"] == model and c["lang"] == lang and c["regime"] == regime
           and (degree is None or c["degree"] == degree)]
    return hit[0] if hit else None


def report():
    cache = json.loads(DEPTH.read_text(encoding="utf-8"))
    cells = sorted(table1_cells(), key=key)
    extra = extra_cells()
    attach_depth(cells, cache)
    attach_depth(list(extra.values()), cache)
    sup = superseded_cells()
    attach_depth(list(sup.values()), cache)
    import run_production as RP
    src_words = {c["lang"]: len(RP.resolve_source(c["meta"]).split()) for c in cells}
    src_depth = {c["lang"]: cache[c["file"]]["source"]["ds"] for c in cells}
    src_cl = {c["lang"]: sum(cache[c["file"]]["source"]["inv"].values()) for c in cells}
    P = print

    P("# Table 1 numbers, recomputed 2026-11-20\n")
    P(f"Generated by `scripts/table1_numbers.py report` from the {len(cells)} cells of "
      "`make_production_table.py --table1`. Source: IT "
      f"{src_words['IT']} words, {src_cl['IT']} clauses, clause depth {src_depth['IT']}; EN "
      f"{src_words['EN']} words, {src_cl['EN']} clauses, depth {src_depth['EN']}. Means over each cell's texts. Tests: Welch's t on texts, "
      "two-sided, 95% interval on the difference.\n")
    mism = sum(1 for c in cells for r in c["rows"] if not r["_ok"])
    P(f"Re-parse check: {mism} of {sum(len(c['rows']) for c in cells)} texts disagree with "
      "the recorded depth_gap, inventory or nesting.\n")

    # ---- 1. the cells and their parameters
    P("## Cells and parameters\n")
    P("| model | lang | deg | arm | n | model id | via | max tok | temp | run date | at cap | mid-sentence |")
    P("|---|---|---|---|---:|---|---|---:|---|---|---:|---:|")
    for c in cells:
        r = c["meta"]
        via = r.get("backend")
        pr = r.get("provider_routing") or {}
        if pr.get("resolved_provider"):
            via += f" ({pr['resolved_provider']}, {pr.get('quant') or '?'})"
        n_len, mid = cut_counts(c)
        P(f"| {c['model']} | {c['lang']} | {c['degree']} | {c['regime']} | {len(c['rows'])} | "
          f"`{r.get('model')}` | {via} | {r.get('max_tokens')} | {temp_sent(c)} | {run_date(c)} | "
          f"{'—' if n_len is None else n_len} | {mid} |")
    P("\n*at cap*: texts the API reports as stopped by max_tokens (— where the run file kept no "
      "stop reason). *mid-sentence*: texts whose last character is not end punctuation.\n")

    # ---- 2. the full table, depth-sim in place of depth-gap
    cols = [("clause_inv", "inventory"), ("clause_nest", "nesting"), ("depth_sim", "depth-sim"),
            ("s0_pos", "surf-synt"), ("lemma_jacc", "lex-sim"), ("frame_div", "sem-diverg"),
            ("coherence", "coherence")]
    P("## Full table (appendix candidate)\n")
    P("| model | lang | deg | arm | " + " | ".join(t for _, t in cols) + " | words | embargo |")
    P("|---|---|---|---|" + "---:|" * (len(cols) + 2))
    for c in cells:
        P(f"| {c['model']} | {c['lang']} | {c['degree']} | {c['regime']} | " +
          " | ".join(f"{m(c, k):.2f}" for k, _ in cols) +
          f" | {m(c, 'words'):.0f} | {sum(len(r.get('embargo_hits') or []) for r in c['rows'])} |")
    P()

    # ---- 3. draft compact body table
    P("## Draft compact body table (D2, with and without patterns)\n")
    P("inventory / nesting / coherence\n")
    P("| model | IT, patterns | IT, none | EN, patterns | EN, none |")
    P("|---|---|---|---|---|")
    for model in ORDER:
        row = []
        for lang in ("IT", "EN"):
            for reg in ("patterns", "none"):
                c = find(cells, model, lang, "D2", reg)
                row.append(f"{m(c, 'clause_inv'):.2f} / {m(c, 'clause_nest'):.2f} / "
                           f"{m(c, 'coherence'):.2f}" if c else "—")
        if any(x != "—" for x in row):
            P(f"| {model} | " + " | ".join(row) + " |")
    P()

    length_section(P, cells, sup, src_cl)
    structure(P, cells, extra, src_words, "clause_inv", "clause_nest", "")
    # ---- 11. gaming: high fidelity, low coherence, one text at a time
    P("## Texts the judge finds incoherent (coherence below 0.5)\n")
    low = sorted(((c, r) for c in cells for r in c["rows"]
                  if r.get("coherence") is not None and r["coherence"] < 0.5),
                 key=lambda x: -x[1]["clause_inv"])
    P("| cell | text | inventory | nesting | coherence | words |")
    P("|---|---:|---:|---:|---:|---:|")
    for c, r in low:
        P(f"| {c['model']} {c['lang']} {c['degree'] if c['regime'] == 'patterns' else 'no patterns'} | "
          f"#{r['i']} | {r['clause_inv']:.3f} | {r['clause_nest']:.3f} | {r['coherence']:.2f} | "
          f"{r['words']} |")
    P(f"\n{len(low)} texts of {sum(len(c['rows']) for c in cells)}.\n")

    P("# The same numbers on proportions (length taken out)\n")
    P("Inventory and nesting recomputed as the shared share of the two clause mixes "
      "(sum of min(p, q) over clause types, or over typed attachment edges), so a text's "
      "length no longer enters. Coherence, depth-sim and words are unchanged. The prompts "
      "state counts (\"which clause types and how many of each\", or the nesting schema) and "
      "a length per period (\"about 106 words\"), so the count-based scores above measure "
      "compliance with what was asked; these measure the mix alone.\n")
    structure(P, cells, extra, src_words, "inv_prop", "nest_prop", " — proportions")


def pearson(x, y):
    mx, my = st.mean(x), st.mean(y)
    sx = sum((a - mx) ** 2 for a in x) ** 0.5
    sy = sum((b - my) ** 2 for b in y) ** 0.5
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / (sx * sy) if sx and sy else float("nan")


def length_section(P, cells, sup, src_cl):
    """How much of inventory and nesting is length. The scorer's F1 compares raw counts,
    so it cannot exceed 2*min(ns, ng)/(ns + ng) whatever the mix of clauses."""
    P("## Length and the count-based scores\n")
    P("Inventory and nesting are F1 on raw clause counts. On totals alone a text with "
      "*k* times the source's clauses cannot score above 2/(1 + *k*) (*k* ≥ 1), even with "
      "the source's exact mix. *ceiling* is that bound per text; *inv-prop* and *nest-prop* "
      "compare the mixes as proportions, which takes length out.\n")
    P("| model | lang | deg | arm | words | clauses ÷ source | ceiling | inventory | inv-prop | nesting | nest-prop |")
    P("|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for c in cells:
        P(f"| {c['model']} | {c['lang']} | {c['degree']} | {c['regime']} | {m(c, 'words'):.0f} | "
          f"{m(c, 'clause_ratio'):.2f} | {m(c, 'f1_ceiling'):.2f} | {m(c, 'clause_inv'):.2f} | "
          f"{m(c, 'inv_prop'):.2f} | {m(c, 'clause_nest'):.2f} | {m(c, 'nest_prop'):.2f} |")
    P()
    P(f"Source clauses: IT {src_cl['IT']}, EN {src_cl['EN']}.\n")
    for name, cs in (("all texts", cells),
                     ("open models, patterns", [c for c in cells if c["model"] != "Opus 5"
                                                and c["regime"] == "patterns"]),
                     ("Opus 5", [c for c in cells if c["model"] == "Opus 5"])):
        w, r = vals(cs, "clause_ratio"), vals(cs, "f1_ceiling")
        P(f"- **{name}** ({len(w)} texts): clause ratio against inventory r = "
          f"{pearson(w, vals(cs, 'clause_inv')):+.2f}, against inv-prop r = "
          f"{pearson(w, vals(cs, 'inv_prop')):+.2f}; against nesting r = "
          f"{pearson(w, vals(cs, 'clause_nest')):+.2f}, against nest-prop r = "
          f"{pearson(w, vals(cs, 'nest_prop')):+.2f}. Ceiling against inventory r = "
          f"{pearson(r, vals(cs, 'clause_inv')):+.2f}.")
    P()
    P("### The reruns: lower cap against higher cap, same cell\n")
    P("| cell | same texts | words | inventory | inv-prop | nesting | nest-prop | coherence |")
    P("|---|---:|---|---|---|---|---|---|")
    for c in cells:
        olds = [o for o in sup.values() if T.MT.sub("", o["file"]) == T.MT.sub("", c["file"])]
        for o in sorted(olds, key=lambda o: o["meta"].get("max_tokens") or 0):
            cap = f"{o['meta'].get('max_tokens')}→{c['meta'].get('max_tokens')}"
            same = sum(1 for x, y in zip(o["meta"]["gens"], c["meta"]["gens"])
                       if x.get("text") == y.get("text"))
            P(f"| {c['model']} {c['lang']} {c['degree']} ({cap}) | {same} | {m(o, 'words'):.0f} → "
              f"{m(c, 'words'):.0f} | " +
              " | ".join(f"{m(o, k):.3f} → {m(c, k):.3f}"
                         for k in ("clause_inv", "inv_prop", "clause_nest", "nest_prop",
                                   "coherence")) + " |")
    P("\n*same texts*: generations byte-identical to the lower-cap run. Where all ten are, "
      "the rerun reproduced the original byte for byte (cause not established; vLLM's "
      "sampling seed is the likely one) and is not a replicate, but its rescoring is a "
      "test-retest of the judge.\n")


def rng(cs, k):
    v = [m(c, k) for c in cs]
    return min(v), max(v)


def structure(P, cells, extra, src_words, ik, nk, sfx):
    """Sections 4-7 and 10, on the inventory and nesting measures named."""
    # ---- 4. the wall
    P(f"## The wall: inventory above nesting{sfx}\n")
    for label, sub in (("all cells", cells),
                       ("patterns only", [c for c in cells if c["regime"] == "patterns"])):
        gaps = [(r[ik] - r[nk], c, r) for c in sub for r in c["rows"]]
        cg = [(m(c, ik) - m(c, nk), c) for c in sub]
        lo = min(gaps, key=lambda g: g[0])
        clo, chi = min(cg, key=lambda g: g[0]), max(cg, key=lambda g: g[0])
        P(f"- **{label}:** inventory above nesting in {sum(1 for g, _ in cg if g > 0)} of {len(sub)} "
          f"cells and {sum(1 for g, _, _ in gaps if g > 0)} of {len(gaps)} texts. Mean gap "
          f"{st.mean(g for g, _, _ in gaps):.3f}; cell range {clo[0]:.3f} ({clo[1]['model']} "
          f"{clo[1]['lang']} {clo[1]['degree']}) to {chi[0]:.3f} ({chi[1]['model']} "
          f"{chi[1]['lang']} {chi[1]['degree']}); smallest single gap {lo[0]:+.3f} "
          f"({lo[1]['model']} {lo[1]['lang']} {lo[1]['degree']} #{lo[2]['i']}).")
    P()

    # ---- 5. the two bands
    P(f"## Two bands (patterns arm){sfx}\n")
    pat = [c for c in cells if c["regime"] == "patterns"]
    opus = [c for c in pat if c["model"] == "Opus 5"]
    open_ = [c for c in pat if c["model"] != "Opus 5"]

    for name, cs in (("Opus 5", opus), ("open models", open_)):
        P(f"- **{name}** ({len(cs)} cells): inventory {rng(cs, ik)[0]:.2f}–"
          f"{rng(cs, ik)[1]:.2f}, nesting {rng(cs, nk)[0]:.2f}–"
          f"{rng(cs, nk)[1]:.2f}, depth-sim {rng(cs, 'depth_sim')[0]:.2f}–"
          f"{rng(cs, 'depth_sim')[1]:.2f}, coherence {rng(cs, 'coherence')[0]:.2f}–"
          f"{rng(cs, 'coherence')[1]:.2f}, words {rng(cs, 'words')[0]:.0f}–{rng(cs, 'words')[1]:.0f}.")
    for k, t in ((ik, "inventory"), (nk, "nesting")):
        gap = rng(opus, k)[0] - rng(open_, k)[1]
        P(f"- Space between the bands on {t}: {gap:+.2f} (lowest Opus cell minus highest open cell).")
    for k, t in ((ik, "inventory"), (nk, "nesting")):
        d = st.mean(m(c, k) for c in opus) - st.mean(m(c, k) for c in open_)
        P(f"- Mean of the Opus cells minus mean of the open cells, {t}: {d:+.3f}.")
    top = max(open_, key=lambda c: m(c, ik))
    P(f"- Highest open cell: {top['model']} {top['lang']} {top['degree']} "
      f"({m(top, ik):.3f} / {m(top, nk):.3f}).")
    tx_open = vals(open_, ik)
    tx_opus = vals(opus, ik)
    P(f"- Text level: open-model texts reach {max(tx_open):.3f} inventory; Opus texts go down to "
      f"{min(tx_opus):.3f}; {sum(1 for x in tx_open if x >= min(tx_opus))} of {len(tx_open)} open "
      "texts sit at or above the lowest Opus text.")
    P()

    # ---- 6. the cost of the patterns
    P(f"## The cost of the patterns (D2 with patterns minus no patterns){sfx}\n")
    P("| model | lang | n | inventory | nesting | depth-sim | coherence | words |")
    P("|---|---|---|---|---|---|---|---|")
    for model in ("Opus 5", "Qwen3-235B (ACES)", "Qwen3-235B (OpenRouter)"):
        for lang in ("IT", "EN"):
            a, b = find(cells, model, lang, "D2"), find(cells, model, lang, None, "none")
            if not (a and b):
                continue
            # only the Opus cells are both July runs; the Qwen3 pairs set a November
            # rerun against a July (ACES) or a clean-prompt (OpenRouter) no-pattern run
            arms = [("July" if model == "Opus 5" else "", [a], [b])]
            if model == "Opus 5":
                l = lang.lower()
                arms.append(("pooled", [a, extra[EXTRA[0].format(l=l)]],
                             [b, extra[EXTRA[1].format(l=l)]]))
            for tag, A, B in arms:
                n = f"{len(vals(A, ik))} v {len(vals(B, ik))}"
                P(f"| {model}{f' ({tag})' if tag else ''} | {lang} | {n} | " +
                  " | ".join(fmt_test(vals(A, k), vals(B, k))
                             for k in (ik, nk, "depth_sim", "coherence")) +
                  f" | {m(A, 'words'):.0f} v {m(B, 'words'):.0f} |")
    P("\nWith D3 as well (means; no patterns → D2 → D3):\n")
    for model in ("Opus 5", "Qwen3-235B (ACES)", "Qwen3-235B (OpenRouter)"):
        for lang in ("IT", "EN"):
            b = find(cells, model, lang, None, "none")
            a2, a3 = find(cells, model, lang, "D2"), find(cells, model, lang, "D3")
            if not (b and a2 and a3):
                continue
            P(f"- {model} {lang}: inventory {m(b, ik):.3f} → {m(a2, ik):.3f} → "
              f"{m(a3, ik):.3f}; nesting {m(b, nk):.3f} → "
              f"{m(a2, nk):.3f} → {m(a3, nk):.3f}; depth-sim "
              f"{m(b, 'depth_sim'):.2f} → {m(a2, 'depth_sim'):.2f} → {m(a3, 'depth_sim'):.2f}; "
              f"words {m(b, 'words'):.0f} → {m(a2, 'words'):.0f} → {m(a3, 'words'):.0f} "
              f"(source {src_words[lang]}).")
    P("\nThe Qwen3 ACES no-pattern cells ran at 2048 tokens with the July prompt; the OpenRouter "
      "ones at 4096 with the clean prompt.\n")

    # ---- 7. D2 against D3
    P(f"## D2 against D3, model by model (patterns arm; D2 minus D3){sfx}\n")
    P("| model | lang | inventory | nesting | D2 ahead on both? |")
    P("|---|---|---|---|---|")
    for model in ORDER:
        for lang in ("IT", "EN"):
            a, b = find(cells, model, lang, "D2"), find(cells, model, lang, "D3")
            if not (a and b):
                continue
            di = m(a, ik) - m(b, ik)
            dn = m(a, nk) - m(b, nk)
            verdict = "yes" if di > 0 and dn > 0 else "no: " + ", ".join(
                t for t, d in (("inventory", di), ("nesting", dn)) if d <= 0)
            P(f"| {model} | {lang} | {fmt_test(vals(a, ik), vals(b, ik))} | "
              f"{fmt_test(vals(a, nk), vals(b, nk))} | {verdict} |")
    P()

    # ---- 10. ownership ceiling
    P(f"## The ownership ceiling: Qwen2.5-72B{sfx}\n")
    q72 = [c for c in pat if c["model"] == "Qwen2.5-72B (ACES)"]
    rest = [c for c in open_ if c["model"] != "Qwen2.5-72B (ACES)"]
    for name, cs in (("Qwen2.5-72B", q72), ("other open models", rest)):
        P(f"- {name}: inventory {rng(cs, ik)[0]:.2f}–{rng(cs, ik)[1]:.2f}, "
          f"nesting {rng(cs, nk)[0]:.2f}–{rng(cs, nk)[1]:.2f}, coherence "
          f"{rng(cs, 'coherence')[0]:.2f}–{rng(cs, 'coherence')[1]:.2f}.")
    P()


    # ---- 8. deployment
    P(f"## Deployment: Qwen3-235B on ACES against OpenRouter (ACES minus OpenRouter){sfx}\n")
    P("| cell | inventory | nesting | coherence | words |")
    P("|---|---|---|---|---|")
    for lang in ("IT", "EN"):
        for deg, reg in (("D2", "patterns"), ("D3", "patterns"), (None, "none")):
            a = find(cells, "Qwen3-235B (ACES)", lang, deg, reg)
            b = find(cells, "Qwen3-235B (OpenRouter)", lang, deg, reg)
            if not (a and b):
                continue
            P(f"| {lang} {deg or 'no patterns'} | " +
              " | ".join(fmt_test(vals(a, k), vals(b, k))
                         for k in (ik, nk, "coherence")) +
              f" | {m(a, 'words'):.0f} v {m(b, 'words'):.0f} |")
    P("\nThe no-pattern pair also differs in prompt (July against clean) and cap (2048 against 4096).\n")

    # ---- 9. precision
    P(f"## Precision: Maverick fp8 (ACES) against bf16 (Delta){sfx}\n")
    P("| cell | fp8 4096 v bf16 2048: inventory | nesting | coherence | fp8 2048 v bf16 2048: inventory | nesting | coherence |")
    P("|---|---|---|---|---|---|---|")
    for lang in ("IT", "EN"):
        for deg in ("D2", "D3"):
            a = find(cells, "Llama-4-Maverick (ACES)", lang, deg)
            b = find(cells, "Llama-4-Maverick (bf16, Delta)", lang, deg)
            o = extra[FP8_2048.format(d=deg, l=lang.lower())]
            P(f"| {lang} {deg} | " +
              " | ".join(fmt_test(vals(x, k), vals(b, k)) for x in (a, o)
                         for k in (ik, nk, "coherence")) + " |")
    P()


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else ""
    if stage == "depth":
        depth_stage()
    elif stage == "report":
        report()
    else:
        print(__doc__)
        sys.exit(1)
