#!/usr/bin/env python3
"""test1_signs_without_hypotheses.py — loop-A test 1: find the signs without knowing what to look for.

Spec: PoeticsToday/S3-4_S4-3_context-leakage_draft_2026-11-26.md, note 15, test 1.

For every content lemma (NOUN, PROPN, VERB, ADJ, ADV; stanza 1.12.2, CPU) count the texts of each
of the six Opus-5 D2 arms that contain it (document frequency, 0-10). Then compare groups of arms:

  first    = july + replicate          (20 texts, first formalization)
  revised  = revised + stranger        (20 texts, revised formalization)
  nopat    = nopat + nopat-noemb       (20 texts, no patterns)

The null is july against the replicate: the same prompt sent twice, so whatever a lemma does between
them is what ten texts do by chance. Differences are in proportions of texts. A 10-vs-10 difference
is noisier than a 20-vs-20 one by a factor sqrt(2) (binomial), so the null for the group contrasts is
the july/replicate curve read at t*sqrt(2). The threshold is the smallest t at which that scaled null
admits no lemma; a "watch" tier one step below (0.1 less) is listed too, with its own false-positive
count. A lemma must move by the threshold in a group contrast to be listed.

Each lemma that moves is sorted by its profile across the three groups:
  first-wording    first high, revised and nopat low   (the July wording carried it)
  revised-wording  revised high, first and nopat low   (second turn: left in by the revision)
  pattern          first and revised high, nopat low   (travels with the pattern, in both wordings)
  anti-pattern     nopat high, first and revised low   (the patterns push it out)
and flagged when it occurs in the 1840 incipit (provenance candidate), in the July prompt, in the
revised prompt, or in the no-pattern prompt (wording candidates). Provenance itself stays a reader's
judgment: the flags only shorten the list.

Read-only on the run files. Writes results/test1_signs_<tag>.json and .md, plus a lemma cache
(results/test1_lemmas_<tag>.json) so the ranking can be redone without re-parsing.

  ~/.local/share/manzoni-stanza-venv/bin/python scripts/test1_signs_without_hypotheses.py
"""
import argparse, hashlib, json, math, sys
from pathlib import Path

HS = Path(__file__).resolve().parent.parent
RES = HS / "results"
ARMS = {  # same files and sha1s as the loop-A output probe (specs/probe/loopA_outputs_it_D2.json)
    "july":        ("production_D2_it_constr-1840_claude-opus-5.json",          "c579ae31f3cae48be6ffad23768a7b5b70b10b9d"),
    "replicate":   ("production_D2_it_constr-1840-rep_claude-opus-5.json",      "8c54e1f1ff4405101c795f187f6344395dd08a39"),
    "revised":     ("production_D2_it_effectfree-1840_claude-opus-5.json",      "4a6611451c5dc01dc6b8829dabdf0765cef5a8ce"),
    "stranger":    ("production_D2_it_effectfree-noemb-1840_claude-opus-5.json", "b9bff3c29974716e26596753311da9cf5ec67aac"),
    "nopat":       ("unrestricted_it_1840_opus5_clean_n10.json",                "2c22e198d0d3b353b670400462fe0927bbcf8f6a"),
    "nopat-noemb": ("unrestricted_it_1840_opus5_noemb_n10.json",                "8dc2e3155a9cbe76fdee4ae95e8d74aa373c5ab0"),
}
GROUPS = {"first": ("july", "replicate"), "revised": ("revised", "stranger"),
          "nopat": ("nopat", "nopat-noemb")}
CONTENT = {"NOUN", "PROPN", "VERB", "ADJ", "ADV"}
INCIPIT = HS / "texts" / "incipit_it.txt"


def lemmatize(nlp, text):
    out = set()
    for s in nlp(text).sentences:
        for w in s.words:
            if w.upos in CONTENT and w.lemma and any(c.isalpha() for c in w.lemma):
                out.add(w.lemma.lower())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="1126")
    ap.add_argument("--reparse", action="store_true", help="ignore the lemma cache")
    a = ap.parse_args()
    cache = RES / f"test1_lemmas_{a.tag}.json"

    runs, prompts = {}, {}
    for arm, (fn, sha) in ARMS.items():
        raw = (RES / fn).read_bytes()
        got = hashlib.sha1(raw).hexdigest()
        if got != sha:
            sys.exit(f"{fn}: sha1 {got} != {sha}")
        d = json.loads(raw)
        runs[arm] = [g["text"] for g in sorted(d["gens"], key=lambda g: g["i"])]
        prompts[arm] = d["prompt"]
        assert len(runs[arm]) == 10, (arm, len(runs[arm]))

    if cache.exists() and not a.reparse:
        L = json.loads(cache.read_text())
    else:
        import stanza
        nlp = stanza.Pipeline("it", processors="tokenize,mwt,pos,lemma", use_gpu=False,
                              download_method=None, verbose=False)
        L = {"stanza": stanza.__version__, "upos": sorted(CONTENT),
             "texts": {arm: [sorted(lemmatize(nlp, t)) for t in ts] for arm, ts in runs.items()},
             "incipit": sorted(lemmatize(nlp, INCIPIT.read_text())),
             "prompts": {k: sorted(lemmatize(nlp, prompts[k])) for k in ("july", "revised", "nopat")}}
        cache.write_text(json.dumps(L, ensure_ascii=False, indent=0))

    df = {}  # lemma -> arm -> texts containing it
    for arm, docs in L["texts"].items():
        for doc in docs:
            for lem in doc:
                df.setdefault(lem, dict.fromkeys(ARMS, 0))[arm] += 1
    inc = set(L["incipit"])
    pr = {k: set(v) for k, v in L["prompts"].items()}

    def p(lem, grp):
        return sum(df[lem][x] for x in GROUPS[grp]) / (10 * len(GROUPS[grp]))

    rows = []
    for lem in df:
        g = {k: p(lem, k) for k in GROUPS}
        rows.append({"lemma": lem, "df": df[lem], **{f"p_{k}": v for k, v in g.items()},
                     "null": abs(df[lem]["july"] - df[lem]["replicate"]) / 10,
                     "first_vs_nopat": g["first"] - g["nopat"],
                     "first_vs_revised": g["first"] - g["revised"],
                     "revised_vs_nopat": g["revised"] - g["nopat"],
                     "in_incipit": lem in inc, "in_prompt_july": lem in pr["july"],
                     "in_prompt_revised": lem in pr["revised"], "in_prompt_nopat": lem in pr["nopat"]})

    # empirical false-positive curve: lemmas over threshold t under the null vs in each contrast
    ts = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    cmp_keys = ["first_vs_nopat", "first_vs_revised", "revised_vs_nopat"]
    eps = 1e-9
    curve = {t: {"null": sum(r["null"] >= t - eps for r in rows),
                 "null_scaled": sum(r["null"] >= t * math.sqrt(2) - eps for r in rows),
                 **{c: sum(abs(r[c]) >= t - eps for r in rows) for c in cmp_keys}} for t in ts}
    null_max = max(r["null"] for r in rows)
    thr = next(t for t in ts if curve[t]["null_scaled"] == 0)
    watch = round(thr - 0.1, 1)

    def profile(r, thr):
        hi = {k: r[f"p_{k}"] for k in GROUPS}
        f, v, n = hi["first"], hi["revised"], hi["nopat"]
        if f - v >= thr and f - n >= thr: return "first-wording"
        if v - f >= thr and v - n >= thr: return "revised-wording"
        if f - n >= thr and v - n >= thr: return "pattern"
        if n - f >= thr and n - v >= thr: return "anti-pattern"
        if max(abs(r[c]) for c in cmp_keys) >= thr: return "partial"
        return None

    for r in rows:
        f, v, n = (r[f"p_{k}"] for k in GROUPS)
        r["first_vs_nopat"], r["first_vs_revised"], r["revised_vs_nopat"] = (
            round(f - n, 3), round(f - v, 3), round(v - n, 3))
        r["profile"] = profile(r, thr - eps)
        r["tier"] = "main" if r["profile"] else None
        if not r["profile"]:
            r["profile"] = profile(r, watch - eps)
            r["tier"] = "watch" if r["profile"] else None
    order = ["first-wording", "pattern", "revised-wording", "anti-pattern", "partial"]
    short = sorted((r for r in rows if r["profile"]),
                   key=lambda r: (r["tier"] != "main", order.index(r["profile"]),
                                  -max(abs(r[c]) for c in cmp_keys), r["lemma"]))

    out = {"spec": "PT context-leakage draft 2026-11-26, note 15, test 1", "tag": a.tag,
           "stanza": L["stanza"], "upos": L["upos"],
           "arms": {k: {"file": f"results/{fn}", "sha1": sha} for k, (fn, sha) in ARMS.items()},
           "groups": GROUPS, "n_lemmas": len(rows), "null_max": null_max, "threshold": thr,
           "watch": watch,
           "fp_curve": curve, "short_list": short,
           "null_movers": sorted((r for r in rows if r["null"] >= 0.5), key=lambda r: -r["null"])}
    (RES / f"test1_signs_{a.tag}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))

    arms = list(ARMS)
    md = [f"# Test 1: signs without hypotheses ({a.tag})", "",
          f"Content lemmas: {len(rows)}. Null (july vs replicate, 10 vs 10): largest move {null_max:.1f}. "
          f"Group contrasts are 20 vs 20; their null is the july/replicate curve read at t*sqrt(2). "
          f"Threshold {thr:.1f} (scaled null admits no lemma); watch tier {watch:.1f}.", "",
          "Lemmas over threshold t (null = raw 10 vs 10; scaled = expected false positives at 20 vs 20):", "",
          "| t | null | scaled | first vs nopat | first vs revised | revised vs nopat |",
          "|---|---|---|---|---|---|"]
    md += [f"| {t} | {c['null']} | {c['null_scaled']} | {c['first_vs_nopat']} | {c['first_vs_revised']} | {c['revised_vs_nopat']} |"
           for t, c in curve.items()]
    md += ["", "Short list (texts of ten per arm; I = incipit; prompts: J july, R revised, N no-pattern):", "",
           "| tier | profile | lemma | " + " | ".join(arms) + " | I | J | R | N |",
           "|---|---|---|" + "---|" * (len(arms) + 4)]
    for r in short:
        fl = ["I" if r["in_incipit"] else "", "J" if r["in_prompt_july"] else "",
              "R" if r["in_prompt_revised"] else "", "N" if r["in_prompt_nopat"] else ""]
        md.append(f"| {r['tier']} | {r['profile']} | {r['lemma']} | " + " | ".join(str(r['df'][x]) for x in arms)
                  + " | " + " | ".join(fl) + " |")
    (RES / f"test1_signs_{a.tag}.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
