#!/usr/bin/env python3
"""test1_signs_without_hypotheses.py — loop-A test 1: find the signs without knowing what to look for.

Spec: PoeticsToday/S3-4_S4-3_context-leakage_draft_2026-11-26.md, note 15, test 1.

For every content lemma (NOUN, PROPN, VERB, ADJ, ADV; stanza 1.12.2, CPU) count the texts of each
of the six Opus-5 D2 arms that contain it (document frequency, 0-10). Then compare groups of arms:

  first    = july + replicate          (20 texts, first formalization)
  revised  = revised + stranger        (20 texts, revised formalization)
  nopat    = nopat + nopat-noemb       (20 texts, no patterns)

The null is july against the replicate: the same prompt sent twice, so whatever a lemma does between
them is what ten texts do by chance. The criterion is per arm, so that it is on the null's own scale
and no lemma passes on the strength of one arm of a pooled group (the two arms of a group differ in
the embargo, which adds noise of its own; the within-pair moves are reported for all three pairs).
For a contrast "hi over lo" the score is

  every-arm difference = (min over the arms of hi - max over the arms of lo) / 10

i.e. the smallest of the four 10-vs-10 differences. Tiers are fixed in advance on the raw null
(the share of lemmas whose july/replicate difference reaches t):
  firm       >= 0.7   (the null reaches 0.7 for 2 of ~2,650 lemmas, for one pair, not four)
  candidate  >= 0.6   (3)
  below      >= 0.5   (13: listed for completeness, not findings)
(The first version of this script pooled each group and scaled the null by sqrt(2); that threshold
was chosen after a strict first run and is withdrawn. Journal, 2026-11-26.)

Each lemma that moves is sorted by its profile across the three groups (both contrasts at the tier):
  first-wording    first high, revised and nopat low   (the July wording carried it)
  revised-wording  revised high, first and nopat low   (second turn: left in by the revision)
  pattern          first and revised high, nopat low   (travels with the pattern, in both wordings)
  anti-pattern     nopat high, first and revised low   (the patterns push it out)
  partial          one contrast only (named in the table)
and flagged when it occurs in the 1840 incipit (provenance candidate), in the July prompt, in the
revised prompt, or in the no-pattern prompt (wording candidates). Provenance itself stays a reader's
judgment: the flags only shorten the list.

Read-only on the run files. Writes results/test1_signs_<tag>.json and .md, plus a lemma cache
(results/test1_lemmas_<tag>.json) so the ranking can be redone without re-parsing.

  ~/.local/share/manzoni-stanza-venv/bin/python scripts/test1_signs_without_hypotheses.py
"""
import argparse, hashlib, json, sys
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

    def every(v, hi, lo):
        return (min(v[x] for x in GROUPS[hi]) - max(v[x] for x in GROUPS[lo])) / 10

    TIERS = [("firm", 0.7), ("candidate", 0.6), ("below", 0.5)]
    PROFILES = [("first-wording", ("first", "revised"), ("first", "nopat")),
                ("revised-wording", ("revised", "first"), ("revised", "nopat")),
                ("pattern", ("first", "nopat"), ("revised", "nopat")),
                ("anti-pattern", ("nopat", "first"), ("nopat", "revised"))]
    CONTRASTS = [(h, l) for h in GROUPS for l in GROUPS if h != l]
    eps = 1e-9

    rows = []
    for lem, v in df.items():
        e = {f"{h}>{l}": round(every(v, h, l), 1) for h, l in CONTRASTS}
        # profile: the one whose weaker contrast is strongest; else the single strongest contrast
        prof, score, contrast = max(((name, min(e[f"{a[0]}>{a[1]}"], e[f"{b[0]}>{b[1]}"]), None)
                                     for name, a, b in PROFILES), key=lambda x: x[1])
        prof2, score2 = prof, score   # the best two-contrast profile, kept for partial rows
        best = max(e, key=e.get)
        if e[best] > score + eps:
            # a single contrast clears a higher tier than any two-contrast profile
            tier_of = lambda x: next((n for n, t in TIERS if x >= t - eps), None)
            if tier_of(score) is None or tier_of(e[best]) != tier_of(score):
                prof, score, contrast = "partial", e[best], best
        tier = next((n for n, t in TIERS if score >= t - eps), None)
        rows.append({"lemma": lem, "df": v, **{f"p_{k}": p(lem, k) for k in GROUPS},
                     "every_arm": e, "profile": prof if tier else None, "contrast": contrast,
                     "score": score, "tier": tier, "profile2": prof2, "score2": score2,
                     "null": abs(v["july"] - v["replicate"]) / 10,
                     "in_incipit": lem in inc, "in_prompt_july": lem in pr["july"],
                     "in_prompt_revised": lem in pr["revised"], "in_prompt_nopat": lem in pr["nopat"]})

    pairs = {"july/replicate": ("july", "replicate"), "revised/stranger": ("revised", "stranger"),
             "nopat/nopat-noemb": ("nopat", "nopat-noemb")}
    ts = [0.5, 0.6, 0.7, 0.8]
    within = {t: {k: sum(abs(r["df"][a] - r["df"][b]) / 10 >= t - eps for r in rows)
                  for k, (a, b) in pairs.items()} for t in ts}
    passing = {t: sum(r["score"] >= t - eps for r in rows) for t in ts}
    null_max = max(r["null"] for r in rows)
    order = [n for n, _ in TIERS]
    porder = [n for n, _, _ in PROFILES] + ["partial"]
    short = sorted((r for r in rows if r["tier"]),
                   key=lambda r: (order.index(r["tier"]), porder.index(r["profile"]), -r["score"], r["lemma"]))

    out = {"spec": "PT context-leakage draft 2026-11-26, note 15, test 1", "tag": a.tag,
           "stanza": L["stanza"], "upos": L["upos"],
           "arms": {k: {"file": f"results/{fn}", "sha1": sha} for k, (fn, sha) in ARMS.items()},
           "groups": GROUPS, "criterion": "every-arm difference; tiers fixed on the raw null",
           "tiers": dict(TIERS), "n_lemmas": len(rows), "null_max": null_max,
           "within_pair_moves": within, "lemmas_passing": passing, "short_list": short}
    (RES / f"test1_signs_{a.tag}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))

    arms = list(ARMS)
    md = [f"# Test 1: signs without hypotheses ({a.tag})", "",
          f"Content lemmas: {len(rows)}. Criterion: the every-arm difference (smallest of the four "
          f"10-vs-10 differences between the arms of two groups), on the raw null's scale. Tiers fixed "
          f"on the raw null: firm >= 0.7, candidate >= 0.6, below >= 0.5 (not findings).", "",
          "Lemmas moving by t or more between the two arms of a pair (july/replicate is the null), "
          "and lemmas whose every-arm score reaches t:", "",
          "| t | july/replicate | revised/stranger | nopat/nopat-noemb | every-arm score |",
          "|---|---|---|---|---|"]
    md += [f"| {t} | " + " | ".join(str(within[t][k]) for k in pairs) + f" | {passing[t]} |" for t in ts]
    md += ["", "Short list (texts of ten per arm; I = incipit; prompts: J july, R revised, N no-pattern):", "",
           "| tier | profile | lemma | score | " + " | ".join(arms) + " | I | J | R | N |",
           "|---|---|---|---|" + "---|" * (len(arms) + 4)]
    for r in short:
        fl = ["I" if r["in_incipit"] else "", "J" if r["in_prompt_july"] else "",
              "R" if r["in_prompt_revised"] else "", "N" if r["in_prompt_nopat"] else ""]
        pf = r["profile"] + (f" ({r['contrast']})" if r["contrast"] else "")
        if r["contrast"] and r["score2"] >= TIERS[-1][1] - eps:
            pf += f"; {r['profile2']} at {r['score2']:.1f}"
        md.append(f"| {r['tier']} | {pf} | {r['lemma']} | {r['score']:.1f} | "
                  + " | ".join(str(r['df'][x]) for x in arms) + " | " + " | ".join(fl) + " |")
    (RES / f"test1_signs_{a.tag}.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
