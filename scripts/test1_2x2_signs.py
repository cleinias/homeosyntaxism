#!/usr/bin/env python3
"""test1_2x2_signs.py — loop-A test 1 on the 2x2 dataset only: patterns x embargo, Opus 5, IT, D2.

SF, 2026-11-27: rerun test 1 without the July and replicate arms, on the four runs of the test-3
table (PT context-leakage draft, "What the tests changed", item 1):

  pat+emb  revised patterns, embargo      production_D2_it_effectfree-1840        ("revised")
  pat      revised patterns, no embargo   production_D2_it_effectfree-noemb-1840  ("constrained, no embargo")
  emb      no patterns, embargo           unrestricted_it_1840_opus5_clean_n10    ("no patterns")
  none     no patterns, no embargo        unrestricted_it_1840_opus5_noemb_n10    ("no patterns, no embargo")

Same model and parameters in all four; within each pair the prompts differ only by the embargo
block, which is the same block in both embargo arms (checked below).

What is counted: for every content lemma (NOUN, PROPN, VERB, ADJ, ADV; stanza 1.12.2, CPU), the
number of each arm's ten texts that contain it (0-10). Presence per text, not frequency.

Contrasts, each read in both directions (a lemma can go up or down):
  patterns         every arm: {pat+emb, pat} against {emb, none}
  embargo          every arm: {pat+emb, emb} against {pat, none}
  patterns|no emb  pat against none         (the reading arm against its own control)
  patterns|emb     pat+emb against emb
  embargo|patterns pat+emb against pat
  embargo|no pat   emb against none
Score: the every-arm difference (smallest arm of the high side minus largest arm of the low side)
divided by 10; for a single pair, the plain difference / 10.

Null (there is no identical-prompt pair in this dataset): 2,000 stratified shuffles of the texts,
seed 20261127. A contrast's texts are shuffled only across the factor it tests, within the levels
of the other factor (patterns: within the embargo arms and within the no-embargo arms; embargo:
within the pattern arms and within the no-pattern arms; a pair: within the pair). For each
threshold t the null gives the expected number of lemmas that pass by chance and the family-wise
rate (share of shuffles in which at least one lemma passes).

Tiers, fixed here before the observed lists were read:
  firm       the smallest t at which the family-wise rate is <= 0.05
  candidate  the smallest t at which the expected number of chance passes is < 1 (if below firm)

Flags (they shorten the reading, they do not decide it): I in the 1840 incipit; E in the embargo
block; P in the pattern block (in the pat prompt, not in the none prompt); B in the prompt that all
four share (the none prompt: skeleton, domain, framing). P and B are lemma matches, and the prompts
are in English, so they catch only the Italian words in them (il mare, Lampedusa). T and U, by hand
(GLOSS below, checked against the prompts on 11-27): the lemma is the Italian for an English word of
the pattern block (T) or of the shared prompt (U); the md prints the English word.

Read-only on the run files. Writes results/test1_2x2_signs_<tag>.{json,md} and a lemma cache
results/test1_2x2_lemmas_<tag>.json; the text lemmas are checked against the 11-26 cache.

  ~/.local/share/manzoni-stanza-venv/bin/python scripts/test1_2x2_signs.py
"""
import argparse, hashlib, json, re, sys
from pathlib import Path

import numpy as np

HS = Path(__file__).resolve().parent.parent
RES = HS / "results"
ARMS = {  # same files and sha1s as test 1 (11-26) and the loop-A output probe
    "pat+emb": ("production_D2_it_effectfree-1840_claude-opus-5.json",       "4a6611451c5dc01dc6b8829dabdf0765cef5a8ce"),
    "pat":     ("production_D2_it_effectfree-noemb-1840_claude-opus-5.json", "b9bff3c29974716e26596753311da9cf5ec67aac"),
    "emb":     ("unrestricted_it_1840_opus5_clean_n10.json",                 "2c22e198d0d3b353b670400462fe0927bbcf8f6a"),
    "none":    ("unrestricted_it_1840_opus5_noemb_n10.json",                 "8dc2e3155a9cbe76fdee4ae95e8d74aa373c5ab0"),
}
OLD_NAMES = {"pat+emb": "revised", "pat": "stranger", "emb": "nopat", "none": "nopat-noemb"}
CONTENT = {"NOUN", "PROPN", "VERB", "ADJ", "ADV"}
INCIPIT = HS / "texts" / "incipit_it.txt"
EMBARGO_RE = re.compile(r"LEXICAL EMBARGO.*?\n\n", re.S)

# name: (high side, low side, strata shuffled within)
CONTRASTS = {
    "patterns":         (("pat+emb", "pat"), ("emb", "none"), (("pat+emb", "emb"), ("pat", "none"))),
    "embargo":          (("pat+emb", "emb"), ("pat", "none"), (("pat+emb", "pat"), ("emb", "none"))),
    "patterns|no emb":  (("pat",), ("none",), (("pat", "none"),)),
    "patterns|emb":     (("pat+emb",), ("emb",), (("pat+emb", "emb"),)),
    "embargo|patterns": (("pat+emb",), ("pat",), (("pat+emb", "pat"),)),
    "embargo|no pat":   (("emb",), ("none",), (("emb", "none"),)),
}
# By hand, 11-27 (side-agent note: the P flag cannot see a translation). Every lemma that reaches a
# tier in some contrast was looked up in the English prompts: (T pattern block | U shared prompt,
# the English it renders). Lemmas not listed render nothing in either: onda, superficie, respiro,
# pazienza, conca, insenatura, correre, prefabbricato, chiaro, fondo, cale, recinto, esistere.
GLOSS = {
    "allargare":  ("T", "P5-01 expansion, re-expansion"),
    "dilatare":   ("T", "P5-01 dilation"),
    "dilare":     ("T", "P5-01 dilation (stanza's lemma for some forms of dilatare)"),
    "stringere":  ("T", "P5-01 constriction"),
    "nuovo":      ("T", "P5-01 re-expansion (di nuovo); also 'NEW material'"),
    "specchio":   ("T", "P6-01 specular, mirroring medium"),
    "capovolto":  ("T", "P6-01 inversion marker"),
    "riflesso":   ("T", "P6-01 reflection predicate"),
    "paesaggio":  ("U", "domain brief: migration landscape"),
    "europeo":    ("U", "domain brief: Italian and European policing apparatus"),
}
TS = [round(0.1 * k, 1) for k in range(3, 10)]
N_SHUFFLE, SEED = 2000, 20261127


def lemmatize(nlp, text):
    out = set()
    for s in nlp(text).sentences:
        for w in s.words:
            if w.upos in CONTENT and w.lemma and any(c.isalpha() for c in w.lemma):
                out.add(w.lemma.lower())
    return out


def scores(C, idx, hi, lo):
    """Every-arm difference /10, both directions. C: lemmas x arms counts."""
    h = C[:, [idx[a] for a in hi]]
    l = C[:, [idx[a] for a in lo]]
    up = (h.min(1) - l.max(1)) / 10
    down = (l.min(1) - h.max(1)) / 10
    return up, down


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="1127")
    ap.add_argument("--reparse", action="store_true", help="ignore the lemma cache")
    a = ap.parse_args()
    cache = RES / f"test1_2x2_lemmas_{a.tag}.json"

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
    # the design: within each pair the prompts differ only by one and the same embargo block
    emb_blocks = {k: EMBARGO_RE.search(prompts[k]).group(0) for k in ("pat+emb", "emb")}
    assert emb_blocks["pat+emb"] == emb_blocks["emb"], "embargo blocks differ"
    for k_emb, k_no in (("pat+emb", "pat"), ("emb", "none")):
        assert not EMBARGO_RE.search(prompts[k_no]), k_no
        assert prompts[k_emb].replace(emb_blocks[k_emb], "") == prompts[k_no], (k_emb, k_no)

    if cache.exists() and not a.reparse:
        L = json.loads(cache.read_text())
    else:
        import stanza
        nlp = stanza.Pipeline("it", processors="tokenize,mwt,pos,lemma", use_gpu=False,
                              download_method=None, verbose=False)
        L = {"stanza": stanza.__version__, "upos": sorted(CONTENT),
             "texts": {arm: [sorted(lemmatize(nlp, t)) for t in ts] for arm, ts in runs.items()},
             "incipit": sorted(lemmatize(nlp, INCIPIT.read_text())),
             "embargo": sorted(lemmatize(nlp, emb_blocks["emb"])),
             "prompt_pat": sorted(lemmatize(nlp, prompts["pat"])),
             "prompt_none": sorted(lemmatize(nlp, prompts["none"]))}
        cache.write_text(json.dumps(L, ensure_ascii=False, indent=0))

    # consistency with the 11-26 parse of the same files
    old = RES / "test1_lemmas_1126.json"
    same_as_1126 = None
    if old.exists():
        O = json.loads(old.read_text())["texts"]
        same_as_1126 = all(O[OLD_NAMES[k]] == L["texts"][k] for k in ARMS)

    arms = list(ARMS)
    idx = {k: i for i, k in enumerate(arms)}
    lemmas = sorted({lem for k in arms for doc in L["texts"][k] for lem in doc})
    li = {lem: i for i, lem in enumerate(lemmas)}
    # X: lemmas x 40 texts (arm-major), presence 0/1
    X = np.zeros((len(lemmas), 40), dtype=np.int16)
    for k in arms:
        for j, doc in enumerate(L["texts"][k]):
            for lem in doc:
                X[li[lem], idx[k] * 10 + j] = 1
    cols = {k: np.arange(idx[k] * 10, idx[k] * 10 + 10) for k in arms}

    def counts(Xp):
        return np.stack([Xp[:, cols[k]].sum(1) for k in arms], axis=1)

    C = counts(X)
    rng = np.random.default_rng(SEED)
    null = {}
    for name, (hi, lo, strata) in CONTRASTS.items():
        passes = np.zeros((N_SHUFFLE, len(TS)), dtype=np.int32)
        for s in range(N_SHUFFLE):
            order = np.arange(40)
            for st in strata:
                pool = np.concatenate([cols[k] for k in st])
                order[pool] = rng.permutation(pool)
            up, down = scores(counts(X[:, order]), idx, hi, lo)
            best = np.maximum(up, down)
            passes[s] = [(best >= t - 1e-9).sum() for t in TS]
        up, down = scores(C, idx, hi, lo)
        best = np.maximum(up, down)
        exp = passes.mean(0)
        fwer = (passes > 0).mean(0)
        firm = next((t for t, f in zip(TS, fwer) if f <= 0.05), None)
        cand = next((t for t, e in zip(TS, exp) if e < 1), None)
        if cand is not None and firm is not None and cand >= firm:
            cand = None
        rows = []
        for i in np.argsort(-best, kind="stable"):
            lo_t = cand if cand is not None else firm
            if lo_t is None or best[i] < lo_t - 1e-9:
                break
            lem = lemmas[i]
            rows.append({"lemma": lem, "score": round(float(best[i]), 1),
                         "direction": "up" if up[i] >= down[i] else "down",
                         "tier": "firm" if best[i] >= firm - 1e-9 else "candidate",
                         "df": {k: int(C[i, idx[k]]) for k in arms},
                         "I": lem in L["incipit"], "E": lem in L["embargo"],
                         "P": lem in L["prompt_pat"] and lem not in L["prompt_none"],
                         "B": lem in L["prompt_none"],
                         "T": GLOSS.get(lem, ("",))[0] == "T", "U": GLOSS.get(lem, ("",))[0] == "U",
                         "gloss": GLOSS[lem][1] if lem in GLOSS else None})
        null[name] = {"high": hi, "low": lo, "shuffled_within": strata,
                      "t": TS, "expected_chance": [round(float(e), 3) for e in exp],
                      "familywise": [round(float(f), 3) for f in fwer],
                      "observed": [int((best >= t - 1e-9).sum()) for t in TS],
                      "firm": firm, "candidate": cand, "lemmas": rows}

    out = {"spec": "SF 2026-11-27: test 1 on the 2x2 dataset (patterns x embargo) only",
           "tag": a.tag, "stanza": L["stanza"], "upos": L["upos"], "n_lemmas": len(lemmas),
           "arms": {k: {"file": f"results/{fn}", "sha1": sha} for k, (fn, sha) in ARMS.items()},
           "text_lemmas_same_as_1126_parse": same_as_1126,
           "null": {"method": "stratified shuffles", "n": N_SHUFFLE, "seed": SEED},
           "tier_rule": {"firm": "smallest t with family-wise rate <= 0.05",
                         "candidate": "smallest t with expected chance passes < 1, if below firm"},
           "contrasts": null}
    (RES / f"test1_2x2_signs_{a.tag}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))

    md = [f"# Test 1 on the 2x2 dataset ({a.tag})", "",
          "Arms (ten texts each): **pat+emb** revised patterns with the embargo; **pat** revised "
          "patterns, no embargo (the reading arm); **emb** no patterns, embargo; **none** no patterns, "
          "no embargo. Cells: texts (of ten) containing the lemma.", "",
          f"Content lemmas: {len(lemmas)}. Null: {N_SHUFFLE} stratified shuffles (seed {SEED}). "
          "Tiers fixed before reading: firm = smallest t with family-wise rate <= 0.05; candidate = "
          "smallest t with fewer than one chance pass expected.", "",
          "Flags: I incipit, E embargo block, P pattern block, B prompt shared by all four (P and B "
          "are lemma matches: the prompts are in English, so they catch only Italian words in them); "
          "by hand, T renders an English word of the pattern block, U one of the shared prompt "
          "(the English is in the last column).", ""]
    for name, r in null.items():
        hi, lo = " + ".join(r["high"]), " + ".join(r["low"])
        md += [f"## {name}: {hi} against {lo}", "",
               "| t | " + " | ".join(str(t) for t in TS) + " |",
               "|---|" + "---|" * len(TS),
               "| expected by chance | " + " | ".join(f"{e:.2f}" for e in r["expected_chance"]) + " |",
               "| family-wise rate | " + " | ".join(f"{f:.2f}" for f in r["familywise"]) + " |",
               "| observed | " + " | ".join(str(o) for o in r["observed"]) + " |", "",
               f"Firm at t >= {r['firm']}" + (f", candidate at t >= {r['candidate']}" if r["candidate"] else "")
               + ".", ""]
        if r["lemmas"]:
            md += ["| tier | lemma | dir | score | " + " | ".join(arms) + " | flags | renders |",
                   "|---|---|---|---|" + "---|" * len(arms) + "---|---|"]
            for x in r["lemmas"]:
                fl = "".join(f for f in "IEPBTU" if x[f])
                md.append(f"| {x['tier']} | {x['lemma']} | {x['direction']} | {x['score']:.1f} | "
                          + " | ".join(str(x['df'][k]) for k in arms) + f" | {fl} | {x['gloss'] or ''} |")
            md.append("")
        else:
            md += ["No lemma reaches a tier.", ""]
    md += [f"Text lemmas identical to the 11-26 parse: {same_as_1126}."]
    (RES / f"test1_2x2_signs_{a.tag}.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
