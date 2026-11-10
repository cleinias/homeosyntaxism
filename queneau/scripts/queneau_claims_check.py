#!/usr/bin/env python3
"""Standing check: re-test the paper's STRUCTURAL claims against whatever is on disk.

WHY THIS EXISTS

Verifying that every printed number still matches the data catches one kind of staleness
and misses the more dangerous kind. The paper's load-bearing statements are mostly
RELATIONAL -- the gradient climbs, the peak sits at 72B, the surface holds at a ~0.3
floor, no source-blind generation clears both bars, the 32B inversion is confined to the
tight rungs -- and a relation can be falsified by data that ARRIVES, without any printed
figure changing.

That is exactly what happened on 2026-08-09. The sentence "blind, the surface can be
reached or the meaning kept, never both" was true of the corpus it was written against,
when D0 was a single English cell of ten generations. Broadening D0 to nine cells across
three languages (commit 95d99bf) introduced two French Opus generations that clear both
bars at 0.60 coherence. No number on the page changed; the claim simply stopped being
true. Re-checking the figures would never have found it -- and did not, through several
passes -- because every such pass asks whether the printed values are right, not whether
the sentences around them still hold.

So: RUN THIS AFTER ANY NEW RUN LANDS, before believing the prose. It tests relations, not
values, and it prints each claim's status so a change is visible as a change.

The second column applies the contamination filter (see queneau_contamination_check.py)
so the effect of excluding leaked-scaffolding generations on each claim is visible
side by side. Where the two columns agree, the claim is robust to that choice.

    python scripts/queneau_claims_check.py
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
S = ROOT / "Experiments/Queneau/queneau-homo/outputs/scored"
O = ROOT / "Experiments/Queneau/queneau-homo/outputs"
SCEN = ("grocery-checkout", "security-checkpoint", "hospital-emergency-room")
L3, EN = ("en", "fr", "it"), ("en",)

POS = re.compile(r"(?<![\w'])(NOUN|VERB|ADJ|ADV|DET|ADP|PRON|PROPN|AUX|CCONJ|SCONJ|"
                 r"PART|NUM|INTJ|PUNCT|SYM)(?![\w'])")
BRACK = re.compile(r"\[\d+\]")


def dbl(t):
    w = [x.lower().strip(".,;:!?«»\"'()—–-") for x in t.split()]
    return sum(1 for a, b in zip(w, w[1:]) if a == b and len(a) > 2)


def bad(t):
    return len(POS.findall(t)) >= 3 or len(BRACK.findall(t)) >= 3 or dbl(t) >= 10


def rows(pat, exc, langs=L3):
    out = []
    for l in langs:
        for s in SCEN:
            f = S / pat.format(l=l, s=s)
            if not f.exists():
                f = S / pat.format(l=l, s=s).replace("_blind_", "_")
            if not f.exists():
                continue
            d = json.loads(f.read_text(encoding="utf-8"))
            raw = json.loads((O / f.name[len("scored_"):]).read_text(encoding="utf-8"))
            txt = {g["i"]: " ".join(g["text"].split()) for g in raw["gens"]}
            for r in d["rows"]:
                if exc and bad(txt.get(r["i"], "")):
                    continue
                out.append(r)
    return out


def mean(rs, k):
    v = [r[k] for r in rs if k in r and r[k] is not None]
    return sum(v) / len(v) if v else float("nan")


LADDER = [("0.5B", "Qwen_Qwen2.5-0.5B-Instruct"), ("1.5B", "Qwen_Qwen2.5-1.5B-Instruct"),
          ("3B", "Qwen_Qwen2.5-3B-Instruct"), ("7B", "Qwen_Qwen2.5-7B-Instruct"),
          ("14B", "Qwen_Qwen2.5-14B-Instruct"), ("32B", "Qwen_Qwen2.5-32B-Instruct"),
          ("72B", "Qwen_Qwen2.5-72B-Instruct")]
DEG = [("D0", "blind_ladderD0soft", "s0_pos"), ("D1", "blind_ladderD1", "d1_triple"),
       ("D2c", "blind_ladderD2c", "clause_nest"), ("D2", "ladderD2", "d2_clause")]

results = []


def claim(text, fn):
    a, b = fn(False), fn(True)
    results.append((text, a, b))


def t3(exc):
    """fidelity per (model, degree) and coherence per (model, degree)"""
    fid, coh = {}, {}
    for lab, slug in LADDER:
        for deg, tag, key in DEG:
            r = rows(f"scored_homo_{deg}_{{l}}_{{s}}_{tag}_{slug}.json", exc)
            fid[(lab, deg)] = mean(r, key)
            coh[(lab, deg)] = mean(r, "coherence")
    return fid, coh


def main():
    F0, C0 = t3(False)
    F1, C1 = t3(True)

    def gradient_ok(F):
        bad_rows = [lab for lab, _ in LADDER
                    if not (F[(lab, "D0")] <= F[(lab, "D1")] <= F[(lab, "D2c")] <= F[(lab, "D2")])]
        return f"{7-len(bad_rows)}/7 rows monotone" + (f" (breaks: {bad_rows})" if bad_rows else "")

    def peak72(F):
        cols = [deg for deg, _, _ in DEG
                if F[("72B", deg)] < max(F[(l, deg)] for l, _ in LADDER)]
        return f"72B tops {4-len(cols)}/4 fidelity columns" + (f" (not: {cols})" if cols else "")

    def floor(F):
        v = [F[(lab, "D0")] for lab, _ in LADDER]
        return f"D0 surface {min(v):.2f}-{max(v):.2f}"

    def coh_scale(C):
        v = [C[(lab, "D0")] for lab, _ in LADDER]
        inv = [LADDER[i][0] for i in range(1, 7) if v[i] < v[i-1]]
        return f"D0 coherence {v[0]:.2f}->{v[-1]:.2f}, inversions at {inv or 'none'}"

    def b32_vs_14(C):
        out = []
        for deg in ("D0", "D1", "D2c", "D2"):
            out.append(f"{deg}:{'32B' if C[('32B',deg)]>C[('14B',deg)] else '14B'}")
        return " ".join(out)

    claim("Table 3 gradient: fidelity climbs D0<=D1<=D2<=D3",
          lambda e: gradient_ok(F1 if e else F0))
    claim("Table 3: 72B at/near the fidelity peak",
          lambda e: peak72(F1 if e else F0))
    claim("prose: 'surface reproduced at only about 0.3'",
          lambda e: floor(F1 if e else F0))
    claim("Table 5 caption: coherence rises with scale",
          lambda e: coh_scale(C1 if e else C0))
    claim("32B inversion confined to tight rungs (who leads)",
          lambda e: b32_vs_14(C1 if e else C0))

    # Table 1 claims
    def blind_both(exc):
        n = tot = 0
        for deg in ("S0", "D0"):
            for m in ("qwen_qwen-2.5-7b-instruct", "qwen_qwen-2.5-72b-instruct",
                      "meta-llama_llama-4-maverick", "claude-opus-4-8"):
                for r in rows(f"scored_homo_{deg}_{{l}}_{{s}}_blind_b1_{m}.json", exc):
                    tot += 1
                    if r["s0_pos"] >= 0.60 and (r["coherence"] or 0) >= 0.50:
                        n += 1
        return f"{n} of {tot} source-blind generations clear BOTH bars"
    claim("prose: 'blind, no generation gets both'", blind_both)

    def open_held_zero(exc):
        z = []
        for m, lab in (("qwen_qwen-2.5-7b-instruct", "7B"), ("qwen_qwen-2.5-72b-instruct", "72B"),
                       ("meta-llama_llama-4-maverick", "Mav")):
            for deg in ("S0", "D0"):
                r = rows(f"scored_homo_{deg}_{{l}}_{{s}}_blind_b1_{m}.json", exc)
                z.append(f"{lab}/{deg}:{mean(r,'held'):.2f}")
        return " ".join(z)
    claim("prose: 'not one generation in hundreds holds the form' (blind held-rate)",
          open_held_zero)

    def t5_held(exc):
        v = [mean(rows(f"scored_homo_D0_{{l}}_{{s}}_ladder_{slug}.json", exc), "held")
             for _, slug in LADDER]
        return f"max held-rate across the 7 Qwen rows: {max(v):.2f}"
    claim("Table 5: held-rate zero for every model", t5_held)

    # Table 4 claims (English only)
    def t4_flat(exc):
        q4 = rows("scored_homo_D2c_{l}_{s}_mac-7b-q4_qwen2.5_7b-instruct-q4_K_M.json", exc, EN)
        bf = rows("scored_homo_D2c_{l}_{s}_blind_ladderD2c_Qwen_Qwen2.5-7B-Instruct.json", exc, EN)
        return (f"D2 7B q4 {mean(q4,'clause_nest'):.2f} vs BF16 {mean(bf,'clause_nest'):.2f}"
                f" -> quantized {'AHEAD' if mean(q4,'clause_nest')>mean(bf,'clause_nest') else 'behind'}")
    claim("appendix: 'at D2 the quantized models beat their unquantized counterparts'", t4_flat)

    def t4_size(exc):
        out = []
        for lab, pat in (("7B q4", "scored_homo_{d}_{l}_{s}_mac-7b-q4_qwen2.5_7b-instruct-q4_K_M.json"),
                         ("72B", "scored_homo_{d}_{l}_{s}_blind_ladderD2_Qwen_Qwen2.5-72B-Instruct.json")):
            vals = []
            for d in ("S0", "D0", "D1", "D2c", "D2"):
                tag = {"S0": "ladderS0", "D0": "ladderD0soft", "D1": "ladderD1",
                       "D2c": "ladderD2c", "D2": "ladderD2"}[d]
                p = (pat.replace("{d}", d) if "mac-" in pat
                     else f"scored_homo_{d}_{{l}}_{{s}}_blind_{tag}_Qwen_Qwen2.5-72B-Instruct.json")
                r = rows(p, exc, EN)
                if r:
                    vals.append(mean(r, "coherence"))
            out.append(f"{lab} {sum(vals)/len(vals):.2f}")
        return " vs ".join(out)
    claim("appendix: 'coherence rises with size'", t4_size)

    print(f"  {'claim':62} {'all generations':34} {'excluded'}")
    print(f"  {'-'*62} {'-'*34} {'-'*34}")
    for text, a, b in results:
        flag = "" if a == b else "   <-- CHANGES"
        print(f"  {text:62} {str(a):34} {str(b)}{flag}")


if __name__ == "__main__":
    main()
