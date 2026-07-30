#!/usr/bin/env python3
"""score_production.py — score + select Phase D production candidates.

Scores each whole-passage candidate in a production_*.json on all three axes
and ranks them, so we can pick the best homeosyntactic deformation:

  axis 1 (keep-some-syntax, drop-rest):
    d2_clause  : F1 of the clause-type multiset vs source  (HIGH = held architecture)
    depth_gap  : |hypotaxis depth(src) - depth(gen)|       (LOW  = matched nesting)
    s0_pos     : POS-sequence edit similarity, order-SENS  (LOW  = surface freed)
  axis 2 (semantic divergence, no calque):
    lemma_jacc : content-lemma Jaccard(src, gen)           (LOW  = new lexis)
    frame_div  : 1 - LLM-judge "same scene relexicalized?" (HIGH = new scene)
  axis 3 (coherence):
    coherence  : LLM-judge "one coherent passage?"         (HIGH = good)

Selection score = mean of the five "good" directions, each in [0,1]:
    (d2_clause + (1-s0_pos) + (1-lemma_jacc) + frame_div + coherence) / 5
Coherence is the binding axis, so it also prints the best COHERENT candidate.

  python scripts/score_production.py reproducibility/graded-corpus/results/production_*.json
  SEM_JUDGE_MODEL=claude-opus-4-8 python scripts/score_production.py <file>   # stronger judge
Run with the Atlas venv python (Stanza). Needs ANTHROPIC_API_KEY in .env.
"""
import json
import re
import sys
import time
import urllib.error
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import run_production as RP   # noqa: E402
import tree_metric as TM   # noqa: E402
import sem_metrics as M    # noqa: E402

CONTENT = {"NOUN", "VERB", "ADJ", "ADV", "PROPN"}


def lemmas(toks):
    return {t["lemma"] for t in toks if t["upos"] in CONTENT}


def judged(fn, *a):
    delay = 3.0
    for attempt in range(6):
        try:
            return fn(*a)
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 529) and attempt < 5:
                time.sleep(delay); delay = min(delay * 2, 40); continue
            raise
        except Exception:
            if attempt < 5:
                time.sleep(delay); delay = min(delay * 2, 40); continue
            raise


def score_file(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    lang = data.get("lang", "it")
    src = RP.resolve_source(data)
    # the embargo as the run actually stated it, read back out of its own prompt,
    # so old and new runs are both scoreable without consulting a spec file
    body = data.get("prompt", "").split("inflected form", 1)
    spec_embargo = sorted({w.strip().lower() for w in
                           body[1].split("REQUIREMENTS")[0].replace(":", " ")
                           .replace("\n", " ").split(",")
                           if w.strip() and " " not in w.strip()}) if len(body) > 1 else []
    stok = TM.parse(src, lang=lang)
    sl = lemmas(stok)
    rows = []
    for g in data["gens"]:
        txt = g["text"]
        if not txt or txt.startswith("[ERR"):
            continue
        gtok = TM.parse(txt, lang=lang)
        c = TM.compliance(stok, gtok)        # d2_clause, depth_gap, d1_triple, s0_pos
        gl = lemmas(gtok)
        lj = len(sl & gl) / len(sl | gl) if (sl | gl) else 0.0
        fsim = judged(M.frame_judge_sim, src, txt, lang)
        coh = judged(M.coh_judge, txt, lang)
        fdiv = (1 - fsim) if fsim is not None else None
        good = [c["d2_clause"], 1 - c["s0_pos"], 1 - lj,
                fdiv if fdiv is not None else 0.0, coh if coh is not None else 0.0]
        rows.append({
            "i": g["i"], "words": len(txt.split()),
            "embargo_hits": emb_hits(g["text"], spec_embargo),
            "d2_clause": c["d2_clause"], "depth_gap": c["depth_gap"],
            "s0_pos": c["s0_pos"], "lemma_jacc": round(lj, 3),
            "frame_div": round(fdiv, 3) if fdiv is not None else None,
            "coherence": coh, "score": round(sum(good) / 5, 3), "text": txt,
        })
        time.sleep(0.2)
    rows.sort(key=lambda r: r["score"], reverse=True)
    return data, rows


def emb_hits(text, embargo):
    """Embargoed forms present in a generation, by lemma-blind stem match.

    Reported, never enforced: the criterion is the gate, and a violation rate is
    itself a finding — it is how we learned the old 284-word list could not be
    obeyed.  Stems because the embargo forbids "any inflected form".
    """
    if not embargo:
        return []
    words = set(re.findall(r"[a-zà-ÿ']{3,}", text.lower()))
    hits = set()
    for e in embargo:
        for w in words:
            # exact, or one is the other's prefix with at most two characters of
            # inflection between them.  A blunter 5-character stem match reported
            # `strada` as `stradette` and `contrada` as `contrassegno` — different
            # words that merely start alike.
            if w == e or (len(e) > 4
                          and ((w.startswith(e) and len(w) - len(e) <= 2)
                               or (e.startswith(w) and len(e) - len(w) <= 2))):
                hits.add(e)
                break
    return sorted(hits)


def main():
    files = sys.argv[1:]
    if not files:
        print(__doc__); sys.exit(1)
    print(f"judge model: {M.JUDGE_MODEL}\n")
    for path in files:
        data, rows = score_file(path)
        print(f"=== {Path(path).name}  ({data.get('model')}, "
              f"periods={data.get('periods')}, n={len(rows)}) ===")
        print(f"{'cand':>4}{'words':>6}{'d2_cl':>7}{'depth':>6}{'s0_pos':>7}"
              f"{'lem_j':>6}{'fr_div':>7}{'coher':>6}{'SCORE':>7}")
        for r in rows:
            print(f"{r['i']:>4}{r['words']:>6}{r['d2_clause']:>7.3f}{r['depth_gap']:>6.1f}"
                  f"{r['s0_pos']:>7.3f}{r['lemma_jacc']:>6.3f}"
                  f"{(r['frame_div'] if r['frame_div'] is not None else float('nan')):>7.3f}"
                  f"{(r['coherence'] if r['coherence'] is not None else float('nan')):>6.2f}"
                  f"{r['score']:>7.3f}")
        best = rows[0]
        coherent = [r for r in rows if (r["coherence"] or 0) >= 0.5]
        best_coh = coherent[0] if coherent else None
        outp = Path(path).with_name(Path(path).stem.replace("production_", "production_scored_") + ".json")
        outp.write_text(json.dumps({"model": data.get("model"), "source": src_of(data),
                                    "rows": rows}, ensure_ascii=False, indent=1),
                        encoding="utf-8")
        print(f"\nwrote {outp.name}")
        print(f"\n--- TOP by score (cand {best['i']}, score {best['score']}) ---\n{best['text']}\n")
        if best_coh and best_coh["i"] != best["i"]:
            print(f"--- TOP coherent (cand {best_coh['i']}, coher {best_coh['coherence']}, "
                  f"score {best_coh['score']}) ---\n{best_coh['text']}\n")


def src_of(data):
    return RP.resolve_source(data)


if __name__ == "__main__":
    main()
