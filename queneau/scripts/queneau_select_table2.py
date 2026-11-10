#!/usr/bin/env python3
"""Pick Table 2's two Qwen specimens from the ladder cells, once job 2030984 lands.

WHY THIS EXISTS

Table 2 (tab:source-copy-blind-samples) shows four generations on one instruction and
scene -- English, a security checkpoint -- chosen to occupy the corners of the
form/sense tension rather than by any mechanical rule:

    row 2  source-visible   form HELD, sense KEPT      currently 0.85 / 1.00   (b1)
    row 3  blind            form LOST, sense KEPT      currently 0.39 / 1.00   (b1)
    row 4  blind, Opus      form HELD, sense LOST      currently 0.90 / 0.00   (Anthropic)
    row 5  blind, D3        the escape at the inventory currently 0.90 / 1.00  (ACES)

Rows 2 and 3 are Qwen-72B at S0 from the hosted b1 arm. Job 2030984 runs exactly that
cell -- 72B, S0, English, security checkpoint, in BOTH regimes -- on our own nodes, so
those two specimens can move to the ladder and Table 2 becomes three-from-the-ladder
plus Opus for reference, matching Table 1's new shape. Row 4 is closed-weight and
cannot move; row 5 is already ACES. Neither is touched here.

WHAT IT WILL NOT DO

It will not choose for you. A corner has to actually EXIST among the new ten
generations: row 2 needs one that is high-form AND coherent, row 3 one that fails the
form bar clearly while staying readable. If the new cell's best candidate is weaker
than what is printed, that is a real trade -- provenance uniformity against the
clarity of the illustration -- and it should be made deliberately, looking at the
texts. So this ranks and shows; it does not write to the paper.

The incumbent b1 specimen is printed alongside every ranking, because "is the new one
good enough" is only answerable against the one it would replace.

    python scripts/queneau_select_table2.py                 # ladderT1 vs the incumbents
    python scripts/queneau_select_table2.py --tag ladderT1 --preview 400
"""
import argparse
import json
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "Experiments/Queneau/queneau-homo/outputs"
SCORED = OUT / "scored"

MODEL = "Qwen_Qwen2.5-72B-Instruct"          # as it appears in ladder filenames
B1_MODEL = "qwen_qwen-2.5-72b-instruct"      # as it appears in b1 filenames
CELL = "S0_en_security-checkpoint"
HELD_BAR = 0.60                              # the paper's form bar for S0

# what is printed in the paper today, and which generation it is
INCUMBENT = {"vis":   {"i": 2, "form": 0.85, "coh": 1.00},
             "blind": {"i": 0, "form": 0.39, "coh": 1.00}}


def load(regime, tag):
    # `--tag b1` re-runs the whole selection over the cells already in the paper, which
    # is how the ranking logic gets exercised on real generations before the new cells
    # exist: the incumbents should come out at or near the top of their own corners.
    model = B1_MODEL if tag == "b1" else MODEL
    raw = OUT / f"homo_{CELL}_{regime}_{tag}_{model}.json"
    sc = SCORED / f"scored_homo_{CELL}_{regime}_{tag}_{model}.json"
    if not raw.exists():
        return None, f"raw cell missing: {raw.name} (has job 2030984 finished?)"
    if not sc.exists():
        return None, f"cell not scored yet: {sc.name}"
    r = json.loads(raw.read_text(encoding="utf-8"))
    s = json.loads(sc.read_text(encoding="utf-8"))
    text = {g["i"]: " ".join(g["text"].split()) for g in r["gens"]}
    rows = [{**row, "text": text.get(row["i"], "")} for row in s["rows"]]
    return {"rows": rows, "n": s["n"], "refused": s.get("n_refused", 0)}, None


def incumbent(regime):
    """The b1 generation currently printed, so the comparison is like for like."""
    raw = OUT / f"homo_{CELL}_{regime}_b1_{B1_MODEL}.json"
    sc = SCORED / f"scored_homo_{CELL}_{regime}_b1_{B1_MODEL}.json"
    if not (raw.exists() and sc.exists()):
        return None
    i = INCUMBENT[regime]["i"]
    r = json.loads(raw.read_text(encoding="utf-8"))
    s = json.loads(sc.read_text(encoding="utf-8"))
    row = next((x for x in s["rows"] if x["i"] == i), None)
    txt = next((g["text"] for g in r["gens"] if g["i"] == i), "")
    return row and {**row, "text": " ".join(txt.split())}


def show(tag_label, cand, preview):
    f, c, w = cand["s0_pos"], cand["coherence"], cand["words"]
    bar = "HELD" if f >= HELD_BAR else "lost"
    print(f"    {tag_label:9} i={cand['i']:<2} form {f:.2f} ({bar})  coherence {c:.2f}  {w} words")
    if preview:
        print(textwrap.fill(cand["text"][:preview] + ("..." if len(cand["text"]) > preview else ""),
                            96, initial_indent=" " * 8, subsequent_indent=" " * 8))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="ladderT1")
    ap.add_argument("--preview", type=int, default=260, help="chars of text to show (0 = none)")
    a = ap.parse_args()

    print(f"Table 2 specimen selection — {CELL}, Qwen2.5-72B, tag {a.tag}\n")

    # ---- row 2: source-visible. The corner is BOTH: hold the surface and stay sensible.
    #      Rank by form among the coherent ones; a high form with dead prose is row 4's
    #      corner, not this one, so coherence gates rather than trades off.
    print("=" * 96)
    print("ROW 2  source-visible — form HELD with sense kept   (incumbent 0.85 / 1.00, b1 i=2)")
    print("=" * 96)
    cell, err = load("vis", a.tag)
    if err:
        print(f"  -- {err}")
    else:
        inc = incumbent("vis")
        ok = [r for r in cell["rows"] if (r["coherence"] or 0) >= 0.50]
        ok.sort(key=lambda r: -r["s0_pos"])
        print(f"  {cell['n']} generations scored"
              f"{', ' + str(cell['refused']) + ' refused' if cell['refused'] else ''}; "
              f"{len(ok)} clear the coherence bar\n")
        for r in ok[:3]:
            show("candidate", r, a.preview)
        if inc:
            print()
            show("INCUMBENT", inc, a.preview)
        if ok:
            best = ok[0]
            verdict = ("stronger than" if best["s0_pos"] > inc["s0_pos"] else "weaker than") if inc else "?"
            print(f"\n  -> best new candidate is {verdict} the incumbent on form "
                  f"({best['s0_pos']:.2f} vs {inc['s0_pos']:.2f})" if inc else "")
            if best["s0_pos"] < HELD_BAR:
                print("  !! no source-visible generation clears the form bar — the corner is not "
                      "present in this cell, and swapping would break the illustration")

    # ---- row 3: blind. The corner is form LOST while sense survives. The clearest
    #      specimen is the LOWEST form among coherent ones -- the incumbent's 0.39 is
    #      well under the bar, and a candidate at 0.55 would blur the contrast.
    print("\n" + "=" * 96)
    print("ROW 3  blind — form LOST, sense kept   (incumbent 0.39 / 1.00, b1 i=0)")
    print("=" * 96)
    cell, err = load("blind", a.tag)
    if err:
        print(f"  -- {err}")
    else:
        inc = incumbent("blind")
        ok = [r for r in cell["rows"] if (r["coherence"] or 0) >= 0.50]
        ok.sort(key=lambda r: r["s0_pos"])
        print(f"  {cell['n']} generations scored"
              f"{', ' + str(cell['refused']) + ' refused' if cell['refused'] else ''}; "
              f"{len(ok)} clear the coherence bar\n")
        for r in ok[:3]:
            show("candidate", r, a.preview)
        if inc:
            print()
            show("INCUMBENT", inc, a.preview)
        if ok and ok[0]["s0_pos"] >= HELD_BAR:
            print("\n  !! every coherent blind generation clears the form bar — the 'form lost' "
                  "corner is absent, which would itself be a finding, not a selection problem")

    print("\n" + "-" * 96)
    print("Row 4 (Opus, blind, form held / sense lost) is closed-weight and does not move.")
    print("Row 5 (D3, the escape at the clause inventory) is already a ladder cell.")
    print("Nothing is written to the paper: choose by reading the texts, then splice.")


if __name__ == "__main__":
    main()
