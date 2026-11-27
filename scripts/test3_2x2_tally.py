#!/usr/bin/env python3
"""test3_2x2_tally.py — test 3 (output probe, Opus 5) on the four 2x2 arms, both batches pooled.

SF, 2026-11-27: test 1 set aside; test 3 firmed up with ten more texts per arm (scripts/run_2x2_batch.sh,
batch b1127; manifest by scripts/build_loopA_probes.py --batch b1127). This script reads the two
Opus results files, keeps the four arms of the 2x2 (patterns x embargo; the July and replicate arms
of the first file are left out), and counts, per arm, cue and batch, the texts for which Opus names
I promessi sposi (the rule of run_probe_batch.py), out of the texts it answered for (a refusal is
left out of the denominator and reported). Embargo contrasts by Fisher's exact test, two-sided.

Read-only. Writes results/test3_2x2_tally_<tag>.{json,md}.

  python3 scripts/test3_2x2_tally.py
"""
import argparse, json
from pathlib import Path

from scipy.stats import fisher_exact

RES = Path(__file__).resolve().parent.parent / "results"
FILES = {"1126": "probe_loopA_outputs_it_D2_loopA-1126_claude-opus-5.json",
         "b1127": "probe_loopA_outputs_it_D2_b1127_loopA-1127_claude-opus-5.json"}
# manifest keys -> the draft's names
ARMS = {"revised": "patterns, embargo (revised)",
        "stranger": "patterns, no embargo (constrained, no embargo)",
        "nopat": "no patterns, embargo",
        "nopat-noemb": "no patterns, no embargo"}
CUES = ("uncued", "cued")
CONTRASTS = {  # embargo against no embargo
    "embargo, pooled over patterns": (("revised", "nopat"), ("stranger", "nopat-noemb")),
    "embargo | patterns": (("revised",), ("stranger",)),
    "embargo | no patterns": (("nopat",), ("nopat-noemb",)),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="1127")
    a = ap.parse_args()
    cell = {}  # (arm, cue, batch) -> [named, answered, refused]
    src = {}
    for batch, fn in FILES.items():
        d = json.loads((RES / fn).read_text())
        src[batch] = {"file": f"results/{fn}", "manifest": d["manifest"], "params": d["params"]}
        for it in d["items"].values():
            if it["arm"] not in ARMS:
                continue
            c = cell.setdefault((it["arm"], it["cue"], batch), [0, 0, 0])
            if it["failure"]:
                c[2] += 1
            else:
                c[0] += int(it["recognised"] > 0)
                c[1] += 1
    for arm in ARMS:
        for cue in CUES:
            for batch in FILES:
                _, answered, refused = cell.get((arm, cue, batch), [0, 0, 0])
                assert answered + refused == 10, (arm, cue, batch, answered, refused)

    def tot(arms, cue, batches=tuple(FILES)):
        k = [0, 0, 0]
        for arm in arms:
            for b in batches:
                k = [x + y for x, y in zip(k, cell[(arm, cue, b)])]
        return k

    out = {"spec": "SF 2026-11-27: test 3 on the 2x2 arms, two batches of ten texts per arm",
           "sources": src, "arms": ARMS, "cells": {}, "contrasts": {}}
    md = [f"# Test 3 on the 2x2 arms, both batches ({a.tag})", "",
          "Opus 5, one answer per text. Cells: texts for which Opus names *I promessi sposi* / texts "
          "it answered for (refusals noted).", "",
          "| arm | cue | " + " | ".join(FILES) + " | pooled |", "|---|---|" + "---|" * (len(FILES) + 1)]
    for arm, name in ARMS.items():
        for cue in CUES:
            per = [cell[(arm, cue, b)] for b in FILES]
            p = tot((arm,), cue)
            out["cells"][f"{arm}|{cue}"] = {"batches": dict(zip(FILES, per)), "pooled": p}
            f = lambda c: f"{c[0]}/{c[1]}" + (f" ({c[2]} refused)" if c[2] else "")
            md.append(f"| {name} | {cue} | " + " | ".join(f(c) for c in per) + f" | {f(p)} |")
    md += ["", "## Embargo against no embargo (Fisher, two-sided)", "",
           "| contrast | cue | batches | embargo | no embargo | p |", "|---|---|---|---|---|---|"]
    for cname, (e, ne) in CONTRASTS.items():
        for cue in CUES:
            for label, bs in [(b, (b,)) for b in FILES] + [("pooled", tuple(FILES))]:
                x, y = tot(e, cue, bs), tot(ne, cue, bs)
                p = fisher_exact([[x[0], x[1] - x[0]], [y[0], y[1] - y[0]]])[1]
                out["contrasts"][f"{cname}|{cue}|{label}"] = {"embargo": x, "no_embargo": y, "p": p}
                md.append(f"| {cname} | {cue} | {label} | {x[0]}/{x[1]} | {y[0]}/{y[1]} | {p:.3g} |")
    (RES / f"test3_2x2_tally_{a.tag}.json").write_text(json.dumps(out, indent=1))
    (RES / f"test3_2x2_tally_{a.tag}.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
