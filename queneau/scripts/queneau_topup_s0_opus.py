#!/usr/bin/env python3
"""queneau_topup_s0_opus.py --- refill the Opus S0 cells that classifier refusals left thin.

Opus 4.8's refusals on this corpus are stochastic and sharply structured: every one of
the 51 falls in ENGLISH, and all but one in the BLIND regime (showing the source text
switches the classifier off entirely).  Two cells were left far below N=10 --
en/hospital-emergency-room at 1/10 (95% refusal) and en/security-checkpoint at 8/10
(69%) -- which is a problem for the row they feed: the Opus blind mean pools nine cells
but those two contribute 9 generations between them instead of 20, so the row
under-weights exactly the condition Opus finds hardest.  The lone survivor of a
95%-refusal cell is also not a random draw -- it is the one that got past the gate.

This TOPS UP rather than re-runs: the June generations keep their text and their index,
the new ones continue the numbering, and the original refusal counts stay readable
beside the new ones so the June rate remains reportable.  The prompt is replayed
VERBATIM from the stored record rather than rebuilt from the spec files, so the added
generations answer byte-identically the same question as the originals.

    python scripts/queneau_topup_s0_opus.py            # top up to N=10
    python scripts/queneau_topup_s0_opus.py --dry-run
"""
import argparse
import json
import sys
import time
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import run_production as RP   # noqa: E402

OUT = ROOT / "Experiments" / "Queneau" / "queneau-homo" / "outputs"
CELLS = [
    "homo_S0_en_hospital-emergency-room_blind_b1_claude-opus-4-8.json",
    "homo_S0_en_security-checkpoint_blind_b1_claude-opus-4-8.json",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=10)
    ap.add_argument("--retries", type=int, default=25,
                    help="refusal retries per sample (refused calls are unbilled)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    for name in CELLS:
        p = OUT / name
        if not p.exists():
            print(f"!! missing {name}"); continue
        d = json.loads(p.read_text(encoding="utf-8"))
        have = len(d["gens"])
        need = a.target - have
        label = name.replace("homo_S0_", "").replace("_b1_claude-opus-4-8.json", "")
        if need <= 0:
            print(f"[{label}] already at {have}/{a.target}"); continue
        print(f"[{label}] {have}/{a.target} — requesting {need} more "
              f"(June: {d.get('refused')} refusals in {d.get('attempts')} attempts)", flush=True)
        if a.dry_run:
            continue

        # A generous retry budget: at June's 95% refusal rate on the hospital cell, 8
        # retries per sample would clear barely a third of them and leave the cell short
        # for a second time. Refused calls return no content and are not billed, so the
        # only cost of a high ceiling is wall-clock.
        stats, t0 = {}, time.time()
        gens = RP.gen_anthropic(d["model"], d["prompt"], need, d["temperature"],
                                d["max_tokens"], stats=stats, refusal_retries=a.retries)
        dt = round(time.time() - t0, 1)

        nxt = max((g["i"] for g in d["gens"]), default=-1) + 1
        d["gens"].extend({"i": nxt + k, "text": t} for k, t in enumerate(gens))
        d["n"] = len(d["gens"])
        d.setdefault("topups", []).append({
            "date": date.today().isoformat(), "added": len(gens),
            "refused": stats.get("refused"), "attempts": stats.get("attempts"),
            "elapsed_s": dt,
            "note": "prompt replayed verbatim from this record; original rows untouched",
        })
        p.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")
        short = "" if len(d["gens"]) >= a.target else f"  !! STILL SHORT {len(d['gens'])}/{a.target}"
        print(f"[{label}] +{len(gens)} -> {d['n']}/{a.target} in {dt}s "
              f"({stats.get('refused')} refusals in {stats.get('attempts')} attempts){short}",
              flush=True)


if __name__ == "__main__":
    main()
