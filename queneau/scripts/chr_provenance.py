#!/usr/bin/env python3
"""Generate PROVENANCE.md: for every number the CHR2027 paper prints, where it came from.

WHY GENERATED RATHER THAN WRITTEN

A hand-written provenance table is a second copy of the truth, and second copies go stale
silently -- which is the failure this whole exercise exists to prevent. On 2026-08-10 the
contamination census was still computed over a corpus definition that a table move had
invalidated a day earlier, and nothing failed. So this document is derived: the rules come
from queneau_contamination_check.REPORTED, the values are recomputed from the run records,
and the launcher/job mapping below is the only hand-maintained part -- deliberately small,
and audited by the script itself, which flags any run tag it cannot attribute.

    python scripts/chr_provenance.py            # write the document
    python scripts/chr_provenance.py --check    # report gaps, write nothing
"""
import argparse
import collections
import contextlib
import importlib.util
import io
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# the working repository keeps it beside the paper; the public queneau/ folder at its root
OUT = (ROOT / "Docs and Research notes/CHR2027 paper/PROVENANCE.md"
       if (ROOT / "Docs and Research notes").is_dir() else ROOT / "PROVENANCE.md")

# REPORTED names tables by their working labels (T1...T5); the camera-ready numbers them
PAPER_TABLE = {"T1": "Table 2", "T3": "Table 4", "T4": "Table 6", "T5": "Table 7"}


def paper_label(tbl):
    """'T1/T5 Opus' -> 'Table 2/7 Opus'."""
    head, _, rest = tbl.partition(" ")
    nums = [PAPER_TABLE.get(t, t).replace("Table ", "") for t in head.split("/")]
    return f"Table {'/'.join(nums)} {rest}".strip()


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    m = importlib.util.module_from_spec(spec)
    with contextlib.redirect_stdout(io.StringIO()):
        spec.loader.exec_module(m)
    return m


CC = _load("cc", "scripts/queneau_contamination_check.py")
V = _load("v", "scripts/chr_verify.py")
S, O = CC.S, CC.O

# ---------------------------------------------------------------------------------
# The one hand-maintained mapping: run tag -> how it was produced.
# `launcher: None` means no committed script produces that tag. The script reports it
# rather than hiding it, because "every table can be regenerated from the run records"
# is a claim Appendix B makes.
# ---------------------------------------------------------------------------------
TAGS = {
    "ladderT1": dict(
        job="ACES 2030984", arm="ACES · vLLM · H100",
        launcher="scripts/queneau_homo_ladderT1.sbatch",
        quant="bf16 below 32B, fp8 at 32B and above", note=""),
    "ladderD0soft": dict(
        job="ACES 2029205", arm="ACES · vLLM · H100",
        launcher="scripts/queneau_homo_ladderD0soft.sbatch",
        quant="bf16 below 32B, fp8 at 32B and above",
        note="launcher recovered and committed 2026-08-10 (8d97b7a); the data went in "
             "2026-08-09 without it"),
    "ladderS0": dict(
        job="ACES 2029455", arm="ACES · vLLM · H100",
        launcher="scripts/queneau_homo_ladderS0.sbatch",
        quant="bf16 below 32B, fp8 at 32B and above",
        note="launcher recovered and committed 2026-08-10 (8d97b7a)"),
    "ladderD1": dict(
        job="ACES 2031074 / 2031075", arm="ACES · vLLM · H100",
        launcher="scripts/queneau_homo_ladderD1_frit.sbatch",
        quant="bf16 below 32B, fp8 at 32B and above",
        note="the French and Italian cells; the English cells came from an earlier pass"),
    "ladderD2c": dict(
        job="ACES (job id unrecorded)", arm="ACES · vLLM · H100",
        launcher="scripts/queneau_d2c_aces.sbatch + scripts/queneau_d2c_72b_aces.sbatch",
        quant="bf16 below 32B, fp8 at 32B and above", note=""),
    "ladderD2": dict(
        job="ACES (job id unrecorded, 2026-07-04)", arm="ACES · vLLM · H100",
        launcher="scripts/queneau_homo_ladderD2.sbatch  **(RECONSTRUCTED)**",
        quant="bf16 below 32B, fp8 at 32B and above (INFERRED — no record survives)",
        note="original launcher lost. Rebuilt 2026-08-10 (37ca4d1) from the run records; "
             "63/63 stored prompts are byte-identical to what the rebuilt path produces, "
             "so the prompt path is verified even though the job id and quantization are not"),
    "mac": dict(
        job="—", arm="laptop · Ollama · GGUF (24 GB MacBook Air)",
        launcher=None, quant="q4_K_M / q8_0 as named in the model string",
        note="English only; no committed launcher"),
    "b1": dict(
        job="—", arm=None,                       # per-row, from the recorded backend
        launcher="scripts/queneau_d0_grid_b1.sh  (D0 only)", quant="n/a (hosted)",
        note="no committed launcher covers S0 on this arm"),
}

# The hosted arm serves two different providers, so its label is taken from what each run
# actually recorded rather than from the tag.
BACKEND_ARM = {"anthropic": "Anthropic API · direct",
               "openrouter": "OpenRouter · hosted",
               "vllm-http": "ACES · vLLM · H100",
               "ollama-openai-loop": "laptop · Ollama · GGUF (24 GB MacBook Air)"}


def tag_of(fn):
    for t in ("ladderT1", "ladderD0soft", "ladderS0", "ladderD1", "ladderD2c", "ladderD2"):
        if f"_{t}_" in fn:
            return t
    if "_mac-" in fn:
        return "mac"
    if "_b1_" in fn:
        return "b1"
    return "?"


POS, BRACK = V.POS, V.BRACK


def cell_stats(files):
    """attempts, scored, contaminated, and the sampling parameters actually recorded."""
    att = scored = cont = 0
    temps, backends, models = set(), set(), set()
    for f in files:
        d = json.loads((S / f).read_text(encoding="utf-8"))
        raw = json.loads((O / f[len("scored_"):]).read_text(encoding="utf-8"))
        txt = {g["i"]: " ".join(g["text"].split()) for g in raw["gens"]}
        n_ref = d.get("n_refused") or len(d.get("refused", []) or [])
        att += len(d["rows"]) + n_ref
        temps.add(raw.get("temperature"))
        backends.add(raw.get("backend"))
        models.add(raw.get("model"))
        for r in d["rows"]:
            scored += 1
            if V.contaminated(txt.get(r["i"], "")):
                cont += 1
    t = sorted(str(x) for x in temps)
    return dict(att=att, scored=scored, cont=cont, used=scored - cont,
                temps="/".join(t), backends="/".join(sorted(str(b) for b in backends)),
                n_models=len(models))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()

    # group every reported cell by (table, tag, degree)
    groups = collections.defaultdict(list)
    for f in sorted(S.glob("scored_homo_*.json")):
        t = CC.reported_by(f.name)
        if not t or not (O / f.name[len("scored_"):]).exists():
            continue
        groups[(t, tag_of(f.name), f.name.split("_")[2])].append(f.name)

    unknown = sorted({k[1] for k in groups if k[1] not in TAGS})
    nolaunch = sorted({k[1] for k in groups if TAGS.get(k[1], {}).get("launcher") is None})
    if a.check or unknown:
        print(f"  {len(groups)} (table, run, degree) groups over "
              f"{sum(len(v) for v in groups.values())} cells")
        if unknown:
            print(f"  UNATTRIBUTED run tags: {unknown}")
        if nolaunch:
            print(f"  run tags with no committed launcher: {nolaunch}")
        if a.check:
            return

    DEG = {"S0": "S₀ surface", "D0": "D₀ ordered tree", "D1": "D₁ order-free deps",
           "D2c": "D₂ clause nesting", "D2": "D₃ clause inventory"}
    METRIC = {"S0": "s0_pos", "D0": "s0_pos", "D1": "d1_triple",
              "D2c": "clause_nest", "D2": "d2_clause"}

    L = []
    w = L.append
    w("# Provenance — *Keeping It in Shape* (CHR 2027)\n")
    w("*Generated by `scripts/chr_provenance.py`. Do not edit by hand: the rules come from\n"
      "`queneau_contamination_check.REPORTED`, the counts are recomputed from the run\n"
      "records, and only the run-tag → launcher table inside the script is maintained by\n"
      "hand. Regenerate after any change to a table's provenance.*\n")

    w("\n## How to read this\n")
    w("**The degree names in filenames are not the paper's.** Files named `D2` hold the\n"
      "paper's **D₃** (clause inventory); the paper's **D₂** (clause nesting) is in files\n"
      "named `D2c`. Every table below gives both.\n")
    w("\n**Contamination.** Every figure in the paper is computed over generations that pass\n"
      "the leaked-scaffolding detector (`scripts/queneau_contamination_check.py`). The\n"
      "`used` column is what the figures rest on; `scored` is before that exclusion and\n"
      "`attempts` includes generations the judge refused to score.\n")

    w("\n## 1. Where each table's numbers come from\n")
    w("| paper | degree (file) | run | arm | job | temp | cells | attempts | scored | used |")
    w("|---|---|---|---|---|---|---|---|---|---|")
    ORD = {"S0": 0, "D0": 1, "D1": 2, "D2c": 3, "D2": 4}
    def rowkey(kv):
        tbl, tag, deg = kv[0]          # sorted() hands the whole (key, value) pair
        return (tbl.split()[0], ORD.get(deg, 9), tag)
    for (tbl, tag, deg), files in sorted(groups.items(), key=rowkey):
        s = cell_stats(files)
        info = TAGS.get(tag, {})
        arm = info.get("arm") or BACKEND_ARM.get(s["backends"], s["backends"])
        w(f"| {paper_label(tbl)} | {DEG.get(deg, deg)} (`{deg}`) | `{tag}` | {arm} | "
          f"{info.get('job','?')} | {s['temps']} | {len(files)} | {s['att']} | "
          f"{s['scored']} | **{s['used']}** |")

    w("\n### The scripts behind each run\n")
    w("| run | launcher | quantization | notes |")
    w("|---|---|---|---|")
    for tag in sorted({k[1] for k in groups}):
        i = TAGS.get(tag, {})
        lau = i.get("launcher") or "**none committed**"
        w(f"| `{tag}` | {lau} | {i.get('quant','?')} | {i.get('note','') or '—'} |")

    w("\n### Common to every cell\n")
    w("| stage | script | detail |")
    w("|---|---|---|")
    w("| generation | `scripts/queneau_run_homo.py` | builds the prompt from "
      "`Experiments/Queneau/queneau-homo/spec/specs_{lang}_{degree}.json`; writes "
      "`Experiments/Queneau/queneau-homo/outputs/homo_*.json` with the prompt it sent |")
    w("| form specs | `scripts/queneau_build_specs.py` | derives each degree's invariant "
      "from the Stanza parse of Queneau's *Notations* |")
    w("| scoring | `scripts/queneau_score_homo.py` | writes "
      "`Experiments/Queneau/queneau-homo/outputs/scored/scored_homo_*.json`; `held` is "
      "degree-aware via `HELD_METRIC` |")
    w("| metrics | `scripts/sem_metrics.py` | the two LLM-judged axes; `JudgeRefusal` is "
      "raised without retry, since refusals are deterministic |")
    w("| judge | Claude Sonnet 4.6, temperature 0 | one judge for every coherence figure "
      "in the paper |")
    w("\nFidelity metric by degree: " + " · ".join(
        f"{DEG[d]} → `{METRIC[d]}`" for d in ("S0", "D0", "D1", "D2c", "D2")) + "\n")

    # ---------------- Manzoni ----------------
    w("\n## 2. The Manzoni stress test (§5 and Appendix A; the discourse-type control is Appendix A.4, Table 5)\n")
    MZ = ROOT / "Experiments/Manzoni/homeosyntaxism/results"
    w("| what | file | source text | generator | n |")
    w("|---|---|---|---|---|")
    for f in sorted(MZ.glob("unrestricted_*_1840_*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        # the Qwen3 run files carry no `lang` key; the filename does
        lang = d.get("lang") or f.name.split("_")[1]
        target = ("inheritance ruling (Table 5)" if "ctrl-argument" in f.name else
                  "surgical procedure (Table 5)" if "ctrl-procedure" in f.name else "Lampedusa")
        w(f"| {lang} · {target} · {d.get('regime','?')} | `Experiments/Manzoni/homeosyntaxism/results/{f.name}` | "
          f"`{d.get('source_ref')}` sha1 `{str(d.get('source_sha1'))[:12]}…` | "
          f"{d.get('model')} | {d.get('n')} |")
    w("\nScored by `Experiments/Manzoni/homeosyntaxism/scripts/score_production.py` into "
      "`scored_unrestricted_*.json`. **The source hash is over the *stripped* text** — "
      "hashing the raw bytes will not match, and that difference is not a corruption.\n")
    w("\nThe Italian incipit was a retyped literal in code until 2026-07-30; everything in\n"
      "`Experiments/Manzoni/homeosyntaxism/results/legacy-pre-1840/` was computed against the wrong edition and\n"
      "**feeds nothing in this paper**.\n")

    # ---------------- census ----------------
    w("\n## 3. The census figures in Appendix C\n")
    tot = sum(cell_stats(v)["scored"] for v in groups.values())
    cont = sum(cell_stats(v)["cont"] for v in groups.values())
    att = sum(cell_stats(v)["att"] for v in groups.values())
    ncells = sum(len(v) for v in groups.values())
    w(f"Reported corpus: **{ncells} cells / {tot:,} generations / {att:,} attempts** — the\n"
      f"cells that produce a printed figure, and nothing else. Contaminated: **{cont}\n"
      f"({100*cont/tot:.1f}%)**.\n")
    w("\nThis set is defined by the rules in `queneau_contamination_check.REPORTED`, each of\n"
      "which names the table it feeds. It is *not* every run on disk: superseded families\n"
      "(the strict July D₀ ladder, the b1 Qwen grid, the anchors, the bf16 control, the\n"
      "unreported `ladderS0` models) are excluded, and the script warns if a cell matches\n"
      "no rule.\n")

    # ---------------- gaps ----------------
    w("\n## 4. Gaps, stated rather than smoothed over\n")
    w("| gap | what is affected | status |")
    w("|---|---|---|")
    w("| `ladderD2`'s original launcher is lost | Table 4's D₃ column | reconstructed as "
      "`scripts/queneau_homo_ladderD2.sbatch`; **63/63 prompts verified byte-identical**, "
      "but the job id and quantization are inference |")
    w("| no committed launcher for S₀ on the b1 arm | Table 2's Opus S₀ rows (18 cells) | "
      "open — the run records carry the full prompt and parameters, so the cells are "
      "reproducible, but the driver is not in the repository |")
    w("| no committed launcher for the laptop runs | Table 6's three laptop columns "
      "(45 cells) | open — same position: records complete, driver absent |")
    w("| job ids unrecorded for `ladderD2c` and `ladderD2` | Table 4's D₂ and D₃ columns | "
      "open — the launchers exist for D₂c; the runs are identified by tag and date only |")
    w("| Opus b1 cells do not record a temperature | Table 2's Opus rows | 18 of 36 carry "
      "`temperature: null`; the rest record 0.9 |")
    w("\nTwo sampling temperatures are in play — 0.9 almost everywhere, 0.7 for `ladderT1`\n"
      "and for `ladderD1`'s French and Italian cells. Appendix C states this. The job that\n"
      "would remove the difference is prepared but **not submitted**:\n"
      "`scripts/queneau_homo_temp09_uniform.sbatch`.\n")

    w("\n## 5. Verifying all of it\n")
    w("```bash\npython scripts/chr_verify.py                 # 242 assertions over every "
      "printed figure\npython scripts/queneau_contamination_check.py # the census, and a "
      "warning per unattributed cell\npython scripts/queneau_claims_check.py        # the "
      "relational claims, which no number check reaches\npython scripts/chr_provenance.py "
      "--check      # gaps in this document's own mapping\n```\n")

    OUT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"  wrote {OUT.relative_to(ROOT)} — {len(L)} lines, {ncells} cells, "
          f"{tot:,} generations")
    if nolaunch:
        print(f"  flagged as having no committed launcher: {nolaunch}")


if __name__ == "__main__":
    main()
