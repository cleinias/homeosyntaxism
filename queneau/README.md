# The Queneau ladder — data and scripts for the CHR 2027 paper

Marchesini & Franchi, *Keeping It in Shape: Semantic Transpositions of a Form-Bound Text*,
Computational Humanities Research 2027 (Manchester). This folder holds everything behind the
paper's Queneau experiments (§4, the asymmetry of §6, Appendices C and D); the Manzoni stress
test (§5, Appendix A, the discourse-type control) lives in the repository root.

## Layout

The folder mirrors the working repository the runs were made in, so that every script runs
unchanged from here (each one resolves its paths from `scripts/..`):

| path | contents |
|---|---|
| `scripts/` | generation (`queneau_run_homo.py`), form specifications (`queneau_build_specs.py`), scoring (`queneau_score_homo.py`), the metric and judge libraries it imports (`tree_metric.py`, `sem_metrics.py`, `run_production.py`), summaries and selections (`queneau_summarize.py`, `queneau_select_table2.py`, `queneau_compile_texts.py`), the audits (`queneau_contamination_check.py`, `queneau_judge_agreement*.py`, `queneau_openweights_judge_check.py`, `queneau_aligned_judge_refusal_check.py`, `queneau_claims_check.py`), the paper checks (`chr_verify.py`, `chr_provenance.py`), and the SLURM / shell launchers |
| `Experiments/Queneau/queneau_prototype/` | the three source texts: Queneau's *Notations* (French), Barbara Wright's English and Umberto Eco's Italian |
| `Experiments/Queneau/queneau-homo/spec/` | the form specification of each degree, per language |
| `Experiments/Queneau/queneau-homo/outputs/` | every generation (`homo_*.json`, with the exact prompt sent) and every scored cell (`outputs/scored/`) |
| `Experiments/Queneau/queneau-homo/results/` | the summaries the tables are read from, and the audit results |
| `Experiments/Queneau/queneau-homo/mac/` | the minimal-compute (laptop, Ollama) runs of Table 6 |
| `Experiments/Manzoni/homeosyntaxism/` | two symlinks (`results`, `texts`) back to the repository root, where the scripts expect the Manzoni files |
| `PROVENANCE.md` | run by run: which job, temperature, quantization and launcher produced each cell, and how many generations were kept |

The library modules in `scripts/` are the versions the Queneau runs used. They are not
identical to the same-named modules in the root `scripts/`, which serve the Manzoni pipeline;
keep them apart.

## Notation: paper and code

The paper's degree names were fixed after the code was written, so two of them differ:

| paper | what it holds | Queneau code | fidelity metric (code) | paper metric |
|---|---|---|---|---|
| S₀ | the ordered part-of-speech string | `S0` | `s0_pos` | *surf-synt* |
| D₀ | the full ordered tree | `D0` | `s0_pos` | *surf-synt* |
| D₁ | order-free dependencies | `D1` | `d1_triple` | *dep-triple* |
| D₂ | clause nesting | **`D2c`** | `clause_nest` | *clause-nest* |
| D₃ | clause inventory | **`D2`** | `d2_clause` | *clause-inv* |
| D₄ | depth of subordination | — (not run on Queneau) | — | — |

Run tags (`b1`, `ladderT1`, `ladderD0soft`, `ladderD1`, `ladderD2c`, `ladderD2`,
`ladderS0`, `mac`) appear in every file name; `PROVENANCE.md` maps each to its launcher.

## Where each table comes from

`PROVENANCE.md` uses the paper's table numbers. The scripts still name the tables by their
working labels ("T1"–"T5", in `queneau_contamination_check.REPORTED`); the last column gives
them.

| paper | content | runs | working label |
|---|---|---|---|
| Table 1 | the measures | — (definitions; validation data in the root `results/metric_validation*`) | — |
| Table 2 | blind vs. source-visible at S₀ and D₀ | Opus `b1`; Qwen `ladderT1` (temperature 0.7) | T1 |
| Table 3 | four sample generations | the Table 2 cells, plus one D₃ generation | — |
| Table 4 | the Qwen 2.5 ladder, D₀–D₃ | `ladderD0soft`, `ladderD1`, `ladderD2c`, `ladderD2` | T3 |
| Table 5 | the discourse-type control (Appendix A.4): the Manzoni D₂ prompt into a surgical procedure and an inheritance ruling | root `results/scored_unrestricted_*_1840_opus5_ctrl-*_n10.json` (Opus 5); the Lampedusa rows are root `results/scored_unrestricted_*_1840_opus5_n10.json` | — |
| Table 6 | minimal compute (laptop) | `mac`, beside the cluster's 72B | T4 |
| Table 7 | blind homosyntaxism (Appendix D) | `ladderD0soft` blind; Opus and Maverick `b1`; Fable 5's refusal (`FABLE5_REFUSAL_RECORD.md`) | T5 |

Contaminated generations (the model recited the specification instead of writing prose) are
excluded from every table by one rule, `queneau_contamination_check.REPORTED`.

## Checking the numbers

```
python3 scripts/queneau_claims_check.py     # recomputes, from the scored cells, the claims the prose makes
python3 scripts/chr_verify.py               # checks every number in the paper against the data
```

`chr_verify.py` reads the paper's LyX source to find the numbers it checks. Both scripts only
read; the summary scripts (`queneau_summarize.py` and the like) rewrite files in `results/`. Regenerating text needs the API keys or a vLLM endpoint described in the root
`README.md`, and the Stanza 1.12.2 English, French and Italian models for scoring.

## Copyright

Queneau's *Notations* (Gallimard, 1947) and the Wright and Eco translations are in copyright.
The single short passage analysed here, with its parses and the generations derived from it,
is included as quotation for scholarly analysis; no complete text of any of the three books
is. Scripts are GPL-3.0; texts, specifications, generations and scores CC BY-SA 4.0 (see the
root `LICENSE.md`).
