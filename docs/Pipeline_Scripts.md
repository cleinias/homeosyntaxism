# Pipeline scripts (Phase 3 production)

Every script that ships in the public repo: the constrained‑homeosyntactic deformation
pipeline, end to end. For each: what it does · what it **consumes** · what it **produces**.
(Data‑file structures are documented separately in `Data_Files.md`.)

The rubric (the 8 patterns) is **produced by `build_rubric.py`** (Stage 0) from the human
workbook `Phase_1_Workbook_Manzoni.md`, so it is reproducible and editable post‑clone. The
downstream scripts consume the emitted `spec/patterns.json` / `constraints.json` (see
`Data_Files.md`). The transposition **domain** is separate: it is authored in
`spec/brief_<lang>.json` (see `Target_Domain_Brief.md`).

## Pipeline at a glance
```
 Phase_1_Workbook_Manzoni.md ──► build_rubric.py ──► spec/{patterns,constraints,constraints_en}.json
 spec/brief_<lang>.json (the domain) ───────────────────────────┐
 source text ───────────────────────────────────────────────────┤
                                                                ▼
 build_specs.py --lang --degree  ──►  spec/specs_<lang>_<degree>.json   (parse + render rubric + brief)
                           ▼
 run_homeo.py (+run_production.py)  ──►  results/production_<label>_*.json   (generate)
                           ▼
 score_production.py  ──►  results/production_scored_*.json              (3-axis gate)

 brief_to_tex.py  ──►  a LaTeX rendering of the brief (slides / paper)
 sem_metrics.py  ──►  metric_validation_scored_*.json   (one-time: validate the judges)
 monitor.py / *_aces.sbatch  : self-hosted serving + output retrieval (ACES)
 tree_metric.py, overlap_metrics.py : libraries (no standalone I/O)
```

---

## Stage 0 — build the rubric (reproducible & editable)

### `build_rubric.py`
The single editable source of the rubric: holds the 8 patterns + the verifiable constraints
as well‑commented Python (each item tied to a claim in `Phase_1_Workbook_Manzoni.md`), emits
the spec JSONs, and schema‑validates them (legal field values; every `constraint.pattern_id`
resolves to a pattern; etc.). Edit this (or the workbook) and re‑run to re‑derive or modify the
rubric. (Extracted from the authoring notebook; the notebook itself stays private. The
transposition **domain** is *not* here — it lives in `spec/brief_<lang>.json`.)
- **Consumes:** nothing at runtime (the rubric lives in the file); `Phase_1_Workbook_Manzoni.md` is the human reference it transcribes.
- **Produces:** `spec/patterns.json`, `spec/constraints.json`, `spec/constraints_en.json`.

## Stage 1 — build the degree specification

### `build_specs.py`
Parses the source incipit (Stanza UD) and, per period, emits the form target at the chosen
**degree** (`--degree S|D0|D1|D2|D3`); D2 carries two grains — **(a)** clause‑type inventory +
hypotaxis depth, **(c)** clause‑nesting schema. It then folds in the rubric (constraint block),
the **brief** (domain + elements from `spec/brief_<lang>.json`), and the source‑word embargo,
into one self‑contained spec. Bilingual via `--lang it|en`; **errors out if the brief's `lang`
field does not match `--lang`**.
- **Consumes:** the source text (Italian incipit embedded in `run_production.py`; English from `texts/incipit_en.txt`, which you supply); `spec/constraints.json` / `constraints_en.json`; `spec/brief_<lang>.json`.
- **Produces:** `spec/specs_<lang>_<degree>.json`.
- **Imports:** `tree_metric`, `run_production`, `overlap_metrics`.

## Stage 2 — generate

### `run_homeo.py`
Builds the constrained‑homeomorphism prompt from a `specs_<lang>_<degree>.json` file (the
degree's form target + patterns + brief + embargo + the threefold instruction) and generates N
oversampled candidates, **source‑blind**. For D2 it does one run per grain. Backends:
`vllm-http` (self‑hosted), `anthropic` (Opus), `openrouter` (open‑weight, provider/fp8‑pinned).
`--lang`, `--degree`, `--grains a,c`, `--n`, `--provider`.
- **Consumes:** `spec/specs_<lang>_<degree>.json`.
- **Produces:** `results/production_<label>_<tag>_<model>.json` (`label` = degree + grain, e.g. `d2a`).
- **Imports:** `run_production`.

### `run_production.py`
Backend library for `run_homeo`: `gen_vllm` (OpenAI‑compatible, n‑sampling), `gen_anthropic`
(omits the temperature param for opus‑4‑8), `gen_openrouter` (pins
`provider:{order,quantizations:["fp8"],allow_fallbacks:false}`, records the resolved
provider). Also embeds the **public‑domain Italian 1840 incipit** and reads API keys from a
local `.env`.
- **Consumes:** `.env` (`ANTHROPIC_API_KEY`, `OPENROUTER_API_KEY`); a vLLM endpoint if self‑hosting.
- **Produces:** nothing on its own (library); when run standalone, `results/production_<tag>_<model>.json`.

## Stage 3 — score / select

### `score_production.py`
The **three‑axis gate** = the operational definition of "constrained homeomorphism".
For every candidate computes clause‑F1 + depth_gap + s0_pos (axis 1, D₂), content‑lemma
Jaccard + frame‑divergence (axis 2), coherence (axis 3), a composite, ranks them, and
emits the best. Auto‑detects language from each file's `lang` field.
- **Consumes:** `results/production_*.json` (any degree/grain).
- **Produces:** `results/production_scored_*.json` (+ printed ranked table).
- **Imports:** `tree_metric`, `sem_metrics`.

## Rendering (optional)

### `brief_to_tex.py`
Renders `spec/brief_<lang>.json` as a self‑standing LaTeX document (the domain header + the
scene‑world elements as a bulleted brief), for slides / the paper. Compile with a
Unicode‑native engine (`lualatex` / `xelatex` — the brief carries `—`, `'`, `°`).
- **Consumes:** `spec/brief_<lang>.json`.
- **Produces:** LaTeX on stdout (or `--out FILE`).

## Metric validation (one‑time)

### `sem_metrics.py`
Builds and **validates** the two judge‑based metrics (frame‑divergence, coherence) used
by the gate, against a hand‑labelled gold set — and is the home of the judge functions
(`frame_judge_sim`, `coh_judge`; Italian/English prompts) that `score_production` calls.
- **Consumes:** `results/metric_validation.tsv`; `.env`.
- **Produces:** `results/metric_validation_scored_<judge-model>.json` (+ a separation summary).
- **Imports:** `tree_metric`.

## Libraries (no standalone I/O)

### `tree_metric.py`
Stanza UD parse (per `--lang`) + all structural metrics: clause‑type multiset & F1
(`clause`/`d2_clause`), hypotaxis depth & `depth_gap`, POS‑sequence edit similarity
(`s0_pos`), dependency‑triple overlap. The single parse from which S (ordered POS) and
the Dₙ degrees are both projected.
- **Imports:** `stanza` (downloads it/en UD models on first use).

### `overlap_metrics.py`
Light regex tokenizer + content‑word extractor, used to build the source‑word **embargo**.

## Self‑hosting (cluster; optional)

### `monitor.py`
Generic ACES job watcher: `python monitor.py <job_id> <tag> <model_slug>` — polls the
queue and pulls the two grain output files back when the job finishes.
- **Consumes:** a SLURM job id (over SSH).
- **Produces:** local copies of `results/production_<label>_<tag>_<slug>.json`.

### `qwen_aces.sbatch`, `maverick_aces.sbatch`
SLURM recipes to serve a large open‑weight model at FP8 on H100s (vLLM) and run the generator
against it. Encodes the env requirements: `atlas-llm` env, `VLLM_USE_DEEP_GEMM=0`,
Triton/Inductor/XDG caches redirected off `$HOME`, and `module load GCC + CUDA` for the
FP8 JIT toolchain. (As provided they target the research cluster's working tree; adjust paths
and the generator invocation to your own layout.)
- **Consumes:** the staged FP8 checkpoint (HF cache); the generator + `run_production.py`; `spec/specs_<lang>_<degree>.json`.
- **Produces:** `results/production_<label>_<tag>_<model>.json` (on the cluster).

---

*Not in the production path (excluded from the public set):* `homeo_prompts.py` (the bare
D₀/D₁ degree prompts from the POS‑only study) — `run_homeo` builds its own prompt for every
degree, so this is not needed.
