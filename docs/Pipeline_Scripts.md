# Pipeline scripts (Phase 3 production)

Every script that ships in the public repo: the constrained‑homeosyntactic deformation
pipeline, end to end. For each: what it does · what it **consumes** · what it **produces**.
(Data‑file structures are documented separately in `Data_Files.md`.)

The rubric (the 8 patterns) is **produced by `build_rubric.py`** (Stage 0) from the human
workbook `Phase_1_Workbook_Manzoni.md`, so it is reproducible and editable post‑clone. The
downstream scripts consume the emitted `spec/patterns.json` / `constraints.json` /
`correspondence.json` (see `Data_Files.md`).

## Pipeline at a glance
```
 Phase_1_Workbook_Manzoni.md ──► build_rubric.py ──► spec/{patterns,constraints,correspondence}.json
 source text ──────────────────────────────────────────────────┐
                                                                ▼
 build_d2_specs.py  ──►  spec/d2_specs{,_en}.json      (parse + render rubric)
                           ▼
 run_d2.py (+run_production.py)  ──►  results/production_d2{a,c}_*.json   (generate)
                           ▼
 score_production.py  ──►  results/production_scored_*.json              (3-axis gate)

 sem_metrics.py  ──►  metric_validation_scored_*.json   (one-time: validate the judges)
 monitor.py / *_aces.sbatch  : self-hosted serving + output retrieval (ACES)
 tree_metric.py, overlap_metrics.py : libraries (no standalone I/O)
```

---

## Stage 0 — build the rubric (reproducible & editable)

### `build_rubric.py`
The single editable source of the rubric: holds the 8 patterns + the verifiable constraints +
the source→target correspondence as well‑commented Python (each item tied to a claim in
`Phase_1_Workbook_Manzoni.md`), emits the three spec JSONs, and schema‑validates them (legal
field values; every `constraint.pattern_id` resolves to a pattern; etc.). Edit this (or the
workbook) and re‑run to re‑derive or modify the rubric. (Extracted from the authoring notebook;
the notebook itself stays private.)
- **Consumes:** nothing at runtime (the rubric lives in the file); `Phase_1_Workbook_Manzoni.md` is the human reference it transcribes.
- **Produces:** `spec/patterns.json`, `spec/constraints.json`, `spec/constraints_en.json`, `spec/correspondence.json`.

## Stage 1 — build the D₂ specification

### `build_d2_specs.py`
Parses the source incipit (Stanza UD) and, per period, extracts the D₂ clause
architecture at two grains — **(a)** clause‑type inventory + hypotaxis depth, **(c)**
clause‑nesting schema — then renders the rubric (constraints + functional brief) and the
source‑word embargo into one self‑contained spec. Bilingual via `--lang it|en`.
- **Consumes:** the source text (Italian incipit embedded in `run_production.py`; English from `Texts/Manzoni/incipit_en_*.txt`); `spec/constraints.json` / `constraints_en.json`; `spec/correspondence.json`.
- **Produces:** `spec/d2_specs.json` (it) or `spec/d2_specs_en.json` (en).
- **Imports:** `tree_metric`, `run_production`, `overlap_metrics`.

## Stage 2 — generate

### `run_d2.py`
Builds the constrained‑homeomorphism prompt from a `d2_specs` file (D₂ target +
patterns + brief + embargo + the threefold instruction) and generates N oversampled
candidates per grain, **source‑blind**. Backends: `vllm-http` (self‑hosted), `anthropic`
(Opus), `openrouter` (open‑weight, provider/fp8‑pinned). `--lang`, `--grains a,c`,
`--n`, `--provider`.
- **Consumes:** `spec/d2_specs{,_en}.json`.
- **Produces:** `results/production_d2{a,c}_<tag>_<model>.json`.
- **Imports:** `run_production`.

### `run_production.py`
Backend library for `run_d2`: `gen_vllm` (OpenAI‑compatible, n‑sampling), `gen_anthropic`
(omits the temperature param for opus‑4‑8), `gen_openrouter` (pins
`provider:{order,quantizations:["fp8"],allow_fallbacks:false}`, records the resolved
provider). Also embeds the **public‑domain Italian 1840 incipit** and the border/migration
domain brief, and reads API keys from a local `.env`.
- **Consumes:** `.env` (`ANTHROPIC_API_KEY`, `OPENROUTER_API_KEY`); a vLLM endpoint if self‑hosting.
- **Produces:** nothing on its own (library); when run standalone, `results/production_<tag>_<model>.json`.

## Stage 3 — score / select

### `score_production.py`
The **three‑axis gate** = the operational definition of "constrained homeomorphism".
For every candidate computes clause‑F1 + depth_gap + s0_pos (axis 1, D₂), content‑lemma
Jaccard + frame‑divergence (axis 2), coherence (axis 3), a composite, ranks them, and
emits the best. Auto‑detects language from each file's `lang` field.
- **Consumes:** `results/production_d2{a,c}_*.json`.
- **Produces:** `results/production_scored_d2{a,c}_*.json` (+ printed ranked table).
- **Imports:** `tree_metric`, `sem_metrics`.

## Metric validation (one‑time)

### `sem_metrics.py`
Builds and **validates** the two judge‑based metrics (frame‑divergence, coherence) used
by the gate, against a hand‑labelled gold set — and is the home of the judge functions
(`frame_judge_sim`, `coh_judge`; Italian/English prompts) that `score_production` calls.
- **Consumes:** `reproducibility/graded-corpus/metric_validation.tsv`; `.env`.
- **Produces:** `metric_validation_scored_<judge-model>.json` (+ a separation summary).
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
- **Produces:** local copies of `results/production_d2{a,c}_<tag>_<slug>.json`.

### `qwen_aces.sbatch`, `maverick_aces.sbatch`
SLURM recipes to serve a large open‑weight model at FP8 on H100s (vLLM) and run `run_d2`
against it. Encodes the env requirements: `atlas-llm` env, `VLLM_USE_DEEP_GEMM=0`,
Triton/Inductor/XDG caches redirected off `$HOME`, and `module load GCC + CUDA` for the
FP8 JIT toolchain.
- **Consumes:** the staged FP8 checkpoint (HF cache); `run_d2.py`, `run_production.py`, `spec/d2_specs.json`.
- **Produces:** `results/production_d2{a,c}_<tag>_<model>.json` (on the cluster).

---

*Not in the production path (excluded from the public set):* `homeo_prompts.py` (the bare
D₀/D₁ degree prompts from the POS‑only study) — `run_d2` builds its own prompt, so this is
kept only if the public repo later exposes the full Dₙ degree ladder as a knob.
