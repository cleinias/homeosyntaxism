# Data files (Phase 3 production + rubric)

The JSON artifacts of the pipeline, as data structures — kept distinct from the scripts
(`Pipeline_Scripts.md`) on purpose, though the producer/consumer links are noted. Three
tiers: the **rubric** (emitted by `build_rubric.py` from the workbook), the **spec** (built), the **results** (generated/scored).

Lineage in one line:
`Phase_1_Workbook_Manzoni.md` → `build_rubric.py` → `patterns.json` + `constraints.json`;
plus the authored `brief_<lang>.json` (the domain) → `build_specs.py` →
`specs_<lang>_<degree>.json` → `production_*.json` → `production_scored_*.json`;
and `metric_validation*` validates the gate.

---

## Tier 1 — the rubric (emitted by `build_rubric.py` from the workbook; full & editable)

The rubric lives in `build_rubric.py` (one editable Python source, each item tied to a claim in
`Phase_1_Workbook_Manzoni.md`) and is emitted to the JSONs below; edit + re‑run to change it.

### `spec/patterns.json`
The full rubric: the 8 patterns that operationalize the critic's claims, with the interpretive
layer (published in full — the rubric must be reproducible & editable post‑clone).
- **By/for:** produced by `build_rubric.py` (transcribed from the workbook); the *source of* `constraints.json`.
- **Structure:** `{source, rubric_version, schema_version, notes, field_vocabularies, patterns:[…]}` where each pattern = `{id (P_1…P_8), name, form, effect_claim, mapping_prediction:{autonomous, with_content, hollow}, scope (clause|paragraph|work-level), generation_task, source_tag (Critic-explicit|near-explicit|My inference), paratext_required}`. The `mapping_prediction` triple is what the Phase‑4 human audit classifies each generation against.

### `spec/constraints.json`, `spec/constraints_en.json`
The **operational** subset of the rubric: each pattern as a single mechanically‑checkable
requirement. This is what actually enters the prompt (as the "constraint block").
- **By/for:** produced by `build_rubric.py`; **consumed by** `build_specs.py`.
- **Structure:** `{meta, constraints:[{id (C-P1-01…), pattern_id, property, specification, priority (MUST|SHOULD)}]}`. The `_en` file is the English derivative (P_2 re‑derived as iambic pentameter; P_7 formula = "swaying").

### `spec/brief_it.json`, `spec/brief_en.json`  — the transposition domain
The **only** place the target domain is set: a source-blind scene-world the generator deploys
in its own words while holding the form. (Authored directly; the rich `source→target`
correspondence reasoning lives in the private research repo — only its target column survives
here, as this brief. See `Target_Domain_Brief.md`.)
- **By/for:** authored by hand; **consumed by** `build_specs.py` (→ the prompt's `TARGET DOMAIN`).
- **Structure:** `{lang (it|en), domain (string → the prompt header), elements:[string,…]}`. `build_specs.py` exits with an error if `lang` ≠ the requested `--lang`.

## Tier 2 — the spec (built from rubric + brief + source)

### `spec/specs_<lang>_<degree>.json`  (degree ∈ {S, D0, D1, D2, D3})
The self‑contained generation spec for one degree: the form target per period, plus the
rendered rubric, the brief, the domain, and the embargo. The single file `run_homeo.py`
needs (no Stanza at gen time). Each period carries one `spec` for the file's degree (D2 =
nesting, D3 = inventory, D4 = depth profile; pre-2026-07-30 D2 files carried both
`inventory` and `schema` keys for the two grains,
c = schema); other degrees carry one `spec` string per period.
- **By/for:** produced by `build_specs.py`; **consumed by** `run_homeo.py`.
- **Structure:** `{lang, degree, source, n_periods, periods:[{k, words, …degree-specific…}], embargo:[…content words…], constraint_block (string), brief (string), domain (string)}`.
- **Note:** `specs_en_*.json` embeds the (copyright) English source in `source` → **not** committed; regenerated from a local copy of the English incipit.

## Tier 3 — the results (generated, then scored)

### `results/production_<label>_<tag>_<model>.json`  (label e.g. `d2a`, `d2c`, `S`, `D0`…)
One file per (degree, run): the N oversampled candidates with full provenance.
- **By/for:** produced by `run_homeo.py`; **consumed by** `score_production.py`.
- **Structure:** `{model, backend, degree, lang, n, temperature, max_tokens, provider_routing (resolved OpenRouter provider + quant, or null), source, prompt, elapsed_s, gens:[{i, text}]}`. Files written before 2026-07-30 carry `grain` (c = nesting, now D2; a = inventory, now D3) in place of `degree`.
- **Tags seen:** `opus48` / `opus48-en` (Opus), `qwen-or` / `qwen-or-en` (Qwen OpenRouter), `kimi-or` / `kimi-or-en` (Kimi‑K2 OpenRouter), `qwen3-aces` (Qwen self‑hosted), `maverick-aces`. EN files embed the copyright source → gitignored.

### `results/production_scored_d2{a,c}_<tag>_<model>.json`
The gate output: every candidate scored on all three axes, ranked, with text retained.
- **By/for:** produced by `score_production.py`.
- **Structure:** `{model, source, rows:[{i, words, d2_clause, depth_gap, s0_pos, lemma_jacc, frame_div, coherence, score (composite), text}]}` (rows sorted by `score`).

## Validation (one‑time, gate calibration)

### `results/metric_validation.tsv`  *(input; TSV, not JSON)*
Hand‑labelled (source, candidate) pairs spanning frame‑calque / divergent / incoherent,
with gold labels — the substrate that earns trust in the two LLM‑judge metrics.
- **Structure (columns):** `id, frame (calque|divergent), coh (yes|no), src, cand`.

### `metric_validation_scored_<judge-model>.json`
The pairs re‑scored by each metric, to show the judge separates the labels (and the
structural frame metric does not).
- **By/for:** produced by `sem_metrics.py`.
- **Structure:** `[{id, frame, coh, src, cand, f_struct, f_judge, coh_jdg, (f_mpnet if --mpnet)}]`; one file per judge (`claude-sonnet-4-6`, `claude-opus-4-8`).

---

*Phase‑2 development JSONs* (`d2_batch_*`, `d2_compare.json`, `score_axes23.json`,
`far_*`) belong to the gold‑corpus / degree‑ladder study, **not** to Phase‑3 production —
excluded from the public repo.
