# Data files (Phase 3 production + rubric)

The JSON artifacts of the pipeline, as data structures — kept distinct from the scripts
(`Pipeline_Scripts.md`) on purpose, though the producer/consumer links are noted. Three
tiers: the **rubric** (emitted by `build_rubric.py` from the workbook), the **spec** (built), the **results** (generated/scored).

Lineage in one line:
`Phase_1_Workbook_Manzoni.md` → `build_rubric.py` → `patterns.json` + `constraints.json` +
`correspondence.json` → `d2_specs*.json` → `production_*.json` → `production_scored_*.json`;
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
- **By/for:** produced by `build_rubric.py`; **consumed by** `build_d2_specs.py`.
- **Structure:** `{meta, constraints:[{id (C-P1-01…), pattern_id, property, specification, priority (MUST|SHOULD)}]}`. The `_en` file is the English derivative (P_2 re‑derived as iambic pentameter; P_7 formula = "swaying").

### `spec/correspondence.json`
The source→target domain map (Manzoni → border/migration), incl. the `justification` for each
mapping (published in full — editable rubric).
- **By/for:** produced by `build_rubric.py`; **consumed by** `build_d2_specs.py` (→ the functional brief).
- **Structure:** `{meta, correspondence_table:[{source_element, source_function, target_element, justification}]}`.

## Tier 2 — the spec (built from rubric + source)

### `spec/d2_specs.json` (IT), `spec/d2_specs_en.json` (EN)
The self‑contained generation spec: the D₂ target per period at both grains, plus the
rendered rubric, brief, and embargo. The single file `run_d2` needs (no Stanza at gen time).
- **By/for:** produced by `build_d2_specs.py`; **consumed by** `run_d2.py`.
- **Structure:** `{lang, source, n_periods, periods:[{k, words, inventory (grain a: clause‑type counts + depth, prose), schema (grain c: nesting tree, prose), n_clauses, depth}], embargo:[…content words…], constraint_block (string), brief (string)}`.
- **Note:** `d2_specs_en.json` embeds the (copyright) English source in `source` → **not** committed; regenerated from a local copy.

## Tier 3 — the results (generated, then scored)

### `results/production_d2{a,c}_<tag>_<model>.json`
One file per (grain, run): the N oversampled candidates with full provenance.
- **By/for:** produced by `run_d2.py`; **consumed by** `score_production.py`.
- **Structure:** `{model, backend, grain (a|c), lang, n, temperature, max_tokens, provider_routing (resolved OpenRouter provider + quant, or null), source, prompt, elapsed_s, gens:[{i, text}]}`.
- **Tags seen:** `opus48` / `opus48-en` (Opus), `qwen-or` / `qwen-or-en` (Qwen OpenRouter), `kimi-or` / `kimi-or-en` (Kimi‑K2 OpenRouter), `qwen3-aces` (Qwen self‑hosted), `maverick-aces`. EN files embed the copyright source → gitignored.

### `results/production_scored_d2{a,c}_<tag>_<model>.json`
The gate output: every candidate scored on all three axes, ranked, with text retained.
- **By/for:** produced by `score_production.py`.
- **Structure:** `{model, source, rows:[{i, words, d2_clause, depth_gap, s0_pos, lemma_jacc, frame_div, coherence, score (composite), text}]}` (rows sorted by `score`).

## Validation (one‑time, gate calibration)

### `reproducibility/graded-corpus/metric_validation.tsv`  *(input; TSV, not JSON)*
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
