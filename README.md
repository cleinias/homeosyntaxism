# homeosyntax

Machinery for **constrained homeosyntactic transposition**: hold a text's syntactic
*form* at a chosen degree while inventing entirely new *content*, then measure — with an
automatic gate — which of a literary critic's formal claims actually survive the move.

The worked case is the incipit of Manzoni's *I promessi sposi* (1840), transposed into the
border/migration landscape of the central Mediterranean. It is the proof-of-concept for the
"Inverse Queneau" method (ACH 2026, Marchesini & Franchi): where Queneau holds content and
varies form, we hold form and vary content, to reveal how form naturalizes meaning.

## The two-dimensional design space

Every run is a point in a 2-D space:

- **Degree axis** — *how much* form is held. A single dependency parse is projected to
  successively coarser targets — **a partial order, not a chain**:
  - `D0` — the **full ordered dependency tree**: POS + linear order + relations. *The top.*
  - `S`  — the **surface**: the ordered part-of-speech sequence (the tree's leaves). Forgets
    the arcs. The calque / Oulipian end.
  - `D1` — the dependency relations, **order-free**. Forgets the order.
  - `D2` — the **clause architecture** (inventory and nesting). *The operative rung.*
  - `D3` — the **hypotaxis profile** only (clause count + max depth). The loosest.

  `S` and `D1` are **two independent forgetful projections of `D0`** — one drops the arcs,
  the other drops the order — so they are *incomparable*: neither is "more form" than the
  other. (Hence the warning against writing `S ⊃ D0 ⊃ D1` as a line: that would smuggle the
  *kind* of structure into the *amount* of structure and collapse the two axes into one.)
  `D2` and `D3` coarsen `D1` further. The metrics mirror the rungs: `s0_pos`↔`S`,
  `d1_triple`↔`D1`, `d2_clause`↔`D2`, `depth`↔`D3`.
- **Constraint-load axis** — *which critical claims* are additionally imposed: from `0`
  (pure homomorphism — form only) up to `full` (the eight critic-derived patterns + the
  threefold criterion = the **constrained homeomorphism**). This axis is orthogonal to the
  degree axis: you can demand any pattern set at any degree.

Corners of the space: `(D0 × 0)` ≈ a near-copy of the syntactic skeleton (iso-syntacticism);
`(S × full)` ≈ the **calque** (surface locked, patterns can't rescue coherence — see the
results); `(D2 × full)` = the **critical instrument** used here; `(S × low)` = the
Borgesian-Oulipian generator.

## The threefold criterion (the gate)

A transposition is accepted only if all three hold together:

1. **Form preserved** at the chosen degree — `clause-F1` high (for D2).
2. **Divergent** — a genuinely new scene, no lexical calque — `s0_pos`, `lemma_jacc` low,
   `frame_div` high.
3. **Coherent** — one continuous, sensible description — `coherence` high.

`score_production.py` computes all five axes per candidate and the combined SCORE.

## Pipeline

```
docs/Phase_1_Workbook_Manzoni.md      ← critic's claims, provenance (the rubric's source)
        │  build_rubric.py            ← edit + (re)generate + validate the rubric
        ▼
spec/{patterns,constraints,constraints_en,correspondence}.json
        │  build_specs.py --lang it --degree D2     ← project to a source-blind form target
        ▼
spec/specs_it_D2.json
        │  run_homeo.py --degree D2 --grains a,c --backend … --model …
        ▼
results/production_<label>_<tag>_<model>.json        ← the generations
        │  score_production.py
        ▼
results/production_scored_*.json                     ← the 3-axis gate
```

### Quickstart

```bash
pip install -r requirements.txt          # Stanza, anthropic, etc.
python -c "import stanza; stanza.download('it')"

python scripts/build_rubric.py                                  # validate/emit the rubric
python scripts/build_specs.py --lang it --degree D2            # → spec/specs_it_D2.json
python scripts/run_homeo.py --backend anthropic --model claude-opus-4-8 \
       --lang it --degree D2 --grains a,c --n 6 --tag opus48
python scripts/score_production.py                             # → results/production_scored_*.json
```

`run_homeo.py` accepts `--degree S|D0|D1|D2|D3` and three backends (`anthropic`,
`openrouter`, `vllm-http`); the same rubric drives every degree. The Opus/Italian
gradient over all five rungs is in `results/` — the two tight rungs (`D0`, `S`) drive
coherence to zero, which only recovers from `D1` down.

## Layout

| dir        | contents |
|------------|----------|
| `scripts/` | the pipeline (`build_rubric`, `build_specs`, `run_homeo`, `score_production`, the metric libraries, ACES SLURM scripts) |
| `spec/`    | the rubric (patterns/constraints/correspondence) + the built Italian form-targets `specs_it_{S,D0,D1,D2,D3}.json` |
| `texts/`   | the public-domain Italian source (Manzoni 1840) |
| `results/` | the generations + scored outputs + metric-validation data |
| `docs/`    | the rubric workbook + `Pipeline_Scripts.md`, `Data_Files.md`, `Tools_and_Stack.md` |

## Reproducibility & models

Runs span Opus 4.8 (Anthropic), Qwen3-235B (OpenRouter **and** an ACES H100 self-host, fp8),
and Kimi-K2 1T (OpenRouter, fp8). OpenRouter routing is pinned per provider with
`quantizations:["fp8"]` and `allow_fallbacks:false`, and the resolved provider is recorded
in every result file. See `MANIFEST.md` for slugs, parameters, and environment, and
`docs/Tools_and_Stack.md` for the full stack.

## A note on the English track and copyright

The Italian source is public-domain (Manzoni, 1840) and is included. The English source used
in the paper (the revised Colquhoun translation, ed. Reynolds) is **in copyright and is not
included**; the English specs and English result files are therefore omitted from this repo.
To reproduce the English track, place your own copy at `texts/incipit_en.txt` and run
`build_specs.py --lang en`.
