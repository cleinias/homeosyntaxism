# MANIFEST

What this repository contains, the runs it records, and how to reproduce them.

## Models & providers

| label                 | model id                                   | backend / provider        | precision |
|-----------------------|--------------------------------------------|---------------------------|-----------|
| Opus 4.8              | `claude-opus-4-8`                          | Anthropic API             | —         |
| Qwen3-235B (OR)       | `qwen/qwen3-235b-a22b-2507`                | OpenRouter → DeepInfra    | fp8       |
| Qwen3-235B (ACES)     | `Qwen/Qwen3-235B-A22B-Instruct-2507-FP8`   | self-host, ACES H100, vLLM| fp8       |
| Kimi-K2 1T            | `moonshotai/kimi-k2`                       | OpenRouter → Novita       | fp8       |

DeepSeek-V4-pro was tried and **dropped**: as a reasoning model it spent its whole token
budget on the hidden trace and returned empty content (the result file is retained as a
negative finding).

## Generation parameters

`n = 6` candidates per cell · `temperature = 0.9` · `max_tokens = 2048`. Each run holds
degree **D2** (clause nesting) and **D3** (clause inventory) — plus the
eight patterns and the threefold criterion. Source-blind: the model never sees the source
text, only the projected form target + the rubric + the lexical embargo.

OpenRouter reproducibility: provider is pinned with
`provider:{order:[<slug>], quantizations:["fp8"], allow_fallbacks:false}`; the resolved
provider is written into each result file under `provider_routing`.

## Runs recorded (Italian)

For each model × degree: `results/production_<degree>_<tag>_<model>.json` (raw) and
`results/production_scored_d2{a,c}_<tag>_<model>.json` (gated). Tags: `opus48`, `qwen-or`,
`qwen3-aces`, `kimi-or`.

Findings in brief: Opus holds D2 (clause-F1 ~0.82–0.89) **and** stays coherent; Qwen-235B
holds it loosely (~0.67–0.70) but coherent; Kimi-K2 closes the *form* gap by
over-subordination and its **coherence collapses** — the gate catches this. ACES and
OpenRouter Qwen (both fp8) agree on the structural axes to ~0.003.

## Metric validation

`results/metric_validation.tsv` + `results/metric_validation_scored*.json` — the gate scored
against a hand-built reference set, to show the axes measure what they claim.

## Environment

- Parsing: **Stanza** (UD; Italian model). The degree metrics in `tree_metric.py` mirror the
  spec degrees: `s0_pos`↔`S`/`D0`, `d1_triple`↔`D1`, `d2_clause`↔`D2`, `depth`↔`D3`.
- Self-hosting FP8-MoE on ACES (vLLM): the SLURM scripts set `VLLM_USE_DEEP_GEMM=0`, redirect
  Triton/Inductor/XDG caches off `$HOME`, and `module load GCC/13.3.0 CUDA/13.1.0` so the
  FP8 JIT finds `nvcc` + `cc1plus`. See `scripts/qwen_aces.sbatch`.
- Secrets (`ANTHROPIC_API_KEY`, `OPENROUTER_API_KEY`) are read from a local `.env`
  (git-ignored); none are committed.

## Excluded (copyright)

The English source (revised Colquhoun, ed. Reynolds) and everything derived from it — the
English specs (`specs_en_*`) and the English result files (`*-en_*`) — are omitted. See the
README for how to reproduce the English track from your own copy.

See `docs/Pipeline_Scripts.md` and `docs/Data_Files.md` for the per-file inventory.
