# Fable 5 — refusal record (Phase-0 D0 source-blind, closed-weight anchor)

*2026-07-05. The frontier closed-weight anchor `claude-fable-5` **refused** the source-blind D0
homosyntactic task on every attempt. This is recorded as a finding, not a failed run.*

## What happened

Run config: `claude-fable-5`, thinking always-on (param omitted), **no `temperature`** (removed on
Fable 5 — 400s otherwise), `max_tokens=4096`; source-blind D0 spec (full ordered dependency tree,
`[pos] rel->[head]`), English, scenarios {security-checkpoint, hospital-emergency-room,
grocery-checkout}, N=1. Backend: `scripts/run_production.py::gen_anthropic`.

All three cells returned **HTTP 200 with `stop_reason: "refusal"`**, `stop_details.category:
"cyber"` ("restrictions on violative cyber content ... blocked under Anthropic's Usage Policy").
Zero passages generated.

## Isolation probes (why it refused)

The scenario content is innocuous (a grocery checkout), so the trigger is the **form of the
specification**, not the topic. Probes on `claude-fable-5`:

| Probe | Prompt | Verdict |
|---|---|---|
| A | **1 period**, full D0 arrow spec, grocery scenario | ✅ **ACCEPTED** — produced a correct 8-word match ("In the Kroger aisle near the dairy section.") |
| B | **10 periods**, bare POS letters only (no arrows/heads) | ❌ REFUSED (cyber) |
| C | Full prompt (10 periods, prose-styled spec, any scenario) | ❌ REFUSED (cyber) |

**Conclusion:** the trigger is the **bulk density of structured token sequences** — ten lines of
dense uppercase token streams pattern-match to the cyber classifier as encoded/obfuscated payload
— *independent of notation style (arrows vs bare POS) and scenario content*. A single period passes
and the model **executes it correctly**; the whole-passage spec is rejected on format alone.

## Contrast (same prompt, same session)

- **Opus 4.8** ran the identical full prompt and **held the tree** (s0_pos 0.81, held_rate 1.00,
  coherence 0.00 — see `summary_D0.json`). One of four Opus calls hit a stochastic `bio` refusal;
  it passed on retry.
- The **open-weight ladder** (Qwen 0.5→72B, Llama-4-Maverick) ran the identical prompt with **no
  gatekeeping at all** — they just don't *execute* the spec (s0_pos flat ~0.3).

## Interpretation (for the paper)

The frontier closed-weight model's safety layer rejects the **format of the science** — a dense
structural specification — as a false-positive cyber threat, while the open-weight models accept it
freely. This is a concrete instance of the project's ethico-political argument for open-weight /
locally-hosted models in computational stylistics: closed frontier models can gate not just
*content* but the *methodological apparatus itself*. (The vendor's own remedy — configure a
fallback model — would silently re-serve the task on a different model, defeating the point of
benchmarking *this* one.)
