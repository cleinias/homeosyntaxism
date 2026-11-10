# Phase-0 re-run — homosyntactic *Notations* at D0 (paper Ch. 3, "Minimal sizes")

Re-grounds the original Queneau *Notations* proof-of-concept on the category-theoretic
form-object: the source is **locked at D0** (the full ordered dependency tree — POS + linear
order + relations, the top of the lattice), and we run **homosyntactic** generation (hold the
whole tree *identically*, move only the content into a target scenario). The question is a
capability floor: **at what model size does coherence survive the identical-structure
constraint?** Results feed **Chapter 3 ("Minimal sizes")** of the revised outline.

Metric orientation is the inverse of the D2 production line: here **high** structural overlap
is *success*; the size signal is coherence holding up as the model shrinks.

## Matrix (locked 2026-07-04)

- **Level:** D0 only (full tree), homosyntactic.
- **Languages (3):** fr (Queneau original), it (Eco), en (Wright).
- **Scenarios (3):** `security-checkpoint`, `hospital-emergency-room`, `grocery-checkout`.
- **Cells:** 3 × 3 = **9 per model**; **N = 10** candidates per cell.
- Sources → `Experiments/Queneau/queneau_prototype/{Notations_FR,Notazioni_IT,Notation_EN}.md`.
- Specs built → `spec/specs_{fr,it,en}_D0.json` (fr 10 periods, it 11, en 10). **DONE.**

## Test 1 — the size ladder (find S\*)

Single-family ladder + one cross-family open-weight ceiling, all self-hosted on ACES H100/fp8:

| Model (HF id) | GPUs |
|---|---|
| `Qwen/Qwen2.5-0.5B-Instruct` … `-1.5B` … `-3B` … `-7B` … `-14B` | 1×H100 |
| `Qwen/Qwen2.5-32B-Instruct`, `Qwen/Qwen2.5-72B-Instruct` | 1×H100 (fp8) |
| `meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8` (anchor) | 4×H100 node |

→ capability-vs-size curve; **S\*** = smallest rung that holds the tree *and* stays coherent
across all 9 cells.

## Test 2 — family robustness at S\* (conditional on Test 1)

Same 3×3×10 across ~4 *same-size, different-family* models, no anchors. E.g. if S\*≈8B:
`meta-llama/Llama-3.1-8B-Instruct`, `mistralai/Ministral-8B-Instruct-2410`,
`google/gemma-2-9b-it`, `Qwen/Qwen2.5-7B-Instruct` (within-family reference). Answers: is the
floor a property of *scale* or of *Qwen*?

## Commands

Build specs (once, local, Atlas/Stanza venv):
```
../Oulipo-Atlas-2-0/.venv/bin/python scripts/queneau_build_specs.py
```
One generation cell (on ACES, against a running vLLM server):
```
python scripts/queneau_run_homo.py --backend vllm-http --model Qwen/Qwen2.5-7B-Instruct \
    --lang it --scenario security-checkpoint --n 10 --tag qwen7b
```
Outputs → `Experiments/Queneau/queneau-homo/outputs/homo_D0_{lang}_{scenario}_{tag}_{slug}.json`.

## ACES sbatch skeleton (Qwen ladder, one job)

```bash
#!/bin/bash
#SBATCH -J queneau-homo-ladder
#SBATCH -p gpu
#SBATCH --gres=gpu:h100:1        # 4 for the Maverick anchor (separate job)
#SBATCH --ntasks=1 --cpus-per-task=16 --mem=120G
#SBATCH -t 04:00:00
module load CUDA
source $SCRATCH/atlas-llm/bin/activate
export VLLM_USE_DEEP_GEMM=0 HF_HOME=$SCRATCH/.cache/huggingface HF_HUB_OFFLINE=1
LADDER="Qwen/Qwen2.5-0.5B-Instruct Qwen/Qwen2.5-1.5B-Instruct Qwen/Qwen2.5-3B-Instruct \
        Qwen/Qwen2.5-7B-Instruct Qwen/Qwen2.5-14B-Instruct Qwen/Qwen2.5-32B-Instruct \
        Qwen/Qwen2.5-72B-Instruct"
for M in $LADDER; do
  vllm serve "$M" --quantization fp8 --port 8000 &   # tensor-parallel for 72B if needed
  VLLM_PID=$!
  # wait for the server to come up (poll /health), then:
  for L in fr it en; do
    for S in security-checkpoint hospital-emergency-room grocery-checkout; do
      python scripts/queneau_run_homo.py --backend vllm-http --model "$M" \
          --lang "$L" --scenario "$S" --n 10 --tag "${M##*/}"
    done
  done
  kill $VLLM_PID; sleep 10
done
```
Maverick anchor: same body, `--gres=gpu:h100:4`, `vllm serve … --tensor-parallel-size 4`.

## SU estimate (ACES billing ≈ 160 SU / 4×H100 node-hour ≈ 40 SU / H100-hr)

Cost is dominated by **model-load wall-time × node**, not generation count (N=10 over 9 short
~90-word cells is seconds of inference per model). Distinct models + their GPU footprint are
the driver.

| | Dominant cost | ~node-hours | ~SU |
|---|---|---|---|
| **Test 1** | 7 Qwen loads (fast) + Llama-Maverick on 4×H100 | 1.5–2.0 | **~250–320** |
| **Test 2** | ~4 small-model loads, all 1×H100 | 0.5–1.0 | **~100–160** |
| **Total** | | | **~350–480 SU** (~5–7% of the ~6,950 SU balance) |

## Scoring (Test-1 analysis, after the runs)

Parse each generation (Stanza) and compare to the target D0 tree: **s0_pos** (POS-sequence
overlap) and a tree/arc-overlap should be **high** = structure held; plus an LLM-judge
**coherence** pass. S\* = smallest rung where both clear the bar across all 9 cells. (Reuse
`scripts/tree_metric.py`; a `queneau_score_homo.py` is the next artifact.)

## Status

- [x] D0 specs built + verified (fr/it/en)
- [x] `queneau_build_specs.py`, `queneau_run_homo.py` (structure-only homosyntactic prompt, fr wired)
- [ ] `queneau_score_homo.py`
- [ ] Test 1 launched on ACES
- [ ] S\* identified → Test 2 roster fixed → Test 2 launched
