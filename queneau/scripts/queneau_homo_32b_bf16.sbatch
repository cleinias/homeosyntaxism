#!/bin/bash
#SBATCH --job-name=queneau-32b-bf16
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/group/p.cis260959.000/jobs/queneau-32b-bf16-%j.out
#SBATCH --error=/scratch/group/p.cis260959.000/jobs/queneau-32b-bf16-%j.err

#=============================================================================
# Qwen2.5-32B at bf16, D0 — separating scale from precision at the one point
# where the ladder inverts.
#
# THE OBSERVATION
#
#   D0 coherence climbs across the scale ladder with exactly one exception: the
#   32B trails the 14B (0.63 against 0.77 softened; 0.63 against 0.70 strict, so
#   the inversion predates the prompt change). It is broad -- 32B loses in 7 of 9
#   cells across all three languages -- and two of its cells carry judge refusals,
#   so 0.63 is an UPPER bound and the real gap is wider. It is also specific to the
#   tight rungs: 32B trails at D0 and D1 but is AHEAD of the 14B at D2 (0.73 vs
#   0.69) and D3 (0.76 vs 0.74).
#
# THE CONFOUND
#
#   32B is the smallest quantized model on the ladder. 0.5B-14B run bf16; 32B and
#   72B run fp8-at-load. So the single inversion in the size series sits exactly at
#   the precision boundary, and scale and precision cannot be separated there.
#
#   We had recorded that choice as forced by memory. It was not: the 32B bf16
#   weights are 62 GB on disk and an H100 has 80, so it fits on ONE card with room
#   for a KV cache, and comfortably on two. Only the 72B (~145 GB) genuinely needs
#   quantization. The confound was avoidable.
#
# WHAT THIS RUN DECIDES
#
#   Same model, same prompt, same hardware, same degree -- bf16 instead of fp8. If
#   coherence rises to somewhere near the 14B's, the inversion is a quantization
#   artifact and the ladder's story is intact once stated carefully. If it stays at
#   ~0.63, precision is exonerated and the 32B is simply weaker where the constraint
#   is tightest, which is the more interesting result and the one the D2/D3 reversal
#   already hints at.
#
#   Either way the paper stops having to hedge a number it cannot explain.
#
#   TP=2 rather than 1: 62 GB of weights against 72 GB usable at 0.90 utilization
#   would leave ~7 GB of KV cache -- workable but tight, and an OOM costs the whole
#   queue slot. Two cards make it comfortable and still schedule quickly (only
#   ac041/ac045 have eight GPUs, but six nodes have two or more).
#
#   Tag is bf16ctrl. It deliberately does NOT begin with `ladder`: this is a control
#   run, not a ladder rung, and `ladder*` globs must not sweep it up. That trap has
#   already pooled 60 rows into a 30-row cell once.
#
#   Prompt: the SOFTENED D0, checked against the same b1 hashes the other jobs use,
#   so these cells are directly comparable with ladderD0soft.
#=============================================================================

set -uo pipefail
module load Anaconda3/2025.12-2
module load GCC/13.3.0 CUDA/13.1.0
export CUDA_HOME=${CUDA_HOME:-${EBROOTCUDA:-}}
source $(conda info --base)/etc/profile.d/conda.sh
export GROUP_SCRATCH=/scratch/group/p.cis260959.000
export HF_HOME=$GROUP_SCRATCH/huggingface
export TRANSFORMERS_CACHE=$GROUP_SCRATCH/huggingface/hub
export HF_HUB_OFFLINE=1
export VLLM_USE_DEEP_GEMM=0
export PYTHONUNBUFFERED=1
conda activate $GROUP_SCRATCH/conda_envs/atlas-llm

WORKDIR=$GROUP_SCRATCH/manzoni-mapping
cd $WORKDIR || exit 1
MODEL=Qwen/Qwen2.5-32B-Instruct

echo "=== job $SLURM_JOB_ID on $SLURMD_NODENAME ==="; date -Is
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo; echo "=== prompt parity: softened D0, three languages ==="
python - <<'PARITY' || { echo "!!! PROMPT PARITY FAILED — aborting before any GPU work"; exit 1; }
import importlib.util, json, hashlib, sys
from pathlib import Path
spec = importlib.util.spec_from_file_location("qrh", "scripts/queneau_run_homo.py")
qrh = importlib.util.module_from_spec(spec); spec.loader.exec_module(qrh)
want = {k: v for k, v in json.loads(
    Path("Experiments/Queneau/queneau-homo/spec/b1_prompt_hashes.json").read_text()).items()
    if k.startswith("D0|") and k.endswith("|blind")}
bad = 0
for key, h in sorted(want.items()):
    degree, lang, scenario, _ = key.split("|")
    specs = json.loads((qrh.SPEC_DIR / f"specs_{lang}_{degree}.json").read_text(encoding="utf-8"))
    got = hashlib.sha256(qrh.build_prompt(specs, lang, scenario, degree, False).encode()).hexdigest()
    ok = got == h; bad += not ok
    print(f"  {'ok ' if ok else 'MISMATCH'} {degree} {lang} {scenario}")
print(f"  {len(want)-bad}/{len(want)} prompts identical to the softened family")
sys.exit(1 if bad else 0)
PARITY

# --dtype bfloat16 and NO --quantization: this is the whole point of the run
nohup vllm serve $MODEL \
  --tensor-parallel-size 2 --host 127.0.0.1 --port 8000 \
  --dtype bfloat16 --max-model-len 8192 --gpu-memory-utilization 0.90 \
  > vllm-32bbf16-$SLURM_JOB_ID.log 2>&1 &
VLLM_PID=$!
cleanup() { kill $VLLM_PID 2>/dev/null || true; sleep 3; kill -9 $VLLM_PID 2>/dev/null || true; }
trap cleanup EXIT

READY=0
for i in $(seq 1 360); do
  curl -sf http://127.0.0.1:8000/v1/models >/dev/null 2>&1 && { READY=1; echo "vllm ready after $((i*5))s"; break; }
  kill -0 $VLLM_PID 2>/dev/null || { echo "!! vllm died"; tail -60 vllm-32bbf16-$SLURM_JOB_ID.log; exit 1; }
  sleep 5
done
[ $READY -eq 1 ] || { echo "!! vllm never became ready"; exit 1; }

# confirm from the serve log that it really loaded bf16 and not a quantized path --
# the RUNBOOK's serve loop quantizes everything and is not what we want here
echo; echo "=== dtype actually used ==="
grep -iE "dtype|quantization" vllm-32bbf16-$SLURM_JOB_ID.log | head -5

for L in en fr it; do
  for S in grocery-checkout security-checkpoint hospital-emergency-room; do
    echo "--- $MODEL D0 $L $S $(date -Is)"
    python scripts/queneau_run_homo.py --backend vllm-http \
      --endpoint http://127.0.0.1:8000 --model "$MODEL" \
      --degree D0 --lang $L --scenario $S \
      --n 10 --temperature 0.7 --max-tokens 1024 \
      --tag bf16ctrl || echo "!! CELL FAILED $L $S"
  done
done

echo; echo "=== done $(date -Is) ==="
ls -1 Experiments/Queneau/queneau-homo/outputs/homo_D0_*bf16ctrl*.json 2>/dev/null | wc -l
echo "cells written (expected 9)"
