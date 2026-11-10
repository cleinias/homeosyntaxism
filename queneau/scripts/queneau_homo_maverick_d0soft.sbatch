#!/bin/bash
#SBATCH --job-name=queneau-mav-d0soft
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/group/p.cis260959.000/jobs/queneau-mav-d0soft-%j.out
#SBATCH --error=/scratch/group/p.cis260959.000/jobs/queneau-mav-d0soft-%j.err

#=============================================================================
# Maverick at D0 under the softened prompt — a hedge on a decision not yet made.
#
# THE SITUATION
#
#   Table 5 (tab:homosyntaxism) is built from strict-prompt D0 data. Its seven Qwen
#   rows come from the July ladder, its Maverick row from the `anchor` run, its Opus
#   row from `opus48-anchor`. Table 3's D0 column has the same origin.
#
#   Tables 3 and 4 are pulling in opposite directions. Table 4 is already SOFTENED at
#   D0 -- it has to be, since it compares a laptop with a cluster and the laptop has
#   only ever answered the softened prompt -- while Table 3's D0 column is strict and
#   its other three rungs are softened. So the paper currently reports two D0
#   instruments, and Table 3's gradient has its first rung measured differently from
#   its other three. The likely resolution is to move Tables 3 and 5 to softened.
#
#   If that happens, `ladderD0soft` covers all seven Qwen rows -- and neither of the
#   two non-Qwen rows. Opus is closed-weight and can never be re-run here. Maverick
#   can: this job produces its softened D0 so Table 5 need not carry one strict row
#   inside an otherwise softened table.
#
# WHY IT IS SUBMITTED BEFORE THE DECISION
#
#   Purely scheduling. This is the only model in the corpus needing gpu:h100:8, and
#   only ac041 and ac045 have eight; with 116 jobs pending, it is the one piece of
#   work where waiting for the decision costs days rather than minutes. It is nine
#   cells and ~50 SU. If the D0 decision goes the other way, scancel and nothing is
#   lost but queue position.
#
# MODEL
#
#   RedHatAI/Llama-4-Maverick-17B-128E-Instruct-FP8 -- the ungated mirror; the
#   meta-llama original is gated. ~400 GB FP8 across 8xH100 (TP=8), weights already
#   staged in group scratch from the anchor run. Same atlas-llm env, same
#   anti-DeepGEMM guard as the Qwen jobs.
#
#   Tag is mavD0soft. It deliberately does NOT extend `anchor` or `ladder`: the strict
#   anchor cells stay exactly as they are, because published numbers rest on them and
#   strict-vs-softened at fixed model and hardware is a comparison worth having.
#
# PRE-FLIGHT
#
#   The softened D0 prompts are already pinned: spec/b1_prompt_hashes.json holds the
#   SHA-256 of what the b1 run sent for D0 in all three languages, and those prompts
#   are model-independent (verified: no two models disagree on any of the 36). So this
#   job can prove, before generating, that it is sending Maverick exactly the
#   instruction the softened family received. Any mismatch aborts.
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
MODEL=RedHatAI/Llama-4-Maverick-17B-128E-Instruct-FP8

echo "=== job $SLURM_JOB_ID on $SLURMD_NODENAME ==="; date -Is
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo; echo "=== prompt parity: softened D0, all three languages ==="
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
    degree, lang, scenario, regime = key.split("|")
    specs = json.loads((qrh.SPEC_DIR / f"specs_{lang}_{degree}.json").read_text(encoding="utf-8"))
    got = hashlib.sha256(qrh.build_prompt(specs, lang, scenario, degree, False).encode()).hexdigest()
    ok = got == h; bad += not ok
    print(f"  {'ok ' if ok else 'MISMATCH'} {degree} {lang} {scenario}")
print(f"  {len(want)-bad}/{len(want)} prompts identical to the softened family")
sys.exit(1 if bad else 0)
PARITY

nohup vllm serve $MODEL \
  --tensor-parallel-size 8 --host 127.0.0.1 --port 8000 \
  --max-model-len 16384 --gpu-memory-utilization 0.92 --enforce-eager \
  > vllm-mavD0soft-$SLURM_JOB_ID.log 2>&1 &
VLLM_PID=$!
cleanup() { kill $VLLM_PID 2>/dev/null || true; sleep 3; kill -9 $VLLM_PID 2>/dev/null || true; }
trap cleanup EXIT

READY=0
for i in $(seq 1 720); do          # a 400 GB load is slow; allow an hour
  curl -sf http://127.0.0.1:8000/v1/models >/dev/null 2>&1 && { READY=1; echo "vllm ready after $((i*5))s"; break; }
  kill -0 $VLLM_PID 2>/dev/null || { echo "!! vllm died"; tail -60 vllm-mavD0soft-$SLURM_JOB_ID.log; exit 1; }
  sleep 5
done
[ $READY -eq 1 ] || { echo "!! vllm never became ready"; exit 1; }

for L in en fr it; do
  for S in grocery-checkout security-checkpoint hospital-emergency-room; do
    echo "--- $MODEL D0 $L $S $(date -Is)"
    python scripts/queneau_run_homo.py --backend vllm-http \
      --endpoint http://127.0.0.1:8000 --model "$MODEL" \
      --degree D0 --lang $L --scenario $S \
      --n 10 --temperature 0.7 --max-tokens 1024 \
      --tag mavD0soft || echo "!! CELL FAILED $L $S"
  done
done

echo; echo "=== done $(date -Is) ==="
ls -1 Experiments/Queneau/queneau-homo/outputs/homo_D0_*mavD0soft*.json 2>/dev/null | wc -l
echo "cells written (expected 9)"
