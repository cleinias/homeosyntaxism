#!/bin/bash
#SBATCH --job-name=queneau-homo-T1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/group/p.cis260959.000/jobs/queneau-homo-T1-%j.out
#SBATCH --error=/scratch/group/p.cis260959.000/jobs/queneau-homo-T1-%j.err

#=============================================================================
# Table 1's two Qwen models, on our own hardware, in BOTH regimes.
#
# WHAT THIS IS FOR
#
#   Every cell behind Table 1 -- the blind/source-visible dissociation, the
#   paper's headline result -- comes from hosted inference: the open models
#   through OpenRouter with the provider pinned (Qwen-7B on Together, Qwen-72B
#   and Maverick on DeepInfra, all fp8), Opus 4.8 through the Anthropic API.
#   None of it ran on the cluster the paper otherwise describes, and the paper
#   does not currently say so.
#
#   This job puts the two Qwen rows on ACES instead.
#
# WHAT IT CANNOT DO, AND WHY THE DEPENDENCY SURVIVES ANYWAY
#
#   Opus 4.8 is closed-weight: there is nothing to serve, and it is not
#   incidental to the table -- it is the frontier arm of the dissociation, the
#   calque corner. Llama-4-Maverick is open but ~400B parameters (MoE, 17B
#   active); at fp8 its weights alone exceed the 320 GB of HBM on a 4xH100 node,
#   so it would need two nodes and a ~400 GB staging download.
#
#   So this cannot make Table 1 cluster-sourced. It moves two rows of four and
#   leaves the table with THREE provenances instead of two. Whether that is an
#   improvement is a judgement about which inconsistency matters more: running
#   the open models on hardware we control, or keeping all four rows of one
#   table on a single serving precision. Submitted so the queue time is known
#   before that judgement has to be made; scancel is free.
#
# THE ONE THING THAT MAKES THE RESULT COMPARABLE
#
#   ACES has never run the source-visible regime -- all 336 ladder files are
#   blind. The runner supports it (--source-visible), so the regime is new here,
#   not the code. But a regime that has never run is exactly where a silent
#   prompt divergence would hide, and that is the failure that cost us a month
#   in July: Table 4's D0 row compared a laptop and a cluster that had been
#   given materially different instructions, and nothing detected it.
#
#   So before generating anything, this job rebuilds all 36 prompts and compares
#   each against the SHA-256 of the prompt the b1 run actually sent, stored in
#   spec/b1_prompt_hashes.json. All 36 are model-independent (verified: no two
#   models disagree on any of them), so one hash per combination is a complete
#   gate. Any mismatch aborts before a single GPU-second is spent.
#
# COST
#
#   Measured from the ladderS0 run: generation is ~1.0 s/gen at 7B and ~1.4 s/gen
#   at 72B, so 720 generations is ~12 min of actual work. The cost is model
#   loading (~3 min for 7B, ~10 min for 72B at TP=2) and the queue. Expect ~30
#   min wall on 2xH100 ~= 40 SU; --time=02:00:00 bounds the worst case at ~160.
#
# QUANTIZATION follows ladder policy exactly -- bf16 at 7B, fp8-at-load at 72B --
#   so these cells are directly comparable with the ACES cells already in Tables
#   3 and 5. NOTE the consequence for Table 1: its 7B row would move from fp8
#   (Together) to bf16, so the table would no longer be uniform in precision.
#
# Tag is ladderT1. Nothing to overwrite: no ladder run has ever produced S0 or
# D0 in the source-visible regime.
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

echo "=== job $SLURM_JOB_ID on $SLURMD_NODENAME ==="
date -Is
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# ---------------------------------------------------------------- parity gate
echo
echo "=== prompt parity against the b1 run (36 combinations) ==="
python - <<'PARITY' || { echo "!!! PROMPT PARITY FAILED — aborting before any GPU work"; exit 1; }
import importlib.util, json, hashlib, sys
from pathlib import Path
spec = importlib.util.spec_from_file_location("qrh", "scripts/queneau_run_homo.py")
qrh = importlib.util.module_from_spec(spec); spec.loader.exec_module(qrh)
want = json.loads(Path("Experiments/Queneau/queneau-homo/spec/b1_prompt_hashes.json").read_text())
bad = 0
for key, h in sorted(want.items()):
    degree, lang, scenario, regime = key.split("|")
    # loaded exactly as main() does -- a divergence in the spec files themselves is
    # precisely one of the things this gate exists to catch
    specs = json.loads((qrh.SPEC_DIR / f"specs_{lang}_{degree}.json").read_text(encoding="utf-8"))
    p = qrh.build_prompt(specs, lang, scenario, degree, regime == "vis")
    got = hashlib.sha256(p.encode()).hexdigest()
    ok = got == h
    bad += not ok
    print(f"  {'ok ' if ok else 'MISMATCH'} {degree:3} {lang} {regime:5} {scenario}")
print(f"  {len(want)-bad}/{len(want)} prompts identical to the b1 run")
sys.exit(1 if bad else 0)
PARITY

# ------------------------------------------------------------------ the sweep
LANGS="en fr it"
SCENARIOS="grocery-checkout security-checkpoint hospital-emergency-room"
DEGREES="S0 D0"
MODELS="Qwen/Qwen2.5-7B-Instruct Qwen/Qwen2.5-72B-Instruct"

for M in $MODELS; do
  TP=1; QUANT=""
  case "$M" in
    *72B*) QUANT="--quantization fp8"; TP=2 ;;
  esac
  echo; echo "############ MODEL $M (TP=$TP ${QUANT:-bf16}) $(date -Is) ############"

  nohup vllm serve "$M" \
    --tensor-parallel-size $TP --host 127.0.0.1 --port 8000 \
    --max-model-len 8192 --gpu-memory-utilization 0.90 $QUANT \
    > vllm-T1-$(basename "$M").log 2>&1 &
  VLLM_PID=$!
  cleanup() { kill $VLLM_PID 2>/dev/null || true; sleep 3; kill -9 $VLLM_PID 2>/dev/null || true; }
  trap cleanup EXIT

  READY=0
  for i in $(seq 1 360); do
    curl -sf http://127.0.0.1:8000/v1/models >/dev/null 2>&1 && { READY=1; echo "vllm ready after $((i*5))s"; break; }
    kill -0 $VLLM_PID 2>/dev/null || { echo "!! vllm died"; tail -40 vllm-T1-$(basename "$M").log; break; }
    sleep 5
  done
  [ $READY -eq 1 ] || { echo "!! skipping $M"; cleanup; trap - EXIT; continue; }

  for D in $DEGREES; do
    for L in $LANGS; do
      for S in $SCENARIOS; do
        for REG in "" "--source-visible"; do
          echo "--- $M $D $L $S ${REG:-blind} $(date -Is)"
          python scripts/queneau_run_homo.py --backend vllm-http \
            --endpoint http://127.0.0.1:8000 --model "$M" \
            --degree $D --lang $L --scenario $S $REG \
            --n 10 --temperature 0.7 --max-tokens 1024 \
            --tag ladderT1 || echo "!! CELL FAILED $M $D $L $S ${REG:-blind}"
        done
      done
    done
  done

  cleanup; trap - EXIT
  sleep 5
done

echo; echo "=== done $(date -Is) ==="
ls -1 Experiments/Queneau/queneau-homo/outputs/homo_[SD]0_*ladderT1*.json 2>/dev/null | wc -l
echo "cells written (expected 72)"
