#!/bin/bash
#=============================================================================
# Phase D comparison — Llama-4-Maverick-17B-128E-Instruct-FP8 on ACES, self-hosted.
#
# Cross-family open-weight datapoint for the constrained-homeosyntactic D2 run
# (grains a + c), to set beside Qwen3-235B. ~400 GB FP8 -> 8xH100 (TP=8), single
# node (only ac041/ac045 have gpu:h100:8, so this will queue). Likely lands AFTER
# the talk; kept as a datapoint for later work.
#
# Model = RedHatAI mirror (ungated; meta-llama original is gated, Cleinias not on
# Meta's allow-list). Same atlas-llm env (vLLM 0.21.0, Llama4-capable), same
# anti-DeepGEMM guard as the Qwen job.
#=============================================================================
#SBATCH --job-name=maverick-prod
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=01:30:00
#SBATCH --output=/scratch/group/p.cis260959.000/jobs/maverick-prod-%j.out
#SBATCH --error=/scratch/group/p.cis260959.000/jobs/maverick-prod-%j.err

set -uo pipefail
module load Anaconda3/2025.12-2
module load GCC/13.3.0 CUDA/13.1.0   # g++ (cc1plus) + nvcc for the flashinfer/FP8 JIT (torch cu130; node is bare)
export CUDA_HOME=${CUDA_HOME:-${EBROOTCUDA:-}}
source $(conda info --base)/etc/profile.d/conda.sh
export GROUP_SCRATCH=/scratch/group/p.cis260959.000
export HF_HOME=$GROUP_SCRATCH/huggingface
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_DEEP_GEMM=0   # block-scaled FP8 via Triton, not DeepGEMM (deep_gemm not installed)
export TRITON_CACHE_DIR=$GROUP_SCRATCH/.triton          # keep kernel-compile caches OFF $HOME
export TORCHINDUCTOR_CACHE_DIR=$GROUP_SCRATCH/.inductor # ($HOME 10GB quota; Triton compile filled it)
export XDG_CACHE_HOME=$GROUP_SCRATCH/.cache
conda activate $GROUP_SCRATCH/conda_envs/atlas-llm   # vllm 0.21.0 + torch 2.11.0

WORKDIR=$GROUP_SCRATCH/manzoni-mapping
cd $WORKDIR
MODEL=RedHatAI/Llama-4-Maverick-17B-128E-Instruct-FP8

echo "=== job $SLURM_JOB_ID on $SLURMD_NODENAME ==="; date -Is
python -c "import vllm,torch;print('vllm',vllm.__version__,'torch',torch.__version__,'cuda',torch.version.cuda)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

nohup vllm serve $MODEL \
  --tensor-parallel-size 8 --host 127.0.0.1 --port 8000 \
  --max-model-len 16384 --gpu-memory-utilization 0.92 --enforce-eager \
  > vllm-serve-$SLURM_JOB_ID.log 2>&1 &
VLLM_PID=$!
cleanup() { kill $VLLM_PID 2>/dev/null || true; sleep 2; kill -9 $VLLM_PID 2>/dev/null || true; }
trap cleanup EXIT

READY=0
for i in $(seq 1 600); do
  curl -sf http://127.0.0.1:8000/v1/models > /dev/null 2>&1 && { READY=1; echo "vllm ready after $((i*5))s"; break; }
  kill -0 $VLLM_PID 2>/dev/null || { echo "vllm died"; tail -80 vllm-serve-$SLURM_JOB_ID.log; exit 1; }
  sleep 5
done
[ "$READY" -eq 1 ] || { echo "vllm not ready in 50 min"; tail -80 vllm-serve-$SLURM_JOB_ID.log; exit 1; }

echo; echo "=== generate: CONSTRAINED homeosyntactic D2 (grains a + c), border/migration ==="
python scripts/run_d2.py \
  --backend vllm-http --endpoint http://127.0.0.1:8000 \
  --model $MODEL \
  --grains a,c --n 12 --temperature 0.9 --max-tokens 2048 \
  --tag maverick-aces \
  || echo "RUN FAILED"

echo "[sbatch] done"; date -Is
