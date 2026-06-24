#!/bin/bash
#=============================================================================
# Phase D production — Qwen3-235B-A22B-Instruct-2507-FP8 on ACES, self-hosted.
#
# Whole-passage D2 (homeosyntactic) deformation of the Manzoni incipit into the
# border/migration domain, oversample N=16. The atlas-llm-cu121 env (vLLM 0.6.6)
# predates Qwen3; the atlas-llm env has vLLM 0.21.0 + torch 2.11.0 (Qwen3-capable),
# so this uses atlas-llm. FP8 (235 GB) fits 4xH100 (precedent: Mistral-Large
# 123B bf16 = 246 GB ran on the same 4xH100 recipe).
#
# Scoring happens OFF-cluster (judges need the Anthropic API; compute nodes are
# offline): pull production_*.json, run scripts/score_production.py.
#=============================================================================
#SBATCH --job-name=qwen3-prod
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/group/p.cis260959.000/jobs/qwen3-prod-%j.out
#SBATCH --error=/scratch/group/p.cis260959.000/jobs/qwen3-prod-%j.err

set -uo pipefail
module load Anaconda3/2025.12-2
module load GCC/13.3.0 CUDA/13.1.0   # g++ (cc1plus) + nvcc for the flashinfer/FP8-MoE JIT (torch cu130; node is bare)
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
conda activate $GROUP_SCRATCH/conda_envs/atlas-llm   # vllm 0.21.0 + torch 2.11.0 (Qwen3-capable)

WORKDIR=$GROUP_SCRATCH/manzoni-mapping
cd $WORKDIR
MODEL=Qwen/Qwen3-235B-A22B-Instruct-2507-FP8

echo "=== job $SLURM_JOB_ID on $SLURMD_NODENAME ==="; date -Is
python -c "import vllm,torch;print('vllm',vllm.__version__,'torch',torch.__version__,'cuda',torch.version.cuda)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

nohup vllm serve $MODEL \
  --tensor-parallel-size 4 --host 127.0.0.1 --port 8000 \
  --max-model-len 16384 --gpu-memory-utilization 0.92 --enforce-eager \
  > vllm-serve-$SLURM_JOB_ID.log 2>&1 &
VLLM_PID=$!
cleanup() { kill $VLLM_PID 2>/dev/null || true; sleep 2; kill -9 $VLLM_PID 2>/dev/null || true; }
trap cleanup EXIT

READY=0
for i in $(seq 1 480); do
  curl -sf http://127.0.0.1:8000/v1/models > /dev/null 2>&1 && { READY=1; echo "vllm ready after $((i*5))s"; break; }
  kill -0 $VLLM_PID 2>/dev/null || { echo "vllm died"; tail -80 vllm-serve-$SLURM_JOB_ID.log; exit 1; }
  sleep 5
done
[ "$READY" -eq 1 ] || { echo "vllm not ready in 40 min"; tail -80 vllm-serve-$SLURM_JOB_ID.log; exit 1; }

echo; echo "=== generate: CONSTRAINED homeosyntactic D2 (grains a + c), border/migration ==="
python scripts/run_homeo.py \
  --backend vllm-http --endpoint http://127.0.0.1:8000 \
  --model $MODEL \
  --lang it --degree D2 --grains a,c --n 12 --temperature 0.9 --max-tokens 2048 \
  --tag qwen3-aces \
  || echo "RUN FAILED"

echo "[sbatch] done"; date -Is
