#!/bin/bash
#=============================================================================
# Queneau *Notations* --- grain-c (D2c) 72B rung ONLY, on 2xH100 (tensor-parallel).
# The single-GPU online-fp8 load OOM'd (72B loads bf16 to quantize -> ~144 GB);
# tensor-parallel-size 2 splits it across 2x80 GB. Fills the missing D2 (nesting)
# 72B cell in the gradient table. Blind, 3 langs x 3 scenarios, n=10.
#=============================================================================
#SBATCH --job-name=queneau-d2c-72b
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=01:30:00
#SBATCH --output=/scratch/group/p.cis260959.000/jobs/queneau-d2c-72b-%j.out
#SBATCH --error=/scratch/group/p.cis260959.000/jobs/queneau-d2c-72b-%j.err

set -uo pipefail
module load Anaconda3/2025.12-2
module load GCC/13.3.0 CUDA/13.1.0
export CUDA_HOME=${CUDA_HOME:-${EBROOTCUDA:-}}
source $(conda info --base)/etc/profile.d/conda.sh
export GROUP_SCRATCH=/scratch/group/p.cis260959.000
export HF_HOME=$GROUP_SCRATCH/huggingface
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_DEEP_GEMM=0
export TRITON_CACHE_DIR=$GROUP_SCRATCH/.triton
export TORCHINDUCTOR_CACHE_DIR=$GROUP_SCRATCH/.inductor
export XDG_CACHE_HOME=$GROUP_SCRATCH/.cache
conda activate $GROUP_SCRATCH/conda_envs/atlas-llm

WORKDIR=$GROUP_SCRATCH/manzoni-mapping
cd $WORKDIR
M=Qwen/Qwen2.5-72B-Instruct
LANGS="fr it en"
SCEN="security-checkpoint hospital-emergency-room grocery-checkout"

echo "=== job $SLURM_JOB_ID on $SLURMD_NODENAME ==="; date -Is
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

nohup vllm serve "$M" --quantization fp8 --tensor-parallel-size 2 \
  --host 127.0.0.1 --port 8000 --max-model-len 8192 --gpu-memory-utilization 0.90 --enforce-eager \
  > vllm-d2c-72b-$SLURM_JOB_ID.log 2>&1 &
VLLM_PID=$!
READY=0
for i in $(seq 1 360); do
  curl -sf http://127.0.0.1:8000/v1/models >/dev/null 2>&1 && { READY=1; echo "ready after $((i*5))s"; break; }
  kill -0 $VLLM_PID 2>/dev/null || { echo "vllm DIED"; tail -80 vllm-d2c-72b-$SLURM_JOB_ID.log; exit 1; }
  sleep 5
done
[ "$READY" -eq 1 ] || { echo "vllm not ready in 30 min"; tail -80 vllm-d2c-72b-$SLURM_JOB_ID.log; exit 1; }

for L in $LANGS; do
  for S in $SCEN; do
    python scripts/queneau_run_homo.py \
      --backend vllm-http --endpoint http://127.0.0.1:8000 \
      --model "$M" --lang "$L" --degree D2c --scenario "$S" --n 10 \
      --temperature 0.9 --max-tokens 1024 --tag ladderD2c \
      || echo "CELL FAILED: $L $S"
  done
done
kill "$VLLM_PID" 2>/dev/null || true; sleep 8; kill -9 "$VLLM_PID" 2>/dev/null || true
echo "[sbatch] done"; date -Is
