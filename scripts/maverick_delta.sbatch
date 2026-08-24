#!/bin/bash
#=============================================================================
# Phase D comparison — Llama-4-Maverick-17B-128E-Instruct (bf16) on NCSA Delta H200.
#
# Full bf16 (~803 GB) needs 8×H200 (141 GB HBM3e each = 1128 GB total).
# The bf16 reference run that the ACES FP8 run approximated.
#=============================================================================
#SBATCH --job-name=maverick-bf16
#SBATCH --account=bidl-delta-gpu
#SBATCH --partition=gpuH200x8
#SBATCH --gpus-per-node=8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --time=03:00:00
#SBATCH --output=/projects/bidl/mmarchesini/jobs/maverick-bf16-%j.out
#SBATCH --error=/projects/bidl/mmarchesini/jobs/maverick-bf16-%j.err

set -uo pipefail
module load miniforge3-python
module unload cudatoolkit 2>/dev/null || true
source $(conda info --base)/etc/profile.d/conda.sh
conda activate /scratch/bidl/mmarchesini/conda_envs/atlas-llm

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}
export HF_HOME=/work/hdd/bidl/mmarchesini/huggingface
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export TRITON_CACHE_DIR=/scratch/bidl/mmarchesini/.triton
export TORCHINDUCTOR_CACHE_DIR=/scratch/bidl/mmarchesini/.inductor
export XDG_CACHE_HOME=/scratch/bidl/mmarchesini/.cache

WORKDIR=/projects/bidl/mmarchesini/manzoni-mapping
cd $WORKDIR
MODEL=/projects/bidl/mmarchesini/Llama-4-Maverick-17B-128E-Instruct
TAG=maverick-bf16

echo "=== job $SLURM_JOB_ID on $SLURMD_NODENAME ==="; date -Is
echo "model=$MODEL  tag=$TAG  (bf16 reference, 8×H200)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

nohup vllm serve $MODEL \
  --tensor-parallel-size 8 --host 127.0.0.1 --port 8000 \
  --max-model-len 16384 --gpu-memory-utilization 0.92 --enforce-eager \
  > vllm-serve-$SLURM_JOB_ID.log 2>&1 &
VLLM_PID=$!
cleanup() { kill $VLLM_PID 2>/dev/null || true; sleep 2; kill -9 $VLLM_PID 2>/dev/null || true; }
trap cleanup EXIT

READY=0
for i in $(seq 1 1200); do
  curl -sf http://127.0.0.1:8000/v1/models > /dev/null 2>&1 && { READY=1; echo "vllm ready after $((i*5))s"; break; }
  kill -0 $VLLM_PID 2>/dev/null || { echo "vllm died"; tail -80 vllm-serve-$SLURM_JOB_ID.log; exit 1; }
  sleep 5
done
[ "$READY" -eq 1 ] || { echo "vllm not ready in 100 min"; tail -80 vllm-serve-$SLURM_JOB_ID.log; exit 1; }

mkdir -p results
for LANG in it en; do
  for DEG in D2 D3; do
    echo; echo "=== $LANG / $DEG : maverick-bf16 (constrained, 1840) ==="
    python scripts/run_homeo.py \
      --backend vllm-http --endpoint http://127.0.0.1:8000 \
      --model $MODEL --lang $LANG --degree $DEG \
      --n 10 --temperature 0.9 --max-tokens 2048 \
      --domain specs/domain_brief_3item.txt \
      --tag $TAG || echo "RUN FAILED ($LANG/$DEG)"
  done
done

echo; echo "[sbatch] done"; date -Is
ls -la results/production_D*_${TAG}_*.json 2>/dev/null
