#!/bin/bash
#=============================================================================
# Unconstrained (pattern-free) homeosyntactic transposition of the Manzoni
# incipit, self-hosted on ACES.  Re-run of the dom3 arm on the CORRECTED 1840
# source: every earlier run took its clause targets from a retyped literal that
# was not the quarantina (periods 8 and 9 wrong).  Same three-item domain brief,
# same threefold criterion, n=10 per language.
#
# Scoring happens off-cluster (the judges need the Anthropic API).
#=============================================================================
#SBATCH --job-name=manzoni-1840
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=01:30:00
#SBATCH --output=/scratch/group/p.cis260959.000/jobs/manzoni-1840-%j.out
#SBATCH --error=/scratch/group/p.cis260959.000/jobs/manzoni-1840-%j.err

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
MODEL=${MODEL:-Qwen/Qwen3-235B-A22B-Instruct-2507-FP8}
TP=${TP:-4}
N=${N:-10}
SHORT=$(basename $MODEL)

echo "=== job $SLURM_JOB_ID on $SLURMD_NODENAME ==="; date -Is
echo "model=$MODEL tp=$TP n=$N  (1840 source)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

nohup vllm serve $MODEL \
  --tensor-parallel-size $TP --host 127.0.0.1 --port 8000 \
  --max-model-len 16384 --gpu-memory-utilization 0.92 --enforce-eager \
  > vllm-1840-$SLURM_JOB_ID-$SHORT.log 2>&1 &
VLLM_PID=$!
cleanup() { kill $VLLM_PID 2>/dev/null || true; sleep 2; kill -9 $VLLM_PID 2>/dev/null || true; }
trap cleanup EXIT

READY=0
for i in $(seq 1 600); do
  curl -sf http://127.0.0.1:8000/v1/models > /dev/null 2>&1 && { READY=1; echo "vllm ready after $((i*5))s"; break; }
  kill -0 $VLLM_PID 2>/dev/null || { echo "vllm died"; tail -80 vllm-1840-$SLURM_JOB_ID-$SHORT.log; exit 1; }
  sleep 5
done
[ "$READY" -eq 1 ] || { echo "vllm not ready in 50 min"; tail -80 vllm-1840-$SLURM_JOB_ID-$SHORT.log; exit 1; }

mkdir -p results
for LANG in it en; do
  SRC=results/production_D2_${LANG}_pilot-1840_claude-opus-5.json
  OUT=results/unrestricted_${LANG}_1840_${SHORT}_n${N}.json
  echo; echo "=== $LANG : $SHORT (1840) ==="
  python scripts/run_unrestricted_manzoni.py \
    --backend vllm-http --endpoint http://127.0.0.1:8000 \
    --constrained $SRC --model $MODEL \
    --n $N --max-tokens 2048 --temperature 0.9 \
    --out $OUT || echo "RUN FAILED ($LANG)"
done

echo; echo "[sbatch] done"; date -Is
ls -la results/unrestricted_*_1840_*.json 2>/dev/null
