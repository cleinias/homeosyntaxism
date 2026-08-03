#!/bin/bash
#=============================================================================
# CONSTRAINED homeosyntactic transposition of the Manzoni incipit — self-hosted.
#
# The clause target AND the eight critic-derived patterns, on the corrected 1840
# source, with the menu-free constraint specifications (2026-07-30: the patterns
# no longer hand the model its own realization vocabulary) and the filtered
# embargo (36 Italian forms, 14 English: the source's names and markers, not its
# whole content-word list).
#
# Two rungs per language: D2 = clause nesting, D3 = clause inventory.
# Scoring happens off-cluster (the judges need the Anthropic API).
#=============================================================================
#SBATCH --job-name=manzoni-constr
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=01:45:00
#SBATCH --output=/scratch/group/p.cis260959.000/jobs/manzoni-constr-%j.out
#SBATCH --error=/scratch/group/p.cis260959.000/jobs/manzoni-constr-%j.err

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
DEGREES=${DEGREES:-D2 D3}
SHORT=$(basename $MODEL)

# The run tag names the cell in every output filename, so it has to follow MODEL.
# Job 1999824 ran Maverick with the tag left at its Qwen default and wrote four
# files reading `qwen3-1840` — harmless only because the model suffix disambiguated
# them.  Derive it instead, and keep TAG overridable for one-off cells.
case "$SHORT" in
  Qwen3-*)          DEFTAG=qwen3-1840 ;;
  Llama-4-Maverick*) DEFTAG=maverick-1840 ;;
  *)                DEFTAG=$(echo "$SHORT" | tr 'A-Z' 'a-z' | cut -d- -f1)-1840 ;;
esac
TAG=${TAG:-$DEFTAG}
echo "tag=$TAG"

echo "=== job $SLURM_JOB_ID on $SLURMD_NODENAME ==="; date -Is
echo "model=$MODEL tp=$TP n=$N degrees='$DEGREES'  (1840 source, menu-free patterns)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

nohup vllm serve $MODEL \
  --tensor-parallel-size $TP --host 127.0.0.1 --port 8000 \
  --max-model-len 16384 --gpu-memory-utilization 0.92 --enforce-eager \
  > vllm-constr-$SLURM_JOB_ID-$SHORT.log 2>&1 &
VLLM_PID=$!
cleanup() { kill $VLLM_PID 2>/dev/null || true; sleep 2; kill -9 $VLLM_PID 2>/dev/null || true; }
trap cleanup EXIT

READY=0
for i in $(seq 1 600); do
  curl -sf http://127.0.0.1:8000/v1/models > /dev/null 2>&1 && { READY=1; echo "vllm ready after $((i*5))s"; break; }
  kill -0 $VLLM_PID 2>/dev/null || { echo "vllm died"; tail -80 vllm-constr-$SLURM_JOB_ID-$SHORT.log; exit 1; }
  sleep 5
done
[ "$READY" -eq 1 ] || { echo "vllm not ready in 50 min"; tail -80 vllm-constr-$SLURM_JOB_ID-$SHORT.log; exit 1; }

mkdir -p results
for LANG in it en; do
  for DEG in $DEGREES; do
    echo; echo "=== $LANG / $DEG : $SHORT (constrained, 1840) ==="
    python scripts/run_homeo.py \
      --backend vllm-http --endpoint http://127.0.0.1:8000 \
      --model $MODEL --lang $LANG --degree $DEG \
      --n $N --temperature 0.9 --max-tokens 2048 \
      --domain specs/domain_brief_3item.txt \
      --tag $TAG || echo "RUN FAILED ($LANG/$DEG)"
  done
done

echo; echo "[sbatch] done"; date -Is
ls -la results/production_D*_${TAG}_*.json 2>/dev/null
