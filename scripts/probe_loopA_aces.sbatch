#!/bin/bash
#=============================================================================
# LOOP-A PROBES, Qwen3-235B arm (2026-11-26). Nothing is generated: the model is
# asked to find the source, never to write.
#
#   test 2  per-pattern probe: each pattern entry of the Italian D2 prompt alone,
#           July and revised wording (16 specifications), asked for its source's
#           identity and for a reconstruction of the source passage; n = 5.
#   test 3  output probe: each of the 60 Opus 5 IT D2 texts of Table 2's six arms
#           alone, uncued and cued; n = 1.
#
# Prompts: specs/probe/loopA_{patterns,outputs}_it_D2.json, built by
# scripts/build_loopA_probes.py; each item carries its prompt verbatim.
# Results: results/probe_loopA_*_<TAG>_<model>.json (+ _PROMPTS.jsonl). The runner
# writes after every item and resumes on a rerun with the same TAG, so a job that
# hits its walltime can simply be submitted again.
# Why: PoeticsToday/S3-4_S4-3_context-leakage_draft_2026-11-26.md, note 15.
#
#   sbatch scripts/probe_loopA_aces.sbatch
#=============================================================================
#SBATCH --job-name=probe-loopA
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=01:15:00
#SBATCH --exclude=ac096
#SBATCH --output=/scratch/group/p.cis260959.000/jobs/probe-loopA-%j.out
#SBATCH --error=/scratch/group/p.cis260959.000/jobs/probe-loopA-%j.err

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
TAG=${TAG:-loopA-1126}
SHORT=$(basename $MODEL)
RUN=(python scripts/run_probe_batch.py --backend vllm-http --endpoint http://127.0.0.1:8000
     --model $MODEL --temperature 0.9 --max-tokens 1024 --tag $TAG)

#--- pre-flight, before any GPU time: the manifests load and the runner can see them ---
for M in specs/probe/loopA_patterns_it_D2.json specs/probe/loopA_outputs_it_D2.json; do
  "${RUN[@]}" --manifest $M --n 1 --dry-run > /dev/null || { echo "pre-flight FAILED on $M"; exit 1; }
  echo "pre-flight ok: $M sha1 $(sha1sum < $M | cut -c1-12)"
done

echo "=== job $SLURM_JOB_ID on $SLURMD_NODENAME ==="; date -Is
echo "model=$MODEL tp=$TP tag=$TAG"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

nohup vllm serve $MODEL \
  --tensor-parallel-size $TP --host 127.0.0.1 --port 8000 \
  --max-model-len 16384 --gpu-memory-utilization 0.92 --enforce-eager \
  > vllm-probe-$SLURM_JOB_ID-$SHORT.log 2>&1 &
VLLM_PID=$!
cleanup() { kill $VLLM_PID 2>/dev/null || true; sleep 2; kill -9 $VLLM_PID 2>/dev/null || true; }
trap cleanup EXIT

READY=0
for i in $(seq 1 540); do
  curl -sf http://127.0.0.1:8000/v1/models > /dev/null 2>&1 && { READY=1; echo "vllm ready after $((i*5))s"; break; }
  kill -0 $VLLM_PID 2>/dev/null || { echo "vllm died"; tail -80 vllm-probe-$SLURM_JOB_ID-$SHORT.log; exit 1; }
  sleep 5
done
[ "$READY" -eq 1 ] || { echo "vllm not ready in 45 min"; tail -80 vllm-probe-$SLURM_JOB_ID-$SHORT.log; exit 1; }

mkdir -p results
echo; echo "=== test 2: per-pattern probe (n=5) ==="
"${RUN[@]}" --manifest specs/probe/loopA_patterns_it_D2.json --n 5 || echo "TEST 2 STOPPED (see above)"
echo; echo "=== test 3: output probe (n=1) ==="
"${RUN[@]}" --manifest specs/probe/loopA_outputs_it_D2.json --n 1 || echo "TEST 3 STOPPED (see above)"

echo; echo "[sbatch] done"; date -Is
ls -la results/probe_loopA_*_${TAG}_*.json 2>/dev/null
