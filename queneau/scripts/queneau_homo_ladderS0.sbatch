#!/bin/bash
#SBATCH --job-name=queneau-homo-qwenS0
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/group/p.cis260959.000/jobs/queneau-homo-qwenS0-%j.out
#SBATCH --error=/scratch/group/p.cis260959.000/jobs/queneau-homo-qwenS0-%j.err

# S0 on the ACES ladder — the degree that has never been run here.
#
# Why: S0 exists only in the b1 (OpenRouter) and mac (laptop) arms. Table 4's S0 row
# therefore takes its "cluster" columns from OpenRouter rather than from the cluster,
# and its 14B cell is empty for no better reason than that b1's model set stops at 7B
# and 72B. Every other row of that table (D0, D1, D3) is ladder-sourced, so one column
# silently changes machine between its first row and the rest.
#
# This run puts S0 on the same ladder, the same hardware and the same quantization
# policy as the other degrees, fills the 14B cell honestly, and makes all four rows of
# Table 4 comparable. It also gives Table 3 an S0 rung should we ever want one.
#
# No prompt-regime question here, unlike the D0 re-run: S0 was added to the runner on
# 2026-07-14, after the template was softened, so it has only ever existed in the
# softened form. The prompt-check below records that in the job log regardless.
#
# Tag is ladderS0. Nothing to overwrite — no ladder S0 has ever existed.
#
# Quantization matches the other ladder jobs: bf16 for 0.5B-14B, fp8-at-load for 32B
# and 72B, which do not fit one H100 at bf16. Verified against the 1917095 serve logs.

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

cd $GROUP_SCRATCH/manzoni-mapping
echo "=== job $SLURM_JOB_ID on $SLURMD_NODENAME ==="; date -Is
python -c "import vllm,torch;print('vllm',vllm.__version__,'torch',torch.__version__)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Record the exact prompt this job will send, so the regime is auditable from the job
# log alone and we never have to infer it from a commit date again.
python - <<'PROMPTCHECK'
import importlib.util, json
spec = importlib.util.spec_from_file_location("qrh", "scripts/queneau_run_homo.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
s = json.load(open("Experiments/Queneau/queneau-homo/spec/specs_en_S0.json"))
p = m.build_prompt(s, "en", "security-checkpoint", "S0", False)
print(f"[prompt-check] S0/en/security-checkpoint: {len(p)} chars, "
      f"strict={'EXACTLY' in p}")
PROMPTCHECK

LADDER="Qwen/Qwen2.5-0.5B-Instruct Qwen/Qwen2.5-1.5B-Instruct Qwen/Qwen2.5-3B-Instruct \
        Qwen/Qwen2.5-7B-Instruct Qwen/Qwen2.5-14B-Instruct Qwen/Qwen2.5-32B-Instruct \
        Qwen/Qwen2.5-72B-Instruct"

for M in $LADDER; do
  TP=1; QUANT=""
  case "$M" in
    *32B*) QUANT="--quantization fp8" ;;
    *72B*) QUANT="--quantization fp8"; TP=2 ;;
  esac
  SLOG=vllm-serve-$SLURM_JOB_ID-${M##*/}.log
  echo; echo "############ MODEL $M (TP=$TP $QUANT) $(date -Is) ############"
  nohup vllm serve $M $QUANT \
    --tensor-parallel-size $TP --host 127.0.0.1 --port 8000 \
    --max-model-len 4096 --gpu-memory-utilization 0.92 --enforce-eager \
    > $SLOG 2>&1 &
  VLLM_PID=$!
  READY=0
  for i in $(seq 1 360); do
    curl -sf http://127.0.0.1:8000/v1/models > /dev/null 2>&1 && { READY=1; echo "ready after $((i*5))s"; break; }
    kill -0 $VLLM_PID 2>/dev/null || { echo "!! vllm died for $M"; tail -40 $SLOG; break; }
    sleep 5
  done
  if [ "$READY" -eq 1 ]; then
    for L in fr it en; do
      for S in security-checkpoint hospital-emergency-room grocery-checkout; do
        python scripts/queneau_run_homo.py --backend vllm-http --endpoint http://127.0.0.1:8000 \
          --model $M --lang $L --scenario $S --degree S0 --n 10 --temperature 0.9 \
          --max-tokens 1024 --tag ladderS0 || echo "!! CELL FAILED $M $L $S"
      done
    done
  else
    echo "!! SKIP $M (never ready)"
  fi
  kill $VLLM_PID 2>/dev/null || true; sleep 3; kill -9 $VLLM_PID 2>/dev/null || true
  sleep 5
done
echo "[sbatch] ALL DONE"; date -Is
