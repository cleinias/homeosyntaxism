#!/bin/bash
#=============================================================================
# Queneau *Notations* --- GRAIN-C (D2c) size ladder on ACES, self-hosted vLLM (fp8).
#
# Regenerates the D2 backbone at GRAIN C (clause-NESTING SCHEMA --- which clause
# embeds in which) instead of grain A (clause inventory + max depth), so the
# paper's D2 evidence matches the section-2 definition D2 = grain c. Blind
# (source-hidden), 7-rung Qwen2.5 ladder x 3 langs x 3 scenarios, n=10.
# Structure/coherence scoring happens OFF-cluster (compute nodes are offline):
# pull the outputs, then scripts/queneau_score_homo.py (+ an attachment metric).
#
# PREREQS on the cluster (sync from the laptop, tar-over-ssh --- no scp):
#   - spec/specs_{fr,it,en}_D2c.json           (built local: queneau_build_specs.py --degree D2c)
#   - scripts/queneau_run_homo.py              (has the D2c degree prompt)
#   - scripts/queneau_build_specs.py           (has emit_D2_schema; only needed if rebuilding on-cluster)
#=============================================================================
#SBATCH --job-name=queneau-d2c-ladder
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/group/p.cis260959.000/jobs/queneau-d2c-%j.out
#SBATCH --error=/scratch/group/p.cis260959.000/jobs/queneau-d2c-%j.err

set -uo pipefail
module load Anaconda3/2025.12-2
module load GCC/13.3.0 CUDA/13.1.0            # cc1plus + nvcc for the FP8/flashinfer JIT
export CUDA_HOME=${CUDA_HOME:-${EBROOTCUDA:-}}
source $(conda info --base)/etc/profile.d/conda.sh
export GROUP_SCRATCH=/scratch/group/p.cis260959.000
export HF_HOME=$GROUP_SCRATCH/huggingface
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_DEEP_GEMM=0                    # block-scaled FP8 via Triton (no DeepGEMM)
export TRITON_CACHE_DIR=$GROUP_SCRATCH/.triton         # keep compile caches OFF $HOME (10 GB quota)
export TORCHINDUCTOR_CACHE_DIR=$GROUP_SCRATCH/.inductor
export XDG_CACHE_HOME=$GROUP_SCRATCH/.cache
conda activate $GROUP_SCRATCH/conda_envs/atlas-llm     # vllm 0.21.0 + torch 2.11.0

WORKDIR=$GROUP_SCRATCH/manzoni-mapping
cd $WORKDIR

# 7-rung single-family ladder (fp8, 1xH100 each; 72B online-fp8 fits 80 GB).
LADDER="Qwen/Qwen2.5-0.5B-Instruct Qwen/Qwen2.5-1.5B-Instruct Qwen/Qwen2.5-3B-Instruct \
        Qwen/Qwen2.5-7B-Instruct Qwen/Qwen2.5-14B-Instruct Qwen/Qwen2.5-32B-Instruct \
        Qwen/Qwen2.5-72B-Instruct"
LANGS="fr it en"
SCEN="security-checkpoint hospital-emergency-room grocery-checkout"

echo "=== job $SLURM_JOB_ID on $SLURMD_NODENAME ==="; date -Is
python -c "import vllm,torch;print('vllm',vllm.__version__,'torch',torch.__version__,'cuda',torch.version.cuda)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

for M in $LADDER; do
  echo; echo "########## serving $M ##########"; date -Is
  nohup vllm serve "$M" --quantization fp8 --host 127.0.0.1 --port 8000 \
    --max-model-len 8192 --gpu-memory-utilization 0.92 --enforce-eager \
    > vllm-d2c-$SLURM_JOB_ID.log 2>&1 &
  VLLM_PID=$!
  READY=0
  for i in $(seq 1 360); do   # up to 30 min for the 72B load
    curl -sf http://127.0.0.1:8000/v1/models >/dev/null 2>&1 && { READY=1; echo "vllm ready after $((i*5))s"; break; }
    kill -0 $VLLM_PID 2>/dev/null || { echo "vllm DIED for $M"; tail -80 vllm-d2c-$SLURM_JOB_ID.log; break; }
    sleep 5
  done
  if [ "$READY" -eq 1 ]; then
    for L in $LANGS; do
      for S in $SCEN; do
        python scripts/queneau_run_homo.py \
          --backend vllm-http --endpoint http://127.0.0.1:8000 \
          --model "$M" --lang "$L" --degree D2c --scenario "$S" --n 10 \
          --temperature 0.9 --max-tokens 1024 --tag ladderD2c \
          || echo "CELL FAILED: $M $L $S"
      done
    done
  else
    echo "SKIP $M (server not ready)"
  fi
  kill "$VLLM_PID" 2>/dev/null || true; sleep 8; kill -9 "$VLLM_PID" 2>/dev/null || true
done
echo "[sbatch] done --- outputs in Experiments/Queneau/queneau-homo/outputs/homo_D2c_*_blind_ladderD2c_*.json"; date -Is
