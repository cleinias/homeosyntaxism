#!/bin/bash
#SBATCH --job-name=queneau-homo-qwenD2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/group/p.cis260959.000/jobs/queneau-homo-qwenD2-%j.out
#SBATCH --error=/scratch/group/p.cis260959.000/jobs/queneau-homo-qwenD2-%j.err

# RECONSTRUCTED LAUNCHER for the ladderD2 family -- the clause-INVENTORY rung, which the
# paper calls D3 and the filenames call D2. (Files named D2 hold the paper's D3; the
# paper's D2, clause nesting, is in files named D2c. This is the single most confusing
# fact in the corpus and it is worth restating every time.)
#
# WHY IT IS RECONSTRUCTED, AND ON WHAT EVIDENCE
#
# This run produced Table 3's clause-inventory column -- 63 cells, 2026-07-04 -- and its
# data was committed (5cd6dee) while its launcher never was. The original is gone. What
# follows was rebuilt on 2026-08-10 from what the run records themselves carry, and it is
# not guesswork in the part that matters:
#
#   parameters   all 63 records agree: backend vllm-http, temperature 0.9, max_tokens
#                1024, n 10, seven Qwen2.5 models x three languages x three scenarios
#   loop order   the records' ordering is model-ascending, then fr/it/en, then
#                security-checkpoint / hospital-emergency-room / grocery-checkout --
#                the same nesting as the other ladder jobs
#   PROMPTS      63 of 63 stored prompts are BYTE-IDENTICAL to what today's
#                queneau_run_homo.py + the committed specs_{lang}_D2.json rebuild.
#                That is the real check: the launcher's job is to call the runner with
#                the right degree, spec and sampling parameters, and the prompt proves
#                it would.
#
# What is NOT recoverable: the original Slurm job id, its wall time, and the exact
# quantization it used. The policy below (bf16 to 14B, fp8 at 32B and 72B) is the one
# every other ladder job used and the one the paper documents, so it is the defensible
# choice -- but it is inference, not a record, and is flagged as such here.
#
# NO PROMPT-REGIME QUESTION AT THIS DEGREE. The strict template ("...SYNTACTIC STRUCTURE
# EXACTLY ... same tags, same order, same arcs") was a property of the D0 form-target
# block, not of the shared preamble: the July D0 family (tag `ladder`) carries the marker
# and nothing else in the corpus does -- including this run, which predates the 2026-07-05
# softening by a day. So there is no ladderD2soft and none is needed.

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

# ---------------------------------------------------------------------------------
# GUARD. Re-running this overwrites data the paper's Table 3 rests on, and sampling is
# not deterministic, so the numbers would move. Refuse unless told explicitly.
#     sbatch --export=ALL,ALLOW_OVERWRITE=1 scripts/queneau_homo_ladderD2.sbatch
# ---------------------------------------------------------------------------------
EXISTING=$(ls Experiments/Queneau/queneau-homo/outputs/homo_D2_*_ladderD2_*.json 2>/dev/null | wc -l)
if [ "$EXISTING" -gt 0 ] && [ "${ALLOW_OVERWRITE:-0}" != "1" ]; then
  echo "!! $EXISTING ladderD2 run files already exist and Table 3 is computed from them."
  echo "!! Re-running would overwrite them with a fresh sample. To proceed anyway:"
  echo "!!   sbatch --export=ALL,ALLOW_OVERWRITE=1 \$0"
  exit 1
fi

python -c "import vllm,torch;print('vllm',vllm.__version__,'torch',torch.__version__)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Prompt parity, checked against the hash of what the 2026-07-04 run actually sent, so a
# drift in the runner or the specs fails here rather than silently producing a different
# experiment under the same tag.
python - <<'PROMPTCHECK'
import importlib.util, hashlib, json
REF = "6e0e6ce3189a451160855866d49d9820d615e3fe"   # en / security-checkpoint / D2
spec = importlib.util.spec_from_file_location("qrh", "scripts/queneau_run_homo.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
s = json.load(open("Experiments/Queneau/queneau-homo/spec/specs_en_D2.json"))
p = m.build_prompt(s, "en", "security-checkpoint", "D2", False)
h = hashlib.sha1(p.encode()).hexdigest()
print(f"[prompt-check] D2/en/security-checkpoint: {len(p)} chars, sha1 {h}, "
      f"strict={'EXACTLY' in p}")
if h != REF:
    raise SystemExit(f"[prompt-check] FAIL: expected {REF}; the template or the spec "
                     f"has changed since 2026-07-04 and this is no longer that experiment")
print("[prompt-check] matches the 2026-07-04 prompt exactly")
PROMPTCHECK
[ $? -eq 0 ] || { echo "!! prompt parity failed, aborting"; exit 1; }

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
          --model $M --lang $L --scenario $S --degree D2 --n 10 --temperature 0.9 \
          --max-tokens 1024 --tag ladderD2 || echo "!! CELL FAILED $M $L $S"
      done
    done
  else
    echo "!! SKIP $M (never ready)"
  fi
  kill $VLLM_PID 2>/dev/null || true; sleep 3; kill -9 $VLLM_PID 2>/dev/null || true
  sleep 5
done
echo "[sbatch] ALL DONE"; date -Is
