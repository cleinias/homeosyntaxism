#!/bin/bash
#SBATCH --job-name=queneau-homo-D1frit
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/group/p.cis260959.000/jobs/queneau-homo-D1frit-%j.out
#SBATCH --error=/scratch/group/p.cis260959.000/jobs/queneau-homo-D1frit-%j.err

#=============================================================================
# D1 in French and Italian: the ladder's last single-language rung.
#
# THE GAP
#
#   Every degree on the ACES ladder is three-language except D1:
#
#       S0   ladderS0     en,fr,it   63 cells
#       D0   ladder       en,fr,it   63 cells
#       D1   ladderD1     en          21 cells   <-- this job
#       D2   ladderD2c    en,fr,it   63 cells
#       D3   ladderD2     en,fr,it   63 cells
#
#   So Table 3's dep-triple column, and the D1 entry of its coherence column, rest
#   on English alone while the three rungs beside them pool three languages. A
#   reader takes a row of that table as one measurement across a graded series; one
#   rung is drawn from a different population. This is exactly the defect found in
#   the D3 coherence figures on 2026-08-08 -- a column that reads as commensurable
#   with its neighbours and is not -- still present, one column over.
#
#   It is worth fixing regardless of the pending strict-vs-softened D0 decision,
#   which touches a different column.
#
# WHY THE TAG IS UNCHANGED
#
#   These cells carry tag `ladderD1`, the same as the July English ones, because
#   they are the same run family: the pre-flight below proves the prompts this job
#   builds are byte-identical to the ones those cells were built from, and the
#   quantization policy is the ladder's own. Language is already in every filename,
#   so nothing is lost by not inventing a tag -- and inventing one would have made
#   `ladderD1*` match two families, which is the globbing trap that silently pooled
#   60 rows into a 30-row cell once already (`ladderD2*` still matches both
#   ladderD2 and ladderD2c; we are not adding a third).
#
# PRE-FLIGHT
#
#   Rebuilds the three ENGLISH D1 prompts and compares them against the SHA-256 of
#   what the existing English cells actually sent (spec/d1_en_prompt_hashes.json).
#   French and Italian have no incumbent to compare against, so the gate proves the
#   template itself has not drifted since those cells were made; the new cells then
#   come from the same code path with only the language changed. Any mismatch aborts
#   before a GPU-second is spent.
#
# COST
#
#   42 cells, 420 generations. Measured throughput on the ladderS0 run is ~0.4-1.4
#   s/gen across this model range, so ~10 min of generation; the rest is seven model
#   loads. Expect ~30-40 min wall on 2xH100 ~= 50 SU. --time=02:00:00 bounds the
#   worst case at ~160.
#
#   Quantization follows ladder policy: bf16 for 0.5B-14B, fp8-at-load for 32B and
#   72B, which do not fit one H100 at bf16.
#=============================================================================

set -uo pipefail
module load Anaconda3/2025.12-2
module load GCC/13.3.0 CUDA/13.1.0
export CUDA_HOME=${CUDA_HOME:-${EBROOTCUDA:-}}
source $(conda info --base)/etc/profile.d/conda.sh
export GROUP_SCRATCH=/scratch/group/p.cis260959.000
export HF_HOME=$GROUP_SCRATCH/huggingface
export TRANSFORMERS_CACHE=$GROUP_SCRATCH/huggingface/hub
export HF_HUB_OFFLINE=1
export VLLM_USE_DEEP_GEMM=0
export PYTHONUNBUFFERED=1
conda activate $GROUP_SCRATCH/conda_envs/atlas-llm

WORKDIR=$GROUP_SCRATCH/manzoni-mapping
cd $WORKDIR || exit 1
echo "=== job $SLURM_JOB_ID on $SLURMD_NODENAME ==="; date -Is

echo; echo "=== prompt parity: the English D1 prompts must rebuild exactly ==="
python - <<'PARITY' || { echo "!!! D1 TEMPLATE HAS DRIFTED — aborting before any GPU work"; exit 1; }
import importlib.util, json, hashlib, sys
from pathlib import Path
spec = importlib.util.spec_from_file_location("qrh", "scripts/queneau_run_homo.py")
qrh = importlib.util.module_from_spec(spec); spec.loader.exec_module(qrh)
want = json.loads(Path("Experiments/Queneau/queneau-homo/spec/d1_en_prompt_hashes.json").read_text())
bad = 0
for key, h in sorted(want.items()):
    degree, lang, scenario, regime = key.split("|")
    specs = json.loads((qrh.SPEC_DIR / f"specs_{lang}_{degree}.json").read_text(encoding="utf-8"))
    got = hashlib.sha256(qrh.build_prompt(specs, lang, scenario, degree,
                                          regime == "vis").encode()).hexdigest()
    ok = got == h; bad += not ok
    print(f"  {'ok ' if ok else 'MISMATCH'} {degree} {lang} {scenario}")
# the specs this job will actually use must at least exist and parse
for lang in ("fr", "it"):
    s = json.loads((qrh.SPEC_DIR / f"specs_{lang}_D1.json").read_text(encoding="utf-8"))
    print(f"  specs_{lang}_D1.json: {s['n_periods']} periods")
print(f"  {len(want)-bad}/{len(want)} English D1 prompts identical to the existing cells")
sys.exit(1 if bad else 0)
PARITY

LANGS="fr it"
SCENARIOS="grocery-checkout security-checkpoint hospital-emergency-room"
LADDER="Qwen/Qwen2.5-0.5B-Instruct Qwen/Qwen2.5-1.5B-Instruct Qwen/Qwen2.5-3B-Instruct \
Qwen/Qwen2.5-7B-Instruct Qwen/Qwen2.5-14B-Instruct Qwen/Qwen2.5-32B-Instruct \
Qwen/Qwen2.5-72B-Instruct"

for M in $LADDER; do
  TP=1; QUANT=""
  case "$M" in
    *32B*) QUANT="--quantization fp8" ;;
    *72B*) QUANT="--quantization fp8"; TP=2 ;;
  esac
  echo; echo "############ MODEL $M (TP=$TP ${QUANT:-bf16}) $(date -Is) ############"
  nohup vllm serve "$M" \
    --tensor-parallel-size $TP --host 127.0.0.1 --port 8000 \
    --max-model-len 8192 --gpu-memory-utilization 0.90 $QUANT \
    > vllm-D1frit-$(basename "$M").log 2>&1 &
  VLLM_PID=$!
  cleanup() { kill $VLLM_PID 2>/dev/null || true; sleep 3; kill -9 $VLLM_PID 2>/dev/null || true; }
  trap cleanup EXIT

  READY=0
  for i in $(seq 1 360); do
    curl -sf http://127.0.0.1:8000/v1/models >/dev/null 2>&1 && { READY=1; echo "vllm ready after $((i*5))s"; break; }
    kill -0 $VLLM_PID 2>/dev/null || { echo "!! vllm died"; tail -40 vllm-D1frit-$(basename "$M").log; break; }
    sleep 5
  done
  [ $READY -eq 1 ] || { echo "!! skipping $M"; cleanup; trap - EXIT; continue; }

  for L in $LANGS; do
    for S in $SCENARIOS; do
      echo "--- $M D1 $L $S $(date -Is)"
      python scripts/queneau_run_homo.py --backend vllm-http \
        --endpoint http://127.0.0.1:8000 --model "$M" \
        --degree D1 --lang $L --scenario $S \
        --n 10 --temperature 0.7 --max-tokens 1024 \
        --tag ladderD1 || echo "!! CELL FAILED $M D1 $L $S"
    done
  done
  cleanup; trap - EXIT; sleep 5
done

echo; echo "=== done $(date -Is) ==="
ls -1 Experiments/Queneau/queneau-homo/outputs/homo_D1_{fr,it}_*ladderD1*.json 2>/dev/null | wc -l
echo "fr+it cells written (expected 42)"
