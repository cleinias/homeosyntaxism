#!/bin/bash
#SBATCH --job-name=queneau-homo-temp09
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/group/p.cis260959.000/jobs/queneau-homo-temp09-%j.out
#SBATCH --error=/scratch/group/p.cis260959.000/jobs/queneau-homo-temp09-%j.err

# PREPARED, NOT SUBMITTED. Re-run at temperature 0.9 the two families that ran at 0.7,
# so the whole Queneau corpus sits at one sampling temperature.
#
# NOT for the 2026-08-14 resubmission -- the paper discloses the two 0.7 families in
# Appendix B and is correct as it stands. This is for the camera-ready if the paper is
# accepted, and for the Poetics Today version, where a single sampling temperature across
# every table is worth having on its own.
#
# WHAT RAN AT 0.7, AND WHY IT MATTERS
#
#   ladderT1   ACES 2030984. Table 1's whole Qwen half: 7B and 72B x S0 and D0 x blind
#              and source-visible x 3 languages x 3 scenarios = 72 cells.
#   ladderD1   ACES 2031074/2031075. The French and Italian cells of Table 3's
#              order-free dependency column, seven models = 42 cells.
#              (Its English cells ran at 0.9 and do NOT need re-running.)
#
# Everything else in the corpus -- ladderD0soft, ladderS0, ladderD2c, ladderD2, the
# laptop runs, the hosted Opus and Maverick cells -- is already 0.9.
#
# The measured effect of the difference, from the one configuration that exists at both
# temperatures (Qwen at D0 source-blind, same cluster, same prompt):
#
#     7B    0.7 -> coherence 0.65     0.9 -> 0.52      surf-synt 0.35 vs 0.34
#     72B   0.7 -> coherence 0.98     0.9 -> 0.94      surf-synt 0.38 vs 0.38
#
# So the expected consequence of this job is that Table 1's coherence column falls a
# little and its fidelity column does not move. Table 3's D1 column should barely move:
# its English-minus-French/Italian coherence gap is +0.23 against +0.32/+0.32/+0.29 at
# the three uniformly-0.9 degrees, i.e. the language effect already covers it.
#
# NEW TAGS, NOTHING OVERWRITTEN. Output goes to ladderT1t9 and ladderD1t9. The 0.7 data
# stays exactly where it is, because the paper as submitted rests on it and because
# 0.7-vs-0.9 at one degree is a finding worth keeping in its own right. The tags are also
# chosen so the existing selectors cannot match them by accident: the census and verify
# scripts test for "ladderT1_" and "ladderD1_", which "ladderT1t9_" and "ladderD1t9_" do
# not contain.
#
# ONE MODEL LOAD PER MODEL. The two families are interleaved rather than run as two jobs:
# serving is what costs time here, and 7B and 72B are needed by both. Loading each of the
# seven models once and running all of its cells saves roughly an hour over two jobs.
#
# COST. Two H100s for an estimated 1.5-2 h => roughly 120-160 SU at 40 SU/H100-hr, against
# a balance last seen near 6,950. Negligible; the 4 h limit is headroom, not an estimate.

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

# Prompt parity against what the 0.7 runs actually sent. Temperature is the ONLY thing
# this job changes, so if any prompt differs the comparison is void and we stop. The
# reference hashes are of the prompts recorded in the ladderT1 and ladderD1 run files.
python - <<'PROMPTCHECK'
import importlib.util, hashlib, json, glob, os
spec = importlib.util.spec_from_file_location("qrh", "scripts/queneau_run_homo.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
SP = "Experiments/Queneau/queneau-homo/spec"
O = "Experiments/Queneau/queneau-homo/outputs"
bad = 0; checked = 0
for pat, deg in (("homo_S0_*_ladderT1_*.json", "S0"), ("homo_D0_*_ladderT1_*.json", "D0"),
                 ("homo_D1_*_ladderD1_*.json", "D1")):
    for f in sorted(glob.glob(os.path.join(O, pat))):
        d = json.load(open(f))
        if d["lang"] == "en" and deg == "D1":
            continue                      # English D1 already ran at 0.9; not re-run here
        s = json.load(open(f"{SP}/specs_{d['lang']}_{deg}.json"))
        p = m.build_prompt(s, d["lang"], d["scenario"], deg, d.get("source_visible", False))
        checked += 1
        if p != d["prompt"]:
            bad += 1
            print(f"[prompt-check] DIFFERS {os.path.basename(f)}")
print(f"[prompt-check] {checked - bad}/{checked} prompts byte-identical to the 0.7 runs")
if bad:
    raise SystemExit("[prompt-check] FAIL: the template or specs moved; this would not be "
                     "a temperature-only comparison")
PROMPTCHECK
[ $? -eq 0 ] || { echo "!! prompt parity failed, aborting"; exit 1; }

LADDER="Qwen/Qwen2.5-0.5B-Instruct Qwen/Qwen2.5-1.5B-Instruct Qwen/Qwen2.5-3B-Instruct \
        Qwen/Qwen2.5-7B-Instruct Qwen/Qwen2.5-14B-Instruct Qwen/Qwen2.5-32B-Instruct \
        Qwen/Qwen2.5-72B-Instruct"

run_cell () {   # model lang scenario degree regime-flag tag
  python scripts/queneau_run_homo.py --backend vllm-http --endpoint http://127.0.0.1:8000 \
    --model "$1" --lang "$2" --scenario "$3" --degree "$4" --n 10 --temperature 0.9 \
    --max-tokens 1024 $5 --tag "$6" || echo "!! CELL FAILED $1 $2 $3 $4 $6"
}

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

    # -- Table 3's D1 column, French and Italian only: every model ------------------
    for L in fr it; do
      for S in security-checkpoint hospital-emergency-room grocery-checkout; do
        run_cell "$M" "$L" "$S" D1 "" ladderD1t9
      done
    done

    # -- Table 1: only the two models that table reports, both degrees, both regimes -
    case "$M" in
      *Qwen2.5-7B-Instruct|*Qwen2.5-72B-Instruct)
        for D in S0 D0; do
          for L in fr it en; do
            for S in security-checkpoint hospital-emergency-room grocery-checkout; do
              run_cell "$M" "$L" "$S" "$D" ""                 ladderT1t9
              run_cell "$M" "$L" "$S" "$D" "--source-visible" ladderT1t9
            done
          done
        done
        ;;
    esac

  else
    echo "!! SKIP $M (never ready)"
  fi
  kill $VLLM_PID 2>/dev/null || true; sleep 3; kill -9 $VLLM_PID 2>/dev/null || true
  sleep 5
done

echo; echo "[sbatch] expected cell counts: ladderD1t9 42, ladderT1t9 72"
ls Experiments/Queneau/queneau-homo/outputs/homo_D1_*_ladderD1t9_*.json 2>/dev/null | wc -l
ls Experiments/Queneau/queneau-homo/outputs/homo_*_ladderT1t9_*.json 2>/dev/null | wc -l
echo "[sbatch] ALL DONE"; date -Is
