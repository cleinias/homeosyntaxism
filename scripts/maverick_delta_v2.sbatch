#!/bin/bash
#=============================================================================
# Phase D comparison — Llama-4-Maverick-17B-128E-Instruct (bf16) on NCSA Delta H200.
#
# v2 (2026-08-24), after three consecutive loading failures:
#   21369332 (00:20:52), 21373980 (00:50:31), 21395206 (01:41:00).
#
# Measurement ruled out slow storage: /work/hdd reads at 932 MB/s single-stream
# and 2.6 GB/s with 16 streams, striping is healthy (12-20 OSTs), and vLLM's own
# prefetcher pulled all 748 GiB into page cache in 112 s (~6.8 GB/s) on the
# compute node.  Loading nonetheless ran fast to shard 44 and then collapsed to
# 3.5-17+ min/shard.  Standing hypothesis: page-cache eviction under cgroup
# pressure, after which 8 TP workers re-read the same file and thrash.
#
# Changes from v1:
#   1. Run:ai model streamer instead of the mmap/page-cache loader (LOADFMT).
#   2. Memory instrumentation every 20 s, joined against loader progress, so the
#      page-cache hypothesis is confirmed or killed in a single run.
#   3. Stall watchdog + absolute cap: a repeat failure costs ~45-60 min of
#      allocation instead of 100 min.
#   4. All logs live in $JOBDIR next to the Slurm .out, not in the working
#      directory where they were easy to lose.
#   5. One automatic fallback to --load-format auto if the streamer dies early.
#
# Streamer validated on the login node 2026-08-24 with these exact settings
# (concurrency 8, memory_limit 32 GiB): shard 45 -- the one that took 17.5 min
# in 21395206 -- streamed in 29.1 s (704 MB/s), shard 01 in 31.3 s (655 MB/s).
#
# Override at submit time, e.g.:
#   sbatch --export=ALL,LOADFMT=auto,READY_LIMIT=5400 maverick_delta_v2.sbatch
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

# Run:ai streamer tuning (env names verified against vllm 0.27.1
# model_executor/model_loader/runai_streamer_loader.py).
#   CONCURRENCY  8 readers per worker x 8 workers = 64 streams, which is what
#                vLLM's own prefetcher used when it hit 6.8 GB/s on this node.
#   MEMORY_LIMIT bytes of host buffer per worker.  32 GiB > the largest shard
#                (20 GiB), so it never serialises, and 8 x 32 = 256 GB keeps
#                total host pressure far below the 2 TB the node has -- which
#                is the point, given the page-cache-eviction hypothesis.
export RUNAI_STREAMER_CONCURRENCY=${RUNAI_STREAMER_CONCURRENCY:-8}
export RUNAI_STREAMER_MEMORY_LIMIT=${RUNAI_STREAMER_MEMORY_LIMIT:-34359738368}

JOBDIR=/projects/bidl/mmarchesini/jobs
WORKDIR=/projects/bidl/mmarchesini/manzoni-mapping
MODEL=/projects/bidl/mmarchesini/Llama-4-Maverick-17B-128E-Instruct
TAG=maverick-bf16
cd $WORKDIR

LOADFMT=${LOADFMT:-runai_streamer}
READY_LIMIT=${READY_LIMIT:-3600}   # absolute seconds allowed to reach readiness
STALL_LIMIT=${STALL_LIMIT:-600}    # seconds with no loader progress before abort
MEM_LOG=$JOBDIR/meminfo-$SLURM_JOB_ID.log
ABORT_FLAG=$JOBDIR/.abort-$SLURM_JOB_ID
VLLM_LOG=""                        # set per attempt
VLLM_PID=""
MEM_PID=""
WD_PID=""

echo "=== job $SLURM_JOB_ID on $SLURMD_NODENAME ==="; date -Is
echo "model=$MODEL  tag=$TAG  (bf16 reference, 8xH200)"
echo "load-format=$LOADFMT  ready_limit=${READY_LIMIT}s  stall_limit=${STALL_LIMIT}s"
echo "runai: concurrency=$RUNAI_STREAMER_CONCURRENCY memory_limit=$RUNAI_STREAMER_MEMORY_LIMIT"
echo "LOGS:"
echo "  slurm stdout : $JOBDIR/maverick-bf16-$SLURM_JOB_ID.out"
echo "  vllm server  : $JOBDIR/vllm-serve-$SLURM_JOB_ID-<loadfmt>.log"
echo "  memory probe : $MEM_LOG"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

#-- pre-flight ---------------------------------------------------------------
echo "--- pre-flight ---"
python - <<'PY'
import importlib
for m in ("vllm", "runai_model_streamer"):
    try:
        mod = importlib.import_module(m)
        print(f"  {m}: OK {getattr(mod, '__version__', '')}")
    except Exception as e:
        print(f"  {m}: MISSING ({e})")
PY

# The streamer refuses a memory limit below the largest single shard, and does
# so only after the engine has spun up -- an expensive way to learn it.  Check
# it here, in seconds, instead.  (Verified on the login node 2026-08-24:
# 8 GiB raised RunaiStreamerMemoryLimitException against the 20 GiB shards.)
if [ "$LOADFMT" = "runai_streamer" ]; then
  BIGGEST=$(for f in "$MODEL"/*.safetensors; do stat -Lc %s "$f"; done | sort -n | tail -1)
  echo "  largest shard: $BIGGEST bytes; streamer memory_limit: $RUNAI_STREAMER_MEMORY_LIMIT bytes"
  if [ "${RUNAI_STREAMER_MEMORY_LIMIT}" -lt "${BIGGEST}" ]; then
    echo "  FATAL: RUNAI_STREAMER_MEMORY_LIMIT is below the largest shard -- the"
    echo "         streamer would abort mid-load.  Raise it and resubmit."
    exit 1
  fi
  echo "  streamer memory limit OK"
fi

#-- instrumentation ----------------------------------------------------------
cgroup_file() {  # $1 = memory.current | memory.max
  local rel
  rel=$(awk -F: '$1=="0"{print $3}' /proc/self/cgroup 2>/dev/null | head -1)
  for p in "/sys/fs/cgroup${rel}/$1" "/sys/fs/cgroup/$1"; do
    [ -r "$p" ] && { echo "$p"; return; }
  done
}
CG_CUR=$(cgroup_file memory.current)
CG_MAX=$(cgroup_file memory.max)

progress() {  # last "N/55" from the newest vllm log for this job
  # Globs rather than reading $VLLM_LOG: mem_probe is forked before the first
  # attempt sets that variable, and a forked subshell never sees the update.
  local f
  f=$(ls -t "$JOBDIR"/vllm-serve-"$SLURM_JOB_ID"-*.log 2>/dev/null | head -1)
  [ -n "$f" ] || { echo ""; return; }
  grep -a "Loading safetensors checkpoint shards" "$f" 2>/dev/null \
    | tail -1 | grep -oE '[0-9]+/[0-9]+' | tail -1
}

mem_probe() {
  echo "# time  MemAvailable Cached Dirty Writeback cgroup_current shard" >> "$MEM_LOG"
  while true; do
    {
      printf '%s ' "$(date +%H:%M:%S)"
      awk '/^MemTotal:|^MemFree:|^MemAvailable:|^Cached:|^Dirty:|^Writeback:/ \
           {printf "%s=%dG ", substr($1,1,length($1)-1), $2/1048576}' /proc/meminfo
      [ -n "$CG_CUR" ] && printf "cgcur=%dG " $(( $(cat "$CG_CUR" 2>/dev/null || echo 0) / 1073741824 ))
      [ -n "$CG_MAX" ] && printf "cgmax=%s " "$(cat "$CG_MAX" 2>/dev/null)"
      printf "shard=%s\n" "$(progress)"
    } >> "$MEM_LOG" 2>/dev/null
    sleep 20
  done
}

watchdog() {
  local seen="" changed armed=0 now
  changed=$(date +%s)
  while true; do
    sleep 60
    now=$(date +%s)
    local cur; cur=$(progress)
    if [ -n "$cur" ]; then
      armed=1
      if [ "$cur" != "$seen" ]; then seen="$cur"; changed=$now; fi
    fi
    if [ "$armed" -eq 1 ] && [ $((now - changed)) -ge "$STALL_LIMIT" ]; then
      echo "[watchdog] loader stuck at shard $seen for $((now - changed))s -- aborting to save allocation"
      touch "$ABORT_FLAG"
      return
    fi
  done
}

cleanup() {
  [ -n "$MEM_PID" ] && kill $MEM_PID 2>/dev/null
  [ -n "$WD_PID"  ] && kill $WD_PID  2>/dev/null
  [ -n "$VLLM_PID" ] && { kill $VLLM_PID 2>/dev/null; sleep 3; kill -9 $VLLM_PID 2>/dev/null; }
  rm -f "$ABORT_FLAG"
}
trap cleanup EXIT

mem_probe & MEM_PID=$!

#-- serve --------------------------------------------------------------------
try_serve() {  # $1 = load format ; returns 0 if the server became ready
  local fmt=$1 start now elapsed ready=0
  VLLM_LOG=$JOBDIR/vllm-serve-$SLURM_JOB_ID-$fmt.log
  rm -f "$ABORT_FLAG"
  echo; echo "=== starting vllm (--load-format $fmt) ==="; date -Is
  echo "    log: $VLLM_LOG"

  nohup vllm serve $MODEL \
    --tensor-parallel-size 8 --host 127.0.0.1 --port 8000 \
    --max-model-len 16384 --gpu-memory-utilization 0.92 --enforce-eager \
    --load-format "$fmt" \
    > "$VLLM_LOG" 2>&1 &
  VLLM_PID=$!

  watchdog & WD_PID=$!
  start=$(date +%s)
  while true; do
    if curl -sf http://127.0.0.1:8000/v1/models > /dev/null 2>&1; then
      ready=1; echo "vllm ready after $(( $(date +%s) - start ))s"; break
    fi
    kill -0 $VLLM_PID 2>/dev/null || { echo "vllm process exited"; break; }
    [ -f "$ABORT_FLAG" ] && { echo "aborting: watchdog"; break; }
    now=$(date +%s); elapsed=$((now - start))
    [ "$elapsed" -ge "$READY_LIMIT" ] && { echo "not ready within ${READY_LIMIT}s"; break; }
    sleep 5
  done
  kill $WD_PID 2>/dev/null; WD_PID=""
  ATTEMPT_ELAPSED=$(( $(date +%s) - start ))

  if [ "$ready" -eq 0 ]; then
    echo "--- last 60 lines of $VLLM_LOG ---"; tail -60 "$VLLM_LOG"
    echo "--- last 25 memory probes ---"; tail -25 "$MEM_LOG"
    kill $VLLM_PID 2>/dev/null; sleep 3; kill -9 $VLLM_PID 2>/dev/null
    VLLM_PID=""
    return 1
  fi
  return 0
}

SERVED=0
if try_serve "$LOADFMT"; then
  SERVED=1
elif [ "$LOADFMT" != "auto" ] && [ "${ATTEMPT_ELAPSED:-9999}" -lt 900 ]; then
  # Died fast -> looks like the streamer is unusable here rather than a slow
  # load.  Fall back once; a slow load would not have failed this quickly.
  echo; echo "[fallback] $LOADFMT failed after ${ATTEMPT_ELAPSED}s -- retrying with auto"
  try_serve auto && SERVED=1
fi

if [ "$SERVED" -ne 1 ]; then
  echo; echo "[sbatch] server never became ready -- no generation attempted"; date -Is
  echo "memory probe timeline: $MEM_LOG"
  exit 1
fi

#-- generate -----------------------------------------------------------------
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
