#!/usr/bin/env python3
"""Generic ACES job monitor: poll a job; when its production JSONs land,
pull them locally; exit on success or job-gone.
  python scripts/monitor.py <job_id> <tag> <model_slug>
"""
import subprocess
import sys
import time

JOB, TAG, SLUG = sys.argv[1], sys.argv[2], sys.argv[3]
HOST = "login.aces.hprc.tamu.edu"
G = "/scratch/group/p.cis260959.000"
RES = f"{G}/manzoni-mapping/reproducibility/graded-corpus/results"
FILES = [f"production_d2{g}_{TAG}_{SLUG}.json" for g in ("a", "c")]
LOCALDIR = "reproducibility/graded-corpus/results"


def sh(cmd, t=90):
    try:
        return subprocess.run(["ssh", HOST, cmd], capture_output=True, text=True, timeout=t).stdout.strip()
    except Exception as e:
        return f"(ssh err: {e})"


def pull(f):
    d = subprocess.run(["ssh", HOST, f"cat {RES}/{f}"], capture_output=True, text=True, timeout=180).stdout
    if d:
        open(f"{LOCALDIR}/{f}", "w", encoding="utf-8").write(d)
    return len(d)


for _ in range(648):  # ~36 h
    st = sh(f"squeue -j {JOB} -h -o %T")
    present = sh(f"for f in {' '.join(FILES)}; do test -f {RES}/$f && echo $f; done")
    print(time.strftime("%H:%M:%S"), f"[{TAG} {JOB}] state=", st or "(gone)",
          "have=", present.replace(chr(10), ",") or "none", flush=True)
    if present.count("production_d2") >= 2:
        for f in FILES:
            print(f"PULLED {f}: {pull(f)} bytes", flush=True)
        break
    if not st or st == "(gone)":
        for f in FILES:
            n = pull(f)
            if n:
                print(f"PULLED {f}: {n} bytes", flush=True)
        print("JOB_GONE — tail:\n" + sh(f"tail -40 $(ls {G}/jobs/*-{JOB}.out 2>/dev/null | head -1)"), flush=True)
        break
    time.sleep(120)
print("monitor exit", flush=True)
