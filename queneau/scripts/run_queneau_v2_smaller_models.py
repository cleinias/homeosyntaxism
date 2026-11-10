#!/usr/bin/env python3
"""run_queneau_v2_smaller_models.py — ThunderCompute orchestrator for the Queneau v2 open-weight pass.

Spins up three ThunderCompute A100 instances (one per model), runs the four
Queneau (language, target) prompts against each model via run.py's vllm-http
backend, and tears each instance down. Per-model lifecycle:

  1. `tnr create` an A100 instance
  2. wait for RUNNING
  3. add port forwarding for vLLM (:8000)
  4. SSH install vLLM (if not already in the template)
  5. SSH start vLLM serving the model (HF_TOKEN exported for gated models)
  6. poll /v1/models until vLLM is ready
  7. for each of the four (language, target) combos:
       run.py --backend vllm-http --endpoint <url> --model <X> ...
  8. `tnr delete` the instance

Outputs land in ../Oulipo-Atlas-2-0/queneau-notations/outputs/
{LANG}-{TARGET}--{MODEL_WITH_SLASHES_REPLACED}--{DATE}/ via run.py.

Usage:
    python3 scripts/run_queneau_v2_smaller_models.py
    python3 scripts/run_queneau_v2_smaller_models.py --models Qwen/Qwen2.5-7B-Instruct
    python3 scripts/run_queneau_v2_smaller_models.py --dry-run  # spin up, verify, delete

Prerequisites:
    - `tnr` CLI on PATH or at /home/stefano/.tnr/bin/tnr
    - SSH key registered with ThunderCompute (any key valid on this host)
    - HF token at /home/stefano/hf_token (38 chars, hf_... prefix)
    - Queneau v2 .venv at ../Oulipo-Atlas-2-0/.venv with anthropic installed
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

# --- Paths ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
QUENEAU_DIR = PROJECT_ROOT.parent / "Oulipo-Atlas-2-0" / "queneau-notations"
RUN_PY = QUENEAU_DIR / "run.py"
VENV_PYTHON = QUENEAU_DIR.parent / ".venv" / "bin" / "python"
HF_TOKEN_FILE = Path("/home/stefano/hf_token")
LOG_PATH = PROJECT_ROOT / "scripts" / "_queneau_smaller_models.log"

# --- ThunderCompute config (mirrors SocioEroAnalysis's setup) ---
TNR_BIN = "/home/stefano/.tnr/bin/tnr"
TC_GPU = "a100"
TC_MODE = "prototyping"
TC_TEMPLATE = "base"
TC_DISK = 100
TC_VCPUS = 4
TC_VLLM_PORT = 8000
# Note: newer tnr CLI removed --ssh-key flag; keys are now managed at the
# account level via `tnr ssh-keys` and applied automatically to every instance.

# --- Model lineup ---
MODELS = [
    "Qwen/Qwen2.5-7B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "microsoft/Phi-3.5-mini-instruct",
]

# --- Queneau v2 runs (per model) ---
RUNS = [
    ("fr", "security-checkpoint"),
    ("it", "security-checkpoint"),
    ("en", "security-checkpoint"),
    ("en", "academic-hiring"),
]

# --- Timeouts ---
STARTUP_POLL_S = 15
STARTUP_TIMEOUT_S = 360
VLLM_STARTUP_TIMEOUT_S = 900
RUN_TIMEOUT_S = 900

# --- SSH options ---
# Note: in tnr v2.0.61, `tnr create` returns a per-instance SSH private key
# in its JSON response (the "key" field). We write that to a temp file and
# use `-i <tmp_file>` for direct SSH. No reliance on ~/.ssh/id_ed25519 or any
# pre-registered manual key. The temp file is cleaned up at instance delete.
SSH_BASE_OPTS = [
    "-o", "StrictHostKeyChecking=no",
    "-o", "IdentitiesOnly=yes",
    "-o", "UserKnownHostsFile=/dev/null",
    "-o", "LogLevel=ERROR",
    "-o", "ConnectTimeout=10",
]


# ============================================================================
# Logging
# ============================================================================

def log(msg: str, also_print: bool = True) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    if also_print:
        print(line, flush=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


# ============================================================================
# ThunderCompute lifecycle
# ============================================================================

def tnr_cmd(args: list, timeout: int = 60) -> tuple[bool, str]:
    """Run a tnr CLI command. Sources ~/.bash_profile so tnr is on PATH."""
    cmd = f"source ~/.bash_profile 2>/dev/null; {TNR_BIN} {' '.join(args)}"
    result = subprocess.run(
        ["bash", "-c", cmd], capture_output=True, text=True, timeout=timeout,
    )
    output = result.stdout.strip()
    if result.returncode != 0:
        err = result.stderr.strip()
        return False, (output or err)
    return True, output


def _parse_first_json_object(text: str) -> dict | None:
    """Extract the first JSON object from a possibly-noisy CLI output."""
    for line in text.split("\n"):
        line = line.strip()
        if line.startswith("{"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                pass
    # Try the whole output starting at the first `{`
    start = text.find("{")
    if start >= 0:
        try:
            return json.loads(text[start:])
        except json.JSONDecodeError:
            return None
    return None


def tc_create() -> dict:
    log("Creating ThunderCompute instance...")
    ok, output = tnr_cmd([
        "create", "--json",
        "--mode", TC_MODE,
        "--gpu", TC_GPU,
        "--template", TC_TEMPLATE,
        "--primary-disk", str(TC_DISK),
        "--vcpus", str(TC_VCPUS),
    ], timeout=120)
    if not ok:
        log(f"ERROR: tnr create failed: {output}")
        sys.exit(1)
    info = _parse_first_json_object(output)
    if info is None:
        log(f"ERROR: could not parse tnr create output:\n{output}")
        sys.exit(1)
    log(f"Instance created: uuid={info.get('uuid')}, id={info.get('identifier')}")
    return info


def tc_wait_running() -> dict | None:
    log("Waiting for instance to reach RUNNING state...")
    start = time.time()
    while time.time() - start < STARTUP_TIMEOUT_S:
        ok, output = tnr_cmd(["status", "--json"])
        if ok:
            try:
                instances = json.loads(output)
                if isinstance(instances, list) and instances:
                    inst = instances[0]
                    status = inst.get("status", "")
                    log(f"  status: {status}")
                    if status == "RUNNING":
                        return inst
            except json.JSONDecodeError:
                pass
        time.sleep(STARTUP_POLL_S)
    log("ERROR: instance did not reach RUNNING within timeout")
    return None


def tc_add_port(port: int = TC_VLLM_PORT) -> None:
    log(f"Adding port forwarding for {port}...")
    ok, output = tnr_cmd(["ports", "forward", "0", "--add", str(port), "--json"])
    if ok:
        log(f"  port forwarded: {output[:200]}")
    else:
        log(f"  WARNING: port forward may have failed: {output[:200]}")


def tc_delete() -> None:
    log("Deleting ThunderCompute instance...")
    ok, output = tnr_cmd(["delete", "0", "--yes", "--json"], timeout=60)
    if ok:
        log(f"  deleted: {output[:200]}")
    else:
        # Try without --json
        ok2, output2 = tnr_cmd(["delete", "0", "--yes"], timeout=60)
        log(f"  fallback delete: ok={ok2}, output={output2[:200]}")


def tc_endpoint(uuid: str, port: int = TC_VLLM_PORT) -> str:
    return f"https://{uuid}-{port}.thundercompute.net"


# ============================================================================
# SSH + vLLM setup
# ============================================================================

def ssh_cmd(ip: str, port: int, key_path: str, command: str, timeout: int = 300) -> tuple[bool, str]:
    ssh_args = ["ssh", *SSH_BASE_OPTS, "-i", key_path, "-p", str(port), f"ubuntu@{ip}", command]
    try:
        result = subprocess.run(ssh_args, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return False, "ssh timed out"
    output = (result.stdout + result.stderr).strip()
    return result.returncode == 0, output


def wait_ssh_ready(ip: str, port: int, key_path: str, timeout: int = 180) -> bool:
    """Poll ssh until 'echo ready' succeeds. Instance can report RUNNING
    before sshd is actually accepting connections."""
    log(f"Waiting for sshd on {ip}:{port}...")
    start = time.time()
    while time.time() - start < timeout:
        ok, out = ssh_cmd(ip, port, key_path, "echo ready", timeout=15)
        if ok and "ready" in out:
            log(f"  sshd up after {int(time.time()-start)}s")
            return True
        time.sleep(5)
    log(f"  ERROR: sshd didn't become ready within {timeout}s")
    return False


def setup_vllm(ip: str, port: int, key_path: str, model: str, hf_token: str) -> None:
    log("Installing vLLM (if needed)...")
    ssh_cmd(ip, port, key_path, "pip install --quiet vllm 2>&1 | tail -3", timeout=900)
    log("Running sudo ldconfig (CUDA path fix)...")
    ssh_cmd(ip, port, key_path, "sudo ldconfig 2>&1", timeout=30)
    log(f"Starting vLLM serving {model}...")
    vllm_cmd = (
        f"export HF_TOKEN={hf_token}; "
        f"nohup /home/ubuntu/.local/bin/vllm serve {model} "
        f"--port {TC_VLLM_PORT} --dtype bfloat16 --max-model-len 16384 "
        f"--gpu-memory-utilization 0.90 "
        f"> /tmp/vllm.log 2>&1 &"
    )
    ok, output = ssh_cmd(ip, port, key_path, vllm_cmd, timeout=30)
    log(f"  vLLM launch ok={ok}")


def wait_vllm_ready(endpoint: str) -> bool:
    """Poll the vLLM /v1/models endpoint via HTTPS until it returns the model list.

    Note on User-Agent: ThunderCompute's Cloudflare reverse-proxy returns
    HTTP 403 to Python's `urllib.request` default user-agent
    (`Python-urllib/X.Y`) — its anti-bot heuristics flag it as automated.
    `requests` and `curl` are not blocked. Using `requests` here avoids
    the silent-403 polling that wasted 3 model cycles on 2026-05-14
    before we diagnosed it.
    """
    import requests
    log(f"Waiting for vLLM at {endpoint}/v1/models...")
    start = time.time()
    while time.time() - start < VLLM_STARTUP_TIMEOUT_S:
        try:
            r = requests.get(f"{endpoint}/v1/models", timeout=10)
            if r.status_code == 200:
                data = r.json()
                models = [m["id"] for m in data.get("data", [])]
                if models:
                    log(f"  vLLM ready after {int(time.time()-start)}s, models: {models}")
                    return True
            # Non-200 (proxy 502/503 while vLLM is still loading, etc.) — keep polling.
        except requests.RequestException as e:
            # Brief debug-level log so we can see what's actually happening if it stalls.
            log(f"  vLLM not ready: {type(e).__name__}: {str(e)[:120]}")
        time.sleep(15)
    log("ERROR: vLLM did not become ready within timeout")
    return False


# ============================================================================
# Queneau v2 invocation
# ============================================================================

def run_queneau(model: str, endpoint: str, language: str, target: str) -> bool:
    log(f"  -> run.py: language={language}, target={target}")
    cmd = [
        str(VENV_PYTHON), str(RUN_PY),
        "--language", language, "--target", target, "--step", "all",
        "--backend", "vllm-http", "--endpoint", endpoint, "--model", model,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=RUN_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        log("     FAILED: run.py timed out")
        return False
    if result.returncode != 0:
        log(f"     FAILED (exit {result.returncode}): {result.stderr[-500:]}")
        return False
    # Last lines of stdout summarize tokens
    tail = "\n".join(result.stdout.strip().split("\n")[-4:])
    log(f"     done:\n{tail}")
    return True


# ============================================================================
# Per-model cycle
# ============================================================================

def cycle_model(model: str, hf_token: str, dry_run: bool = False) -> bool:
    log("")
    log("=" * 60)
    log(f"MODEL CYCLE: {model}")
    log("=" * 60)
    info = tc_create()
    uuid = info.get("uuid", "")

    # tnr create returns a per-instance SSH private key in the response.
    # Write it to a temp file with 0600 perms so ssh -i can use it.
    ssh_key_text = info.get("key", "")
    if not ssh_key_text:
        log("ERROR: no SSH key returned by tnr create — cannot proceed")
        tc_delete()
        return False
    key_fd, key_path = tempfile.mkstemp(prefix=f"tc-{uuid}-", suffix=".pem")
    os.write(key_fd, ssh_key_text.encode("utf-8"))
    os.close(key_fd)
    os.chmod(key_path, 0o600)
    log(f"  wrote per-instance SSH key to {key_path}")

    try:
        status = tc_wait_running()
        if not status:
            return False
        ip = status.get("ip", "")
        ssh_port = status.get("port", 0)
        uuid = status.get("uuid", uuid) or uuid
        if not ip or not ssh_port:
            log(f"ERROR: missing ip/port in status: {status}")
            return False
        log(f"RUNNING: ip={ip}, ssh_port={ssh_port}, uuid={uuid}")
        tc_add_port(TC_VLLM_PORT)
        endpoint = tc_endpoint(uuid)
        log(f"endpoint: {endpoint}")
        # sshd may not be accepting yet even when status=RUNNING; poll.
        if not wait_ssh_ready(ip, ssh_port, key_path):
            return False
        if dry_run:
            log("Dry run — verifying GPU then tearing down...")
            ok, out = ssh_cmd(ip, ssh_port, key_path,
                              "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader",
                              timeout=30)
            log(f"  GPU test: ok={ok}, output={out}")
            return ok
        setup_vllm(ip, ssh_port, key_path, model, hf_token)
        if not wait_vllm_ready(endpoint):
            return False
        results = []
        for lang, target in RUNS:
            ok = run_queneau(model, endpoint, lang, target)
            results.append((lang, target, ok))
        n_ok = sum(1 for _, _, ok in results if ok)
        log(f"Model {model}: {n_ok}/{len(results)} runs succeeded")
        return n_ok == len(results)
    finally:
        tc_delete()
        # Remove the temp key file
        try:
            os.unlink(key_path)
            log(f"  removed temp SSH key {key_path}")
        except OSError:
            pass


# ============================================================================
# Main
# ============================================================================

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", default=MODELS,
                        help="Models to run (default: all three)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Create one instance, verify SSH + GPU, delete; no vLLM, no queneau runs.")
    args = parser.parse_args()
    if not HF_TOKEN_FILE.exists():
        log(f"ERROR: HF token file not found at {HF_TOKEN_FILE}")
        return 1
    hf_token = HF_TOKEN_FILE.read_text().strip()
    if not hf_token.startswith("hf_"):
        log(f"ERROR: token at {HF_TOKEN_FILE} does not look like an HF token")
        return 1
    log(f"HF token loaded ({len(hf_token)} chars, prefix '{hf_token[:4]}...')")

    def cleanup(signum=None, frame=None):
        log("Caught signal — cleaning up instance...")
        tc_delete()
        sys.exit(1)
    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    log(f"Models to process: {args.models}")
    log(f"Runs per model: {len(RUNS)} ({RUNS})")

    failures = []
    for model in args.models:
        ok = cycle_model(model, hf_token, dry_run=args.dry_run)
        if not ok:
            failures.append(model)
        if args.dry_run:
            break  # one cycle is enough for a dry run

    log("")
    log("=" * 60)
    log("SUMMARY")
    log("=" * 60)
    log(f"Models attempted: {len(args.models)}")
    log(f"Failures: {failures}")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
