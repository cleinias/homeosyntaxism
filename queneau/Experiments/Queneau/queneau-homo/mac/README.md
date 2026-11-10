# Minimal-computing run — MacBook Air (Ollama)

Self-contained overnight run that validates the **student-laptop tier** of the tool on the
**actual target stack**: Apple-silicon unified memory + Metal + GGUF, via Ollama. No repo, no
Python packages beyond the system `python3`. Scoped to the **≤16 GB tier only**:

| Ollama model | quant | ~size | output tag |
|---|---|---|---|
| `qwen2.5:7b-instruct-q4_K_M` | Q4_K_M | 4.7 GB | `mac-7b-q4` |
| `qwen2.5:7b-instruct-q8_0` | Q8_0 | 8.1 GB | `mac-7b-q8` |
| `qwen2.5:14b-instruct-q4_K_M` | Q4_K_M | 9.0 GB | `mac-14b-q4` |

A 24 GB Air is a faithful proxy for a 16 GB machine here: these three configs produce **identical
output** on 16 GB and 24 GB — RAM headroom changes only whether it fits and how fast, never the
tokens. (Do **not** add 14B-Q8 or 32B: they don't fit 16 GB, so they can't back the claim.)

The run is **source-blind and prompt-identical to the ACES/vLLM runs** — the prompt builder is
copied verbatim, so only the box + quantization differ. That makes this a clean
**cross-quant / cross-stack robustness** check, the version that actually earns the claim.

## Prerequisites (once)

1. **Ollama** — install the app from <https://ollama.com/download> (it runs a background server on
   `127.0.0.1:11434`). Verify: `ollama --version`.
2. **python3** — ships with Xcode Command Line Tools; if `python3` is missing, run
   `xcode-select --install`. No `pip` packages are needed (pure stdlib).
3. Free disk: ~22 GB for the three models (stored in `~/.ollama`), plus a few MB of outputs.

## Run

```bash
tar xzf manzoni-mac-mincompute.tar.gz
cd manzoni-mac-mincompute
./run_mac_subset.sh
```

- It **self-caffeinates** (`caffeinate -i`) so the Mac won't sleep mid-run — but also plug in
  power and, if you like, keep the lid open.
- It **pulls all three models first** (retries; a bad pull is reported immediately, not at 3am),
  then generates.
- It is **idempotent**: each cell writes one file and is skipped on re-run, so if it's
  interrupted you can just run it again to finish.
- **Time:** ~3.5–4.5 h for 360 generations (plus the one-time ~22 GB download). Overnight is ample.
  7B-Q4 ≈ 15–20 tok/s, 14B-Q4 ≈ 8–12 tok/s on the M2; the fanless throttling only affects speed.

Knobs (env): `N` (default 10), `TEMP` (0.9), `ENDPOINT` (Ollama URL). To smoke-test first, run one
cell by hand:

```bash
python3 gen_ollama.py --model qwen2.5:7b-instruct-q4_K_M \
    --lang en --degree D2 --scenario security-checkpoint --n 2 --tag smoke
```

## After the run — score on your main machine

Do **not** score on the Mac; keep the same instruments as the cluster runs. Copy `./outputs/`
back into `Experiments/Queneau/queneau-homo/outputs/` in the repo, then:

```bash
<atlas-venv>/bin/python scripts/queneau_score_homo.py --glob 'homo_*mac-*'
```

Same Stanza structural metrics + same Claude coherence judge as the ACES analysis, so the desktop
rows sit directly beside the fp8 rows. **The datum:** does D2 coherence (ACES fp8 ≈ 0.98) survive
`Q4_K_M`/`Q8_0` on a laptop? Structure should be quant-robust; coherence is the axis to watch.

## What's in here

```
gen_ollama.py         standalone generator (stdlib only; loops N because Ollama ignores n>1;
                      prompt logic + output schema copied verbatim from queneau_run_homo.py)
run_mac_subset.sh     driver: pull → generate the EN subset → write outputs/
spec/                 specs_en_S0.json, specs_en_D0.json, specs_en_D1.json, specs_en_D2.json (the source form specs)
outputs/              created at runtime; the generations land here
```
