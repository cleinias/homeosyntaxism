# Tools & computational stack

Reference list of every tool/library/service used across the project's phases —
including those **used and then discarded** — for the talk Q&A and (eventually) the
public-repo README. Each entry: name · site · one-paragraph description · what we used it for.

---

## 1. Linguistic analysis (parsing, POS, metrics)

### Stanza — https://stanfordnlp.github.io/stanza/  *(core)*
Stanford NLP's Python toolkit: neural pipelines (tokenize, POS/UPOS, lemma,
dependency parse) trained on Universal Dependencies treebanks, with models for
~70 languages including Italian and English that share the same UD label set.
**Used for:** the backbone of every structural metric — we parse both the source
incipit and each generation to Universal Dependencies and read off the
clause-architecture (clause-type multiset, hypotaxis depth), POS sequence, lemmas,
and dependency triples. The language-shared UD scheme is why the Italian and
English (Moore) tracks run on the *same* metric code (`tree_metric.py`,
`build_d2_specs.py`).

### spaCy — https://spacy.io/  *(used early, then discarded for the metric pipeline)*
Industrial NLP library with fast statistical/transformer pipelines for tagging,
parsing, and NER. **Used for:** early POS tagging and some corpus-construction
helpers. **Discarded** as the metric parser because the small Italian model
systematically mis-tagged adjectives as nouns (`ADJ`→`NOUN`), which corrupts the
POS-sequence and clause metrics; we moved the measurement pipeline to Stanza
(UPOS ~0.98 on our material). Still appears in some non-metric utility scripts.

### UDPipe 2 — https://ufal.mff.cuni.cz/udpipe  *(POS cross-check)*
Charles University (Prague) trainable UD pipeline (tokenizer, tagger, parser),
available as a library and a public REST service. **Used for:** an *independent*
UD POS tagger to cross-validate Stanza's tagging on the source — a second opinion
for establishing POS ground-truth (the "Stanza/UDPipe2 POS tags" comparison), so
the degree-ladder metrics don't rest on a single tagger's idiosyncrasies.

### sentence-transformers (paraphrase-multilingual-mpnet) — https://sbert.net/  *(discarded)*
Library of pretrained sentence/text embedding models (Sentence-BERT family) for
semantic similarity. **Used for:** a first attempt at the *frame-divergence* metric
(axis 2) via multilingual MPNet cosine similarity. **Discarded** because the
embedding could not separate a frame-mirror calque (same scene, new words) from
genuine divergence — it scored both as "different." Replaced by an LLM-judge.

### fastText (cc.it.300), loaded via gensim — https://fasttext.cc/ · https://radimrehurek.com/gensim/
fastText = Facebook/Meta subword word-embedding vectors (here the 300-d Italian
Common-Crawl model `cc.it.300`); gensim = a Python library for loading and querying
such vector spaces. **Used for:** the domain-distance metric in the domain-transfer
ladder — measuring how far candidate target domains (travel, migration, medicine…)
sit from the source's lexical field, to choose equidistant domains.

### Edit distance (Levenshtein) — in-house
A small in-house implementation (`tree_metric._lev`). **Used for:** the `s0_pos`
metric — normalized edit similarity between the source's and a generation's
part-of-speech *sequences* (order-sensitive surface-form proxy), which tells us
whether the surface was genuinely freed (low) versus calqued (high).

### LLM-as-judge (Claude) — see §3
**Used for:** the two semantic metrics the structural tools couldn't capture —
frame-divergence ("same scene, only relexicalized?") and coherence ("one coherent
passage?"), validated against a hand-labeled gold set before being trusted
(`sem_metrics.py`). This is a *tool role*, implemented on the Anthropic API.

---

## 2. LLM inference & serving

### vLLM — https://docs.vllm.ai/  ·  https://github.com/vllm-project/vllm
High-throughput LLM inference/serving engine (PagedAttention, tensor-parallelism,
an OpenAI-compatible server, FP8/quantized execution). **Used for:** self-hosting
the large open-weight models on the clusters — serving Qwen3-235B and
Llama-4-Maverick at FP8 across 4–8 H100 GPUs and generating against the
OpenAI-compatible endpoint. Its FP8 path pulls low-level GEMM kernels
(**Triton, Marlin, CUTLASS, FlashInfer, DeepGEMM**); we pin the Triton block-FP8
path (`VLLM_USE_DEEP_GEMM=0`) because the DeepGEMM kernel isn't installed.

### HuggingFace Transformers / Accelerate / bitsandbytes — https://huggingface.co/docs/transformers
Transformers = the model-loading/inference library; Accelerate = device/parallelism
orchestration; bitsandbytes = int8/nf4 quantization. **Used for:** the earlier
local-inference path on FASTER (A100s, no working vLLM build there) — loading
Mistral/Qwen checkpoints directly, including 4-bit, before the project moved the
large runs to vLLM on H100s.

### HuggingFace Hub — https://huggingface.co/
Model/dataset registry and download CLI/library. **Used for:** fetching the
open-weight checkpoints (Mistral-Large, Qwen3-235B-FP8, Llama-4-Maverick-FP8 via
the ungated RedHatAI mirror, etc.) into the cluster caches.

---

## 3. Model access (hosted APIs)

### Anthropic API (Claude) — https://www.anthropic.com/  ·  https://docs.anthropic.com/
API for the Claude models (we use **Sonnet 4.6** and **Opus 4.8**). **Used for:**
(a) a generation substrate (the frontier reference + the early runs), and
(b) the **measurement instrument** — the frame-divergence and coherence LLM-judges,
and the compliance auditor (held constant on Sonnet so the judge doesn't vary with
the model under test).

### OpenRouter — https://openrouter.ai/
An aggregator that exposes many providers' models behind one OpenAI-compatible API,
with explicit **provider routing** (you can pin which backend provider and
quantization serve a model). **Used for:** cluster-independent, reproducible
open-weight runs — Qwen3-235B and DeepSeek-V4-pro, each **pinned to an fp8 provider**
(DeepInfra / Baidu, `allow_fallbacks:false`) so precision matches the self-hosted
runs and the resolved provider is recorded for reproducibility. (DeepSeek-V4-pro was
ultimately dropped for this task: as a reasoning model it spent its entire token
budget on the hidden trace and never emitted the structured passage.)

### Mistral API — https://mistral.ai/
Mistral's hosted model API. **Used for:** the early Queneau "notations" ladder and
Mistral-model generations via direct API before the multi-substrate cluster work.

---

## 4. Compute infrastructure

### TAMU HPRC — FASTER — https://hprc.tamu.edu/
Texas A&M High-Performance Research Computing cluster (A100-PCIE-40GB and other
GPUs). **Used for:** the first open-weight survey and the homeosyntax development
runs (transformers/accelerate local inference; vLLM does not build there, capping it
at ~24B models).

### ACCESS / TAMU ACES — https://access-ci.org/  ·  https://hprc.tamu.edu/resources/aces/
ACES is an ACCESS-allocated accelerator cluster (NVIDIA H100, FP8-capable).
**Used for:** the flagship large-model self-hosted runs (Qwen3-235B, Llama-4-Maverick
at FP8 via vLLM across 4–8 H100). Cert-based SSH login; billed in Service Units.

### SLURM — https://slurm.schedmd.com/
The cluster workload manager / job scheduler. **Used for:** submitting and monitoring
all cluster jobs (`sbatch` scripts requesting H100s, partitions, memory, time).

### Conda / Python venv — https://docs.conda.io/  ·  https://www.python.org/
Environment management. **Used for:** the per-machine software environments (the
cluster `atlas-llm` vLLM env; the local analysis venv with Stanza/gensim/etc.).

---

## 5. Models exercised (the *subjects*, not tools)

Open-weight: **Qwen3-235B-A22B-Instruct-2507** (+ FP8), **Llama-4-Maverick-17B-128E**
(FP8), **Llama-3.1-405B** (FP8), **DeepSeek-V4-pro** (dropped — reasoning),
**Mistral-Large-2411**, **Mistral-Small-24B**, **Mistral-Nemo-12B**, **Qwen2.5-14B**.
Closed/frontier: **Claude Sonnet 4.6**, **Claude Opus 4.8**. Auditor held constant on
Sonnet 4.6 throughout.

---

## 6. Authoring, bibliography, version control

### LyX / LaTeX — https://www.lyx.org/
Document processor (LyX is a structured front-end to LaTeX). **Used for:** the talk
slides and the paper/outline (the `.lyx`/`.tex` sources; figures via TikZ).

### biber / biblatex — https://ctan.org/pkg/biblatex
LaTeX bibliography system + its backend. **Used for:** the consolidated bibliography
(`Global-biblio.bib`), including datamodel validation (`biber --tool --validate-datamodel`).

### Git + Codeberg / Bitbucket — https://codeberg.org/  ·  https://bitbucket.org/
Version control and hosting. **Used for:** the private project repo (Bitbucket +
Codeberg mirrors) and the planned public reproducibility repo
(`codeberg.org/Cleinias/homeosyntax`), composed as a submodule per the Oulipian-cube
pattern.

---

## 7. Supporting Python libraries
**NumPy** (numerics/aggregation), **Matplotlib** (figures, dependency-tree renders),
**python-dotenv** (API-key loading), plus the standard library (urllib for the API
calls, json, argparse, pathlib, …).
