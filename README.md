# Anonymous Review Artifact: Trustworthy Deferral for LLM Reasoning

This repository is the anonymous software and data artifact accompanying the
manuscript *Trustworthy Deferral for LLM Reasoning*. It is a fresh snapshot with
no upstream Git history, author metadata, credentials, model weights, or local
machine paths.

The release supports three levels of reproduction:

1. **Numerical checks (CPU, minutes).** Re-derive the submission's numerical
   identities from checked-in caches.
2. **Paper figures (CPU, minutes).** Re-render all main and appendix figures from
   checked-in statistics and aligned held-out predictions.
3. **End-to-end experiments (GPU, hours to days).** Generate solver and judge
   records, extract attention-head activations, fit the probe on a training
   split, and evaluate it on a disjoint test split.

## Quick verification

Python 3.11 is recommended. For the CPU-only artifact checks:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-analysis.txt -r requirements-dev.txt
make verify
```

`make verify` runs the 176 CPU-safe analysis tests, checks the submission's
numerical identities, and renders figures from the archived statistics. It does
not download models or datasets. Two tests are skipped when their optional inputs
are unavailable.

To recompute the figure statistics and bootstrap intervals from the portable
aligned-prediction arrays rather than the archived `statistics.json`:

```bash
make statistics
```

## End-to-end reproduction

The default Transformers path uses the separately pinned environment in
`requirements-ministral-extraction.txt`:

```bash
python3.11 -m venv .venv-model
source .venv-model/bin/activate
python -m pip install -r requirements-ministral-extraction.txt
./run_full.sh
```

The vLLM path for Gemma, Qwen, and Llama-family checkpoints must use a separate
environment because its verified Transformers version conflicts with the
Ministral stack. See [DEPENDENCIES.md](DEPENDENCIES.md) and the comments in
`requirements-vllm.txt`.

Once either complete model environment is installed, run all 363 tests with:

```bash
make test-full
```

Model checkpoints and benchmark datasets are downloaded from their original
providers. Some checkpoints require accepting provider terms and authenticating
with Hugging Face. No credentials are stored here.

## Repository map

| Path | Purpose |
|---|---|
| `run_eval.py` | solver/judge generation entry point |
| `run_full.sh` | clean train/test generation, extraction, and probe evaluation |
| `extract_hidden_rich*.py` | architecture-specific activation extraction |
| `loaders/`, `prompts/` | dataset loading and prompt contracts |
| `metrics/` | correctness, provenance, and verdict-token semantics |
| `filtering/` | probe fitting, baselines, transfer/routing analyses, figures |
| `filtering/figures/ieee_tps_2026/` | aligned predictions, statistics, figures, and manifest |
| `paper/` | numeric caches and verification utilities; no manuscript source |
| `tests/` | unit and synthetic pipeline tests |

See [ARTIFACTS.md](ARTIFACTS.md) for the provenance and inclusion policy.

The manuscript itself is deliberately excluded: this repository contains no
paper sections, bibliography, compiled manuscript PDF, or checked-in LaTeX table
fragments. The table generator is retained as analysis code, but its output is
ignored by Git.

## Scope and limitations

The checked-in aligned predictions reproduce the reported analyses without the
approximately 20 GB activation dumps. Raw generations and activation dumps are
regenerable but are not redistributed because of size and upstream model/dataset
terms. Their hashes, run settings, exclusions, and software versions are retained
in the metadata. Hardware-dependent numerical variation and known limitations are
documented in the artifact package.

During anonymous review, please do not attempt to identify the authors from model
hosting logs, repository traffic, or other external metadata.
