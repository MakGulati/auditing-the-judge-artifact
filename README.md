# Auditing the Judge: Internal Signals for LLM Reasoning Assurance

Public software and data artifact for the accepted IEEE TPS-ISA 2026 paper
*Auditing the Judge: Internal Signals for LLM Reasoning Assurance*, by
**Mayank Gulati and Gerhard Wunder**, Freie Universität Berlin.

- [Fixed camera-ready release](https://github.com/MakGulati/auditing-the-judge-artifact/releases/tag/tps2026-v1)
- [Supplementary material (PDF)](https://github.com/MakGulati/auditing-the-judge-artifact/releases/download/tps2026-v1/supplement.pdf)
- The arXiv identifier and IEEE publication DOI will be added when available.

The supplement contains the proofs, uncertainty protocol, baseline audits,
population accounting, robustness analyses, and reproducibility details removed
from the 12-page camera-ready paper. The main paper links to this fixed release.

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

The canonical end-to-end path uses Gemma 3 12B IT, the study's standard
checkpoint and one of the two models used in the additional routing experiment.
Install the verified vLLM environment:

```bash
python3.11 -m venv .venv-model
source .venv-model/bin/activate
python -m pip install -r requirements-vllm.txt
./run_full.sh
```

`run_full.sh` defaults to the reported Gemma 3/GSM8K configuration. Qwen2.5 and
Llama 3.1 use the same vLLM environment with their model-specific extractor and
head dimension; see [DEPENDENCIES.md](DEPENDENCIES.md).

Once the complete model environment is installed, run all 363 tests with:

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
paper sections, bibliography, compiled main-paper PDF, or checked-in LaTeX table
fragments. The separate supplementary PDF is included under `supplement/`. The table generator is retained as analysis code, but its output is
ignored by Git.

## Scope and limitations

The checked-in aligned predictions reproduce the reported analyses without the
approximately 20 GB activation dumps. Raw generations and activation dumps are
regenerable but are not redistributed because of size and upstream model/dataset
terms. Their hashes, run settings, exclusions, and software versions are retained
in the metadata. Hardware-dependent numerical variation and known limitations are
documented in the artifact package.

## License and citation

The authors' software code and its software documentation are released under
[MIT](LICENSE). The paper, supplementary PDF, research figures, and cached or
derived research data are outside that code license; no additional reuse license
is granted for those materials. Upstream datasets and model checkpoints retain
their provider terms. See [LICENSES.md](LICENSES.md) for scope.

Please cite the paper when using the artifact in research. Citation metadata is
provided in [CITATION.cff](CITATION.cff).
