# Dependency and hardware guide

## CPU analysis environment

Use `requirements-analysis.txt` plus `requirements-dev.txt`. It contains NumPy,
SciPy, scikit-learn, matplotlib, and pytest. This environment is sufficient for
all checked-in artifact verification, table generation, and figure reproduction.
It runs the CPU-safe analysis subset of the tests through `make verify`.

## Transformers generation/extraction environment

Use `requirements-ministral-extraction.txt`. The pinned Transformers and
`mistral_common` versions are required for the verified Ministral configuration.
PyTorch wheels must match the host's supported CUDA version.
This environment also enables the complete synthetic/unit test suite.

## vLLM generation environment

Use `requirements-vllm.txt` in a separate environment. vLLM pins a Transformers
line that is incompatible with the verified Ministral configuration. The file
records the verified package versions and backend constraints.
This environment also enables the complete synthetic/unit test suite.

## External resources

- Linux and Python 3.11 are recommended.
- End-to-end runs require a CUDA-capable NVIDIA GPU. The reported runs used a
  single NVIDIA RTX 5000 Ada Generation GPU with 32 GiB memory; the pinned vLLM
  environment was verified with driver 555.
- Internet access is needed once for benchmark and checkpoint downloads.
- Gated Hugging Face checkpoints require the reviewer to accept the provider's
  license and authenticate locally. Tokens must be supplied through the provider's
  normal credential mechanism and must never be committed.

## Reproducibility boundary

Greedy decoding can vary across GPU/software stacks, and MLP fitting can vary in
the third decimal because of BLAS threading. The paper's point estimates can be
reproduced exactly from the archived aligned predictions; a newly generated
end-to-end run should be treated as an independent replication rather than as a
byte-identical rebuild.
