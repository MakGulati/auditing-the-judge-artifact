# Artifact inventory and provenance

## Included

- Complete generation, judging, activation-extraction, probe-fitting, routing,
  transfer, symbolic-perturbation, and plotting source used by the study.
- Dependency manifests for CPU analysis, Transformers extraction, and vLLM.
- Tests for prompt parity, label semantics, split/run provenance, truncation,
  aggregation, routing policy, and paper-pipeline compatibility.
- Portable aligned held-out arrays for all six model--dataset cells. These arrays
  contain labels, detector scores, shallow features, indices, and routing fields;
  they do not contain model weights or free-form benchmark responses.
- Per-run JSON metadata recording model identifiers, train/test population sizes,
  seeds, selected heads, exclusions, generation settings, source-file hashes, and
  analysis-library versions.
- Cached statistics, CSV tables, rendered PDF/PNG figures, and a SHA-256 manifest.
- Numeric metric caches and scripts that can regenerate tables and verify the
  identities used by the submission. Generated LaTeX tables are not distributed.

## Not included

- Git history or author/committer identity from the private development repository.
- API keys, Hugging Face tokens, local editor/agent settings, virtual environments,
  logs, caches, or absolute workstation paths.
- Public checkpoint weights; these are downloaded from the named model providers.
- Raw benchmark generations and activation dumps. Together these exceed practical
  Git hosting limits and may inherit upstream redistribution constraints.
- Manuscript source, bibliography, compiled manuscript PDF, generated LaTeX table
  fragments, caption drafts, and paper-story prose.

## Provenance chain

```text
provider dataset + checkpoint
  -> run_eval.py (raw generation and self-judgment records)
  -> extract_hidden_rich*.py (activation dumps with prompt/run fingerprints)
  -> filtering/probe_models.py (train-only head selection and fitted scores)
  -> aligned_predictions/*.npz + *.metadata.json
  -> filtering/make_ieee_tps_figures.py
  -> statistics.json, CSVs, figures, and artifact_manifest.json
  -> paper/scripts/{make_tables.py,verify_identities.py}
```

Row fingerprints and prompt identities prevent an activation dump from being
silently paired with a different generation run. Train/test provenance checks
reject overlapping or incompatible splits. The archived metadata preserves hashes
for large source files that are not redistributed.

## Integrity

Run:

```bash
make audit
```

This checks tracked-file anonymity patterns, rejects oversized files, verifies the
portable NumPy archives can be loaded without pickle, and validates the archived
manifest entries whose targets are included in this release.
