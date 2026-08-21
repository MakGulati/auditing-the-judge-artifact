# IEEE TPS 2026 figure package

This directory is generated from the repository's held-out experiment artifacts. The
primary method is fixed to the raw top-32-head, 5-seed MLP ensemble. The
controlled MLP, shallow-feature probe, and LR are ablations only.

## Reproduce

```bash
MPLCONFIGDIR=/tmp .venv/bin/python filtering/make_ieee_tps_figures.py --compute
```

Use `--force-predictions` only when the immutable source dumps or fitting code change.
Aligned predictions are retained so re-rendering does not require loading ~20 GB of
activations.

## Main figures

- `fig1_selective_risk.pdf`: residual wrong-rate reduction at fixed review budgets;
  paired 95% bootstrap intervals.
- `fig2_discrimination.pdf`: raw-MLP minus p(YES) AUROC and error-detection AP;
  paired 95% bootstrap intervals.
- `fig3_selective_routing.pdf`: accuracy gain over random compute allocation on the
  Llama/GSM8K k=5 run, including the raw primary probe.
- `fig4_cross_dataset_transfer.pdf`: in-domain versus cross-dataset AUROC; point
  estimates only because the source transfer cache has no aligned bootstrap draws.

## Appendix figures

- `figA1_risk_coverage.pdf`: full tie-averaged risk–coverage curves.
- `figA2_probe_sensitivity.pdf`: descriptive min–max gaps over the stored sweep grid;
  extrema are test-selected and are not confidence intervals.
- `figA3_symbolic_perturbation.pdf`: paired GSM-Symbolic AUROC changes.
- `figA4_math_level_breakdown.pdf`: descriptive MATH difficulty strata; no intervals.

Every figure is supplied as vector PDF and 600-dpi PNG. Main multi-panel figures use
IEEE double-column width (7.16 in); the routing and compact appendix figures use
single-column width (3.5 in). `statistics.json`, CSV tables, per-run metadata, and
`artifact_manifest.json` trace the plotted values to source files. Manuscript prose,
caption drafts, and LaTeX sources are intentionally excluded from this repository.

The figures use a fixed color-vision-deficiency-safe Okabe--Ito palette and embedded
Liberation Serif typography (a Times-compatible TrueType face). Method, dataset, and
model colors are paired with consistent line styles and markers so distinctions
survive grayscale printing.

## Metric definitions

- AUROC ranks correctness; AP ranks the minority error class.
- Risk–coverage accepts the highest correctness scores first. AURC is the mean
  selective risk over all non-empty coverages (lower is better).
- Fixed-budget recall is the fraction of all wrong answers captured among the lowest
  scoring `round(budget*n)` items.
- Fixed-budget residual risk is the wrong fraction among unreviewed items.
- Routing gain subtracts the expected accuracy of random routing at equal solve cost.
- Cutoffs that bisect score ties use the exact expectation under uniform random
  tie-breaking; this matters for saturated p(YES) values.

The MLP outputs are called *scores*, not calibrated probabilities. No ECE or Brier
score is reported because the current MLP oversamples the minority class and the LR
uses class weights without a separate train-only calibration stage.

## Reproducibility note

The rebuilt MLP scores use scikit-learn 1.9.0; per-run metadata records
the full analysis stack and preserves comparisons with the older paper caches. The
paper should disclose substantive generative-AI assistance as required by the IEEE
TPS 2026 call for papers.
