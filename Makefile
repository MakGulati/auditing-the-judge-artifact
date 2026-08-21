PYTHON ?= python3
MPLCONFIGDIR ?= /tmp/anonymous-review-matplotlib
PREDICTIONS := filtering/figures/ieee_tps_2026/aligned_predictions

.PHONY: verify test test-analysis test-full paper-check tables figures statistics audit

verify: test-analysis paper-check figures audit

test: test-full

test-analysis:
	$(PYTHON) -m pytest -q \
		tests/test_aggregation_consistency.py \
		tests/test_correctness.py \
		tests/test_exclusions.py \
		tests/test_filtering_labels.py \
		tests/test_latex_answer.py \
		tests/test_metrics_semantics.py \
		tests/test_paper_table.py \
		tests/test_paper_tag_selection.py \
		tests/test_route_policy.py \
		tests/test_symbolic_stats.py \
		tests/test_undefined_quantity_formatting.py \
		tests/test_verdict_tokens.py

test-full:
	$(PYTHON) -m pytest -q

paper-check:
	$(PYTHON) paper/scripts/verify_identities.py --predictions $(PREDICTIONS)

tables:
	$(PYTHON) paper/scripts/make_tables.py

figures:
	MPLCONFIGDIR=$(MPLCONFIGDIR) $(PYTHON) filtering/make_ieee_tps_figures.py --render-only

statistics:
	MPLCONFIGDIR=$(MPLCONFIGDIR) $(PYTHON) filtering/make_ieee_tps_figures.py

audit:
	$(PYTHON) scripts/audit_release.py
