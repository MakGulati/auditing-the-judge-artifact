"""The LaTeX table's column spec, header and body must agree — a mismatch only
surfaces when the paper fails to compile."""
import re
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "filtering"))
import make_paper_figures as F


def fake_metrics(auc_p_yes, recall=0.55, precision=0.42, resample_sd=0.03):
    return {
        "title": "Fake", "n_test": 1000, "wrong_test": 120,
        "judge_verdict_balanced_acc": 0.63, "auc_p_yes": auc_p_yes,
        "auc_lr_raw": 0.80, "auc_lr_controlled": 0.78,
        # Ensemble point estimates: the same aggregation the curves and annotations
        # use. `*_seed_sd` is the initialisation-only spread, reported separately.
        "auc_mlp_raw": 0.82, "auc_mlp_controlled": 0.81,
        "auc_mlp_controlled_seed_sd": 0.01, "n_seeds_mlp": 5,
        "auc_shallow": 0.70,
        "auc_mlp_controlled_resample_sd": resample_sd,
        "auc_mlp_controlled_resample_n": 8,
        "judge_rec": 0.33, "judge_prec": 0.31,
        "error_detection": {"mlp_controlled": {"recall": recall, "precision": precision}},
    }


class TableShapeTests(unittest.TestCase):
    """Row labels name the dataset as well as the model. A table listing
    "Gemma3-12B" beside "Gemma3-12B (MATH)" invites the reader to take the
    unlabelled row as the norm rather than as a second dataset."""

    def test_row_label_names_the_dataset(self):
        tex = self._render(0.71)
        self.assertIn("Fake (GSM8K)", tex)

    def _render(self, auc_p_yes, **kw):
        with tempfile.TemporaryDirectory() as tmp:
            F.OUT = Path(tmp)
            F.table_tex({"a": fake_metrics(auc_p_yes, **kw),
                         "b": fake_metrics(auc_p_yes, **kw)})
            # The filename carries the selection, so two selections cannot
            # overwrite each other; see make_paper_figures.selection_slug.
            return (Path(tmp) / "table_probe_auc__a+b.tex").read_text()

    def _check(self, tex):
        spec = re.search(r"\\begin\{tabular\}\{(\w+)\}", tex).group(1)
        header = next(l for l in tex.splitlines() if l.startswith("Judge model"))
        body = [l for l in tex.splitlines() if l.startswith("Fake (GSM8K) &")]
        self.assertEqual(len(spec), len(header.split("&")))
        for row in body:
            self.assertEqual(len(spec), len(row.split("&")))
        # cmidrules must tile the columns after the two leading label columns
        rules = re.findall(r"\\cmidrule\(lr\)\{(\d+)-(\d+)\}", tex)
        self.assertEqual(int(rules[0][0]), 3)
        self.assertEqual(int(rules[-1][1]), len(spec))
        self.assertEqual(int(rules[1][0]), int(rules[0][1]) + 1)

    def test_with_p_yes_column(self):
        tex = self._render(0.71)
        self.assertIn("p_{\\mathrm{YES}}", tex)
        self._check(tex)

    def test_without_p_yes_column(self):
        tex = self._render(None)
        self._check(tex)

    def test_notes_flag_the_verdict_column_as_not_an_auc(self):
        self.assertIn("NOT an AUC", self._render(0.71))

    def test_unreachable_operating_point_renders_dash_not_zero(self):
        """A literal 0 would read as "the probe caught nothing" rather than "no
        operating point existed", so the cell must be an em-dash."""
        tex = self._render(0.71, recall=None, precision=None)
        row = next(l for l in tex.splitlines() if l.startswith("Fake (GSM8K) &"))
        self.assertIn("-- (--)", row)
        self.assertNotIn(" 0 (", row)
        self._check(tex)

    def test_recall_cells_carry_realised_precision(self):
        row = next(l for l in self._render(0.71).splitlines() if l.startswith("Fake (GSM8K) &"))
        # verdict recall 33% at precision 31%, probe recall 55% at precision 42%
        self.assertIn("33 (31)", row)
        self.assertIn("55 (42)", row)

    def test_pm_states_which_variance_it_is(self):
        """The +/- must say what varied: init-only badly understates the real spread."""
        tex = self._render(0.71)
        self.assertIn("head selection", tex)
        self.assertIn("train resamples", tex)

    def test_the_two_uncertainty_sources_are_reported_separately(self):
        """A five-initialisation mean paired with a train-resample sd reads as one
        conventional mean +/- sd while being two unrelated quantities. The point
        estimate is the ensemble, the +/- is named, and the initialisation-only spread
        is stated apart from it."""
        tex = self._render(0.71)
        self.assertIn("initialisation-only", tex)
        self.assertIn("NOT an error bar on the ensemble", tex)
        self.assertIn("not combined", tex)

    def test_no_resample_pass_means_no_error_bar_at_all(self):
        """CONTRACT CHANGE: the +/- used to fall back to the initialisation-only sd
        when no resample pass had run, which silently swapped one uncertainty source
        for a much smaller one. It is now simply absent and the caption says so."""
        tex = self._render(0.71, resample_sd=None)
        self.assertIn("UNAVAILABLE", tex)
        self.assertNotIn("$\\pm$", tex)

    def test_caption_states_the_aggregation(self):
        """Curves, operating points and table values must be one aggregation, and the
        reader has to be able to tell which."""
        tex = self._render(0.71)
        self.assertIn("ENSEMBLE", tex)
        self.assertIn("probabilities averaged", tex)



class OracleExclusionTests(unittest.TestCase):
    """The best-threshold-on-test recall is a best-of-all-thresholds statistic on the
    reporting split. It is kept for reference but must never reach the main results."""

    def test_oracle_recall_is_not_in_the_table(self):
        with tempfile.TemporaryDirectory() as tmp:
            F.OUT = Path(tmp)
            m = fake_metrics(0.71)
            m["error_detection"]["mlp_controlled"]["oracle_recall"] = 0.99
            F.table_tex({"a": m, "b": fake_metrics(0.71)})
            tex = (Path(tmp) / "table_probe_auc__a+b.tex").read_text()
        self.assertNotIn("oracle", tex.lower())
        self.assertNotIn("99", tex)

    def test_table_names_the_precision_as_realised_on_test(self):
        """'matched precision' claimed an equalisation that never happened; the header
        has to say the precision is whatever the train-calibrated rule realises."""
        tex = self._render_for_header()
        self.assertIn("realised test P", tex)
        self.assertIn("train-calibrated op", tex)
        self.assertNotIn("matched", tex.lower())

    def _render_for_header(self):
        with tempfile.TemporaryDirectory() as tmp:
            F.OUT = Path(tmp)
            F.table_tex({"a": fake_metrics(0.71), "b": fake_metrics(0.71)})
            return (Path(tmp) / "table_probe_auc__a+b.tex").read_text()

    def test_caption_denies_precision_matching(self):
        tex = self._render_for_header()
        self.assertIn("does NOT match precision on test", tex)
        self.assertIn("not directly", tex)


if __name__ == "__main__":
    unittest.main()
