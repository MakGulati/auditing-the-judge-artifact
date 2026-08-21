"""Which runs make_paper_figures acts on, and how that selection reaches the output.

RUNS is a registry of every run the paper has ever included, so on any one machine most
entries have no data. The rules pinned here:

  * the selection is always explicit — there is no default, because "whatever is on
    disk" made the output set a property of the machine rather than of the command;
  * naming a tag never silently does nothing;
  * the selection appears in every output filename, so two selections cannot overwrite
    each other's figures.
"""
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "filtering"))
import make_paper_figures as F


class TagSelectionCase(unittest.TestCase):
    """A temp ROOT with `present` fully populated and `absent` registered but empty."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.out = root / "paper"
        self.out.mkdir()

        runs = {
            "present": dict(title="Present", head_dim=128,
                            train=root / "results_present_train",
                            test=root / "results_present_test"),
            "absent": dict(title="Absent", head_dim=128,
                           train=root / "results_absent_train",
                           test=root / "results_absent_test"),
            "present_math": dict(title="Present MATH", head_dim=128, dataset="math",
                                 train=root / "results_present_math_train",
                                 test=root / "results_present_math_test"),
        }
        for tag in ("present", "present_math"):
            cfg = runs[tag]
            ds = cfg.get("dataset", "gsm8k")
            for split in ("train", "test"):
                (cfg[split] / ds).mkdir(parents=True)
                (cfg[split] / "hidden_rich.npz").write_text("")
                (cfg[split] / ds / "raw.jsonl").write_text("")
        (self.out / "metrics_present.json").write_text("{}")

        self.runs = runs
        p1 = mock.patch.object(F, "RUNS", runs)
        p2 = mock.patch.object(F, "OUT", self.out)
        p1.start()
        p2.start()
        self.addCleanup(p1.stop)
        self.addCleanup(p2.stop)
        self.addCleanup(self.tmp.cleanup)


class TestMissingInputs(TagSelectionCase):
    def test_complete_run_has_nothing_missing(self):
        self.assertEqual(F.missing_inputs("present", self.runs["present"]), [])

    def test_dataset_selects_the_raw_subdirectory(self):
        # The MATH run's records live under `math/`, not `gsm8k/`. Checking the wrong
        # subdirectory would report a complete run as missing and vice versa.
        cfg = self.runs["present_math"]
        self.assertEqual(F.missing_inputs("present_math", cfg), [])
        (cfg["train"] / "math" / "raw.jsonl").unlink()
        self.assertEqual([Path(p).name for p in F.missing_inputs("present_math", cfg)],
                         ["raw.jsonl"])

    def test_reports_every_missing_artifact_not_just_the_first(self):
        missing = F.missing_inputs("absent", self.runs["absent"])
        self.assertEqual(len(missing), 4)


class TestParseTagList(unittest.TestCase):
    """Separator handling, split out because it has nothing to do with the filesystem."""

    def test_whitespace_around_separators_is_tolerated(self):
        # "a, b" used to fail with "unknown run tag: ' b'", which reads as a typo in
        # the tag rather than in the separator.
        self.assertEqual(F.parse_tag_list("a, b"), ["a", "b"])
        self.assertEqual(F.parse_tag_list(" a ,b "), ["a", "b"])

    def test_duplicates_collapse(self):
        # "a,a" used to render the same run as two identical panels.
        self.assertEqual(F.parse_tag_list("a,a"), ["a"])
        self.assertEqual(F.parse_tag_list("a, a ,b,a"), ["a", "b"])

    def test_order_is_preserved(self):
        self.assertEqual(F.parse_tag_list("b,a"), ["b", "a"])

    def test_empty_parts_are_dropped(self):
        self.assertEqual(F.parse_tag_list("a,,b,"), ["a", "b"])
        self.assertEqual(F.parse_tag_list(" , "), [])


class TestResolveTags(TagSelectionCase):
    def test_no_selection_is_fatal_and_explains_the_options(self):
        """CONTRACT CHANGE: a bare invocation used to plot whatever happened to be on
        disk, so the same command produced different figures on different machines."""
        with self.assertRaises(SystemExit) as cm:
            F.resolve_tags(None, compute=True)
        msg = str(cm.exception)
        self.assertIn("--runs is required", msg)
        for option in ("--runs all", "--runs present", "--runs a,b"):
            self.assertIn(option, msg)

    def test_no_selection_lists_what_is_actually_available(self):
        with self.assertRaises(SystemExit) as cm:
            F.resolve_tags(None, compute=True)
        self.assertIn("present", str(cm.exception))

    def test_all_is_machine_independent(self):
        """'all' is every REGISTERED tag, so the output set is a property of the
        registry rather than of this machine's disk — and a missing one is fatal."""
        with self.assertRaises(SystemExit) as cm:
            F.resolve_tags("all", compute=True)
        self.assertIn("absent", str(cm.exception))

    def test_all_succeeds_when_every_registered_run_has_data(self):
        with mock.patch.object(F, "RUNS", {k: v for k, v in self.runs.items()
                                           if k != "absent"}):
            self.assertEqual(F.resolve_tags("all", compute=True),
                             ["present", "present_math"])

    def test_present_opts_into_machine_dependence_explicitly(self):
        self.assertEqual(F.resolve_tags("present", compute=True),
                         ["present", "present_math"])

    def test_present_uses_cache_presence_when_replotting(self):
        # present_math has dumps but no cached metrics, so it is computable and not
        # yet plottable. The two modes must not share one availability test.
        self.assertEqual(F.resolve_tags("present", compute=False), ["present"])

    def test_present_with_nothing_available_explains_what_to_do(self):
        with mock.patch.object(F, "RUNS", {"absent": self.runs["absent"]}):
            with self.assertRaises(SystemExit) as cm:
                F.resolve_tags("present", compute=True)
            self.assertIn("RUNS", str(cm.exception))

    def test_explicit_missing_tag_is_fatal_not_skipped(self):
        with self.assertRaises(SystemExit) as cm:
            F.resolve_tags("absent", compute=True)
        self.assertIn("absent", str(cm.exception))

    def test_explicit_error_names_the_missing_files(self):
        with self.assertRaises(SystemExit) as cm:
            F.resolve_tags("absent", compute=True)
        self.assertIn("hidden_rich.npz", str(cm.exception))

    def test_explicit_replot_error_points_at_compute(self):
        with self.assertRaises(SystemExit) as cm:
            F.resolve_tags("present_math", compute=False)
        self.assertIn("--compute", str(cm.exception))

    def test_partially_available_request_fails_rather_than_dropping_one(self):
        # Silently plotting one of two requested panels would produce a figure that
        # looks finished and answers a different question than the one asked.
        with self.assertRaises(SystemExit):
            F.resolve_tags("present,absent", compute=True)

    def test_unknown_tag_lists_known_ones(self):
        with self.assertRaises(SystemExit) as cm:
            F.resolve_tags("nosuch", compute=True)
        msg = str(cm.exception)
        self.assertIn("unknown run tag", msg)
        self.assertIn("present", msg)

    def test_unknown_tag_beats_availability_in_the_error(self):
        # A typo must report the typo, not "no data for nosuch" — the latter sends you
        # looking for a directory that was never supposed to exist.
        with self.assertRaises(SystemExit) as cm:
            F.resolve_tags("nosuch", compute=False)
        self.assertIn("unknown run tag", str(cm.exception))

    def test_explicit_order_is_preserved(self):
        # Panel order on the figure follows the order given on the command line.
        self.assertEqual(F.resolve_tags("present_math,present", compute=True),
                         ["present_math", "present"])

    def test_whitespace_separated_request_resolves(self):
        self.assertEqual(F.resolve_tags("present_math, present", compute=True),
                         ["present_math", "present"])

    def test_duplicate_request_yields_one_panel(self):
        self.assertEqual(F.resolve_tags("present,present", compute=False), ["present"])

    def test_trailing_comma_is_ignored(self):
        self.assertEqual(F.resolve_tags("present,", compute=True), ["present"])


class TestSelectionSlug(unittest.TestCase):
    """Output filenames must identify their inputs, or a re-plot with a different
    selection silently replaces a figure a draft already cites."""

    def test_slug_names_the_runs(self):
        self.assertEqual(F.selection_slug(["gemma3", "llama31_8b"]),
                         "gemma3+llama31_8b")

    def test_different_selections_differ(self):
        self.assertNotEqual(F.selection_slug(["a", "b"]), F.selection_slug(["a"]))

    def test_order_is_part_of_the_identity(self):
        # Panel order differs, so the figures differ and must not share a filename.
        self.assertNotEqual(F.selection_slug(["a", "b"]), F.selection_slug(["b", "a"]))

    def test_long_selections_hash_instead_of_producing_unusable_names(self):
        tags = [f"some_rather_long_run_tag_{i}" for i in range(8)]
        slug = F.selection_slug(tags)
        self.assertLessEqual(len(slug), 24)
        self.assertTrue(slug.startswith("8runs-"))

    def test_long_selections_still_distinguish(self):
        a = [f"some_rather_long_run_tag_{i}" for i in range(8)]
        b = list(a)
        b[0] = "some_rather_long_run_tag_x"
        self.assertNotEqual(F.selection_slug(a), F.selection_slug(b))


class TestCacheConfigBinding(TagSelectionCase):
    """`metrics_{tag}.json` is keyed by tag alone, so reusing a tag for a different run
    left a cache that validated and plotted as if it were current."""

    def test_digest_changes_with_every_bound_field(self):
        base = dict(title="T", head_dim=128, dataset="gsm8k",
                    train=Path("/a"), test=Path("/b"))
        original = F.run_config_digest("tag", base)
        for field, value in (("head_dim", 256), ("dataset", "math"),
                             ("train", Path("/c")), ("test", Path("/d"))):
            self.assertNotEqual(
                original, F.run_config_digest("tag", {**base, field: value}),
                f"{field} does not affect the config digest")

    def test_a_relabel_does_not_invalidate_the_cache(self):
        """CONTRACT CHANGE: `title` was in the digest, so renaming a panel forced a
        20-minute recompute. It is display text and cannot change any computed value;
        the paths, dataset and head_dim are what prove the entry still points at the
        same run."""
        base = dict(title="T", head_dim=128, train=Path("/a"), test=Path("/b"))
        self.assertEqual(F.run_config_digest("tag", base),
                         F.run_config_digest("tag", {**base, "title": "Renamed"}))

    def test_digest_changes_with_the_tag(self):
        cfg = dict(title="T", head_dim=128, train=Path("/a"), test=Path("/b"))
        self.assertNotEqual(F.run_config_digest("a", cfg), F.run_config_digest("b", cfg))

    def test_absent_dataset_defaults_to_gsm8k(self):
        cfg = dict(title="T", head_dim=128, train=Path("/a"), test=Path("/b"))
        self.assertEqual(F.run_config_digest("t", cfg),
                         F.run_config_digest("t", {**cfg, "dataset": "gsm8k"}))

    def test_digest_is_stable_across_calls(self):
        cfg = self.runs["present"]
        self.assertEqual(F.run_config_digest("present", cfg),
                         F.run_config_digest("present", cfg))


if __name__ == "__main__":
    unittest.main()
