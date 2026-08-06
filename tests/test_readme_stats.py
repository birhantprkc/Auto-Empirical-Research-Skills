"""Tests for the README rigor-stats consistency checker."""

from __future__ import annotations

import unittest

from _helpers import ROOT, load_module

check_readme_stats = load_module("scripts/check-readme-stats.py", "aers_check_readme_stats")


class TestExpectedCounts(unittest.TestCase):
    def test_counts_match_committed_toml_files(self):
        n_tasks, n_scenarios, n_rubric = check_readme_stats.expected_counts()
        self.assertEqual(n_tasks, len(list((ROOT / "benchmark" / "tasks").glob("*.toml"))))
        self.assertEqual(n_scenarios, len(list((ROOT / "eval-harness" / "scenarios").glob("*.toml"))))
        self.assertGreater(n_rubric, n_scenarios)  # every scenario has >= 1 rubric item


class TestCheckReadme(unittest.TestCase):
    def setUp(self):
        self.counts = check_readme_stats.expected_counts()

    def test_committed_readmes_are_consistent(self):
        for name in check_readme_stats.READMES:
            with self.subTest(readme=name):
                problems = check_readme_stats.check_readme(ROOT / name, *self.counts)
                self.assertEqual(problems, [])

    def test_stale_bolded_count_is_caught(self):
        import tempfile
        from pathlib import Path

        n_tasks, n_scenarios, n_rubric = self.counts
        stale = (
            f"| Numeric benchmark tasks | **{n_tasks + 1}** | [`benchmark/`](benchmark/) |\n"
            f"| Eval scenarios | **{n_scenarios} / {n_rubric}** | [`eval-harness/`](eval-harness/) |\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "README-test.md"
            path.write_text(stale, encoding="utf-8")
            problems = check_readme_stats.check_readme(path, *self.counts)
        self.assertEqual(len(problems), 1)
        self.assertIn("benchmark row says", problems[0])

    def test_stale_suffix_style_count_is_caught(self):
        import tempfile
        from pathlib import Path

        n_tasks, n_scenarios, n_rubric = self.counts
        stale = (
            f"| **数值基准** | 陷阱 | [`benchmark/`](benchmark/) · {n_tasks} 任务 |\n"
            f"| **评测套件** | 失误 | [`eval-harness/`](eval-harness/) · {n_scenarios - 1} 场景 / {n_rubric} rubric |\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "README-test.md"
            path.write_text(stale, encoding="utf-8")
            problems = check_readme_stats.check_readme(path, *self.counts)
        self.assertEqual(len(problems), 1)
        self.assertIn("eval-harness row says", problems[0])

    def test_missing_rows_are_caught(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "README-test.md"
            path.write_text("# empty\n", encoding="utf-8")
            problems = check_readme_stats.check_readme(path, *self.counts)
        self.assertEqual(len(problems), 2)


if __name__ == "__main__":
    unittest.main()

class TestCollectionCoverage(unittest.TestCase):
    def setUp(self):
        self.catalog_ids, self.total = check_readme_stats.catalog_facts()

    def test_catalog_facts_shape(self):
        self.assertGreaterEqual(len(self.catalog_ids), 70)
        self.assertGreater(self.total, 500)

    def test_committed_collection_tables_are_complete(self):
        for name in check_readme_stats.COLLECTION_TABLE_DOCS:
            with self.subTest(doc=name):
                problems = check_readme_stats.check_collections(
                    ROOT / name, self.catalog_ids, self.total
                )
                self.assertEqual(problems, [])

    def test_missing_collection_is_caught(self):
        import tempfile
        from pathlib import Path

        some_id = sorted(self.catalog_ids)[0]
        rows = "\n".join(
            f"| [{cid}](skills/{cid}/) |" for cid in sorted(self.catalog_ids) if cid != some_id
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "README-test.md"
            path.write_text(rows, encoding="utf-8")
            problems = check_readme_stats.check_collections(path, self.catalog_ids, self.total)
        self.assertTrue(any("missing 1 cataloged collection" in p for p in problems))

    def test_unknown_collection_link_is_caught(self):
        import tempfile
        from pathlib import Path

        rows = "\n".join(
            f"| [{cid}](skills/{cid}/) |" for cid in sorted(self.catalog_ids)
        ) + "\n| [gone](skills/99-deleted-collection/) |"
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "README-test.md"
            path.write_text(rows, encoding="utf-8")
            problems = check_readme_stats.check_collections(path, self.catalog_ids, self.total)
        self.assertTrue(any("not in catalog" in p for p in problems))

    def test_widest_table_is_the_all_collections_table(self):
        # Every entry document carries the all-collections table plus smaller
        # by-theme groupings; the source column only exists on the widest one.
        for name in check_readme_stats.COLLECTION_TABLE_DOCS:
            with self.subTest(doc=name):
                text = (ROOT / name).read_text(encoding="utf-8")
                rows = check_readme_stats.widest_collection_table(text)
                ids = {
                    m for row in rows for m in check_readme_stats.COLLECTION_LINK_RE.findall(row)
                }
                self.assertEqual(ids, self.catalog_ids)

    def test_stale_catalog_cited_total_is_caught(self):
        import tempfile
        from pathlib import Path

        rows = "\n".join(
            f"| [{cid}](skills/{cid}/) |" for cid in sorted(self.catalog_ids)
        ) + "\n| skills | **1,151** | [`catalog/skills.json`](catalog/skills.json) |"
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "README-test.md"
            path.write_text(rows, encoding="utf-8")
            problems = check_readme_stats.check_collections(path, self.catalog_ids, self.total)
        self.assertTrue(any("stale" in p or "catalog total" in p for p in problems))


class TestSourceLinks(unittest.TestCase):
    """The 来源/Source column must stay pinned to catalog/provenance.json."""

    def setUp(self):
        self.sources = check_readme_stats.provenance_sources()

    def test_every_cataloged_collection_has_a_source_url(self):
        catalog_ids, _ = check_readme_stats.catalog_facts()
        self.assertEqual(catalog_ids - set(self.sources), set())

    def test_committed_tables_link_their_upstream(self):
        for name in check_readme_stats.COLLECTION_TABLE_DOCS:
            with self.subTest(doc=name):
                problems = check_readme_stats.check_source_links(ROOT / name, self.sources)
                self.assertEqual(problems, [])

    def test_wrong_source_url_is_caught(self):
        import tempfile
        from pathlib import Path

        cid = sorted(self.sources)[0]
        rows = "\n".join(
            f"| [{c}](skills/{c}/) | [x]({self.sources[c]}) |" if c != cid
            else f"| [{c}](skills/{c}/) | [x](https://github.com/wrong/repo) |"
            for c in sorted(self.sources)
        )
        table = "| a | b |\n|---|---|\n" + rows + "\n"
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "README-test.md"
            path.write_text(table, encoding="utf-8")
            problems = check_readme_stats.check_source_links(path, self.sources)
        self.assertEqual(len(problems), 1)
        self.assertIn(cid, problems[0])
        self.assertIn("does not link its upstream source", problems[0])

    def test_missing_source_column_is_caught(self):
        import tempfile
        from pathlib import Path

        rows = "\n".join(f"| [{c}](skills/{c}/) |" for c in sorted(self.sources))
        table = "| a |\n|---|\n" + rows + "\n"
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "README-test.md"
            path.write_text(table, encoding="utf-8")
            problems = check_readme_stats.check_source_links(path, self.sources)
        self.assertEqual(len(problems), len(self.sources))

    def test_no_collection_table_is_caught(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "README-test.md"
            path.write_text("# nothing here\n", encoding="utf-8")
            problems = check_readme_stats.check_source_links(path, self.sources)
        self.assertEqual(len(problems), 1)
        self.assertIn("no collection table found", problems[0])

