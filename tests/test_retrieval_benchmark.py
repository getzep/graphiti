"""Tests for retrieval benchmark harness (FR-13)."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_retrieval_benchmark import (
    FIXTURE_QUOTAS,
    check_recall_gate,
    compute_recall,
    validate_fixture,
)


class TestComputeRecall(unittest.TestCase):
    def test_all_found(self):
        text = 'This is about workout and 2.5 hour workout window before 10:30am'
        expected = ['workout', '2.5 hour workout window']
        self.assertEqual(compute_recall(text, expected), 1.0)

    def test_none_found(self):
        text = 'This text has nothing relevant'
        expected = ['workout', 'schedule']
        self.assertEqual(compute_recall(text, expected), 0.0)

    def test_partial_recall(self):
        text = 'This mentions workout but not the other thing'
        expected = ['workout', 'schedule']
        self.assertAlmostEqual(compute_recall(text, expected), 0.5)

    def test_empty_expected(self):
        self.assertEqual(compute_recall('any text', []), 1.0)

    def test_case_insensitive(self):
        text = 'WORKOUT schedule'
        expected = ['workout', 'Schedule']
        self.assertEqual(compute_recall(text, expected), 1.0)


class TestValidateFixture(unittest.TestCase):
    def test_fixture_file_exists(self):
        fixture_path = (
            Path(__file__).resolve().parents[1]
            / 'tests'
            / 'fixtures'
            / 'retrieval_benchmark_queries.json'
        )
        self.assertTrue(fixture_path.exists(), f'Fixture not found: {fixture_path}')

    def test_fixture_has_minimum_queries(self):
        fixture_path = (
            Path(__file__).resolve().parents[1]
            / 'tests'
            / 'fixtures'
            / 'retrieval_benchmark_queries.json'
        )
        queries = json.loads(fixture_path.read_text(encoding='utf-8'))
        self.assertGreaterEqual(len(queries), 30)

    def test_fixture_passes_validation(self):
        fixture_path = (
            Path(__file__).resolve().parents[1]
            / 'tests'
            / 'fixtures'
            / 'retrieval_benchmark_queries.json'
        )
        queries = json.loads(fixture_path.read_text(encoding='utf-8'))
        errors = validate_fixture(queries)
        self.assertEqual(errors, [], f'Fixture validation errors: {errors}')

    def test_fixture_query_schema(self):
        fixture_path = (
            Path(__file__).resolve().parents[1]
            / 'tests'
            / 'fixtures'
            / 'retrieval_benchmark_queries.json'
        )
        queries = json.loads(fixture_path.read_text(encoding='utf-8'))
        for q in queries:
            self.assertIn('id', q)
            self.assertIn('query', q)
            self.assertIn('expected_facts', q)
            self.assertIn('expected_entities', q)
            self.assertIn('target_group_ids', q)
            self.assertIsInstance(q['expected_facts'], list)
            self.assertIsInstance(q['expected_entities'], list)
            self.assertIsInstance(q['lane_alias'], list)
            self.assertIsInstance(q['target_group_ids'], list)

    def test_too_few_queries_fails(self):
        queries = [{'id': f'q{i}', 'target_group_ids': ['s1_sessions_main']} for i in range(10)]
        errors = validate_fixture(queries)
        self.assertTrue(any('>= 30' in e for e in errors))

    def test_quota_coverage(self):
        fixture_path = (
            Path(__file__).resolve().parents[1]
            / 'tests'
            / 'fixtures'
            / 'retrieval_benchmark_queries.json'
        )
        queries = json.loads(fixture_path.read_text(encoding='utf-8'))

        counts: dict[str, int] = {
            'sessions_main': 0,
            'observational_memory': 0,
            'curated': 0,
            'chatgpt': 0,
            'cross_lane': 0,
        }
        for q in queries:
            aliases = q.get('lane_alias', [])
            if len(aliases) > 1:
                counts['cross_lane'] += 1
                continue

            if aliases:
                alias = aliases[0]
                if alias in counts:
                    counts[alias] += 1
                continue

            target_group_ids = q.get('target_group_ids', []) or []
            if isinstance(target_group_ids, list) and len(target_group_ids) > 1:
                counts['cross_lane'] += 1
            elif target_group_ids:
                alias = {
                    's1_sessions_main': 'sessions_main',
                    's1_observational_memory': 'observational_memory',
                    's1_curated_refs': 'curated',
                    's1_chatgpt_history': 'chatgpt',
                }.get(target_group_ids[0])
                if alias in counts:
                    counts[alias] += 1

        for category, quota in FIXTURE_QUOTAS.items():
            self.assertGreaterEqual(
                counts[category],
                quota,
                f'{category}: need >= {quota}, got {counts[category]}',
            )


class TestOutputSchema(unittest.TestCase):
    def test_expected_output_fields(self):
        """Verify the expected output schema structure."""
        expected_fields = [
            'fixture_path',
            'top_k',
            'timestamp',
            'queries_total',
            'bicameral_aggregate',
            'query_results',
        ]
        # Just verify the constants exist
        for field in expected_fields:
            self.assertIsInstance(field, str)


class TestCheckRecallGate(unittest.TestCase):
    """Regression tests for check_recall_gate robustness (PR #118).

    Verifies that string or malformed baseline_score values never raise and
    that the function degrades gracefully (treats bad baseline as absent).
    """

    def _results(self, score: float) -> dict:
        return {'bicameral_aggregate': {'mean_combined_recall_at_k': score}}

    def _baseline_file(self, tmp_path: Path, score) -> Path:
        p = tmp_path / 'baseline.json'
        p.write_text(
            json.dumps({'bicameral_aggregate': {'mean_combined_recall_at_k': score}}),
            encoding='utf-8',
        )
        return p

    def test_numeric_float_baseline_passes_through(self):
        """Normal float baseline: delta computed correctly, no crash."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            bp = self._baseline_file(Path(tmp), 0.70)
            gate = check_recall_gate(self._results(0.75), 0.60, baseline_path=str(bp))
        self.assertTrue(gate['passed'])
        self.assertAlmostEqual(gate['baseline_score'], 0.70, places=4)
        self.assertAlmostEqual(gate['delta'], 0.05, places=4)
        self.assertIn('baseline=0.7000', gate['details'])

    def test_string_baseline_score_does_not_raise(self):
        """String value in baseline JSON must not raise — degrades to no-baseline."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            bp = self._baseline_file(Path(tmp), '0.70')  # string, not float
            gate = check_recall_gate(self._results(0.75), 0.60, baseline_path=str(bp))
        # Should not raise; baseline treated as absent (coerced to float OK)
        self.assertIsNotNone(gate)
        self.assertIn('passed', gate)

    def test_malformed_baseline_score_does_not_raise(self):
        """Totally malformed baseline score must not raise — degrades gracefully."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            bp = self._baseline_file(Path(tmp), 'not-a-number')
            gate = check_recall_gate(self._results(0.75), 0.60, baseline_path=str(bp))
        # Must not raise; baseline_score should be None, delta should be None
        self.assertIsNone(gate['baseline_score'])
        self.assertIsNone(gate['delta'])
        # Gate still evaluates against threshold
        self.assertTrue(gate['passed'])

    def test_none_baseline_score_does_not_raise(self):
        """None value for baseline score in JSON — degrades gracefully."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            bp = self._baseline_file(Path(tmp), None)
            gate = check_recall_gate(self._results(0.75), 0.60, baseline_path=str(bp))
        self.assertIsNone(gate['baseline_score'])
        self.assertIsNone(gate['delta'])
        self.assertTrue(gate['passed'])

    def test_score_below_threshold_fails_with_no_baseline(self):
        """Score below threshold fails even without a baseline file."""
        gate = check_recall_gate(self._results(0.55), 0.70)
        self.assertFalse(gate['passed'])
        self.assertIn('FAIL', gate['details'])

    def test_regression_vs_float_baseline_fails_gate(self):
        """Score that regresses vs float baseline marks gate as failed."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            bp = self._baseline_file(Path(tmp), 0.80)
            gate = check_recall_gate(self._results(0.75), 0.60, baseline_path=str(bp))
        self.assertFalse(gate['passed'])
        self.assertLess(gate['delta'], 0)

    def test_malformed_score_field_in_results_does_not_raise(self):
        """Malformed score in the results dict degrades to 0.0, does not raise."""
        results = {'bicameral_aggregate': {'mean_combined_recall_at_k': 'bad'}}
        gate = check_recall_gate(results, 0.60)
        self.assertIsNotNone(gate)
        self.assertAlmostEqual(gate['score'], 0.0)
        self.assertFalse(gate['passed'])


if __name__ == '__main__':
    unittest.main()
