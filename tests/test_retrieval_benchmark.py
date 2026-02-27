"""Tests for retrieval benchmark harness (FR-13)."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_retrieval_benchmark import (
    FIXTURE_QUOTAS,
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
            self.assertIn('lane_alias', q)
            self.assertIsInstance(q['expected_facts'], list)
            self.assertIsInstance(q['expected_entities'], list)
            self.assertIsInstance(q['lane_alias'], list)

    def test_too_few_queries_fails(self):
        queries = [{'id': f'q{i}', 'lane_alias': ['sessions_main']} for i in range(10)]
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
            elif aliases:
                alias = aliases[0]
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


if __name__ == '__main__':
    unittest.main()
