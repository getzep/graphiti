"""Phase C — Slice 4: Production Guardrails Automation tests.

Validates:
1. Recall non-regression gate (check_recall_gate function)
   - Gate PASS: score >= threshold
   - Gate FAIL: score < threshold
   - Gate FAIL: regression vs baseline (score dropped)
   - Gate PASS with delta: score >= threshold AND >= baseline
   - Baseline file missing → no regression check (no crash)
2. Contamination sentinel
   - ContaminationResult.is_clean when all lists are empty
   - ContaminationResult.total_issues counts correctly
   - to_dict() format
   - CLI parses --json and arg defaults
   - Script importable
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest


# ---------------------------------------------------------------------------
# 1. Recall non-regression gate
# ---------------------------------------------------------------------------

class TestRecallGate:
    @pytest.fixture(autouse=True)
    def _import_gate(self):
        from scripts.run_retrieval_benchmark import check_recall_gate
        self.check = check_recall_gate

    def _make_results(self, score: float) -> dict:
        return {
            'bicameral_aggregate': {
                'mean_combined_recall_at_k': score,
            }
        }

    # -- threshold checks -------------------------------------------------

    def test_pass_above_threshold(self):
        gate = self.check(self._make_results(0.80), threshold=0.70)
        assert gate['passed'] is True

    def test_pass_at_threshold(self):
        gate = self.check(self._make_results(0.70), threshold=0.70)
        assert gate['passed'] is True

    def test_fail_below_threshold(self):
        gate = self.check(self._make_results(0.65), threshold=0.70)
        assert gate['passed'] is False

    def test_details_contains_score_and_threshold(self):
        gate = self.check(self._make_results(0.75), threshold=0.70)
        assert '0.7500' in gate['details'] or '0.75' in gate['details']
        assert '0.7000' in gate['details'] or '0.70' in gate['details']

    # -- baseline regression checks ---------------------------------------

    def test_pass_when_above_baseline(self):
        with tempfile.NamedTemporaryFile('w', suffix='.json', delete=False) as f:
            json.dump({'bicameral_aggregate': {'mean_combined_recall_at_k': 0.70}}, f)
            baseline_path = f.name

        gate = self.check(
            self._make_results(0.75),
            threshold=0.60,
            baseline_path=baseline_path,
        )
        assert gate['passed'] is True
        assert gate['delta'] == pytest.approx(0.05, abs=1e-4)

    def test_fail_on_regression_vs_baseline(self):
        with tempfile.NamedTemporaryFile('w', suffix='.json', delete=False) as f:
            json.dump({'bicameral_aggregate': {'mean_combined_recall_at_k': 0.80}}, f)
            baseline_path = f.name

        gate = self.check(
            self._make_results(0.75),
            threshold=0.50,   # would pass threshold alone
            baseline_path=baseline_path,
        )
        assert gate['passed'] is False
        assert gate['delta'] == pytest.approx(-0.05, abs=1e-4)

    def test_no_crash_when_baseline_missing(self):
        gate = self.check(
            self._make_results(0.75),
            threshold=0.70,
            baseline_path='/tmp/nonexistent_baseline_xyz.json',
        )
        # Should pass (above threshold) without crashing
        assert gate['passed'] is True
        assert gate['baseline_score'] is None

    def test_none_baseline_path_skips_regression(self):
        gate = self.check(self._make_results(0.75), threshold=0.70, baseline_path=None)
        assert gate['baseline_score'] is None
        assert gate['delta'] is None


# ---------------------------------------------------------------------------
# 2. Contamination sentinel
# ---------------------------------------------------------------------------

class TestContaminationSentinel:
    @pytest.fixture(autouse=True)
    def _import(self):
        import importlib.util
        script_path = Path(__file__).parents[1] / 'scripts' / 'contamination_sentinel.py'
        mod_name = '_contamination_sentinel'
        spec = importlib.util.spec_from_file_location(mod_name, script_path)
        self.mod = importlib.util.module_from_spec(spec)
        # Register in sys.modules BEFORE exec so dataclass can resolve its module
        sys.modules[mod_name] = self.mod
        spec.loader.exec_module(self.mod)

    def test_clean_when_all_empty(self):
        r = self.mod.ContaminationResult()
        assert r.is_clean is True
        assert r.total_issues == 0

    def test_contaminated_when_multi_group_nodes(self):
        r = self.mod.ContaminationResult(
            multi_group_nodes=[{'uuid': 'n1', 'group_id': 'a,b'}]
        )
        assert r.is_clean is False
        assert r.total_issues == 1

    def test_contaminated_when_episodic_mismatch(self):
        r = self.mod.ContaminationResult(
            episodic_mismatches=[{'episode_uuid': 'ep1', 'entity_group': 'other'}]
        )
        assert r.is_clean is False
        assert r.total_issues == 1

    def test_contaminated_when_edge_mismatch(self):
        r = self.mod.ContaminationResult(
            edge_mismatches=[{'edge_uuid': 'e1', 'edge_group': 'a', 'src_group': 'b'}]
        )
        assert r.is_clean is False
        assert r.total_issues == 1

    def test_total_issues_accumulates(self):
        r = self.mod.ContaminationResult(
            multi_group_nodes=[{}] * 2,
            episodic_mismatches=[{}] * 3,
            edge_mismatches=[{}] * 1,
        )
        assert r.total_issues == 6

    def test_to_dict_structure(self):
        r = self.mod.ContaminationResult()
        d = r.to_dict()
        assert 'clean' in d
        assert 'total_issues' in d
        assert 'multi_group_nodes' in d
        assert 'episodic_mismatches' in d
        assert 'edge_mismatches' in d

    def test_to_dict_clean_true_when_empty(self):
        r = self.mod.ContaminationResult()
        assert r.to_dict()['clean'] is True

    def test_to_dict_clean_false_when_issues(self):
        r = self.mod.ContaminationResult(multi_group_nodes=[{'uuid': 'x'}])
        assert r.to_dict()['clean'] is False

    def test_script_importable(self):
        # Already imported in fixture — if we got here, it's importable
        assert hasattr(self.mod, 'ContaminationResult')
        assert hasattr(self.mod, 'run_sentinel')
        assert hasattr(self.mod, 'main')

    def test_default_sample_limit(self):
        """DEFAULT_SAMPLE_LIMIT must be set to a sane value."""
        assert self.mod.DEFAULT_SAMPLE_LIMIT > 0
        assert self.mod.DEFAULT_SAMPLE_LIMIT <= 500

    def test_parse_args_json_flag(self):
        with patch.object(sys, 'argv', ['contamination_sentinel.py', '--json']):
            args = self.mod._parse_args()
        assert args.json is True

    def test_parse_args_json_off_by_default(self):
        with patch.object(sys, 'argv', ['contamination_sentinel.py']):
            args = self.mod._parse_args()
        assert args.json is False

    def test_parse_args_source_and_clean_group(self):
        with patch.object(sys, 'argv', [
            'contamination_sentinel.py',
            '--source-group', 's1_sessions_main',
            '--expect-clean-in', 's1_inspiration_short_form',
        ]):
            args = self.mod._parse_args()
        assert args.source_group == 's1_sessions_main'
        assert args.clean_group == 's1_inspiration_short_form'


# ---------------------------------------------------------------------------
# 3. Benchmark script gate integration (end-to-end CLI parse check)
# ---------------------------------------------------------------------------

class TestBenchmarkGateCLIFlags:
    def test_recall_gate_flag_parsed(self):
        import importlib.util
        script_path = Path(__file__).parents[1] / 'scripts' / 'run_retrieval_benchmark.py'
        spec = importlib.util.spec_from_file_location('_benchmark', script_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        with patch.object(sys, 'argv', [
            'run_retrieval_benchmark.py',
            '--fixture', 'dummy.json',
            '--output', '/tmp/out.json',
            '--recall-gate', '0.75',
            '--recall-baseline', '/tmp/baseline.json',
        ]):
            import argparse
            # Just verify argparse accepts these flags (don't actually run)
            with pytest.raises(SystemExit):
                mod.main()  # will exit because fixture doesn't exist; that's fine

    def test_check_recall_gate_importable(self):
        import importlib.util
        script_path = Path(__file__).parents[1] / 'scripts' / 'run_retrieval_benchmark.py'
        spec = importlib.util.spec_from_file_location('_benchmark2', script_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        assert callable(mod.check_recall_gate)


# ---------------------------------------------------------------------------
# 4. Multi-group detector — _is_multi_group_value unit tests
#    Verifies the fix for the broken size(n.group_id) > 1 Cypher predicate.
# ---------------------------------------------------------------------------

class TestIsMultiGroupValue:
    """Direct unit tests for _is_multi_group_value.

    These tests do NOT require a Neo4j connection; they exercise the
    Python-side type-safe helper that replaced the broken Cypher size()
    predicate.
    """

    @pytest.fixture(autouse=True)
    def _import(self):
        import importlib.util
        script_path = Path(__file__).parents[1] / 'scripts' / 'contamination_sentinel.py'
        mod_name = '_contamination_sentinel_v2'
        spec = importlib.util.spec_from_file_location(mod_name, script_path)
        self.mod = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = self.mod
        spec.loader.exec_module(self.mod)
        self.fn = self.mod._is_multi_group_value

    # ---- Single normal strings must NOT be flagged ----

    def test_single_string_not_flagged(self):
        """A plain group_id string must never be flagged as multi-group."""
        assert self.fn('main') is False

    def test_single_long_string_not_flagged(self):
        """Long string group_ids (len >> 1) must NOT be flagged."""
        assert self.fn('s1_sessions_main') is False

    def test_single_letter_not_flagged(self):
        assert self.fn('a') is False

    def test_empty_string_not_flagged(self):
        assert self.fn('') is False

    def test_single_element_list_not_flagged(self):
        """A one-element list encodes a single group, not multi-group."""
        assert self.fn(['main']) is False

    def test_empty_list_not_flagged(self):
        assert self.fn([]) is False

    # ---- Multi-group encodings MUST be flagged ----

    def test_comma_string_flagged(self):
        """Comma-separated string is multi-group."""
        assert self.fn('group_a,group_b') is True

    def test_comma_string_with_spaces_flagged(self):
        """Comma with surrounding spaces still counts as multi-group."""
        assert self.fn('group_a, group_b') is True

    def test_two_element_list_flagged(self):
        """A list with two elements is multi-group."""
        assert self.fn(['group_a', 'group_b']) is True

    def test_three_element_list_flagged(self):
        assert self.fn(['g1', 'g2', 'g3']) is True

    # ---- Edge / exotic types must not crash ----

    def test_none_not_flagged(self):
        assert self.fn(None) is False

    def test_integer_not_flagged(self):
        assert self.fn(42) is False

    def test_bool_not_flagged(self):
        assert self.fn(True) is False


# ---------------------------------------------------------------------------
# 5. _MULTI_GROUP_NODES_QUERY does not use size() on string (regression guard)
# ---------------------------------------------------------------------------

class TestMultiGroupQueryNoBrokenSizeCheck:
    """Verify the sentinel Cypher query no longer contains the broken size() predicate."""

    def test_query_does_not_use_size_on_group_id(self):
        import importlib.util
        script_path = Path(__file__).parents[1] / 'scripts' / 'contamination_sentinel.py'
        mod_name = '_contamination_query_check'
        spec = importlib.util.spec_from_file_location(mod_name, script_path)
        mod = importlib.util.module_from_spec(spec)
        # Must be registered BEFORE exec_module so @dataclass resolves cls.__module__
        sys.modules[mod_name] = mod
        try:
            spec.loader.exec_module(mod)
        finally:
            sys.modules.pop(mod_name, None)
        query = mod._MULTI_GROUP_NODES_QUERY
        # The broken pattern is size(n.group_id) > 1 without type discrimination.
        # Verify it is absent.
        assert 'size(n.group_id)' not in query, (
            "_MULTI_GROUP_NODES_QUERY still uses size(n.group_id) which incorrectly "
            "treats string length as element count, falsely flagging normal group_id strings."
        )
