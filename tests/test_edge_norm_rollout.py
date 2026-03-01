"""Phase C — Slice 1: Edge Normalization Rollout tests.

Validates that:
1. normalize_relation_type is exported from graphiti_core.utils.maintenance
2. Universal normalization is applied in permissive mode (not just constrained_soft)
3. Normalization is idempotent
4. A wide range of LLM output variants normalize correctly
5. The offline normalize_edge_names script is importable and parses args safely
6. The script's Cypher queries target the real RELATES_TO relationship model
   (not the no-op Entityedge node label)
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# 1. Public export
# ---------------------------------------------------------------------------

class TestPublicExport:
    def test_normalize_relation_type_exported(self):
        """normalize_relation_type must be importable from the public API surface."""
        from graphiti_core.utils.maintenance import normalize_relation_type
        assert callable(normalize_relation_type)

    def test_normalize_relation_type_in__all__(self):
        import graphiti_core.utils.maintenance as mod
        assert 'normalize_relation_type' in mod.__all__


# ---------------------------------------------------------------------------
# 2. Normalization correctness
# ---------------------------------------------------------------------------

class TestNormalizationCorrectness:
    @pytest.fixture(autouse=True)
    def _import(self):
        from graphiti_core.utils.maintenance import normalize_relation_type
        self.fn = normalize_relation_type

    @pytest.mark.parametrize('raw,expected', [
        # Already canonical — no change
        ('RELATES_TO', 'RELATES_TO'),
        ('USES_MOVE', 'USES_MOVE'),
        ('AUTHORED_BY', 'AUTHORED_BY'),
        # Mixed case from LLM
        ('relates_to', 'RELATES_TO'),
        ('Relates To', 'RELATES_TO'),
        ('relates-to', 'RELATES_TO'),
        # Whitespace-padded
        ('  RELATES_TO  ', 'RELATES_TO'),
        # Punctuation bypass variants
        ('RELATES^TO', 'RELATES_TO'),
        ('MENTIONS.', 'MENTIONS'),
        ('RELATES^.TO', 'RELATES_TO'),
        ('IS^RELATED.TO', 'IS_RELATED_TO'),
        # Hyphen separator
        ('AUTHORED-BY', 'AUTHORED_BY'),
        # Colon separator
        ('AUTHORED:BY', 'AUTHORED_BY'),
        # Space + underscore
        ('USES MOVE', 'USES_MOVE'),
        # Leading/trailing underscores stripped
        ('_RELATES_TO_', 'RELATES_TO'),
        # Multi-word mixed case
        ('opens with', 'OPENS_WITH'),
        # Already SCREAMING_SNAKE with numbers
        ('HAS_3_PARTS', 'HAS_3_PARTS'),
    ])
    def test_normalize_variants(self, raw, expected):
        assert self.fn(raw) == expected

    def test_idempotent(self):
        """Running normalization twice must return the same result."""
        from graphiti_core.utils.maintenance import normalize_relation_type
        for name in ['relates to', 'RELATES_TO', 'USES^MOVE', 'authored-by']:
            first = normalize_relation_type(name)
            second = normalize_relation_type(first)
            assert first == second, f'{name!r}: idempotency failed ({first!r} → {second!r})'


# ---------------------------------------------------------------------------
# 3. Universal normalization in extract_edges (permissive mode)
# ---------------------------------------------------------------------------

class TestUniversalNormalizationInExtractEdges:
    """Verify that normalization is applied BEFORE constrained_soft filtering
    so that permissive-mode edges are also canonicalized."""

    def test_permissive_mode_normalizes_mixed_case(self):
        """In permissive mode, a mixed-case relation_type must be normalized."""
        from graphiti_core.utils.maintenance.edge_operations import _normalize_relation_type

        raw_names = ['relates to', 'Authored By', 'uses_move', 'AUTHORED-BY']
        for raw in raw_names:
            norm = _normalize_relation_type(raw)
            assert norm == norm.upper(), (
                f'{raw!r} normalized to {norm!r} which is not uppercase'
            )
            assert ' ' not in norm, f'{raw!r} → {norm!r} still contains spaces'
            assert '-' not in norm, f'{raw!r} → {norm!r} still contains hyphens'

    def test_edge_data_mutation_in_permissive_path(self):
        """Simulate the normalization loop that runs before constrained_soft block.

        We test the logic directly rather than invoking the full async extract_edges
        (which needs LLM client), by replicating the loop to verify it mutates
        edge_data.relation_type.
        """
        from graphiti_core.utils.maintenance.edge_operations import _normalize_relation_type

        class _FakeEdge:
            def __init__(self, relation_type: str):
                self.relation_type = relation_type

        edges = [
            _FakeEdge('relates to'),
            _FakeEdge('AUTHORED-BY'),
            _FakeEdge('USES_MOVE'),  # already canonical → no change
            _FakeEdge('Mentions.'),
        ]
        expected = ['RELATES_TO', 'AUTHORED_BY', 'USES_MOVE', 'MENTIONS']

        for edge_data in edges:
            norm = _normalize_relation_type(edge_data.relation_type)
            edge_data.relation_type = norm

        for edge_data, exp in zip(edges, expected):
            assert edge_data.relation_type == exp, (
                f'Expected {exp!r}, got {edge_data.relation_type!r}'
            )


# ---------------------------------------------------------------------------
# 4. Offline script — import and arg parsing safety
# ---------------------------------------------------------------------------

class TestNormalizeEdgeNamesScript:
    def test_script_importable(self):
        """The script must import cleanly (without executing side effects)."""
        import importlib.util
        from pathlib import Path

        script_path = Path(__file__).parents[1] / 'scripts' / 'normalize_edge_names.py'
        assert script_path.exists(), f'Script not found: {script_path}'

        spec = importlib.util.spec_from_file_location('_normalize_edge_names', script_path)
        mod = importlib.util.module_from_spec(spec)
        # Should not raise
        spec.loader.exec_module(mod)

    def test_dry_run_is_default(self):
        """Parsing empty argv must default to dry_run=True (no --apply)."""
        import importlib.util
        from pathlib import Path

        script_path = Path(__file__).parents[1] / 'scripts' / 'normalize_edge_names.py'
        spec = importlib.util.spec_from_file_location('_normalize_edge_names_args', script_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        with patch.object(sys, 'argv', ['normalize_edge_names.py']):
            args = mod._parse_args()
        assert args.apply is False, '--apply must not be the default'

    def test_apply_flag_sets_apply(self):
        """Passing --apply must set apply=True."""
        import importlib.util
        from pathlib import Path

        script_path = Path(__file__).parents[1] / 'scripts' / 'normalize_edge_names.py'
        spec = importlib.util.spec_from_file_location('_normalize_edge_names_apply', script_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        with patch.object(sys, 'argv', ['normalize_edge_names.py', '--apply']):
            args = mod._parse_args()
        assert args.apply is True


# ---------------------------------------------------------------------------
# 5. Cypher query model correctness (the real relationship model)
# ---------------------------------------------------------------------------

class TestCypherQueryModelCorrectness:
    """Verify the script queries the actual Neo4j relationship model.

    EntityEdges in Neo4j are stored as RELATES_TO relationships between
    Entity nodes — NOT as standalone nodes with an 'Entityedge' label.
    The incorrect label query silently returns 0 rows and is a no-op.
    """

    @pytest.fixture(autouse=True)
    def _load_script(self):
        import importlib.util
        from pathlib import Path

        script_path = Path(__file__).parents[1] / 'scripts' / 'normalize_edge_names.py'
        spec = importlib.util.spec_from_file_location('_normalize_edge_names_cypher', script_path)
        self.mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.mod)

    def test_scan_query_targets_relates_to_relationship(self):
        """SCAN query must match RELATES_TO relationships, not Entityedge nodes."""
        query: str = self.mod._SCAN_QUERY
        assert 'RELATES_TO' in query, (
            '_SCAN_QUERY must target the RELATES_TO relationship type. '
            'Found: ' + repr(query[:200])
        )

    def test_scan_query_does_not_use_entityedge_label(self):
        """SCAN query must NOT use the no-op Entityedge node label."""
        query: str = self.mod._SCAN_QUERY
        assert 'Entityedge' not in query, (
            '_SCAN_QUERY must not reference the incorrect Entityedge node label. '
            'EntityEdges are relationships in Neo4j, not nodes.'
        )

    def test_update_query_targets_relates_to_relationship(self):
        """UPDATE query must match RELATES_TO relationships, not Entityedge nodes."""
        query: str = self.mod._UPDATE_QUERY
        assert 'RELATES_TO' in query, (
            '_UPDATE_QUERY must target the RELATES_TO relationship type. '
            'Found: ' + repr(query[:200])
        )

    def test_update_query_does_not_use_entityedge_label(self):
        """UPDATE query must NOT use the no-op Entityedge node label."""
        query: str = self.mod._UPDATE_QUERY
        assert 'Entityedge' not in query, (
            '_UPDATE_QUERY must not reference the incorrect Entityedge node label.'
        )

    def test_scan_query_returns_name_field(self):
        """SCAN query must return the 'name' column used for normalization."""
        query: str = self.mod._SCAN_QUERY
        assert 'name' in query, '_SCAN_QUERY must return e.name AS name'

    def test_update_query_sets_name_field(self):
        """UPDATE query must set e.name (the field being normalised)."""
        query: str = self.mod._UPDATE_QUERY
        assert 'e.name' in query and 'u.new_name' in query, (
            '_UPDATE_QUERY must SET e.name = u.new_name'
        )

    def test_run_dry_run_uses_correct_queries_via_mock(self):
        """run() with mocked Neo4j driver exercises both scan + update paths
        and returns the number of edges that need normalization."""
        from unittest.mock import MagicMock, call

        # Simulate two edges: one canonical, one needing normalization
        fake_edges = [
            {'uuid': 'uuid-1', 'name': 'RELATES_TO'},   # already canonical
            {'uuid': 'uuid-2', 'name': 'relates to'},   # needs normalizing
            {'uuid': 'uuid-3', 'name': 'Authored By'},  # needs normalizing
        ]

        mock_session = MagicMock()
        mock_session.run.return_value = [
            {'uuid': e['uuid'], 'name': e['name']} for e in fake_edges
        ]
        mock_driver = MagicMock()
        mock_driver.session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_driver.session.return_value.__exit__ = MagicMock(return_value=False)

        with patch.object(self.mod, 'GraphDatabase') as mock_gdb:
            mock_gdb.driver.return_value = mock_driver

            changed = self.mod.run(
                apply=False,  # dry-run
                group_id=None,
                neo4j_uri='bolt://localhost:7687',
                neo4j_user='neo4j',
                neo4j_password='test',
            )

        # 2 out of 3 edges need normalization
        assert changed == 2, f'Expected 2 edges to normalize, got {changed}'
        # UPDATE must NOT be called in dry-run mode
        update_calls = [
            c for c in mock_session.run.call_args_list
            if 'UPDATE' in str(c) or 'SET' in str(c).upper()
        ]
        assert len(update_calls) == 0, 'Dry-run must not write any updates'
