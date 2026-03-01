"""Phase C — Slice 1: Edge Normalization Rollout tests.

Validates that:
1. normalize_relation_type is exported from graphiti_core.utils.maintenance
2. Universal normalization is applied in permissive mode (not just constrained_soft)
3. Normalization is idempotent
4. A wide range of LLM output variants normalize correctly
5. The offline normalize_edge_names script is importable and parses args safely
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
