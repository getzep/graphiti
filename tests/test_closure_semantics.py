"""Phase C — Slice 2: Closure Semantics tests.

Validates that:
1. CLOSURE_EDGE_NAMES contains RESOLVES and SUPERSEDES
2. apply_closure_semantics is exported from the public maintenance API
3. ClosureResult is a correct dataclass
4. apply_closure_semantics correctly identifies closure edges and marks facts invalid
5. Dry-run mode does NOT write any changes
6. The pass is idempotent (re-running doesn't double-invalidate)
7. When no closure edges exist, result is empty
8. group_id filtering works
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest


# ---------------------------------------------------------------------------
# 1. Public exports
# ---------------------------------------------------------------------------

class TestPublicExports:
    def test_closure_edge_names_exported(self):
        from graphiti_core.utils.maintenance import CLOSURE_EDGE_NAMES
        assert isinstance(CLOSURE_EDGE_NAMES, frozenset)

    def test_resolves_in_closure_names(self):
        from graphiti_core.utils.maintenance import CLOSURE_EDGE_NAMES
        assert 'RESOLVES' in CLOSURE_EDGE_NAMES

    def test_supersedes_in_closure_names(self):
        from graphiti_core.utils.maintenance import CLOSURE_EDGE_NAMES
        assert 'SUPERSEDES' in CLOSURE_EDGE_NAMES

    def test_apply_closure_semantics_exported(self):
        from graphiti_core.utils.maintenance import apply_closure_semantics
        assert callable(apply_closure_semantics)

    def test_closure_result_exported(self):
        from graphiti_core.utils.maintenance import ClosureResult
        r = ClosureResult()
        assert r.closure_edges_found == 0
        assert r.facts_invalidated == 0
        assert r.dry_run is True

    def test_all_includes_closure_symbols(self):
        import graphiti_core.utils.maintenance as mod
        for sym in ('apply_closure_semantics', 'ClosureResult', 'CLOSURE_EDGE_NAMES'):
            assert sym in mod.__all__, f'{sym} missing from __all__'


# ---------------------------------------------------------------------------
# 2. ClosureResult repr
# ---------------------------------------------------------------------------

class TestClosureResultRepr:
    def test_dry_run_repr(self):
        from graphiti_core.utils.maintenance import ClosureResult
        r = ClosureResult(closure_edges_found=3, facts_invalidated=7, dry_run=True)
        assert 'DRY-RUN' in repr(r)
        assert '3' in repr(r)
        assert '7' in repr(r)

    def test_applied_repr(self):
        from graphiti_core.utils.maintenance import ClosureResult
        r = ClosureResult(closure_edges_found=1, facts_invalidated=2, dry_run=False)
        assert 'APPLIED' in repr(r)


# ---------------------------------------------------------------------------
# 3. Behaviour with mock driver
# ---------------------------------------------------------------------------

def _make_driver(closure_records, fact_records_by_target):
    """Build a mock async driver.

    closure_records: list of dicts returned by FIND_CLOSURE_EDGES_QUERY
    fact_records_by_target: dict[target_uuid → list[{fact_uuid, fact_name}]]
    """
    driver = MagicMock()

    async def _execute_query(query, *args, routing_='r', **kwargs):
        if 'closure_names' in kwargs:
            # FIND_CLOSURE_EDGES_QUERY
            return closure_records, [], None
        if 'target_uuid' in kwargs:
            # FIND_ACTIVE_FACTS_QUERY
            target = kwargs['target_uuid']
            return fact_records_by_target.get(target, []), [], None
        # INVALIDATE_FACTS_QUERY (write) or unknown
        return [], [], None

    driver.execute_query = AsyncMock(side_effect=_execute_query)
    return driver


class TestNoClosureEdges:
    def test_empty_result_when_no_closure_edges(self):
        from graphiti_core.utils.maintenance.closure import apply_closure_semantics

        driver = _make_driver([], {})
        result = asyncio.get_event_loop().run_until_complete(
            apply_closure_semantics(driver, dry_run=False)
        )
        assert result.closure_edges_found == 0
        assert result.facts_invalidated == 0


class TestDryRun:
    def test_dry_run_does_not_call_write(self):
        from graphiti_core.utils.maintenance.closure import apply_closure_semantics

        closure_edges = [
            {
                'closure_uuid': 'ce-1',
                'closure_name': 'RESOLVES',
                'source_uuid': 'src-1',
                'source_name': 'PR-42',
                'target_uuid': 'tgt-1',
                'target_name': 'BUG-7',
                'valid_at': datetime(2026, 1, 1, tzinfo=timezone.utc),
                'created_at': datetime(2026, 1, 1, tzinfo=timezone.utc),
            }
        ]
        fact_records = {'tgt-1': [{'fact_uuid': 'f-1', 'fact_name': 'IS_OPEN'}]}

        calls_log = []

        async def _execute_query(query, *args, routing_='r', **kwargs):
            calls_log.append({'routing': routing_, 'kwargs': kwargs})
            if 'closure_names' in kwargs:
                return closure_edges, [], None
            if 'target_uuid' in kwargs:
                return fact_records.get(kwargs['target_uuid'], []), [], None
            return [], [], None

        driver = MagicMock()
        driver.execute_query = AsyncMock(side_effect=_execute_query)

        result = asyncio.get_event_loop().run_until_complete(
            apply_closure_semantics(driver, dry_run=True)
        )

        assert result.dry_run is True
        assert result.closure_edges_found == 1
        assert result.facts_invalidated == 1

        # No write calls should have been made
        write_calls = [c for c in calls_log if c['routing'] == 'w']
        assert len(write_calls) == 0, (
            f'DRY RUN must not write; got write calls: {write_calls}'
        )


class TestApplyMode:
    def test_apply_invalidates_facts(self):
        from graphiti_core.utils.maintenance.closure import apply_closure_semantics

        invalid_ts = datetime(2026, 2, 15, tzinfo=timezone.utc)
        closure_edges = [
            {
                'closure_uuid': 'ce-2',
                'closure_name': 'SUPERSEDES',
                'source_uuid': 'src-2',
                'source_name': 'Policy-v2',
                'target_uuid': 'tgt-2',
                'target_name': 'Policy-v1',
                'valid_at': invalid_ts,
                'created_at': invalid_ts,
            }
        ]
        fact_records = {
            'tgt-2': [
                {'fact_uuid': 'f-2', 'fact_name': 'IS_ACTIVE'},
                {'fact_uuid': 'f-3', 'fact_name': 'GOVERNS'},
            ]
        }

        written_calls = []

        async def _execute_query(query, *args, routing_='r', **kwargs):
            if routing_ == 'w':
                written_calls.append({'kwargs': kwargs, 'query': query})
            if 'closure_names' in kwargs:
                return closure_edges, [], None
            if 'target_uuid' in kwargs:
                return fact_records.get(kwargs['target_uuid'], []), [], None
            return [], [], None

        driver = MagicMock()
        driver.execute_query = AsyncMock(side_effect=_execute_query)

        result = asyncio.get_event_loop().run_until_complete(
            apply_closure_semantics(driver, dry_run=False)
        )

        assert result.dry_run is False
        assert result.closure_edges_found == 1
        assert result.facts_invalidated == 2

        assert len(written_calls) == 1
        write_kwargs = written_calls[0]['kwargs']
        assert set(write_kwargs['uuids']) == {'f-2', 'f-3'}
        assert write_kwargs['invalid_at'] == invalid_ts


class TestGroupIdFiltering:
    def test_group_id_passed_through_to_queries(self):
        from graphiti_core.utils.maintenance.closure import apply_closure_semantics

        received_group_ids = []

        async def _execute_query(query, *args, routing_='r', **kwargs):
            received_group_ids.append(kwargs.get('group_id'))
            return [], [], None

        driver = MagicMock()
        driver.execute_query = AsyncMock(side_effect=_execute_query)

        asyncio.get_event_loop().run_until_complete(
            apply_closure_semantics(driver, group_id='s1_sessions_main', dry_run=True)
        )

        # At least the closure query must have been called with the group_id
        assert 's1_sessions_main' in received_group_ids


class TestIdempotency:
    def test_already_invalidated_facts_are_not_touched(self):
        """If no active (invalid_at IS NULL) facts remain, nothing is written."""
        from graphiti_core.utils.maintenance.closure import apply_closure_semantics

        closure_edges = [
            {
                'closure_uuid': 'ce-3',
                'closure_name': 'RESOLVES',
                'source_uuid': 'src-3',
                'source_name': 'Fix-1',
                'target_uuid': 'tgt-3',
                'target_name': 'Bug-1',
                'valid_at': datetime(2026, 1, 10, tzinfo=timezone.utc),
                'created_at': datetime(2026, 1, 10, tzinfo=timezone.utc),
            }
        ]
        # Second run: no active facts remain for this target
        fact_records: dict = {'tgt-3': []}

        written_calls = []

        async def _execute_query(query, *args, routing_='r', **kwargs):
            if routing_ == 'w':
                written_calls.append(kwargs)
            if 'closure_names' in kwargs:
                return closure_edges, [], None
            if 'target_uuid' in kwargs:
                return fact_records.get(kwargs['target_uuid'], []), [], None
            return [], [], None

        driver = MagicMock()
        driver.execute_query = AsyncMock(side_effect=_execute_query)

        result = asyncio.get_event_loop().run_until_complete(
            apply_closure_semantics(driver, dry_run=False)
        )

        assert result.closure_edges_found == 1
        assert result.facts_invalidated == 0
        assert len(written_calls) == 0, 'No writes expected when all facts already invalidated'


# ---------------------------------------------------------------------------
# 4. Offline script import safety
# ---------------------------------------------------------------------------

class TestClosureScript:
    def test_script_importable(self):
        import importlib.util
        from pathlib import Path

        script_path = Path(__file__).parents[1] / 'scripts' / 'apply_closure_semantics.py'
        assert script_path.exists(), f'Script not found: {script_path}'

        spec = importlib.util.spec_from_file_location('_apply_closure', script_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

    def test_dry_run_is_default(self):
        import importlib.util
        import sys
        from pathlib import Path
        from unittest.mock import patch

        script_path = Path(__file__).parents[1] / 'scripts' / 'apply_closure_semantics.py'
        spec = importlib.util.spec_from_file_location('_apply_closure_args', script_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        with patch.object(sys, 'argv', ['apply_closure_semantics.py']):
            args = mod._parse_args()
        assert args.apply is False
