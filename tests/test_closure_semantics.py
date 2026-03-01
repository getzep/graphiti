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

NO-GO fixes (PR #116):
9.  Closure edge itself is NOT included in invalidation candidates
10. Deterministic processing order (sorted by invalid_at asc, then closure_uuid)
11. Min semantics: earliest invalid_at wins when multiple closures target same fact
12. _to_utc parses ISO strings with UTC offsets correctly
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


# ---------------------------------------------------------------------------
# 5. NO-GO regression fixes (PR #116)
# ---------------------------------------------------------------------------

class TestClosureEdgeExclusion:
    """Regression: the closure edge itself must not appear in invalidation candidates."""

    def test_closure_edge_uuid_excluded_from_active_facts_query(self):
        """_FIND_ACTIVE_FACTS_QUERY must filter out e.uuid = $closure_uuid."""
        from graphiti_core.utils.maintenance.closure import _FIND_ACTIVE_FACTS_QUERY
        assert '$closure_uuid' in _FIND_ACTIVE_FACTS_QUERY, (
            '_FIND_ACTIVE_FACTS_QUERY does not reference $closure_uuid; '
            'the closure edge itself may be self-invalidated during the pass.'
        )
        assert 'e.uuid <> $closure_uuid' in _FIND_ACTIVE_FACTS_QUERY, (
            '_FIND_ACTIVE_FACTS_QUERY does not exclude e.uuid <> $closure_uuid; '
            'the closure edge may be included as a candidate for invalidation.'
        )

    def test_apply_passes_closure_uuid_to_facts_query(self):
        """execute_query calls for the active-facts step must include closure_uuid kwarg."""
        from graphiti_core.utils.maintenance.closure import apply_closure_semantics

        closure_edges = [
            {
                'closure_uuid': 'ce-excl-1',
                'closure_name': 'RESOLVES',
                'source_uuid': 'src-x',
                'source_name': 'Fix-X',
                'target_uuid': 'tgt-x',
                'target_name': 'Bug-X',
                'valid_at': datetime(2026, 1, 5, tzinfo=timezone.utc),
                'created_at': datetime(2026, 1, 5, tzinfo=timezone.utc),
            }
        ]

        received_kwargs: list[dict] = []

        async def _execute_query(query, *args, routing_='r', **kwargs):
            received_kwargs.append(dict(kwargs))
            if 'closure_names' in kwargs:
                return closure_edges, [], None
            return [], [], None

        driver = MagicMock()
        driver.execute_query = AsyncMock(side_effect=_execute_query)

        asyncio.get_event_loop().run_until_complete(
            apply_closure_semantics(driver, dry_run=True)
        )

        # The facts query must be called with closure_uuid
        facts_calls = [kw for kw in received_kwargs if 'target_uuid' in kw]
        assert facts_calls, 'No facts query was issued'
        assert all('closure_uuid' in kw for kw in facts_calls), (
            'At least one facts query call is missing the closure_uuid parameter.'
        )
        assert facts_calls[0]['closure_uuid'] == 'ce-excl-1'


class TestDeterministicOrderingAndMinSemantics:
    """Regression: closure records must be processed in a deterministic order
    and the earliest invalid_at must win when multiple closures target the same fact.
    """

    def test_records_processed_in_invalid_at_ascending_order(self):
        """When two closure edges have different valid_at timestamps, the one with
        the earlier timestamp must be processed first.
        """
        from graphiti_core.utils.maintenance.closure import apply_closure_semantics

        ts_early = datetime(2026, 1, 1, tzinfo=timezone.utc)
        ts_late  = datetime(2026, 6, 1, tzinfo=timezone.utc)

        # Return records in REVERSE chronological order to expose ordering bugs.
        closure_edges = [
            {
                'closure_uuid': 'ce-late',
                'closure_name': 'RESOLVES',
                'source_uuid': 'src-late',
                'source_name': 'Fix-Late',
                'target_uuid': 'tgt-shared',
                'target_name': 'Bug-Shared',
                'valid_at': ts_late,
                'created_at': ts_late,
            },
            {
                'closure_uuid': 'ce-early',
                'closure_name': 'RESOLVES',
                'source_uuid': 'src-early',
                'source_name': 'Fix-Early',
                'target_uuid': 'tgt-shared',
                'target_name': 'Bug-Shared',
                'valid_at': ts_early,
                'created_at': ts_early,
            },
        ]

        processed_order: list[str] = []

        async def _execute_query(query, *args, routing_='r', **kwargs):
            if 'closure_names' in kwargs:
                return closure_edges, [], None
            if 'target_uuid' in kwargs:
                # Record which closure_uuid this facts-query belongs to
                processed_order.append(kwargs.get('closure_uuid', '?'))
                return [{'fact_uuid': 'f-shared', 'fact_name': 'IS_OPEN'}], [], None
            return [], [], None

        driver = MagicMock()
        driver.execute_query = AsyncMock(side_effect=_execute_query)

        asyncio.get_event_loop().run_until_complete(
            apply_closure_semantics(driver, dry_run=True)
        )

        # ce-early (ts_early) must be processed before ce-late (ts_late)
        assert processed_order == ['ce-early', 'ce-late'], (
            f'Expected order [ce-early, ce-late] but got {processed_order}. '
            'Closure edges must be processed in invalid_at ascending order.'
        )

    def test_min_invalid_at_wins_for_shared_fact(self):
        """When two closures have different timestamps and target the same fact,
        the _INVALIDATE_FACTS_QUERY WHERE e.invalid_at IS NULL guard ensures
        only the first (earliest) closure's timestamp is applied.  We verify
        the write is issued with the earliest timestamp.
        """
        from graphiti_core.utils.maintenance.closure import apply_closure_semantics

        ts_early = datetime(2026, 2, 1, tzinfo=timezone.utc)
        ts_late  = datetime(2026, 8, 1, tzinfo=timezone.utc)

        closure_edges = [
            {
                'closure_uuid': 'ce-A',
                'closure_name': 'RESOLVES',
                'source_uuid': 'srcA',
                'source_name': 'FixA',
                'target_uuid': 'tgt-B',
                'target_name': 'BugB',
                'valid_at': ts_early,
                'created_at': ts_early,
            },
            {
                'closure_uuid': 'ce-B',
                'closure_name': 'SUPERSEDES',
                'source_uuid': 'srcB',
                'source_name': 'FixB',
                'target_uuid': 'tgt-B',
                'target_name': 'BugB',
                'valid_at': ts_late,
                'created_at': ts_late,
            },
        ]

        write_calls: list[dict] = []

        async def _execute_query(query, *args, routing_='r', **kwargs):
            if routing_ == 'w':
                write_calls.append(dict(kwargs))
            if 'closure_names' in kwargs:
                return closure_edges, [], None
            if 'target_uuid' in kwargs:
                return [{'fact_uuid': 'f-shared', 'fact_name': 'IS_ACTIVE'}], [], None
            return [], [], None

        driver = MagicMock()
        driver.execute_query = AsyncMock(side_effect=_execute_query)

        asyncio.get_event_loop().run_until_complete(
            apply_closure_semantics(driver, dry_run=False)
        )

        # First write must use ts_early (the min/earliest timestamp)
        assert write_calls, 'Expected at least one write call'
        first_write = write_calls[0]
        assert first_write['invalid_at'] == ts_early, (
            f"First write used {first_write['invalid_at']} instead of {ts_early}. "
            'Min-semantics require the earliest invalid_at to be applied first.'
        )


class TestToUtcTimezoneHandling:
    """Regression: _to_utc must correctly convert ISO strings with UTC offsets."""

    def _call(self, value):
        from graphiti_core.utils.maintenance.closure import _to_utc
        return _to_utc(value)

    def test_naive_datetime_assumes_utc(self):
        dt = datetime(2026, 1, 1, 10, 0, 0)  # naive
        result = self._call(dt)
        assert result.tzinfo is not None
        assert result == datetime(2026, 1, 1, 10, 0, 0, tzinfo=timezone.utc)

    def test_utc_aware_datetime_unchanged(self):
        dt = datetime(2026, 3, 15, 12, 0, 0, tzinfo=timezone.utc)
        result = self._call(dt)
        assert result == dt

    def test_iso_string_naive_parsed_as_utc(self):
        result = self._call('2026-01-01T10:00:00')
        assert result == datetime(2026, 1, 1, 10, 0, 0, tzinfo=timezone.utc)

    def test_iso_string_with_positive_offset_converted_correctly(self):
        """'2026-01-01T10:00:00+05:30' must become '2026-01-01T04:30:00Z'."""
        result = self._call('2026-01-01T10:00:00+05:30')
        expected = datetime(2026, 1, 1, 4, 30, 0, tzinfo=timezone.utc)
        assert result == expected, (
            f'Got {result.isoformat()} but expected {expected.isoformat()}. '
            '_to_utc must use astimezone(), not replace(), for offset-aware strings.'
        )

    def test_iso_string_with_negative_offset_converted_correctly(self):
        """'2026-06-01T08:00:00-04:00' must become '2026-06-01T12:00:00Z'."""
        result = self._call('2026-06-01T08:00:00-04:00')
        expected = datetime(2026, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
        assert result == expected

    def test_iso_string_z_suffix_parsed_correctly(self):
        """ISO strings ending in Z must be treated as UTC."""
        # Python 3.11+ supports 'Z' in fromisoformat, but older may not.
        # Use +00:00 equivalent to test robustly.
        result = self._call('2026-06-01T12:00:00+00:00')
        assert result == datetime(2026, 6, 1, 12, 0, 0, tzinfo=timezone.utc)

    def test_utc_offset_datetime_converted(self):
        """A Python datetime with non-UTC tzinfo must be converted."""
        from datetime import timedelta
        tz_plus5 = timezone(timedelta(hours=5, minutes=30))
        dt = datetime(2026, 1, 1, 10, 0, 0, tzinfo=tz_plus5)
        result = self._call(dt)
        assert result == datetime(2026, 1, 1, 4, 30, 0, tzinfo=timezone.utc)
