#!/usr/bin/env python3
"""Unit tests for read-tool group scoping.

Covers the `'*'` (ALL_GROUPS) sentinel, the `searched_group_ids` echo on every
read response, and the invariant that widening the scope never happens by
omission. These drive the tool functions against a mocked Graphiti client, so
they need no database and run in the default (non-integration) suite.
"""

import sys
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from graphiti_core import Graphiti
from graphiti_core.errors import GroupIdValidationError
from graphiti_core.helpers import validate_group_id
from graphiti_core.nodes import EpisodeType, EpisodicNode
from graphiti_core.search.search_config import SearchResults

# Add the src directory to the path (mirrors the other unit tests)
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import graphiti_mcp_server as server  # noqa: E402
from config.schema import GraphitiConfig  # noqa: E402
from utils.type_config import ALL_GROUPS, resolve_read_group_ids  # noqa: E402

NOW = datetime(2026, 8, 21, 12, 0, tzinfo=timezone.utc)


@pytest.fixture
def fake_service(monkeypatch):
    client = AsyncMock(spec=Graphiti)
    client.driver = object()
    service = AsyncMock()
    service.get_client = AsyncMock(return_value=client)
    monkeypatch.setattr(server, 'graphiti_service', service)
    return client


def _install_config(monkeypatch, group_id: str) -> None:
    """`config` is a module global bound in main(), so tests must supply it."""
    cfg = GraphitiConfig()
    cfg.graphiti.group_id = group_id
    monkeypatch.setattr(server, 'config', cfg, raising=False)


@pytest.fixture
def configured_group(monkeypatch):
    """Server configured with a default group, as a single-tenant deployment is."""
    _install_config(monkeypatch, 'tenant-a')


class TestResolveReadGroupIds:
    def test_omitted_uses_configured_default(self):
        assert resolve_read_group_ids(None, 'tenant-a') == ['tenant-a']

    def test_omitted_with_no_default_is_all_groups(self):
        assert resolve_read_group_ids(None, None) is None
        assert resolve_read_group_ids(None, '') is None

    def test_explicit_groups_pass_through(self):
        assert resolve_read_group_ids(['x', 'y'], 'tenant-a') == ['x', 'y']

    def test_star_widens_to_all_groups(self):
        assert resolve_read_group_ids([ALL_GROUPS], 'tenant-a') is None

    def test_star_wins_anywhere_in_the_list(self):
        # A group literally named '*' cannot exist, so this is unambiguous.
        assert resolve_read_group_ids(['x', ALL_GROUPS], 'tenant-a') is None

    def test_star_is_not_a_legal_group_id(self):
        """The sentinel is safe precisely because no real group can be named it."""
        with pytest.raises(GroupIdValidationError):
            validate_group_id(ALL_GROUPS)


class TestNoAccidentalWidening:
    """Omitting group_ids must never broaden a configured deployment's scope."""

    @pytest.mark.asyncio
    async def test_facts_stay_on_configured_group(self, fake_service, configured_group):
        fake_service.search.return_value = []

        result = await server.search_memory_facts(query='q')

        assert fake_service.search.call_args.kwargs['group_ids'] == ['tenant-a']
        assert result['searched_group_ids'] == ['tenant-a']

    @pytest.mark.asyncio
    async def test_nodes_stay_on_configured_group(self, fake_service, configured_group):
        fake_service.search_.return_value = SearchResults(nodes=[])

        result = await server.search_nodes(query='q')

        assert fake_service.search_.call_args.kwargs['group_ids'] == ['tenant-a']
        assert result['searched_group_ids'] == ['tenant-a']


class TestStarReadsEverything:
    @pytest.mark.asyncio
    async def test_facts_star_passes_none_to_core(self, fake_service, configured_group):
        """None is what graphiti-core's search path reads as 'no group filter'."""
        fake_service.search.return_value = []

        result = await server.search_memory_facts(query='q', group_ids='*')

        assert fake_service.search.call_args.kwargs['group_ids'] is None
        assert result['searched_group_ids'] is None

    @pytest.mark.asyncio
    async def test_nodes_star_passes_none_to_core(self, fake_service, configured_group):
        fake_service.search_.return_value = SearchResults(nodes=[])

        result = await server.search_nodes(query='q', group_ids='*')

        assert fake_service.search_.call_args.kwargs['group_ids'] is None
        assert result['searched_group_ids'] is None


class TestEmptyResultReportsItsScope:
    """The bug this closes: an empty result that reads as absence of memory."""

    @pytest.mark.asyncio
    async def test_empty_facts_carry_scope(self, fake_service, configured_group):
        fake_service.search.return_value = []

        result = await server.search_memory_facts(query='q')

        assert result['facts'] == []
        assert result['searched_group_ids'] == ['tenant-a']

    @pytest.mark.asyncio
    async def test_empty_nodes_carry_scope(self, fake_service, configured_group):
        fake_service.search_.return_value = SearchResults(nodes=[])

        result = await server.search_nodes(query='q')

        assert result['nodes'] == []
        assert result['searched_group_ids'] == ['tenant-a']


class TestGetEpisodes:
    @pytest.mark.asyncio
    async def test_scope_echoed_on_results(self, fake_service, configured_group):
        episode = EpisodicNode(
            uuid='e1',
            name='ep',
            group_id='tenant-a',
            source=EpisodeType.text,
            source_description='',
            content='body',
            valid_at=NOW,
            created_at=NOW,
        )
        with patch.object(EpisodicNode, 'get_by_group_ids', AsyncMock(return_value=[episode])):
            result = await server.get_episodes()

        assert result['searched_group_ids'] == ['tenant-a']
        assert len(result['episodes']) == 1

    @pytest.mark.asyncio
    async def test_no_resolvable_group_errors_instead_of_empty(self, fake_service, monkeypatch):
        """Returning [] here would read as 'this graph has no episodes'."""
        _install_config(monkeypatch, '')

        result = await server.get_episodes()

        assert 'error' in result
        assert 'concrete group' in result['error']

    @pytest.mark.asyncio
    async def test_star_is_rejected_not_silently_empty(self, fake_service, configured_group):
        result = await server.get_episodes(group_ids='*')

        assert 'error' in result
        assert "'*' is not supported" in result['error']


class TestClearGraphUnchanged:
    """The destructive tool must not inherit read semantics."""

    @pytest.mark.asyncio
    async def test_star_does_not_wipe_the_whole_graph(self, fake_service, configured_group):
        with patch.object(server, 'clear_data', AsyncMock()) as cleared:
            result = await server.clear_graph(group_ids='*')

        # '*' must not reach clear_data: it would match no group and still report
        # success, which is the worst possible answer from a destructive tool.
        assert 'error' in result
        assert 'not accepted by clear_graph' in result['error']
        cleared.assert_not_called()

    @pytest.mark.asyncio
    async def test_omitted_still_uses_configured_group(self, fake_service, configured_group):
        with patch.object(server, 'clear_data', AsyncMock()) as cleared:
            await server.clear_graph()

        assert cleared.call_args.kwargs['group_ids'] == ['tenant-a']
