import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from graphiti_core.driver.driver import GraphProvider
from graphiti_core.search.search_filters import SearchFilters
from utils.node_name_lookup import search_nodes_by_name_fallback


def _make_record(name: str, group_id: str = 'group-1') -> dict:
    return {
        'uuid': f'{name}-uuid',
        'name': name,
        'group_id': group_id,
        'created_at': '2026-03-20T00:00:00Z',
        'summary': f'{name} summary',
        'labels': ['Entity'],
        'attributes': {
            'uuid': f'{name}-uuid',
            'name': name,
            'group_id': group_id,
            'created_at': '2026-03-20T00:00:00Z',
            'summary': f'{name} summary',
            'labels': ['Entity'],
        },
    }


@pytest.mark.asyncio
async def test_node_name_fallback_returns_exact_matches_before_contains():
    driver = SimpleNamespace(
        provider=GraphProvider.NEO4J,
        execute_query=AsyncMock(return_value=([_make_record('Exact Match')], None, None)),
    )

    nodes = await search_nodes_by_name_fallback(
        driver=driver,
        query='Exact Match',
        search_filter=SearchFilters(),
        group_ids=['group-1'],
        limit=5,
    )

    assert [node.name for node in nodes] == ['Exact Match']
    assert driver.execute_query.await_count == 1


@pytest.mark.asyncio
async def test_node_name_fallback_uses_contains_when_exact_is_empty():
    driver = SimpleNamespace(
        provider=GraphProvider.NEO4J,
        execute_query=AsyncMock(
            side_effect=[
                ([], None, None),
                ([_make_record('Codex Smoke Tester')], None, None),
            ]
        ),
    )

    nodes = await search_nodes_by_name_fallback(
        driver=driver,
        query='Smoke Tester',
        search_filter=SearchFilters(),
        group_ids=['group-1'],
        limit=5,
    )

    assert [node.name for node in nodes] == ['Codex Smoke Tester']
    assert driver.execute_query.await_count == 2
