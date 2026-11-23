"""
Copyright 2024, Zep Software, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

# ruff: noqa: E402

"""Regression coverage for group-scoped episode UUID resolution."""

from datetime import datetime, timezone
from typing import Any
from unittest.mock import Mock

import pytest

from graphiti_core.driver.driver import GraphProvider
from graphiti_core.edges import EntityEdge
from graphiti_core.errors import NodeNotFoundError
from graphiti_core.graphiti import Graphiti
from graphiti_core.models.edges.edge_db_queries import (
    HAS_EPISODE_EDGE_SAVE,
    NEXT_EPISODE_EDGE_SAVE,
)
from graphiti_core.nodes import EntityNode, EpisodeType, EpisodicNode


def episode_record(uuid: str = 'episode-a', group_id: str = 'group-a') -> dict[str, Any]:
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    return {
        'uuid': uuid,
        'name': 'episode',
        'group_id': group_id,
        'created_at': now,
        'source': EpisodeType.text.value,
        'source_description': 'test',
        'content': 'safe content',
        'valid_at': now,
        'entity_edges': [],
    }


class StubDriver:
    provider = GraphProvider.NEO4J

    def __init__(self, records: list[dict[str, Any]] | None = None):
        self.graph_operations_interface: Any = None
        self.records = records or []
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def execute_query(self, query: str, **params: Any):
        self.calls.append((query, params))
        return self.records, None, None


class ForeignEpisodeOperations:
    async def episodic_node_get_by_uuid(self, cls, driver, uuid):
        return EpisodicNode(
            uuid=uuid,
            name='foreign',
            group_id='group-b',
            source=EpisodeType.text,
            source_description='test',
            content='foreign content',
            valid_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        )


@pytest.mark.asyncio
async def test_episode_uuid_query_binds_requested_group():
    driver = StubDriver([episode_record()])

    episode = await EpisodicNode.get_by_uuid(  # type: ignore[arg-type]
        driver, 'episode-a', group_id='group-a'
    )

    query, params = driver.calls[0]
    assert 'WHERE e.group_id = $group_id' in query
    assert params['group_id'] == 'group-a'
    assert episode.group_id == 'group-a'


@pytest.mark.asyncio
async def test_predecessor_list_rejects_missing_scoped_uuid():
    driver = StubDriver([episode_record(), episode_record(uuid='episode-b', group_id='group-b')])

    with pytest.raises(NodeNotFoundError, match='episode-b'):
        await EpisodicNode.get_by_uuids(  # type: ignore[arg-type]
            driver,
            ['episode-a', 'episode-b'],
            group_id='group-a',
            require_all=True,
        )

    query, params = driver.calls[0]
    assert 'AND e.group_id = $group_id' in query
    assert params['group_id'] == 'group-a'


@pytest.mark.asyncio
async def test_custom_episode_lookup_rejects_foreign_result():
    driver = StubDriver()
    driver.graph_operations_interface = ForeignEpisodeOperations()

    with pytest.raises(NodeNotFoundError, match='episode-b'):
        await EpisodicNode.get_by_uuid(  # type: ignore[arg-type]
            driver, 'episode-b', group_id='group-a'
        )

    assert driver.calls == []


@pytest.mark.asyncio
async def test_unscoped_episode_lookup_remains_supported():
    driver = StubDriver([episode_record()])

    episode = await EpisodicNode.get_by_uuid(driver, 'episode-a')  # type: ignore[arg-type]

    query, _ = driver.calls[0]
    assert 'WHERE e.group_id = $group_id' not in query
    assert episode.uuid == 'episode-a'


@pytest.mark.asyncio
async def test_saga_predecessor_query_binds_saga_and_episode_group():
    driver = StubDriver()
    graphiti = Graphiti.__new__(Graphiti)

    result = await graphiti._saga_get_previous_episode_uuid(
        'saga-a',
        'episode-a',
        'group-a',
        driver=driver,  # type: ignore[arg-type]
    )

    query, params = driver.calls[0]
    assert 'Saga {uuid: $saga_uuid, group_id: $group_id}' in query
    assert 'Episodic {group_id: $group_id}' in query
    assert params['group_id'] == 'group-a'
    assert result is None


@pytest.mark.asyncio
async def test_concurrent_ingestion_rejects_foreign_uuid_before_scheduler():
    driver = StubDriver([episode_record(uuid='episode-b', group_id='group-b')])
    graphiti = Graphiti.__new__(Graphiti)
    graphiti._resolve_request_scope = lambda group_id: (  # type: ignore[method-assign]
        group_id,
        driver,
        object(),
    )
    graphiti._concurrent_ingestion_enabled = lambda group_id: True  # type: ignore[method-assign]
    get_ingestor = Mock(side_effect=AssertionError('scheduler must not run'))
    graphiti._get_concurrent_ingestor = get_ingestor  # type: ignore[method-assign]

    with pytest.raises(NodeNotFoundError, match='episode-b'):
        await graphiti.add_episode(
            name='episode',
            episode_body='caller content',
            source_description='test',
            reference_time=datetime(2026, 1, 1, tzinfo=timezone.utc),
            group_id='group-a',
            uuid='episode-b',
        )

    get_ingestor.assert_not_called()


def test_saga_edge_queries_bind_both_endpoints():
    assert 'Saga {uuid: $saga_uuid, group_id: $group_id}' in HAS_EPISODE_EDGE_SAVE
    assert 'Episodic {uuid: $episode_uuid, group_id: $group_id}' in HAS_EPISODE_EDGE_SAVE
    assert 'Episodic {uuid: $source_episode_uuid, group_id: $group_id}' in NEXT_EPISODE_EDGE_SAVE
    assert 'Episodic {uuid: $target_episode_uuid, group_id: $group_id}' in NEXT_EPISODE_EDGE_SAVE


@pytest.mark.asyncio
async def test_triplet_rejects_mismatched_group_before_processing():
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    source = EntityNode(name='source', group_id='group-b')
    target = EntityNode(name='target', group_id='group-a')
    edge = EntityEdge(
        source_node_uuid=source.uuid,
        target_node_uuid=target.uuid,
        name='RELATES_TO',
        fact='source relates to target',
        group_id='group-a',
        created_at=now,
    )

    with pytest.raises(ValueError, match='does not belong to group group-a'):
        await Graphiti.__new__(Graphiti).add_triplet(source, edge, target)
