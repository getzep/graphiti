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

# Regression coverage for group-scoped episode UUID resolution.

from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest

import graphiti_core.graphiti as graphiti_module
from graphiti_core.driver.driver import GraphProvider
from graphiti_core.edges import EntityEdge, HasEpisodeEdge
from graphiti_core.errors import EdgeNotFoundError, NodeNotFoundError
from graphiti_core.graphiti import Graphiti
from graphiti_core.models.edges.edge_db_queries import (
    HAS_EPISODE_EDGE_SAVE,
    NEXT_EPISODE_EDGE_SAVE,
)
from graphiti_core.nodes import EntityNode, EpisodeType, EpisodicNode, Node, SagaNode
from graphiti_core.utils.maintenance.graph_data_operations import clear_data


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


def entity_edge_record(uuid: str = 'edge-a', group_id: str = 'group-a') -> dict[str, Any]:
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    return {
        'uuid': uuid,
        'source_node_uuid': 'source-a',
        'target_node_uuid': 'target-a',
        'fact': 'source relates to target',
        'fact_embedding': None,
        'name': 'RELATED_TO',
        'group_id': group_id,
        'episodes': [],
        'created_at': now,
        'expired_at': None,
        'valid_at': None,
        'invalid_at': None,
        'reference_time': None,
        'attributes': {},
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


class EntityEdgeOperations:
    def __init__(self, edge, edges=None):
        self.edge = edge
        self.edges = edges or [edge]

    async def edge_get_by_uuid(self, cls, driver, uuid):
        return self.edge

    async def edge_get_by_uuids(self, cls, driver, uuids):
        return self.edges


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


@pytest.mark.asyncio
async def test_entity_edge_uuid_lookup_filters_requested_group():
    driver = StubDriver([entity_edge_record()])

    edge = await EntityEdge.get_by_uuid(  # type: ignore[arg-type]
        driver, 'edge-a', group_id='group-a'
    )

    query, params = driver.calls[0]
    assert 'WHERE e.group_id = $group_id' in query
    assert params['group_id'] == 'group-a'
    assert edge.group_id == 'group-a'


@pytest.mark.asyncio
async def test_entity_edge_uuid_lookup_rejects_foreign_query_result():
    driver = StubDriver([entity_edge_record(group_id='group-b')])

    with pytest.raises(EdgeNotFoundError, match='edge-a'):
        await EntityEdge.get_by_uuid(  # type: ignore[arg-type]
            driver, 'edge-a', group_id='group-a'
        )


@pytest.mark.asyncio
async def test_entity_edge_uuid_lookup_keeps_unscoped_behavior():
    driver = StubDriver([entity_edge_record(group_id='group-b')])

    edge = await EntityEdge.get_by_uuid(driver, 'edge-a')  # type: ignore[arg-type]

    query, params = driver.calls[0]
    assert 'WHERE e.group_id = $group_id' not in query
    assert params['group_id'] is None
    assert edge.group_id == 'group-b'


@pytest.mark.asyncio
async def test_entity_edge_bulk_lookup_filters_foreign_edges():
    driver = StubDriver(
        [entity_edge_record(), entity_edge_record(uuid='edge-b', group_id='group-b')]
    )

    edges = await EntityEdge.get_by_uuids(  # type: ignore[arg-type]
        driver, ['edge-a', 'edge-b'], group_id='group-a'
    )

    query, params = driver.calls[0]
    assert 'AND e.group_id = $group_id' in query
    assert params['group_id'] == 'group-a'
    assert [edge.uuid for edge in edges] == ['edge-a']


@pytest.mark.asyncio
async def test_entity_edge_uuid_lookup_filters_graph_operations_result():
    edge = EntityEdge(
        uuid='edge-b',
        group_id='group-b',
        source_node_uuid='source-b',
        target_node_uuid='target-b',
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        name='RELATED_TO',
        fact='source relates to target',
    )
    driver = StubDriver()
    driver.graph_operations_interface = EntityEdgeOperations(edge)

    with pytest.raises(EdgeNotFoundError, match='edge-b'):
        await EntityEdge.get_by_uuid(  # type: ignore[arg-type]
            driver, 'edge-b', group_id='group-a'
        )

    assert driver.calls == []


@pytest.mark.asyncio
async def test_entity_edge_bulk_lookup_filters_graph_operations_result():
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    edges = [
        EntityEdge(
            uuid='edge-a',
            group_id='group-a',
            source_node_uuid='source-a',
            target_node_uuid='target-a',
            created_at=now,
            name='RELATED_TO',
            fact='source relates to target',
        ),
        EntityEdge(
            uuid='edge-b',
            group_id='group-b',
            source_node_uuid='source-b',
            target_node_uuid='target-b',
            created_at=now,
            name='RELATED_TO',
            fact='source relates to target',
        ),
    ]
    driver = StubDriver()
    driver.graph_operations_interface = EntityEdgeOperations(edges[0], edges)

    result = await EntityEdge.get_by_uuids(  # type: ignore[arg-type]
        driver, ['edge-a', 'edge-b'], group_id='group-a'
    )

    assert [edge.uuid for edge in result] == ['edge-a']
    assert driver.calls == []


@pytest.mark.asyncio
async def test_episode_provenance_forwards_group_to_entity_edge_lookup(monkeypatch):
    driver = StubDriver()
    episode = EpisodicNode(
        uuid='episode-a',
        name='episode',
        group_id='group-a',
        source=EpisodeType.text,
        source_description='test',
        content='content',
        valid_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        entity_edges=['edge-a'],
    )
    edge_group_ids = []

    async def get_episodes(driver, uuids, group_id=None, require_all=False):
        return [episode]

    async def get_edges(driver, uuids, group_id=None):
        edge_group_ids.append(group_id)
        return []

    async def get_nodes(driver, episodes):
        return []

    monkeypatch.setattr(EpisodicNode, 'get_by_uuids', staticmethod(get_episodes))
    monkeypatch.setattr(EntityEdge, 'get_by_uuids', staticmethod(get_edges))
    monkeypatch.setattr(graphiti_module, 'get_mentioned_nodes', get_nodes)
    graphiti = Graphiti.__new__(Graphiti)
    graphiti.driver = driver
    graphiti.max_coroutines = 1
    graphiti._resolve_request_scope = lambda group_id: (  # type: ignore[method-assign]
        group_id,
        driver,
        SimpleNamespace(driver=driver),
    )

    await graphiti.get_nodes_and_edges_by_episode(['episode-a'], group_id='group-a')

    assert edge_group_ids == ['group-a']


@pytest.mark.asyncio
async def test_remove_episode_forwards_group_to_entity_edge_lookup(monkeypatch):
    driver = StubDriver()
    episode = EpisodicNode(
        uuid='episode-a',
        name='episode',
        group_id='group-a',
        source=EpisodeType.text,
        source_description='test',
        content='content',
        valid_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        entity_edges=['edge-a'],
    )
    edge_group_ids = []

    async def get_episode(driver, uuid, group_id=None):
        return episode

    async def get_edges(driver, uuids, group_id=None):
        edge_group_ids.append(group_id)
        return []

    async def get_nodes(driver, episodes):
        return []

    monkeypatch.setattr(EpisodicNode, 'get_by_uuid', staticmethod(get_episode))
    monkeypatch.setattr(EntityEdge, 'get_by_uuids', staticmethod(get_edges))
    monkeypatch.setattr(EpisodicNode, 'delete', AsyncMock())
    monkeypatch.setattr(Node, 'delete_by_uuids', AsyncMock())
    monkeypatch.setattr(graphiti_module, 'get_mentioned_nodes', get_nodes)
    graphiti = Graphiti.__new__(Graphiti)
    graphiti.driver = driver
    graphiti.clients = SimpleNamespace(llm_client=object())
    graphiti.max_coroutines = 1
    graphiti._resolve_request_scope = lambda group_id: (  # type: ignore[method-assign]
        group_id,
        driver,
        graphiti.clients,
    )

    await graphiti.remove_episode('episode-a', group_id='group-a')

    assert edge_group_ids == ['group-a']


@pytest.mark.asyncio
async def test_summarize_saga_rejects_foreign_group_before_llm_or_write(monkeypatch):
    driver = StubDriver()
    saga = SagaNode(uuid='saga-b', name='saga', group_id='group-b')
    saga_lookup_drivers = []

    async def get_saga(driver, uuid):
        saga_lookup_drivers.append(driver)
        return saga

    monkeypatch.setattr(SagaNode, 'get_by_uuid', staticmethod(get_saga))
    summary_fetch = AsyncMock()
    llm_call = AsyncMock()
    save_call = AsyncMock()
    monkeypatch.setattr(graphiti_module, 'generate_prompt_response', llm_call)
    monkeypatch.setattr(SagaNode, 'save', save_call)
    graphiti = Graphiti.__new__(Graphiti)
    graphiti.driver = StubDriver()
    graphiti.clients = SimpleNamespace()
    graphiti._resolve_request_scope = lambda group_id: (  # type: ignore[method-assign]
        group_id,
        driver,
        graphiti.clients,
    )
    graphiti._saga_get_episode_contents = summary_fetch

    with pytest.raises(NodeNotFoundError, match='saga-b'):
        await graphiti.summarize_saga('saga-b', group_id='group-a')

    assert saga_lookup_drivers == [driver]
    summary_fetch.assert_not_awaited()
    llm_call.assert_not_awaited()
    save_call.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.parametrize('graph_driver', [GraphProvider.NEO4J], indirect=True)
async def test_summarize_saga_excludes_legacy_foreign_group_episode(graph_driver):
    group_a = f'saga-summary-a-{uuid4()}'
    group_b = f'saga-summary-b-{uuid4()}'
    created_at = datetime.now(timezone.utc)
    saga = SagaNode(
        uuid=str(uuid4()),
        name='saga summary group test',
        group_id=group_a,
        created_at=created_at,
    )
    episode_a = EpisodicNode(
        uuid=str(uuid4()),
        name='group A episode',
        group_id=group_a,
        source=EpisodeType.text,
        source_description='test',
        content='alphamarker group A episode content',
        valid_at=created_at,
        created_at=created_at,
    )
    episode_b = EpisodicNode(
        uuid=str(uuid4()),
        name='group B episode',
        group_id=group_b,
        source=EpisodeType.text,
        source_description='test',
        content='betamarker group B episode content',
        valid_at=created_at,
        created_at=created_at,
    )

    captured_messages = []

    async def generate_response(messages, **_kwargs):
        captured_messages.append(messages)
        return {'summary': 'updated summary'}

    llm_client = Mock()
    llm_client.generate_response = AsyncMock(side_effect=generate_response)
    graphiti = Graphiti.__new__(Graphiti)
    graphiti.driver = graph_driver
    graphiti.llm_client = llm_client
    graphiti._resolve_request_scope = lambda requested_group_id: (  # type: ignore[method-assign]
        requested_group_id,
        graph_driver,
        None,
    )

    try:
        await saga.save(graph_driver)
        await episode_a.save(graph_driver)
        await episode_b.save(graph_driver)
        await HasEpisodeEdge(
            source_node_uuid=saga.uuid,
            target_node_uuid=episode_a.uuid,
            group_id=group_a,
            created_at=created_at,
        ).save(graph_driver)
        await graph_driver.execute_query(
            """
            MATCH (s:Saga {uuid: $saga_uuid}), (e:Episodic {uuid: $episode_uuid})
            CREATE (s)-[link:HAS_EPISODE]->(e)
            SET link.uuid = randomUUID(),
                link.group_id = s.group_id,
                link.created_at = datetime()
            """,
            saga_uuid=saga.uuid,
            episode_uuid=episode_b.uuid,
        )

        await graphiti.summarize_saga(saga.uuid, group_id=group_a)
        scoped_prompt = '\n'.join(message.content for message in captured_messages[-1])
        assert 'alphamarker group A episode content' in scoped_prompt
        assert 'betamarker' not in scoped_prompt

        await graph_driver.execute_query(
            """
            MATCH (s:Saga {uuid: $saga_uuid})
            SET s.last_summarized_at = null,
                s.last_summarized_episode_valid_at = null,
                s.summary = ''
            """,
            saga_uuid=saga.uuid,
        )
        await graphiti.summarize_saga(saga.uuid)
        unscoped_prompt = '\n'.join(message.content for message in captured_messages[-1])
        assert 'alphamarker group A episode content' in unscoped_prompt
        assert 'betamarker' not in unscoped_prompt
    finally:
        await clear_data(graph_driver, [group_a, group_b])


@pytest.mark.asyncio
async def test_add_triplet_rejects_foreign_edge_uuid_without_saving(monkeypatch):
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    source = EntityNode(name='source', group_id='group-a', name_embedding=[0.1])
    target = EntityNode(name='target', group_id='group-a', name_embedding=[0.1])
    edge = EntityEdge(
        uuid='edge-b',
        source_node_uuid=source.uuid,
        target_node_uuid=target.uuid,
        name='RELATED_TO',
        fact='source relates to target',
        fact_embedding=[0.1],
        group_id='group-a',
        created_at=now,
    )
    foreign_edge = EntityEdge(
        uuid='edge-b',
        source_node_uuid='foreign-source',
        target_node_uuid='foreign-target',
        name='RELATED_TO',
        fact='foreign fact',
        group_id='group-b',
        created_at=now,
    )
    edge_lookup_calls = []

    async def get_node(driver, uuid):
        return source if uuid == source.uuid else target

    async def get_edge(driver, uuid, *args, **kwargs):
        edge_lookup_calls.append((uuid, kwargs))
        return foreign_edge

    persist = AsyncMock()
    monkeypatch.setattr(EntityNode, 'get_by_uuid', staticmethod(get_node))
    monkeypatch.setattr(EntityEdge, 'get_by_uuid', staticmethod(get_edge))
    monkeypatch.setattr(graphiti_module, 'add_nodes_and_edges_bulk', persist)
    graphiti = Graphiti.__new__(Graphiti)
    graphiti.driver = StubDriver()
    graphiti.clients = SimpleNamespace()
    graphiti.embedder = object()

    with pytest.raises(EdgeNotFoundError, match='edge-b'):
        await graphiti.add_triplet(source, edge, target)

    assert edge_lookup_calls == [('edge-b', {})]
    persist.assert_not_awaited()
