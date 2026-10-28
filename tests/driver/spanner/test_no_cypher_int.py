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

# Proof that graphiti_core's hook paths never reach the Spanner driver with Cypher: the real
# core entry points (node/edge classes, graph_data_operations, search_utils, bulk_utils) are
# driven against the emulator while every query the driver receives is recorded.
#
#   SPANNER_EMULATOR_HOST=localhost:9010 pytest tests/driver/spanner/test_no_cypher_int.py -q

import re
from collections.abc import Awaitable, Callable
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest
import pytest_asyncio

from graphiti_core.driver.spanner import SpannerDriver
from graphiti_core.driver.spanner import driver as driver_module
from graphiti_core.driver.spanner.errors import SpannerCypherNotSupportedError
from graphiti_core.edges import CommunityEdge, Edge, EntityEdge, EpisodicEdge
from graphiti_core.embedder.client import EmbedderClient
from graphiti_core.nodes import (
    CommunityNode,
    EntityNode,
    EpisodeType,
    EpisodicNode,
    Node,
    SagaNode,
)
from graphiti_core.search.search_filters import SearchFilters
from graphiti_core.search.search_utils import (
    community_fulltext_search,
    edge_bfs_search,
    edge_similarity_search,
    episode_fulltext_search,
    episode_mentions_reranker,
    get_communities_by_nodes,
    get_mentioned_nodes,
    node_bfs_search,
    node_distance_reranker,
    node_fulltext_search,
    node_similarity_search,
)
from graphiti_core.utils.bulk_utils import add_nodes_and_edges_bulk
from graphiti_core.utils.maintenance.graph_data_operations import clear_data, retrieve_episodes

pytestmark = pytest.mark.asyncio(loop_scope='session')

DIM = 384
UTC = timezone.utc
T0 = datetime(2026, 1, 1, tzinfo=UTC)
GROUP = 'g_nocypher'
ALLOWED_START = re.compile(r'^(GRAPH|SELECT|WITH|DELETE|INSERT|UPDATE)\b', re.IGNORECASE)
DOLLAR_PARAM = re.compile(r'\$[A-Za-z_]')  # not '$.key' JSON paths
FILTERS = SearchFilters()


def emb(index: int) -> list[float]:
    """A 384-float one-hot vector."""
    vector = [0.0] * DIM
    vector[index] = 1.0
    return vector


class FakeEmbedder(EmbedderClient):
    """Deterministic 384-float embeddings, no network."""

    async def create(self, input_data: Any) -> list[float]:
        return emb(sum(map(ord, str(input_data))) % DIM)

    async def create_batch(self, input_data_list: list[str]) -> list[list[float]]:
        return [await self.create(text) for text in input_data_list]


class Recorder:
    """Everything the driver was asked to run, plus how often the Cypher guard fired."""

    def __init__(self) -> None:
        self.queries: list[tuple[str, str]] = []  # (entry point, sql)
        self.guard_calls = 0
        self.guard_raised = 0

    def bad_queries(self) -> list[str]:
        bad = []
        for entry, sql in self.queries:
            text = sql.strip()
            if (
                not ALLOWED_START.match(text)
                or DOLLAR_PARAM.search(text)
                or 'MERGE' in text.upper()
            ):
                bad.append(f'[{entry}] {text[:200]}')
        return bad

    def assert_clean(self) -> None:
        assert self.queries, 'nothing was recorded; the recorder is not wired up'
        assert self.guard_raised == 0, f'{self.guard_raised} queries were rejected as Cypher'
        assert not self.bad_queries()


@pytest_asyncio.fixture(loop_scope='session')
async def recorded(
    clean_driver: SpannerDriver, monkeypatch: pytest.MonkeyPatch
) -> tuple[SpannerDriver, Recorder]:
    recorder = Recorder()

    def wrap(name: str) -> None:
        original = getattr(clean_driver, name)

        async def recording(sql: str, *args: Any, **kwargs: Any) -> Any:
            recorder.queries.append((name, sql))
            return await original(sql, *args, **kwargs)

        monkeypatch.setattr(clean_driver, name, recording)

    for method in ('execute_query', 'read', 'execute_dml', 'execute_partitioned_dml'):
        wrap(method)

    original_guard = driver_module.ensure_not_cypher

    def counting_guard(query: str) -> None:
        recorder.guard_calls += 1
        try:
            original_guard(query)
        except SpannerCypherNotSupportedError:
            recorder.guard_raised += 1
            raise

    monkeypatch.setattr(driver_module, 'ensure_not_cypher', counting_guard)
    return clean_driver, recorder


def entity(uuid: str, name: str, summary: str, index: int) -> EntityNode:
    return EntityNode(
        uuid=uuid,
        name=name,
        group_id=GROUP,
        labels=['Entity', 'Person'],
        summary=summary,
        name_embedding=emb(index),
        created_at=T0,
        attributes={'age': 30},
    )


def relates(uuid: str, source: str, target: str, fact: str, index: int) -> EntityEdge:
    return EntityEdge(
        uuid=uuid,
        group_id=GROUP,
        source_node_uuid=source,
        target_node_uuid=target,
        created_at=T0,
        name='KNOWS',
        fact=fact,
        fact_embedding=emb(index),
        episodes=['ep1'],
        valid_at=T0,
    )


def episode(uuid: str, content: str, valid_at: datetime) -> EpisodicNode:
    return EpisodicNode(
        uuid=uuid,
        name=uuid,
        group_id=GROUP,
        source=EpisodeType.message,
        source_description='chat',
        content=content,
        valid_at=valid_at,
        created_at=valid_at,
        entity_edges=['e1'],
    )


ALICE = entity('n1', 'Alice Anderson', 'A cheerful engineer', 0)
BOB = entity('n2', 'Bob Brown', 'A thoughtful designer', 1)
CAROL = entity('n3', 'Carol Clark', 'A careful analyst', 2)
EP1 = episode('ep1', 'Alice met Bob at the conference', T0)
EP2 = episode('ep2', 'Bob called Carol about the roadmap', T0 + timedelta(hours=1))
COMMUNITY = CommunityNode(
    uuid='c1', name='Zephyr guild', group_id=GROUP, created_at=T0, name_embedding=emb(5)
)
SAGA = SagaNode(uuid='s1', name='Saga One', group_id=GROUP, created_at=T0)


async def seed_through_core(driver: SpannerDriver) -> None:
    """Populate the graph using only the real save() entry points."""
    for node in (ALICE, BOB, CAROL, EP1, EP2, COMMUNITY, SAGA):
        await node.save(driver)
    for edge in (
        relates('e1', 'n1', 'n2', 'Alice works with Bob', 3),
        relates('e2', 'n2', 'n3', 'Bob reports to Carol', 4),
        EpisodicEdge(
            uuid='m1', group_id=GROUP, source_node_uuid='ep1', target_node_uuid='n1', created_at=T0
        ),
        EpisodicEdge(
            uuid='m2', group_id=GROUP, source_node_uuid='ep2', target_node_uuid='n1', created_at=T0
        ),
        EpisodicEdge(
            uuid='m3', group_id=GROUP, source_node_uuid='ep2', target_node_uuid='n2', created_at=T0
        ),
        CommunityEdge(
            uuid='hm1', group_id=GROUP, source_node_uuid='c1', target_node_uuid='n1', created_at=T0
        ),
    ):
        await edge.save(driver)


# --- save / get / delete ---------------------------------------------------------------------


async def test_node_and_edge_crud_never_sends_cypher(
    recorded: tuple[SpannerDriver, Recorder],
) -> None:
    driver, recorder = recorded
    await seed_through_core(driver)

    node = await EntityNode.get_by_uuid(driver, 'n1')
    assert node.name == 'Alice Anderson' and node.attributes == {'age': 30}
    assert node.name_embedding == emb(0)
    by_group = await EntityNode.get_by_group_ids(driver, [GROUP])
    assert sorted(n.uuid for n in by_group) == ['n1', 'n2', 'n3']
    assert (await EpisodicNode.get_by_uuid(driver, 'ep1')).content == EP1.content

    edge = await EntityEdge.get_by_uuid(driver, 'e1')
    assert edge.fact == 'Alice works with Bob' and edge.fact_embedding == emb(3)
    between = await EntityEdge.get_between_nodes(driver, 'n1', 'n2')
    assert [e.uuid for e in between] == ['e1']

    await EntityEdge.delete_by_uuids(driver, ['e2', 'm3'])
    assert [e.uuid for e in await EntityEdge.get_between_nodes(driver, 'n2', 'n3')] == []
    await EntityNode.delete_by_uuids(driver, ['n3', 'c1', 's1', 'ep2', 'missing'])
    remaining = await EntityNode.get_by_group_ids(driver, [GROUP])
    assert sorted(n.uuid for n in remaining) == ['n1', 'n2']

    # The generic delete_by_uuids paths are reached through the base classes too.
    await Node.delete_by_uuids(driver, ['n2'])
    await Edge.delete_by_uuids(driver, ['e1'])
    assert [n.uuid for n in await EntityNode.get_by_group_ids(driver, [GROUP])] == ['n1']

    recorder.assert_clean()


# --- episodes and maintenance ----------------------------------------------------------------


async def test_retrieve_episodes_and_clear_data_never_send_cypher(
    recorded: tuple[SpannerDriver, Recorder],
) -> None:
    driver, recorder = recorded
    await seed_through_core(driver)

    episodes = await retrieve_episodes(driver, T0 + timedelta(days=1), 10, [GROUP])
    assert [e.uuid for e in episodes] == ['ep1', 'ep2']  # oldest first
    only_message = await retrieve_episodes(
        driver, T0 + timedelta(days=1), 1, [GROUP], EpisodeType.message
    )
    assert [e.uuid for e in only_message] == ['ep2']
    assert (
        await retrieve_episodes(driver, T0 + timedelta(days=1), 5, [GROUP], EpisodeType.json) == []
    )

    await clear_data(driver, [GROUP])
    assert await EntityNode.get_by_group_ids(driver, [GROUP]) == []
    assert await retrieve_episodes(driver, T0 + timedelta(days=1), 10, [GROUP]) == []

    await seed_through_core(driver)
    await clear_data(driver)
    assert await EntityNode.get_by_group_ids(driver, [GROUP]) == []

    recorder.assert_clean()


# --- search ----------------------------------------------------------------------------------


async def test_search_utils_never_send_cypher(recorded: tuple[SpannerDriver, Recorder]) -> None:
    driver, recorder = recorded
    await seed_through_core(driver)
    groups = [GROUP]

    found = await node_fulltext_search(driver, 'Alice', FILTERS, groups, 10)
    assert [n.uuid for n in found] == ['n1']
    similar = await node_similarity_search(driver, emb(1), FILTERS, groups, 5, 0.9)
    assert [n.uuid for n in similar] == ['n2']
    edges = await edge_similarity_search(driver, emb(3), None, None, FILTERS, groups, 5, 0.9)
    assert [e.uuid for e in edges] == ['e1']
    bfs_edges = await edge_bfs_search(driver, ['n1'], 2, FILTERS, groups, 10)
    assert {e.uuid for e in bfs_edges} == {'e1', 'e2'}
    bfs_nodes = await node_bfs_search(driver, ['n1'], FILTERS, 1, groups, 10)
    assert 'n2' in {n.uuid for n in bfs_nodes}
    episodes = await episode_fulltext_search(driver, 'conference', FILTERS, groups, 10)
    assert [e.uuid for e in episodes] == ['ep1']
    communities = await community_fulltext_search(driver, 'Zephyr', groups, 10)
    assert [c.uuid for c in communities] == ['c1']

    mentioned = await get_mentioned_nodes(driver, [EP1, EP2])
    assert sorted(n.uuid for n in mentioned) == ['n1', 'n2']
    member_of = await get_communities_by_nodes(driver, [ALICE, BOB])
    assert [c.uuid for c in member_of] == ['c1']

    uuids, scores = await node_distance_reranker(driver, ['n3', 'n2', 'n1'], 'n1')
    assert uuids == ['n1', 'n2', 'n3'] and scores == [10.0, 1.0, 0.0]
    uuids, scores = await episode_mentions_reranker(driver, [['n3', 'n2', 'n1']])
    assert uuids == ['n1', 'n2', 'n3'] and scores == [2, 1, 0]

    recorder.assert_clean()


# --- bulk ------------------------------------------------------------------------------------


async def test_add_nodes_and_edges_bulk_never_sends_cypher(
    recorded: tuple[SpannerDriver, Recorder],
) -> None:
    driver, recorder = recorded
    alice = entity('b1', 'Bulk Alice', 'bulk summary one', 0)
    bob = entity('b2', 'Bulk Bob', 'bulk summary two', 1)
    ep = episode('bep1', 'bulk episode content', T0)
    mention = EpisodicEdge(
        uuid='bm1', group_id=GROUP, source_node_uuid='bep1', target_node_uuid='b1', created_at=T0
    )
    knows = relates('be1', 'b1', 'b2', 'Bulk Alice knows Bulk Bob', 3)
    knows.fact_embedding = None  # the embedder fills it in
    alice.name_embedding = None

    await add_nodes_and_edges_bulk(driver, [ep], [mention], [alice, bob], [knows], FakeEmbedder())

    saved = await EntityNode.get_by_uuid(driver, 'b1')
    assert saved.name == 'Bulk Alice' and saved.attributes == {'age': 30}
    assert saved.name_embedding is not None and len(saved.name_embedding) == DIM
    saved_edge = await EntityEdge.get_by_uuid(driver, 'be1')
    assert saved_edge.fact == knows.fact
    assert saved_edge.fact_embedding is not None and len(saved_edge.fact_embedding) == DIM
    assert (await EpisodicNode.get_by_uuid(driver, 'bep1')).content == 'bulk episode content'
    found = await node_fulltext_search(driver, 'Bulk', FILTERS, [GROUP], 10)
    assert {n.uuid for n in found} == {'b1', 'b2'}

    recorder.assert_clean()


# --- the guard itself ------------------------------------------------------------------------


async def test_recorder_catches_cypher_if_it_were_ever_sent(
    recorded: tuple[SpannerDriver, Recorder],
) -> None:
    """Sanity check that the instrumentation can fail: a real Cypher call is counted."""
    driver, recorder = recorded
    run: Callable[[], Awaitable[Any]] = lambda: driver.execute_query(  # noqa: E731
        'MATCH (n) RETURN n'
    )
    with pytest.raises(SpannerCypherNotSupportedError):
        await run()
    assert recorder.guard_calls == 1 and recorder.guard_raised == 1
    assert recorder.bad_queries()
