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

# Integration tests for concurrent ingestion on embedded FalkorDB.
#
# The tests use the real graph reads and the real graph writes. They make the change
# sets with no model call. They need Python 3.12 or later and the `falkordblite` extra.

import asyncio
import hashlib
import random

import pytest

pytest.importorskip('redislite')

from redislite.async_falkordb_client import AsyncFalkorDB  # noqa: E402

from graphiti_core import Graphiti  # noqa: E402
from graphiti_core.driver.falkordb_driver import FalkorDriver  # noqa: E402
from graphiti_core.edges import EntityEdge  # noqa: E402
from graphiti_core.embedder.client import EmbedderClient  # noqa: E402
from graphiti_core.ingestion import (  # noqa: E402
    AddEpisodesToEdge,
    ChangeSet,
    ConcurrentIngestionConfig,
    GraphReader,
    PatchNode,
    phases,
    scheduler,
)
from graphiti_core.ingestion.phases import commit_unit, enrich_unit  # noqa: E402
from graphiti_core.nodes import EntityNode, EpisodeType, EpisodicNode  # noqa: E402
from graphiti_core.utils.bulk_utils import (  # noqa: E402
    add_nodes_and_edges_bulk,
    execute_write_with_timeout,
    update_entity_edges_tx,
    update_entity_nodes_tx,
)
from tests.ingestion.ingestion_fixtures import T0, T1  # noqa: E402

pytestmark = pytest.mark.integration

GROUP = 'concurrent_ingestion'
# Names that the fuzzy name rule does not match with each other.
PEOPLE = [
    f'{first} {last}'
    for first in ('Ingrid', 'Mateo', 'Yuki', 'Farah', 'Tobias', 'Chloe')
    for last in ('Okafor', 'Lindqvist', 'Navarro', 'Petrov')
]
DIMENSION = 1024


def vector(text: str) -> list[float]:
    rng = random.Random(int(hashlib.md5(text.encode()).hexdigest(), 16))
    # Values with the two signs give two different texts a cosine similarity near zero.
    return [rng.uniform(-1.0, 1.0) for _ in range(DIMENSION)]


class StubEmbedder(EmbedderClient):
    async def create(self, input_data):
        return vector(str(input_data))

    async def create_batch(self, input_data_list):
        return [vector(text) for text in input_data_list]


def node(uuid: str, name: str, **values) -> EntityNode:
    values.setdefault('labels', ['Entity'])
    return EntityNode(
        uuid=uuid, name=name, group_id=GROUP, created_at=T0, name_embedding=vector(name), **values
    )


def edge(uuid: str, source: str, target: str, fact: str, **values) -> EntityEdge:
    values.setdefault('episodes', [])
    return EntityEdge(
        uuid=uuid,
        group_id=GROUP,
        source_node_uuid=source,
        target_node_uuid=target,
        name='RELATES',
        fact=fact,
        created_at=T0,
        fact_embedding=vector(fact),
        **values,
    )


@pytest.fixture
async def graphiti(tmp_path, monkeypatch):
    driver = FalkorDriver(
        falkor_db=AsyncFalkorDB(dbfilename=str(tmp_path / 'graph.db')), database=GROUP
    )
    instance = Graphiti(
        graph_driver=driver,
        embedder=StubEmbedder(),
        concurrent_ingestion=ConcurrentIngestionConfig(window=4, poll_interval_seconds=0.01),
    )
    await instance.build_indices_and_constraints()

    async def extract_attributes(clients, nodes, episode=None, *args, **kwargs):
        for item in nodes:
            item.summary = f'{item.summary}+{episode.name}'
        return nodes

    monkeypatch.setattr(phases, 'extract_attributes_from_nodes', extract_attributes)
    yield instance
    await instance.close()


async def admit(graphiti: Graphiti, name: str):
    config = graphiti.concurrent_ingestion
    return await phases.admit_episode(
        graphiti,
        config.get_state_store(),
        config,
        unit_id=name,
        group_id=GROUP,
        name=name,
        episode_body=f'content of {name}',
        source_description='test',
        reference_time=T1,
        source=EpisodeType.text,
    )


async def analyze(graphiti: Graphiti, unit, **fields) -> ChangeSet:
    store = graphiti.concurrent_ingestion.get_state_store()
    analysis = await store.start_analysis(GROUP, unit.seq)
    episode = unit.episodes[0]
    create_nodes = fields.get('create_nodes', [])
    create_edges = fields.get('create_edges', [])
    for item in create_edges:
        item.episodes = [episode.uuid]
    fields.setdefault('node_uuids', [item.uuid for item in create_nodes])
    written = episode.model_copy(deep=True)
    written.entity_edges = [item.uuid for item in create_edges]
    change_set = ChangeSet(
        group_id=GROUP,
        unit_id=unit.unit_id,
        seq=unit.seq,
        snapshot_seq=analysis.snapshot_seq,
        analysis_started_at=analysis.started_at,
        episodes=[written],
        node_first_episode=dict.fromkeys(fields['node_uuids'], episode.uuid),
        **fields,
    )
    await store.save_change_set(GROUP, unit.seq, change_set.to_bytes())
    return change_set


async def commit(graphiti: Graphiti, unit, change_set: ChangeSet):
    config = graphiti.concurrent_ingestion
    return await commit_unit(graphiti, config.get_state_store(), config, unit, change_set)


async def enrich(graphiti: Graphiti, unit, result):
    config = graphiti.concurrent_ingestion
    return await enrich_unit(graphiti, config.get_state_store(), config, unit, result)


async def test_two_units_of_one_window_merge_a_node_and_an_edge(graphiti):
    driver = graphiti.driver
    unit_1 = await admit(graphiti, 'episode-1')
    unit_2 = await admit(graphiti, 'episode-2')
    # The episode nodes are visible after admission, with no entity edges.
    stored_episode = await EpisodicNode.get_by_uuid(driver, unit_2.episodes[0].uuid)
    assert stored_episode.entity_edges == []

    change_set_1 = await analyze(
        graphiti,
        unit_1,
        create_nodes=[node('a1', 'Acme Corporation'), node('b1', 'Bo Larsen')],
        create_edges=[edge('e1', 'b1', 'a1', 'Bo Larsen works at Acme Corporation')],
    )
    change_set_2 = await analyze(
        graphiti,
        unit_2,
        create_nodes=[
            node('a2', 'acme corporation', labels=['Entity', 'Company']),
            node('b2', 'Bo Larsen'),
        ],
        create_edges=[edge('e2', 'b2', 'a2', 'bo larsen works at acme corporation')],
    )
    result_1 = await commit(graphiti, unit_1, change_set_1)
    result_2 = await commit(graphiti, unit_2, change_set_2)

    final = result_2.change_set
    assert final.uuid_map == {'a2': 'a1', 'b2': 'b1'}
    assert final.create_nodes == [] and final.create_edges == []
    assert result_2.report['repair.c3.node_merged'] == 2
    assert result_2.report['repair.c5.edge_merged'] == 1
    assert result_2.report['repair.llm_calls'] == 0

    nodes = await EntityNode.get_by_group_ids(driver, [GROUP])
    assert sorted(item.uuid for item in nodes) == ['a1', 'b1']
    edges = await EntityEdge.get_by_group_ids(driver, [GROUP])
    assert [item.uuid for item in edges] == ['e1']
    assert edges[0].episodes == [unit_1.episodes[0].uuid, unit_2.episodes[0].uuid]
    stored_episode = await EpisodicNode.get_by_uuid(driver, unit_2.episodes[0].uuid)
    assert stored_episode.entity_edges == ['e1']

    await enrich(graphiti, unit_1, result_1)
    await enrich(graphiti, unit_2, result_2)
    (company,) = await EntityNode.get_by_uuids(driver, ['a1'])
    assert company.summary == '+episode-1+episode-2'
    assert set(company.labels) == {'Entity', 'Company'}


async def test_update_queries_do_not_create_a_deleted_record(graphiti):
    driver, embedder = graphiti.driver, graphiti.embedder
    stored_nodes = [node('s1', 'Stored One'), node('s2', 'Stored Two')]
    stored_edge = edge('e0', 's1', 's2', 'Stored One knows Stored Two')
    await add_nodes_and_edges_bulk(driver, [], [], stored_nodes, [stored_edge], embedder)

    # An update changes a record that exists.
    stored_edge.episodes = ['ep-x']
    stored_nodes[0].summary = 'changed'
    await execute_write_with_timeout(
        driver, 30.0, update_entity_edges_tx, [stored_edge], embedder, driver=driver
    )
    await execute_write_with_timeout(
        driver, 30.0, update_entity_nodes_tx, [stored_nodes[0]], embedder, driver=driver
    )
    assert (await EntityEdge.get_by_uuids(driver, ['e0']))[0].episodes == ['ep-x']
    assert (await EntityNode.get_by_uuids(driver, ['s1']))[0].summary == 'changed'

    # After a delete, the same update writes nothing.
    await stored_edge.delete(driver)
    await stored_nodes[1].delete(driver)
    await execute_write_with_timeout(
        driver, 30.0, update_entity_edges_tx, [stored_edge], embedder, driver=driver
    )
    await execute_write_with_timeout(
        driver, 30.0, update_entity_nodes_tx, [stored_nodes[1]], embedder, driver=driver
    )
    reader = GraphReader(graphiti.clients)
    assert await reader.edges_by_uuids(['e0']) == []
    assert [item.uuid for item in await reader.nodes_by_uuids(['s1', 's2'], GROUP)] == ['s1']


async def test_commit_drops_an_operation_on_a_deleted_edge(graphiti):
    driver, embedder = graphiti.driver, graphiti.embedder
    stored_edge = edge('e0', 's1', 's2', 'Stored One knows Stored Two')
    await add_nodes_and_edges_bulk(
        driver,
        [],
        [],
        [node('s1', 'Stored One'), node('s2', 'Stored Two')],
        [stored_edge],
        embedder,
    )
    unit = await admit(graphiti, 'episode-1')
    change_set = await analyze(
        graphiti,
        unit,
        operations=[
            AddEpisodesToEdge(edge_uuid='e0', episode_uuids=[unit.episodes[0].uuid]),
            PatchNode(node_uuid='s1', labels=['Entity', 'Person']),
        ],
        node_uuids=['s1'],
    )
    await stored_edge.delete(driver)

    result = await commit(graphiti, unit, change_set)
    await enrich(graphiti, unit, result)

    assert result.report['operation.dropped'] == 1
    assert await GraphReader(graphiti.clients).edges_by_uuids(['e0']) == []
    (person,) = await EntityNode.get_by_uuids(driver, ['s1'])
    assert set(person.labels) == {'Entity', 'Person'}
    assert person.summary == '+episode-1'


async def test_scheduler_runs_many_units_on_one_graph(graphiti, monkeypatch):
    names = ['Acme Corporation', 'Bolt Systems', 'Cedar Labs']
    seq_by_episode: dict[str, int] = {}

    async def scripted_analysis(instance, store, config, unit, **kwargs):
        await asyncio.sleep(0.05)
        seq_by_episode[unit.episodes[0].name] = unit.seq
        company = names[unit.seq % len(names)]
        person = PEOPLE[unit.seq - 1]
        return await analyze(
            instance,
            unit,
            create_nodes=[node(f'c{unit.seq}', company), node(f'p{unit.seq}', person)],
            create_edges=[
                edge(f'e{unit.seq}', f'p{unit.seq}', f'c{unit.seq}', f'{person} works at {company}')
            ],
        )

    monkeypatch.setattr(scheduler, 'analyze_episode', scripted_analysis)

    # All 24 calls are in flight at one time. The window is 4.
    episode_names = [f'episode-{index:02d}' for index in range(24)]
    results = await asyncio.gather(
        *[
            graphiti.add_episode(
                name=name,
                episode_body=f'content of {name}',
                source_description='test',
                reference_time=T1,
                source=EpisodeType.text,
                group_id=GROUP,
            )
            for name in episode_names
        ]
    )

    assert [seq_by_episode[name] for name in episode_names] == list(range(1, 25))
    driver = graphiti.driver
    nodes = await EntityNode.get_by_group_ids(driver, [GROUP])
    by_name: dict[str, list[EntityNode]] = {}
    for item in nodes:
        by_name.setdefault(item.name, []).append(item)
    # Each company exists one time. Each person exists one time.
    assert all(len(items) == 1 for items in by_name.values()), by_name.keys()
    assert len(nodes) == len(names) + 24
    edges = await EntityEdge.get_by_group_ids(driver, [GROUP])
    assert len(edges) == 24
    company_uuids = {by_name[name][0].uuid for name in names}
    assert {item.target_node_uuid for item in edges} == company_uuids
    # The enrichment of each company ran one time for each unit that has it.
    for name in names:
        assert by_name[name][0].summary.count('+episode-') == 8
    assert all(len(result.nodes) == 2 for result in results)
