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

from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pytest
from pydantic import BaseModel

from graphiti_core.errors import IdentityPropertyValidationError
from graphiti_core.graphiti import Graphiti
from graphiti_core.ingestion import phases, scheduler
from graphiti_core.ingestion.phases import CommitResult
from graphiti_core.ingestion.state_store import InMemoryIngestionStateStore
from graphiti_core.nodes import NODE_DEDUP_CANDIDATE_LIMIT, EntityNode, EpisodeType
from graphiti_core.utils.maintenance import node_operations
from tests.ingestion.ingestion_fixtures import GROUP, T0, Harness, make_episode, make_node

IDENTITY_PROPERTIES = {'Person': ['email']}
IDENTITY_VALUE = 'ada@example.test'


class PersonProperties(BaseModel):
    email: str


def make_graphiti(harness: Harness) -> Graphiti:
    graphiti = Graphiti.__new__(Graphiti)
    graphiti.driver = harness.clients.driver
    graphiti.clients = harness.clients
    graphiti.tracer = harness.tracer
    graphiti.max_coroutines = None
    graphiti.store_raw_episode_content = True
    graphiti.concurrent_ingestion = harness.config
    graphiti._concurrent_ingestor = None
    graphiti._resolve_request_scope = lambda group_id: (
        group_id or GROUP,
        harness.clients.driver,
        harness.clients,
    )
    graphiti._concurrent_ingestion_enabled = lambda group_id: True
    graphiti.retrieve_episodes = harness.graphiti.retrieve_episodes
    graphiti._extract_and_resolve_edges = AsyncMock(return_value=([], [], []))
    return graphiti


async def run_concurrent_episode(monkeypatch, identity_properties):
    harness = Harness(monkeypatch)
    graphiti = make_graphiti(harness)
    extracted = make_node(
        'extracted',
        'Ada Lovelace',
        labels=['Entity', 'Person'],
        attributes={},
    )
    stored = make_node(
        'stored',
        'Different name',
        labels=['Entity', 'Person'],
        attributes={'email': IDENTITY_VALUE},
    )
    attribute_calls = []
    lookup_calls = []
    commit_calls = []

    async def admit(graphiti, store, config, *, unit_id, **kwargs):
        return await harness.admit(unit_id, make_episode(f'ep-{unit_id}'))

    async def extract_nodes(*args, **kwargs):
        return [extracted], {extracted.uuid: [0]}

    async def extract_identity_attributes(clients, nodes, *args, **kwargs):
        attribute_calls.append((list(nodes), kwargs))
        for node in nodes:
            node.attributes = {'email': IDENTITY_VALUE}
        return nodes

    async def get_by_attribute_values(cls, driver, group_id, attributes, labels=None, limit=None):
        lookup_calls.append((group_id, attributes, labels, limit))
        return [stored]

    async def commit(graphiti, store, config, unit, change_set, **kwargs):
        commit_calls.append((change_set, kwargs))
        return CommitResult(change_set=change_set)

    async def enrich(*args, **kwargs):
        return []

    monkeypatch.setattr(scheduler, 'admit_episode', admit)
    monkeypatch.setattr(scheduler, 'commit_unit', commit)
    monkeypatch.setattr(scheduler, 'enrich_unit', enrich)
    monkeypatch.setattr(phases, 'extract_nodes', extract_nodes)
    monkeypatch.setattr(
        phases,
        'extract_attributes_only_from_nodes',
        extract_identity_attributes,
        raising=False,
    )
    monkeypatch.setattr(
        node_operations, '_collect_exact_name_candidate_nodes', AsyncMock(return_value=[[]])
    )
    monkeypatch.setattr(node_operations, '_semantic_candidate_search', AsyncMock(return_value=[[]]))
    monkeypatch.setattr(EntityNode, 'get_by_attribute_values', classmethod(get_by_attribute_values))

    result = await graphiti.add_episode(
        name='identity episode',
        episode_body='Ada uses an identity key.',
        source_description='test',
        reference_time=T0,
        source=EpisodeType.text,
        group_id=GROUP,
        entity_types={'Person': PersonProperties},
        identity_properties=identity_properties,
    )
    return harness, result, attribute_calls, lookup_calls, commit_calls


async def test_concurrent_add_episode_uses_identity_properties_for_lookup(monkeypatch):
    _, _, attribute_calls, lookup_calls, commit_calls = await run_concurrent_episode(
        monkeypatch, IDENTITY_PROPERTIES
    )

    assert len(attribute_calls) == 1
    assert attribute_calls[0][0][0].uuid == 'extracted'
    assert attribute_calls[0][1] == {'validate_shape': False}
    assert lookup_calls == [
        (GROUP, {'email': IDENTITY_VALUE}, ['Person'], NODE_DEDUP_CANDIDATE_LIMIT)
    ]
    assert commit_calls[0][1]['identity_properties'] == IDENTITY_PROPERTIES


async def test_concurrent_add_episode_resolves_same_identity_without_dedupe_model_call(monkeypatch):
    harness, _, _, lookup_calls, commit_calls = await run_concurrent_episode(
        monkeypatch, IDENTITY_PROPERTIES
    )
    change_set = commit_calls[0][0]

    assert lookup_calls
    assert change_set.create_nodes == []
    assert change_set.node_uuids == ['stored']
    assert change_set.operations[0].node_uuid == 'stored'
    harness.llm.assert_not_awaited()


async def test_concurrent_add_episode_without_identity_properties_skips_identity_calls(monkeypatch):
    _, _, attribute_calls, lookup_calls, _ = await run_concurrent_episode(monkeypatch, None)

    assert attribute_calls == []
    assert lookup_calls == []


@pytest.mark.parametrize('method', ['add_episode', 'add_episode_bulk'])
async def test_invalid_identity_properties_fail_before_concurrent_writes(monkeypatch, method):
    store = InMemoryIngestionStateStore()
    begin_admit = AsyncMock(wraps=store.begin_admit)
    monkeypatch.setattr(store, 'begin_admit', begin_admit)
    graph_write = AsyncMock()
    harness = Harness(monkeypatch)
    harness.config.state_store = store
    graphiti = make_graphiti(harness)

    async def admit(graphiti, store, config, *, unit_id, group_id, **kwargs):
        await store.begin_admit(group_id, unit_id, config.lease_seconds)
        await graph_write()
        raise RuntimeError('concurrent admission reached')

    monkeypatch.setattr(scheduler, 'admit_episode', admit)
    monkeypatch.setattr(scheduler, 'admit_bulk', admit)

    with pytest.raises(IdentityPropertyValidationError):
        if method == 'add_episode':
            await graphiti.add_episode(
                name='invalid episode',
                episode_body='content',
                source_description='test',
                reference_time=datetime.now(timezone.utc),
                group_id=GROUP,
                entity_types={'Person': PersonProperties},
                identity_properties={'Missing': ['email']},
            )
        else:
            await graphiti.add_episode_bulk(
                bulk_episodes=[],
                group_id=GROUP,
                entity_types={'Person': PersonProperties},
                identity_properties={'Missing': ['email']},
            )

    begin_admit.assert_not_awaited()
    graph_write.assert_not_awaited()
