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
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import BaseModel

from graphiti_core.driver.driver import GraphProvider
from graphiti_core.edges import EntityEdge
from graphiti_core.errors import IdentityPropertyValidationError
from graphiti_core.graphiti import Graphiti
from graphiti_core.graphiti_types import GraphitiClients
from graphiti_core.nodes import EntityNode, EpisodeType, EpisodicNode
from graphiti_core.utils.datetime_utils import utc_now
from graphiti_core.utils.maintenance.node_operations import (
    _collect_identity_property_candidate_nodes,
    extract_attributes_only_from_nodes,
)
from graphiti_core.utils.ontology_utils.entity_types_utils import validate_identity_properties


class Person(BaseModel):
    email: str


class PersonWithRequiredFields(BaseModel):
    email: str
    full_name: str


@pytest.mark.parametrize(
    ('identity_properties', 'entity_types', 'message'),
    [
        ({'Person': ['email']}, {'Company': Person}, 'entity type is not defined'),
        ({'Person': []}, {'Person': Person}, 'non-empty list'),
        ({'Person': ['phone']}, {'Person': Person}, 'not a field'),
        ({'Person': ['email']}, None, 'entity type is not defined'),
    ],
)
def test_validate_identity_properties_rejects_invalid_configuration(
    identity_properties,
    entity_types,
    message,
):
    with pytest.raises(IdentityPropertyValidationError, match=message):
        validate_identity_properties(identity_properties, entity_types)


def test_validate_identity_properties_accepts_configured_fields():
    assert validate_identity_properties({'Person': ['email']}, {'Person': Person})


@pytest.mark.asyncio
@pytest.mark.parametrize('provider', [GraphProvider.NEO4J, GraphProvider.FALKORDB])
async def test_get_by_attribute_values_builds_safe_query(provider):
    driver = MagicMock()
    driver.provider = provider
    driver._database = 'group'
    driver.graph_operations_interface = None
    driver.execute_query = AsyncMock(return_value=([], [], None))

    result = await EntityNode.get_by_attribute_values(
        driver,
        'group',
        {'email': 'alex@example.com', 'employee_id': '42'},
        labels=['Person'],
        limit=7,
    )

    query = driver.execute_query.await_args.args[0]
    parameters = driver.execute_query.await_args.kwargs
    assert result == []
    assert 'MATCH (n:Entity:Person)' in query
    assert 'n.group_id = $group_id' in query
    assert 'n.`email` = $attribute_0' in query
    assert 'n.`employee_id` = $attribute_1' in query
    assert 'alex@example.com' not in query
    assert parameters['attribute_0'] == 'alex@example.com'
    assert parameters['attribute_1'] == '42'
    assert parameters['group_id'] == 'group'
    assert parameters['limit'] == 7


@pytest.mark.asyncio
async def test_get_by_attribute_values_returns_empty_for_empty_attributes():
    driver = MagicMock()

    assert await EntityNode.get_by_attribute_values(driver, 'group', {}) == []
    driver.execute_query.assert_not_called()


@pytest.mark.asyncio
async def test_get_by_attribute_values_uses_supplied_falkordb_driver():
    driver = MagicMock()
    driver.provider = GraphProvider.FALKORDB
    driver._database = 'default_db'
    driver.graph_operations_interface = None
    driver.execute_query = AsyncMock(return_value=([], [], None))

    await EntityNode.get_by_attribute_values(
        driver,
        'group',
        {'email': 'alex@example.com'},
        labels=['Person'],
    )

    driver.clone.assert_not_called()
    driver.execute_query.assert_awaited_once()


@pytest.mark.asyncio
async def test_get_by_attribute_values_rejects_unsafe_attribute_name():
    driver = MagicMock()
    driver.provider = GraphProvider.NEO4J
    driver.graph_operations_interface = None
    driver.execute_query = AsyncMock(return_value=([], [], None))

    with pytest.raises(ValueError, match='Invalid attribute name'):
        await EntityNode.get_by_attribute_values(
            driver,
            'group',
            {'email` OR true //': 'alex@example.com'},
        )

    driver.execute_query.assert_not_awaited()


@pytest.mark.asyncio
async def test_get_by_attribute_values_rejects_unsafe_label():
    driver = MagicMock()
    driver.provider = GraphProvider.NEO4J
    driver.graph_operations_interface = None
    driver.execute_query = AsyncMock(return_value=([], [], None))

    with pytest.raises(ValueError, match='node_labels'):
        await EntityNode.get_by_attribute_values(
            driver,
            'group',
            {'email': 'alex@example.com'},
            labels=['Person` MATCH (n)'],
        )

    driver.execute_query.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize('provider', ['neptune'])
async def test_identity_candidate_collection_ignores_unsupported_provider(provider):
    driver = MagicMock()
    driver.provider = SimpleNamespace(value=provider)
    driver.graph_operations_interface = None
    driver.execute_query = AsyncMock()
    clients = GraphitiClients.model_construct(
        driver=driver,
        embedder=MagicMock(),
        cross_encoder=MagicMock(),
        llm_client=MagicMock(),
        max_coroutines=1,
    )
    node = EntityNode(
        name='Alex',
        group_id='group',
        labels=['Entity', 'Person'],
        attributes={'email': 'alex@example.com'},
    )

    result = await _collect_identity_property_candidate_nodes(
        clients,
        [node],
        {'Person': ['email']},
    )

    assert result == [[]]
    driver.execute_query.assert_not_awaited()


@pytest.mark.asyncio
async def test_identity_candidate_collection_calls_entity_attribute_lookup(monkeypatch):
    driver = MagicMock()
    driver.graph_operations_interface = None
    clients = GraphitiClients.model_construct(
        driver=driver,
        embedder=MagicMock(),
        cross_encoder=MagicMock(),
        llm_client=MagicMock(),
        max_coroutines=1,
    )
    node = EntityNode(
        name='Alex',
        group_id='group',
        labels=['Entity', 'Person'],
        attributes={'email': 'alex@example.com'},
    )
    candidate = EntityNode(name='Alex', group_id='group', labels=['Entity', 'Person'])
    lookup = AsyncMock(return_value=[candidate])
    monkeypatch.setattr(EntityNode, 'get_by_attribute_values', lookup)

    result = await _collect_identity_property_candidate_nodes(
        clients,
        [node],
        {'Person': ['email']},
    )

    assert result == [[candidate]]
    lookup.assert_awaited_once_with(
        driver,
        'group',
        {'email': 'alex@example.com'},
        ['Person'],
        15,
    )


def _make_graphiti_for_episode_test():
    driver = MagicMock()
    clients = GraphitiClients.model_construct(
        driver=driver,
        embedder=MagicMock(),
        cross_encoder=MagicMock(),
        llm_client=MagicMock(),
        max_coroutines=1,
    )
    graphiti = Graphiti.__new__(Graphiti)
    graphiti.driver = driver
    graphiti.clients = clients
    graphiti.tracer = MagicMock()
    graphiti.tracer.start_span.return_value.__enter__.return_value = MagicMock()
    graphiti.max_coroutines = 1
    graphiti._resolve_request_scope = MagicMock(return_value=('group', driver, clients))
    graphiti.retrieve_episodes = AsyncMock(return_value=[])
    graphiti._extract_and_resolve_edges = AsyncMock(return_value=([], [], []))

    async def process_episode(episode, *_args, **_kwargs):
        return [], episode

    graphiti._process_episode_data = AsyncMock(side_effect=process_episode)
    return graphiti, clients


@pytest.mark.asyncio
async def test_pre_resolution_attribute_extraction_skips_shape_validation(monkeypatch):
    _, clients = _make_graphiti_for_episode_test()
    node = EntityNode(name='Alex', group_id='group', labels=['Entity', 'Person'])
    response = AsyncMock(return_value={'email': 'alex@example.com'})
    monkeypatch.setattr(
        'graphiti_core.utils.maintenance.node_operations.generate_prompt_response',
        response,
    )

    await extract_attributes_only_from_nodes(
        clients,
        [node],
        entity_types={'Person': PersonWithRequiredFields},
        validate_shape=False,
    )

    assert node.attributes == {'email': 'alex@example.com'}
    response.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize('configured', [True, False])
async def test_add_episode_extracts_identity_attributes_before_resolution(
    monkeypatch,
    configured,
):
    graphiti, clients = _make_graphiti_for_episode_test()
    node = EntityNode(
        name='Alex',
        group_id='group',
        labels=['Entity', 'Person'],
    )
    generic_node = EntityNode(
        name='Unknown entity',
        group_id='group',
        labels=['Entity'],
    )
    episode = EpisodicNode(
        name='conversation',
        group_id='group',
        source=EpisodeType.message,
        source_description='test',
        content='Alex uses alex@example.com.',
        valid_at=datetime.now(timezone.utc),
    )
    order = []
    extracted_nodes = [node, generic_node]
    extract_nodes = AsyncMock(
        return_value=(extracted_nodes, {node.uuid: [0], generic_node.uuid: [0]})
    )
    monkeypatch.setattr('graphiti_core.graphiti.extract_nodes', extract_nodes)

    async def extract_identity_attributes(_clients, nodes, *_args, **kwargs):
        order.append('extract')
        assert nodes == [node]
        assert kwargs['validate_shape'] is False
        nodes[0].attributes = {'email': 'alex@example.com'}
        return nodes

    extract_attributes = AsyncMock(side_effect=extract_identity_attributes)
    monkeypatch.setattr(
        'graphiti_core.graphiti.extract_attributes_only_from_nodes',
        extract_attributes,
    )

    async def resolve_nodes(_clients, nodes, *_args, **kwargs):
        order.append('resolve')
        if configured:
            assert nodes[0].attributes == {'email': 'alex@example.com'}
            assert nodes[1].attributes == {}
        assert kwargs['identity_properties'] == ({'Person': ['email']} if configured else None)
        return nodes, {node.uuid: node.uuid for node in nodes}, []

    resolve = AsyncMock(side_effect=resolve_nodes)
    monkeypatch.setattr('graphiti_core.graphiti.resolve_extracted_nodes', resolve)

    async def extract_late_attributes(_clients, nodes, *_args, **kwargs):
        order.append('late')
        assert 'skip_attribute_extraction' not in kwargs
        return nodes

    late_extract = AsyncMock(side_effect=extract_late_attributes)
    monkeypatch.setattr('graphiti_core.graphiti.extract_attributes_from_nodes', late_extract)

    await graphiti.add_episode(
        name='conversation',
        episode_body=episode.content,
        source_description='test',
        reference_time=episode.valid_at,
        group_id='group',
        entity_types={'Person': Person},
        identity_properties={'Person': ['email']} if configured else None,
    )

    if configured:
        assert order == ['extract', 'resolve', 'late']
        extract_attributes.assert_awaited_once()
    else:
        assert order == ['resolve', 'late']
        extract_attributes.assert_not_awaited()
    assert clients is graphiti.clients


@pytest.mark.asyncio
async def test_add_episode_extracts_attributes_after_generic_node_resolves_to_typed_node(
    monkeypatch,
):
    graphiti, _ = _make_graphiti_for_episode_test()
    generic_node = EntityNode(name='Alex', group_id='group', labels=['Entity'])
    stored_node = EntityNode(
        name='Alex Smith',
        group_id='group',
        labels=['Entity', 'Person'],
        attributes={'email': 'alex@example.com'},
    )
    episode = EpisodicNode(
        name='conversation',
        group_id='group',
        source=EpisodeType.message,
        source_description='test',
        content='Alex uses alex@example.com.',
        valid_at=utc_now(),
    )
    monkeypatch.setattr(
        'graphiti_core.graphiti.extract_nodes',
        AsyncMock(return_value=([generic_node], {generic_node.uuid: [0]})),
    )
    pre_extract = AsyncMock()
    monkeypatch.setattr(
        'graphiti_core.graphiti.extract_attributes_only_from_nodes',
        pre_extract,
    )
    monkeypatch.setattr(
        'graphiti_core.graphiti.resolve_extracted_nodes',
        AsyncMock(return_value=([stored_node], {generic_node.uuid: stored_node.uuid}, [])),
    )
    late_extract = AsyncMock(return_value=[stored_node])
    monkeypatch.setattr('graphiti_core.graphiti.extract_attributes_from_nodes', late_extract)

    await graphiti.add_episode(
        name=episode.name,
        episode_body=episode.content,
        source_description='test',
        reference_time=episode.valid_at,
        group_id='group',
        entity_types={'Person': Person},
        identity_properties={'Person': ['email']},
    )

    pre_extract.assert_not_awaited()
    late_extract.assert_awaited_once()
    assert late_extract.await_args.args[1] == [stored_node]
    assert 'skip_attribute_extraction' not in late_extract.await_args.kwargs


@pytest.mark.asyncio
async def test_bulk_extraction_gets_identity_attributes_before_deduplication(monkeypatch):
    graphiti, clients = _make_graphiti_for_episode_test()
    node = EntityNode(name='Alex', group_id='group', labels=['Entity', 'Person'])
    generic_node = EntityNode(name='Unknown entity', group_id='group', labels=['Entity'])
    episode = EpisodicNode(
        name='conversation',
        group_id='group',
        source=EpisodeType.message,
        source_description='test',
        content='Alex uses alex@example.com.',
        valid_at=utc_now(),
    )
    monkeypatch.setattr(
        'graphiti_core.graphiti.extract_nodes_and_edges_bulk',
        AsyncMock(return_value=([[node, generic_node]], [[]])),
    )

    async def extract_identity_attributes(_clients, nodes, *_args, **kwargs):
        assert nodes == [node]
        assert kwargs['validate_shape'] is False
        nodes[0].attributes = {'email': 'alex@example.com'}
        return nodes

    extract_attributes = AsyncMock(side_effect=extract_identity_attributes)
    monkeypatch.setattr(
        'graphiti_core.graphiti.extract_attributes_only_from_nodes',
        extract_attributes,
    )

    async def dedupe(_clients, node_groups, _episodes, _types, **kwargs):
        assert node_groups[0][0].attributes == {'email': 'alex@example.com'}
        assert kwargs['identity_properties'] == {'Person': ['email']}
        return {episode.uuid: [node]}, {node.uuid: node.uuid}

    dedupe_nodes = AsyncMock(side_effect=dedupe)
    monkeypatch.setattr('graphiti_core.graphiti.dedupe_nodes_bulk', dedupe_nodes)

    result = await graphiti._extract_and_dedupe_nodes_bulk(
        [(episode, [])],
        {},
        None,
        {'Person': Person},
        None,
        identity_properties={'Person': ['email']},
        clients=clients,
    )

    assert result[0][episode.uuid] == [node]
    extract_attributes.assert_awaited_once()
    dedupe_nodes.assert_awaited_once()


@pytest.mark.asyncio
async def test_bulk_resolution_extracts_attributes_after_resolution(monkeypatch):
    graphiti, clients = _make_graphiti_for_episode_test()
    node = EntityNode(name='Alex', group_id='group', labels=['Entity', 'Person'])
    episode = EpisodicNode(
        name='conversation',
        group_id='group',
        source=EpisodeType.message,
        source_description='test',
        content='Alex uses alex@example.com.',
        valid_at=utc_now(),
    )
    resolve = AsyncMock(return_value=([node], {node.uuid: node.uuid}, []))
    monkeypatch.setattr('graphiti_core.graphiti.resolve_extracted_nodes', resolve)
    late_extract = AsyncMock(return_value=[node])
    monkeypatch.setattr('graphiti_core.graphiti.extract_attributes_from_nodes', late_extract)
    monkeypatch.setattr(
        'graphiti_core.graphiti.resolve_extracted_edges',
        AsyncMock(return_value=([], [], [])),
    )

    await graphiti._resolve_nodes_and_edges_bulk(
        {episode.uuid: [node]},
        {episode.uuid: []},
        [(episode, [])],
        {'Person': Person},
        None,
        {},
        [episode],
        identity_properties={'Person': ['email']},
        clients=clients,
    )

    resolve.assert_awaited_once()
    assert resolve.await_args.kwargs['identity_properties'] == {'Person': ['email']}
    late_extract.assert_awaited_once()
    assert 'skip_attribute_extraction' not in late_extract.await_args.kwargs


def _make_bulk_hydration_episode(
    name: str,
    content: str,
    valid_at: datetime,
) -> EpisodicNode:
    return EpisodicNode(
        name=name,
        group_id='group',
        source=EpisodeType.message,
        source_description='test',
        content=content,
        valid_at=valid_at,
    )


def _make_bulk_hydration_test_data():
    episode_one = _make_bulk_hydration_episode(
        'episode one',
        'Alex works at Acme.',
        datetime(2024, 1, 1, tzinfo=timezone.utc),
    )
    episode_two = _make_bulk_hydration_episode(
        'episode two',
        'Alex changed teams.',
        datetime(2024, 1, 2, tzinfo=timezone.utc),
    )
    previous_one = [
        _make_bulk_hydration_episode(
            'previous one',
            'Alex joined Acme.',
            datetime(2023, 12, 30, tzinfo=timezone.utc),
        )
    ]
    previous_two = [
        _make_bulk_hydration_episode(
            'previous two',
            'Alex was a manager.',
            datetime(2023, 12, 31, tzinfo=timezone.utc),
        )
    ]
    alex = EntityNode(
        name='Alex',
        group_id='group',
        labels=['Entity', 'Person'],
        attributes={'email': 'alex@example.com'},
    )
    acme = EntityNode(name='Acme', group_id='group', labels=['Entity'])
    return episode_one, episode_two, previous_one, previous_two, alex, acme


async def _run_bulk_hydration_test(
    monkeypatch,
    graphiti,
    clients,
    episode_context,
    nodes_by_episode,
    identity_properties,
    hydrate=True,
):
    async def resolve_nodes(_clients, nodes, *_args, **_kwargs):
        return nodes, {node.uuid: node.uuid for node in nodes}, []

    monkeypatch.setattr('graphiti_core.graphiti.resolve_extracted_nodes', resolve_nodes)
    extract_attributes = AsyncMock(side_effect=lambda _clients, nodes, *_args, **_kwargs: nodes)
    monkeypatch.setattr(
        'graphiti_core.graphiti.extract_attributes_from_nodes',
        extract_attributes,
    )
    monkeypatch.setattr(
        'graphiti_core.graphiti.resolve_extracted_edges',
        AsyncMock(return_value=([], [], [])),
    )
    episodes = [episode for episode, _ in episode_context]
    result = await graphiti._resolve_nodes_and_edges_bulk(
        nodes_by_episode,
        {episode.uuid: [] for episode in episodes},
        episode_context,
        {'Person': Person},
        None,
        {},
        episodes,
        identity_properties=identity_properties,
        clients=clients,
        hydrate=hydrate,
    )
    return result, extract_attributes


@pytest.mark.asyncio
async def test_bulk_resolution_hydrates_multi_mention_identity_with_all_episodes(monkeypatch):
    graphiti, clients = _make_graphiti_for_episode_test()
    episode_one, episode_two, previous_one, previous_two, alex, acme = (
        _make_bulk_hydration_test_data()
    )
    result, extract_attributes = await _run_bulk_hydration_test(
        monkeypatch,
        graphiti,
        clients,
        [(episode_one, previous_one), (episode_two, previous_two)],
        {
            episode_one.uuid: [alex, acme],
            episode_two.uuid: [alex],
        },
        {'Person': ['email']},
        hydrate=True,
    )

    alex_calls = [call for call in extract_attributes.await_args_list if call.args[1] == [alex]]
    acme_calls = [call for call in extract_attributes.await_args_list if call.args[1] == [acme]]
    assert len(alex_calls) == 1
    assert alex_calls[0].args[2] == [episode_one, episode_two]
    assert alex_calls[0].args[3] == previous_one
    assert len(acme_calls) == 1
    assert acme_calls[0].args[2] is episode_one
    assert sum(node.uuid == alex.uuid for node in result[0]) == 1


@pytest.mark.asyncio
async def test_bulk_resolution_keeps_multi_mention_identity_without_hydration(monkeypatch):
    graphiti, clients = _make_graphiti_for_episode_test()
    episode_one, episode_two, previous_one, previous_two, alex, _ = _make_bulk_hydration_test_data()
    result, _ = await _run_bulk_hydration_test(
        monkeypatch,
        graphiti,
        clients,
        [(episode_one, previous_one), (episode_two, previous_two)],
        {
            episode_one.uuid: [alex],
            episode_two.uuid: [alex],
        },
        {'Person': ['email']},
        hydrate=False,
    )

    assert sum(node.uuid == alex.uuid for node in result[0]) == 1


@pytest.mark.asyncio
async def test_bulk_resolution_sorts_mixed_timezone_identity_episodes_by_utc(monkeypatch):
    graphiti, clients = _make_graphiti_for_episode_test()
    episode_one = _make_bulk_hydration_episode(
        'episode one',
        'Alex started a new role.',
        datetime(2024, 1, 2),
    )
    episode_two = _make_bulk_hydration_episode(
        'episode two',
        'Alex moved to a new team.',
        datetime(2024, 1, 1, tzinfo=timezone.utc),
    )
    previous_one = [
        _make_bulk_hydration_episode('previous one', 'Earlier context.', datetime(2023, 1, 1))
    ]
    previous_two = [
        _make_bulk_hydration_episode(
            'previous two',
            'Alex joined the company.',
            datetime(2023, 12, 31, tzinfo=timezone.utc),
        )
    ]
    alex = EntityNode(
        name='Alex',
        group_id='group',
        labels=['Entity', 'Person'],
        attributes={'email': 'alex@example.com'},
    )

    _, extract_attributes = await _run_bulk_hydration_test(
        monkeypatch,
        graphiti,
        clients,
        [(episode_one, previous_one), (episode_two, previous_two)],
        {
            episode_one.uuid: [alex],
            episode_two.uuid: [alex],
        },
        {'Person': ['email']},
    )

    alex_calls = [call for call in extract_attributes.await_args_list if call.args[1] == [alex]]
    assert len(alex_calls) == 1
    assert alex_calls[0].args[2] == [episode_two, episode_one]
    assert alex_calls[0].args[3] == previous_two


@pytest.mark.asyncio
async def test_bulk_resolution_without_identity_properties_keeps_episode_hydration(
    monkeypatch,
):
    graphiti, clients = _make_graphiti_for_episode_test()
    episode_one, episode_two, previous_one, previous_two, alex, acme = (
        _make_bulk_hydration_test_data()
    )
    _, extract_attributes = await _run_bulk_hydration_test(
        monkeypatch,
        graphiti,
        clients,
        [(episode_one, previous_one), (episode_two, previous_two)],
        {
            episode_one.uuid: [alex, acme],
            episode_two.uuid: [alex],
        },
        None,
    )

    assert extract_attributes.await_count == 2
    calls_by_episode = {call.args[2].uuid: call for call in extract_attributes.await_args_list}
    assert calls_by_episode[episode_one.uuid].args[1] == [alex, acme]
    assert calls_by_episode[episode_two.uuid].args[1] == []
    assert all(
        isinstance(call.args[2], EpisodicNode) for call in extract_attributes.await_args_list
    )


@pytest.mark.asyncio
async def test_bulk_resolution_hydrates_single_mention_identity_in_its_episode(monkeypatch):
    graphiti, clients = _make_graphiti_for_episode_test()
    episode = _make_bulk_hydration_episode(
        'episode',
        'Alex joined Acme.',
        datetime(2024, 1, 1, tzinfo=timezone.utc),
    )
    alex = EntityNode(name='Alex', group_id='group', labels=['Entity', 'Person'])
    result, extract_attributes = await _run_bulk_hydration_test(
        monkeypatch,
        graphiti,
        clients,
        [(episode, [])],
        {episode.uuid: [alex]},
        {'Person': ['email']},
    )

    extract_attributes.assert_awaited_once()
    assert extract_attributes.await_args.args[1] == [alex]
    assert extract_attributes.await_args.args[2] is episode
    assert result[0] == [alex]


@pytest.mark.asyncio
async def test_add_triplet_passes_identity_properties_to_both_node_resolutions(monkeypatch):
    graphiti, clients = _make_graphiti_for_episode_test()
    graphiti.embedder = clients.embedder
    graphiti.llm_client = clients.llm_client
    source = EntityNode(
        name='Alex',
        group_id='group',
        labels=['Entity', 'Person'],
        name_embedding=[0.1],
        attributes={'email': 'alex@example.com'},
    )
    target = EntityNode(
        name='Sam',
        group_id='group',
        labels=['Entity', 'Person'],
        name_embedding=[0.2],
        attributes={'email': 'sam@example.com'},
    )
    edge = EntityEdge(
        name='KNOWS',
        fact='Alex knows Sam',
        group_id='group',
        source_node_uuid=source.uuid,
        target_node_uuid=target.uuid,
        fact_embedding=[0.3],
        created_at=utc_now(),
    )
    node_lookup = AsyncMock(side_effect=lambda _driver, uuid: _raise_node_not_found(uuid))
    edge_lookup = AsyncMock(side_effect=lambda _driver, uuid: _raise_edge_not_found(uuid))
    monkeypatch.setattr(EntityNode, 'get_by_uuid', node_lookup)
    monkeypatch.setattr(EntityEdge, 'get_by_uuid', edge_lookup)
    resolve = AsyncMock(side_effect=lambda _clients, nodes, **_kwargs: (nodes, {}, []))
    monkeypatch.setattr('graphiti_core.graphiti.resolve_extracted_nodes', resolve)
    monkeypatch.setattr(
        'graphiti_core.graphiti.search',
        AsyncMock(return_value=MagicMock(edges=[])),
    )
    monkeypatch.setattr(
        'graphiti_core.graphiti.resolve_extracted_edge',
        AsyncMock(return_value=(edge, [], [])),
    )
    monkeypatch.setattr(
        'graphiti_core.graphiti.create_entity_edge_embeddings',
        AsyncMock(),
    )
    monkeypatch.setattr(
        'graphiti_core.graphiti.create_entity_node_embeddings',
        AsyncMock(),
    )
    monkeypatch.setattr('graphiti_core.graphiti.add_nodes_and_edges_bulk', AsyncMock())

    await graphiti.add_triplet(
        source,
        edge,
        target,
        identity_properties={'Person': ['email']},
    )

    assert resolve.await_count == 2
    assert all(
        call.kwargs['identity_properties'] == {'Person': ['email']}
        for call in resolve.await_args_list
    )


def _raise_node_not_found(uuid: str):
    from graphiti_core.errors import NodeNotFoundError

    raise NodeNotFoundError(uuid)


def _raise_edge_not_found(uuid: str):
    from graphiti_core.errors import EdgeNotFoundError

    raise EdgeNotFoundError(uuid)
