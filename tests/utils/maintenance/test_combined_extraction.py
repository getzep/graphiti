from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import BaseModel

from graphiti_core.graphiti_types import GraphitiClients
from graphiti_core.nodes import EpisodeType, EpisodicNode
from graphiti_core.utils.datetime_utils import utc_now
from graphiti_core.utils.maintenance.combined_extraction import (
    TIMESTAMP_BATCH_SIZE,
    extract_nodes_and_edges,
)


def _make_clients():
    driver = MagicMock()
    embedder = MagicMock()
    cross_encoder = MagicMock()
    llm_client = MagicMock()
    llm_generate = AsyncMock()
    llm_client.generate_response = llm_generate

    clients = GraphitiClients.model_construct(
        driver=driver,
        embedder=embedder,
        cross_encoder=cross_encoder,
        llm_client=llm_client,
    )

    return clients, llm_generate


def _make_episode() -> EpisodicNode:
    return EpisodicNode(
        name='test_episode',
        group_id='group',
        source=EpisodeType.text,
        source_description='test',
        content='Acme Robotics announced the Atlas launch.',
        valid_at=utc_now(),
    )


@pytest.mark.asyncio
async def test_combined_extraction_excluding_default_entity_remaps_custom_type_to_zero():
    class Company(BaseModel):
        """A named business or organization."""

        pass

    clients, llm_generate = _make_clients()
    llm_generate.side_effect = [
        {
            'extracted_entities': [
                {'name': 'Acme Robotics', 'entity_type_id': 0},
                {'name': 'Atlas launch', 'entity_type_id': 0},
            ],
            'edges': [
                {
                    'source_entity_name': 'Acme Robotics',
                    'target_entity_name': 'Atlas launch',
                    'relation_type': 'ANNOUNCED',
                    'fact': 'Acme Robotics announced the Atlas launch.',
                    'episode_indices': [0],
                }
            ],
        },
        {'timestamps': [{'valid_at': None, 'invalid_at': None}]},
    ]

    nodes, edges, node_episode_index_map = await extract_nodes_and_edges(
        clients,
        _make_episode(),
        previous_episodes=[],
        entity_types={'Company': Company},
        excluded_entity_types=['Entity'],
    )

    assert {node.name for node in nodes} == {'Acme Robotics', 'Atlas launch'}
    assert all(set(node.labels) == {'Entity', 'Company'} for node in nodes)
    assert len(edges) == 1
    assert len(node_episode_index_map) == len(nodes)
    assert all(episode_indices == [0] for episode_indices in node_episode_index_map.values())

    prompt_text = '\n'.join(message.content for message in llm_generate.call_args_list[0].args[0])
    assert "'entity_type_name': 'Entity'" not in prompt_text
    assert "'entity_type_id': 0" in prompt_text
    assert "'entity_type_name': 'Company'" in prompt_text


@pytest.mark.asyncio
async def test_combined_extraction_strict_edge_types_filters_derived_relations():
    class Product(BaseModel):
        """A named commercial product."""

        pass

    class INTEGRATES_WITH(BaseModel):
        """One product integrates with another product."""

        pass

    clients, llm_generate = _make_clients()
    llm_generate.side_effect = [
        {
            'extracted_entities': [
                {'name': 'Atlas Home Robot', 'entity_type_id': 0},
                {'name': 'Acme Smart Hub', 'entity_type_id': 0},
            ],
            'edges': [
                {
                    'source_entity_name': 'Atlas Home Robot',
                    'target_entity_name': 'Acme Smart Hub',
                    'relation_type': 'INTEGRATES_WITH',
                    'fact': 'Atlas Home Robot integrates with Acme Smart Hub.',
                    'episode_indices': [0],
                },
                {
                    'source_entity_name': 'Atlas Home Robot',
                    'target_entity_name': 'Acme Smart Hub',
                    'relation_type': 'SOLD_WITH',
                    'fact': 'Atlas Home Robot is sold with Acme Smart Hub.',
                    'episode_indices': [0],
                },
            ],
        },
        {'timestamps': [{'valid_at': None, 'invalid_at': None}]},
    ]

    nodes, edges, _ = await extract_nodes_and_edges(
        clients,
        _make_episode(),
        previous_episodes=[],
        entity_types={'Product': Product},
        excluded_entity_types=['Entity'],
        edge_types={'INTEGRATES_WITH': INTEGRATES_WITH},
        edge_type_map={('Product', 'Product'): ['INTEGRATES_WITH']},
        strict_edge_types=True,
    )

    assert {node.name for node in nodes} == {'Atlas Home Robot', 'Acme Smart Hub'}
    assert [edge.name for edge in edges] == ['INTEGRATES_WITH']
    prompt_text = '\n'.join(message.content for message in llm_generate.call_args_list[0].args[0])
    assert 'If a relationship does not match any FACT TYPE, skip it' in prompt_text
    assert 'derive a new relation_type' in prompt_text


@pytest.mark.asyncio
async def test_combined_extraction_materializes_missing_target_endpoint():
    """Dangling target names become EntityNodes; the edge is kept, not dropped."""
    clients, llm_generate = _make_clients()
    llm_generate.side_effect = [
        {
            'extracted_entities': [
                {'name': 'AMI', 'entity_type_id': 0},
            ],
            'edges': [
                {
                    'source_entity_name': 'AMI',
                    'target_entity_name': 'version 2.18.1',
                    'relation_type': 'PINNED_AT',
                    'fact': 'AMI is pinned at version 2.18.1.',
                    'episode_indices': [0],
                }
            ],
        },
        {'timestamps': [{'valid_at': None, 'invalid_at': None}]},
    ]

    nodes, edges, node_episode_index_map = await extract_nodes_and_edges(
        clients,
        _make_episode(),
        previous_episodes=[],
    )

    assert {node.name for node in nodes} == {'AMI', 'version 2.18.1'}
    assert len(edges) == 1
    assert edges[0].name == 'PINNED_AT'
    assert edges[0].fact == 'AMI is pinned at version 2.18.1.'

    name_to_uuid = {node.name: node.uuid for node in nodes}
    assert edges[0].source_node_uuid == name_to_uuid['AMI']
    assert edges[0].target_node_uuid == name_to_uuid['version 2.18.1']
    assert edges[0].source_node_uuid != edges[0].target_node_uuid
    assert set(node_episode_index_map) == set(name_to_uuid.values())


@pytest.mark.asyncio
async def test_combined_extraction_materializes_missing_source_endpoint():
    """Dangling source names become EntityNodes; the edge is kept, not dropped."""
    clients, llm_generate = _make_clients()
    llm_generate.side_effect = [
        {
            'extracted_entities': [
                {'name': 'production cluster', 'entity_type_id': 0},
            ],
            'edges': [
                {
                    'source_entity_name': 'version 6.8.6',
                    'target_entity_name': 'production cluster',
                    'relation_type': 'DEPLOYED_ON',
                    'fact': 'version 6.8.6 is deployed on the production cluster.',
                    'episode_indices': [0],
                }
            ],
        },
        {'timestamps': [{'valid_at': None, 'invalid_at': None}]},
    ]

    nodes, edges, _ = await extract_nodes_and_edges(
        clients,
        _make_episode(),
        previous_episodes=[],
    )

    assert {node.name for node in nodes} == {'version 6.8.6', 'production cluster'}
    assert len(edges) == 1
    name_to_uuid = {node.name: node.uuid for node in nodes}
    assert edges[0].source_node_uuid == name_to_uuid['version 6.8.6']
    assert edges[0].target_node_uuid == name_to_uuid['production cluster']


@pytest.mark.asyncio
async def test_combined_extraction_keeps_edge_when_both_endpoints_present():
    """Happy path: both endpoints already extracted — no materialization needed."""
    clients, llm_generate = _make_clients()
    llm_generate.side_effect = [
        {
            'extracted_entities': [
                {'name': 'AMI', 'entity_type_id': 0},
                {'name': 'version 2.18.1', 'entity_type_id': 0},
            ],
            'edges': [
                {
                    'source_entity_name': 'AMI',
                    'target_entity_name': 'version 2.18.1',
                    'relation_type': 'PINNED_AT',
                    'fact': 'AMI is pinned at version 2.18.1.',
                    'episode_indices': [0],
                }
            ],
        },
        {'timestamps': [{'valid_at': None, 'invalid_at': None}]},
    ]

    nodes, edges, _ = await extract_nodes_and_edges(
        clients,
        _make_episode(),
        previous_episodes=[],
    )

    assert {node.name for node in nodes} == {'AMI', 'version 2.18.1'}
    assert len(edges) == 1
    assert all(set(node.labels) == {'Entity'} for node in nodes)

    prompt_text = '\n'.join(message.content for message in llm_generate.call_args_list[0].args[0])
    assert 'Every edge endpoint must also appear as an entity' in prompt_text
    assert 'version 2.18.1' in prompt_text


@pytest.mark.asyncio
async def test_combined_extraction_resolves_dangling_endpoint_case_insensitively():
    """Case/whitespace differences still match existing nodes; no duplicate materialization."""
    clients, llm_generate = _make_clients()
    llm_generate.side_effect = [
        {
            'extracted_entities': [
                {'name': 'AMI', 'entity_type_id': 0},
                {'name': 'Version 2.18.1', 'entity_type_id': 0},
            ],
            'edges': [
                {
                    'source_entity_name': 'ami',
                    'target_entity_name': 'version 2.18.1',
                    'relation_type': 'PINNED_AT',
                    'fact': 'AMI is pinned at version 2.18.1.',
                    'episode_indices': [0],
                }
            ],
        },
        {'timestamps': [{'valid_at': None, 'invalid_at': None}]},
    ]

    nodes, edges, _ = await extract_nodes_and_edges(
        clients,
        _make_episode(),
        previous_episodes=[],
    )

    assert len(nodes) == 2
    assert len(edges) == 1
    name_to_uuid = {node.name: node.uuid for node in nodes}
    assert edges[0].source_node_uuid == name_to_uuid['AMI']
    assert edges[0].target_node_uuid == name_to_uuid['Version 2.18.1']


@pytest.mark.asyncio
async def test_combined_extraction_does_not_resurrect_excluded_entity_endpoint():
    """Excluded entity types stay out of the graph even if used as edge endpoints."""

    class Person(BaseModel):
        """A person."""

        pass

    clients, llm_generate = _make_clients()
    llm_generate.side_effect = [
        {
            'extracted_entities': [
                {'name': 'Alice', 'entity_type_id': 0},
                {'name': 'secret value', 'entity_type_id': 1},
            ],
            'edges': [
                {
                    'source_entity_name': 'Alice',
                    'target_entity_name': 'secret value',
                    'relation_type': 'KNOWS',
                    'fact': 'Alice knows secret value.',
                    'episode_indices': [0],
                }
            ],
        },
        {'timestamps': [{'valid_at': None, 'invalid_at': None}]},
    ]

    nodes, edges, _ = await extract_nodes_and_edges(
        clients,
        _make_episode(),
        previous_episodes=[],
        entity_types={'Person': Person},
        excluded_entity_types=['Entity'],
    )

    assert 'secret value' not in {node.name for node in nodes}
    assert edges == []


@pytest.mark.asyncio
async def test_combined_extraction_pairwise_null_batch_does_not_retry():
    clients, llm_generate = _make_clients()
    llm_generate.side_effect = [
        {
            'extracted_entities': [
                {'name': 'Alice', 'entity_type_id': 0},
                {'name': 'Bob', 'entity_type_id': 0},
            ],
            'edges': [
                {
                    'source_entity_name': 'Alice',
                    'target_entity_name': 'Bob',
                    'relation_type': 'KNOWS',
                    'fact': 'Alice knows Bob.',
                    'episode_indices': [0],
                }
            ],
        },
        {'timestamps': [{'valid_at': None, 'invalid_at': None}]},
    ]

    _nodes, edges, _ = await extract_nodes_and_edges(
        clients,
        _make_episode(),
        previous_episodes=[],
    )

    assert len(edges) == 1
    assert edges[0].valid_at is None
    assert llm_generate.call_count == 2
    assert llm_generate.call_args_list[1].kwargs['prompt_name'] == (
        'extract_edges.extract_timestamps_batch'
    )


@pytest.mark.asyncio
async def test_combined_extraction_pairwise_keeps_an_end_before_its_start():
    """A pairwise edge keeps the extracted end time even when it precedes its start."""
    clients, llm_generate = _make_clients()
    start = '2024-02-01T00:00:00+00:00'
    end_before_start = '2024-01-01T00:00:00+00:00'
    llm_generate.side_effect = [
        {
            'extracted_entities': [
                {'name': 'Alice', 'entity_type_id': 0},
                {'name': 'Bob', 'entity_type_id': 0},
            ],
            'edges': [
                {
                    'source_entity_name': 'Alice',
                    'target_entity_name': 'Bob',
                    'relation_type': 'KNOWS',
                    'fact': 'Alice knows Bob.',
                    'episode_indices': [0],
                }
            ],
        },
        {'timestamps': [{'valid_at': start, 'invalid_at': end_before_start}]},
    ]

    _nodes, edges, _ = await extract_nodes_and_edges(
        clients,
        _make_episode(),
        previous_episodes=[],
    )

    assert len(edges) == 1
    assert edges[0].valid_at is not None
    assert edges[0].invalid_at is not None
    assert edges[0].valid_at.isoformat() == start
    assert edges[0].invalid_at.isoformat() == end_before_start


@pytest.mark.asyncio
async def test_combined_extraction_renders_timeline_with_context_turns():
    clients, llm_generate = _make_clients()
    llm_generate.side_effect = [
        {
            'extracted_entities': [{'name': 'Acme Robotics', 'entity_type_id': 0}],
            'edges': [],
        },
        {'timestamps': [{'valid_at': None, 'invalid_at': None}]},
    ]

    target_a = _make_episode()
    target_a.content = 'user turn one'
    context_b = _make_episode()
    context_b.content = 'ignored assistant turn'
    target_c = _make_episode()
    target_c.content = 'user turn two'

    await extract_nodes_and_edges(
        clients,
        [target_a, target_c],
        previous_episodes=[],
        timeline=[target_a, context_b, target_c],
    )

    prompt_text = '\n'.join(message.content for message in llm_generate.call_args_list[0].args[0])
    episode_0 = prompt_text.index('[Episode 0]')
    context = prompt_text.index('[CONTEXT EPISODE] (timestamp:')
    episode_1 = prompt_text.index('[Episode 1]')
    assert episode_0 < context < episode_1
    assert 'ignored assistant turn' in prompt_text
    assert '[CONTEXT EPISODE] are NOT' in prompt_text
    assert 'PREVIOUS MESSAGES and [CONTEXT EPISODE] messages to resolve' in prompt_text
    assert 'CONTEXT EPISODES:' not in prompt_text


@pytest.mark.asyncio
async def test_combined_extraction_timeline_indices_attribute_to_targets_only():
    """Indices in the LLM response point at numbered targets, not context turns."""
    clients, llm_generate = _make_clients()
    llm_generate.side_effect = [
        {
            'extracted_entities': [
                {'name': 'Acme Robotics', 'entity_type_id': 0},
                {'name': 'Atlas launch', 'entity_type_id': 0},
            ],
            'edges': [
                {
                    'source_entity_name': 'Acme Robotics',
                    'target_entity_name': 'Atlas launch',
                    'relation_type': 'ANNOUNCED',
                    'fact': 'Acme Robotics announced the Atlas launch.',
                    'episode_indices': [1],
                }
            ],
        },
        {'timestamps': [{'valid_at': None, 'invalid_at': None}]},
    ]

    target_a = _make_episode()
    target_a.content = 'user turn one'
    context_b = _make_episode()
    context_b.content = 'ignored assistant turn'
    target_c = _make_episode()
    target_c.content = 'user turn two'

    _nodes, edges, node_episode_index_map = await extract_nodes_and_edges(
        clients,
        [target_a, target_c],
        previous_episodes=[],
        timeline=[target_a, context_b, target_c],
    )

    assert len(edges) == 1
    assert edges[0].episodes == [target_c.uuid]
    assert context_b.uuid not in edges[0].episodes
    assert node_episode_index_map
    assert all(indices == [1] for indices in node_episode_index_map.values())


@pytest.mark.asyncio
async def test_combined_extraction_renders_custom_instructions_after_fact_types():
    class Product(BaseModel):
        """A named product."""

        pass

    class SOLD(BaseModel):
        """One entity sold another entity."""

        pass

    clients, llm_generate = _make_clients()
    llm_generate.side_effect = [
        {'extracted_entities': [], 'edges': []},
        {'timestamps': []},
    ]

    await extract_nodes_and_edges(
        clients,
        _make_episode(),
        previous_episodes=[],
        entity_types={'Product': Product},
        edge_types={'SOLD': SOLD},
        edge_type_map={('Product', 'Product'): ['SOLD']},
        custom_extraction_instructions='Only extract products launched after 2020.',
    )

    prompt_text = '\n'.join(message.content for message in llm_generate.call_args_list[0].args[0])
    assert '<CUSTOM INSTRUCTIONS>' in prompt_text
    assert 'Only extract products launched after 2020.' in prompt_text
    assert 'the CUSTOM INSTRUCTION wins' in prompt_text
    assert prompt_text.index('</FACT TYPES>') < prompt_text.index('<CUSTOM INSTRUCTIONS>')
    assert prompt_text.index('<CUSTOM INSTRUCTIONS>') < prompt_text.index('<PREVIOUS MESSAGES>')


@pytest.mark.asyncio
async def test_combined_extraction_omits_custom_instructions_block_when_empty():
    clients, llm_generate = _make_clients()
    llm_generate.side_effect = [
        {'extracted_entities': [], 'edges': []},
        {'timestamps': []},
    ]

    await extract_nodes_and_edges(
        clients,
        _make_episode(),
        previous_episodes=[],
    )

    prompt_text = '\n'.join(message.content for message in llm_generate.call_args_list[0].args[0])
    assert '<CUSTOM INSTRUCTIONS>' not in prompt_text


@pytest.mark.asyncio
async def test_combined_extraction_dates_facts_in_groups():
    fact_count = TIMESTAMP_BATCH_SIZE + 5
    extract_response = {
        'extracted_entities': [
            {'name': f'Entity {i}', 'entity_type_id': 0} for i in range(fact_count + 1)
        ],
        'edges': [
            {
                'source_entity_name': f'Entity {i}',
                'target_entity_name': f'Entity {i + 1}',
                'relation_type': 'LINKS_TO',
                'fact': f'Entity {i} links to Entity {i + 1}.',
                'episode_indices': [0],
            }
            for i in range(fact_count)
        ],
    }
    group_sizes: list[int] = []

    async def generate(messages, **kwargs):
        if kwargs['prompt_name'] != 'extract_edges.extract_timestamps_batch':
            return extract_response
        prompt = '\n'.join(message.content for message in messages)
        size = sum(f'Entity {i} links to Entity' in prompt for i in range(fact_count))
        group_sizes.append(size)
        if len(group_sizes) == 1:
            raise RuntimeError('batch timestamps failed')
        return {'timestamps': [{'valid_at': '2024-01-01T00:00:00Z', 'invalid_at': None}] * size}

    clients, llm_generate = _make_clients()
    llm_generate.side_effect = generate

    _nodes, edges, _ = await extract_nodes_and_edges(
        clients,
        _make_episode(),
        previous_episodes=[],
    )

    assert len(edges) == fact_count
    assert sorted(group_sizes) == sorted([TIMESTAMP_BATCH_SIZE, TIMESTAMP_BATCH_SIZE, 5])
    assert all(edge.valid_at is not None for edge in edges)
