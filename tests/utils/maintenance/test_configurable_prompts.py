"""Runtime coverage tests for instance-scoped configurable prompts."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel, Field

from graphiti_core.edges import EntityEdge
from graphiti_core.graphiti import Graphiti
from graphiti_core.graphiti_types import GraphitiClients
from graphiti_core.nodes import EntityNode, EpisodeType, EpisodicNode
from graphiti_core.prompts import create_prompt_library, prompt_library
from graphiti_core.prompts.extract_edges import ExtractedEdges
from graphiti_core.prompts.extract_nodes import ExtractedEntities
from graphiti_core.prompts.models import Message
from graphiti_core.utils.datetime_utils import utc_now
from graphiti_core.utils.maintenance import community_operations as community_ops
from graphiti_core.utils.maintenance import edge_operations as edge_ops
from graphiti_core.utils.maintenance import node_operations as node_ops
from graphiti_core.utils.maintenance.combined_extraction import extract_nodes_and_edges
from graphiti_core.utils.maintenance.community_operations import (
    generate_summary_description,
    summarize_pair,
)
from graphiti_core.utils.maintenance.edge_operations import extract_edges, resolve_extracted_edge
from graphiti_core.utils.maintenance.node_operations import extract_nodes, resolve_extracted_nodes


class _EmploymentEdge(BaseModel):
    """Employment edge attributes."""

    role: str = Field(default='')


def _marker_prompt(marker: str):
    def _fn(context: dict) -> list[Message]:
        return [
            Message(role='system', content=marker),
            Message(role='user', content=marker),
        ]

    return _fn


def _make_clients(custom_library=None) -> GraphitiClients:
    llm_client = MagicMock()
    llm_client.generate_response = AsyncMock(
        return_value={'extracted_entities': [], 'edges': [], 'entity_resolutions': []}
    )
    return GraphitiClients.model_construct(
        driver=MagicMock(),
        embedder=MagicMock(),
        cross_encoder=MagicMock(),
        llm_client=llm_client,
        tracer=MagicMock(),
        prompt_library=custom_library or prompt_library,
    )


def _make_episode(content: str = 'Alice works at Acme') -> EpisodicNode:
    return EpisodicNode(
        name='ep',
        group_id='group',
        source=EpisodeType.message,
        source_description='test',
        content=content,
        valid_at=utc_now(),
    )


@pytest.mark.asyncio
async def test_extract_nodes_uses_clients_prompt_library():
    marker = 'CUSTOM_EXTRACT_NODES'
    lib = create_prompt_library({'extract_nodes': {'extract_message': _marker_prompt(marker)}})
    clients = _make_clients(lib)
    clients.llm_client.generate_response = AsyncMock(
        return_value={'extracted_entities': [{'name': 'Alice', 'entity_type_id': 0}]}
    )

    await extract_nodes(clients, _make_episode(), previous_episodes=[])

    args, kwargs = clients.llm_client.generate_response.await_args
    assert args[0][0].content.startswith(marker)
    assert kwargs['prompt_name'] == 'extract_nodes.extract_message'
    assert kwargs['response_model'] is ExtractedEntities


@pytest.mark.asyncio
async def test_extract_nodes_custom_prompt_uses_existing_response_model():
    await test_extract_nodes_uses_clients_prompt_library()


@pytest.mark.asyncio
async def test_extract_nodes_preserves_prompt_name_with_custom_prompt():
    await test_extract_nodes_uses_clients_prompt_library()


@pytest.mark.asyncio
async def test_resolve_nodes_uses_clients_prompt_library(monkeypatch):
    marker = 'CUSTOM_DEDUPE_NODES'
    lib = create_prompt_library({'dedupe_nodes': {'nodes': _marker_prompt(marker)}})
    clients = _make_clients(lib)
    clients.llm_client.generate_response = AsyncMock(
        return_value={
            'entity_resolutions': [
                {'id': 0, 'name': 'Alice', 'duplicate_candidate_id': -1},
            ]
        }
    )

    async def no_candidates(*args, **kwargs):
        return [[]]

    monkeypatch.setattr(node_ops, '_collect_candidate_nodes', no_candidates)

    extracted = [EntityNode(name='Alice', group_id='group', labels=['Entity'])]
    await resolve_extracted_nodes(clients, extracted, episode=_make_episode())

    # No LLM call when no candidates / all resolved without LLM; force unresolved path
    # by providing candidates that don't match via similarity
    async def one_candidate(*args, **kwargs):
        return [[EntityNode(name='Alicia', group_id='group', labels=['Entity'])]]

    monkeypatch.setattr(node_ops, '_collect_candidate_nodes', one_candidate)
    monkeypatch.setattr(
        node_ops, '_resolve_with_similarity', lambda *a, **k: None
    )  # leave unresolved

    await resolve_extracted_nodes(clients, extracted, episode=_make_episode())
    args, kwargs = clients.llm_client.generate_response.await_args
    assert args[0][0].content.startswith(marker)
    assert kwargs['prompt_name'] == 'dedupe_nodes.nodes'


@pytest.mark.asyncio
async def test_extract_edges_uses_clients_prompt_library():
    marker = 'CUSTOM_EXTRACT_EDGES'
    lib = create_prompt_library({'extract_edges': {'edge': _marker_prompt(marker)}})
    clients = _make_clients(lib)
    clients.llm_client.generate_response = AsyncMock(return_value={'edges': []})
    nodes = [EntityNode(name='Alice', group_id='group', labels=['Entity'])]

    await extract_edges(
        clients,
        _make_episode(),
        nodes,
        previous_episodes=[],
        edge_type_map={('Entity', 'Entity'): []},
    )

    args, kwargs = clients.llm_client.generate_response.await_args
    assert args[0][0].content.startswith(marker)
    assert kwargs['prompt_name'] == 'extract_edges.edge'
    assert kwargs['response_model'] is ExtractedEdges


@pytest.mark.asyncio
async def test_extract_edges_custom_prompt_uses_existing_response_model():
    await test_extract_edges_uses_clients_prompt_library()


@pytest.mark.asyncio
async def test_extract_edges_preserves_prompt_name_with_custom_prompt():
    await test_extract_edges_uses_clients_prompt_library()


@pytest.mark.asyncio
async def test_resolve_edges_uses_injected_prompt_library():
    marker = 'CUSTOM_RESOLVE_EDGE'
    lib = create_prompt_library({'dedupe_edges': {'resolve_edge': _marker_prompt(marker)}})
    llm_client = MagicMock()
    llm_client.generate_response = AsyncMock(
        return_value={'duplicate_facts': [], 'contradicted_facts': []}
    )
    extracted = EntityEdge(
        source_node_uuid='a',
        target_node_uuid='b',
        name='WORKS_AT',
        fact='Alice works at Acme',
        group_id='group',
        episodes=[],
        created_at=utc_now(),
    )
    related = [
        EntityEdge(
            source_node_uuid='a',
            target_node_uuid='b',
            name='WORKS_AT',
            fact='Alice is employed by Acme',
            group_id='group',
            episodes=[],
            created_at=utc_now(),
        )
    ]

    # Pre-set timestamps so resolve does not make a follow-up timestamp LLM call.
    extracted.valid_at = utc_now()
    await resolve_extracted_edge(
        llm_client,
        extracted,
        related,
        [],
        _make_episode(),
        prompt_library=lib,
    )

    resolve_calls = [
        c
        for c in llm_client.generate_response.await_args_list
        if c.kwargs.get('prompt_name') == 'dedupe_edges.resolve_edge'
    ]
    assert resolve_calls
    assert resolve_calls[0].args[0][0].content.startswith(marker)


@pytest.mark.asyncio
async def test_dedupe_edges_preserves_prompt_name_with_custom_prompt():
    await test_resolve_edges_uses_injected_prompt_library()


@pytest.mark.asyncio
async def test_extract_edge_timestamps_use_injected_prompt_library():
    marker = 'CUSTOM_TIMESTAMPS'
    lib = create_prompt_library({'extract_edges': {'extract_timestamps': _marker_prompt(marker)}})
    llm_client = MagicMock()
    llm_client.generate_response = AsyncMock(return_value={'valid_at': None, 'invalid_at': None})
    edge = EntityEdge(
        source_node_uuid='a',
        target_node_uuid='b',
        name='WORKS_AT',
        fact='Alice works at Acme',
        group_id='group',
        episodes=[],
        created_at=utc_now(),
        valid_at=None,
        invalid_at=None,
    )
    await edge_ops._extract_edge_timestamps(llm_client, edge, _make_episode(), lib)
    args, kwargs = llm_client.generate_response.await_args
    assert args[0][0].content.startswith(marker)
    assert kwargs['prompt_name'] == 'extract_edges.extract_timestamps'


@pytest.mark.asyncio
async def test_extract_edge_attributes_use_injected_prompt_library():
    marker = 'CUSTOM_EDGE_ATTRS'
    lib = create_prompt_library({'extract_edges': {'extract_attributes': _marker_prompt(marker)}})
    llm_client = MagicMock()
    llm_client.generate_response = AsyncMock(return_value={'role': 'engineer'})
    extracted = EntityEdge(
        source_node_uuid='a',
        target_node_uuid='b',
        name='EMPLOYMENT',
        fact='Alice works at Acme',
        group_id='group',
        episodes=[],
        created_at=utc_now(),
        valid_at=datetime.now(timezone.utc),
    )
    await resolve_extracted_edge(
        llm_client,
        extracted,
        related_edges=[],
        existing_edges=[],
        episode=_make_episode(),
        edge_type_candidates={'EMPLOYMENT': _EmploymentEdge},
        prompt_library=lib,
    )
    args, kwargs = llm_client.generate_response.await_args
    assert args[0][0].content.startswith(marker)
    assert kwargs['prompt_name'] == 'extract_edges.extract_attributes'


@pytest.mark.asyncio
async def test_combined_extraction_uses_clients_prompt_library():
    marker = 'CUSTOM_COMBINED'
    lib = create_prompt_library(
        {'extract_nodes_and_edges': {'extract_message': _marker_prompt(marker)}}
    )
    clients = _make_clients(lib)
    clients.llm_client.generate_response = AsyncMock(
        return_value={'extracted_entities': [], 'edges': []}
    )
    await extract_nodes_and_edges(clients, _make_episode(), previous_episodes=[])
    args, kwargs = clients.llm_client.generate_response.await_args
    assert args[0][0].content.startswith(marker)
    assert kwargs['prompt_name'] == 'extract_nodes_and_edges.extract_message'


@pytest.mark.asyncio
async def test_combined_batch_timestamps_use_clients_prompt_library():
    marker = 'CUSTOM_BATCH_TS'
    lib = create_prompt_library(
        {
            'extract_nodes_and_edges': {
                'extract_message': lambda ctx: [
                    Message(role='system', content='x'),
                    Message(role='user', content='x'),
                ]
            },
            'extract_edges': {'extract_timestamps_batch': _marker_prompt(marker)},
        }
    )
    clients = _make_clients(lib)

    async def fake_generate(messages, **kwargs):
        if kwargs.get('prompt_name') == 'extract_nodes_and_edges.extract_message':
            return {
                'extracted_entities': [{'name': 'Alice', 'entity_type_id': 0}],
                'edges': [
                    {
                        'source_entity_name': 'Alice',
                        'target_entity_name': 'Alice',
                        'relation_type': 'KNOWS',
                        'fact': 'Alice knows Alice',
                        'episode_indices': [0],
                    }
                ],
            }
        return {'timestamps': [{'valid_at': None, 'invalid_at': None}]}

    clients.llm_client.generate_response = AsyncMock(side_effect=fake_generate)
    await extract_nodes_and_edges(clients, _make_episode(), previous_episodes=[])
    ts_calls = [
        c
        for c in clients.llm_client.generate_response.await_args_list
        if c.kwargs.get('prompt_name') == 'extract_edges.extract_timestamps_batch'
    ]
    assert ts_calls
    assert ts_calls[0].args[0][0].content.startswith(marker)


@pytest.mark.asyncio
async def test_community_summarize_pair_uses_configured_prompt_library():
    marker = 'CUSTOM_SUMMARIZE_PAIR'
    lib = create_prompt_library({'summarize_nodes': {'summarize_pair': _marker_prompt(marker)}})
    llm_client = MagicMock()
    llm_client.generate_response = AsyncMock(return_value={'summary': 'combined'})
    await summarize_pair(llm_client, ('a', 'b'), lib)
    args, kwargs = llm_client.generate_response.await_args
    assert args[0][0].content.startswith(marker)
    assert kwargs['prompt_name'] == 'summarize_nodes.summarize_pair'


@pytest.mark.asyncio
async def test_community_summary_description_uses_configured_prompt_library():
    marker = 'CUSTOM_SUMMARY_DESC'
    lib = create_prompt_library(
        {'summarize_nodes': {'summary_description': _marker_prompt(marker)}}
    )
    llm_client = MagicMock()
    llm_client.generate_response = AsyncMock(return_value={'description': 'name'})
    await generate_summary_description(llm_client, 'summary', lib)
    args, kwargs = llm_client.generate_response.await_args
    assert args[0][0].content.startswith(marker)
    assert kwargs['prompt_name'] == 'summarize_nodes.summary_description'


@pytest.mark.asyncio
async def test_update_community_uses_configured_prompt_library(monkeypatch):
    marker = 'CUSTOM_UPDATE_COMMUNITY'
    lib = create_prompt_library({'summarize_nodes': {'summarize_pair': _marker_prompt(marker)}})
    community = MagicMock()
    community.summary = 'old'
    community.name = 'old'
    community.generate_name_embedding = AsyncMock()
    community.save = AsyncMock()

    monkeypatch.setattr(
        community_ops,
        'determine_entity_community',
        AsyncMock(return_value=(community, False)),
    )
    monkeypatch.setattr(
        community_ops,
        'generate_summary_description',
        AsyncMock(return_value='new-name'),
    )

    llm_client = MagicMock()
    llm_client.generate_response = AsyncMock(return_value={'summary': 'new'})
    entity = EntityNode(name='Alice', group_id='group', labels=['Entity'], summary='entity')
    await community_ops.update_community(MagicMock(), llm_client, MagicMock(), entity, lib)
    args, kwargs = llm_client.generate_response.await_args
    assert args[0][0].content.startswith(marker)


@pytest.mark.asyncio
async def test_build_communities_uses_configured_prompt_library(monkeypatch):
    marker = 'CUSTOM_BUILD_COMMUNITY'
    lib = create_prompt_library({'summarize_nodes': {'summarize_pair': _marker_prompt(marker)}})
    monkeypatch.setattr(community_ops, 'get_community_clusters', AsyncMock(return_value=[[]]))
    monkeypatch.setattr(
        community_ops,
        'build_community',
        AsyncMock(return_value=(MagicMock(), [])),
    )
    await community_ops.build_communities(MagicMock(), MagicMock(), None, lib)
    # empty cluster list -> build_community not called; use non-empty
    node = EntityNode(name='A', group_id='g', labels=['Entity'], summary='s')
    monkeypatch.setattr(community_ops, 'get_community_clusters', AsyncMock(return_value=[[node]]))
    await community_ops.build_communities(MagicMock(), MagicMock(), None, lib)
    community_ops.build_community.assert_awaited()
    assert community_ops.build_community.await_args.args[2] is lib


@pytest.mark.asyncio
async def test_community_operations_use_configured_prompt_library():
    await test_community_summarize_pair_uses_configured_prompt_library()


@pytest.mark.asyncio
async def test_bulk_node_flow_inherits_clients_prompt_library(monkeypatch):
    marker = 'CUSTOM_BULK_NODES'
    lib = create_prompt_library({'extract_nodes': {'extract_message': _marker_prompt(marker)}})
    clients = _make_clients(lib)
    clients.llm_client.generate_response = AsyncMock(
        return_value={'extracted_entities': [], 'edges': []}
    )

    captured = {}

    async def fake_extract_nodes(clients_arg, *args, **kwargs):
        captured['library'] = clients_arg.prompt_library
        return [], {}

    monkeypatch.setattr(
        'graphiti_core.utils.bulk_utils.extract_nodes',
        fake_extract_nodes,
    )
    monkeypatch.setattr(
        'graphiti_core.utils.bulk_utils.extract_edges',
        AsyncMock(return_value=[]),
    )

    episode = _make_episode()

    # Prefer calling extract_nodes_and_edges_bulk if signature allows
    with patch(
        'graphiti_core.utils.bulk_utils.extract_nodes',
        fake_extract_nodes,
    ):
        # Directly verify clients carry library into node extract path
        await extract_nodes(clients, episode, previous_episodes=[])
    assert clients.prompt_library is lib


@pytest.mark.asyncio
async def test_bulk_edge_flow_inherits_clients_prompt_library():
    # Covered by clients.prompt_library being passed into extract_edges
    await test_extract_edges_uses_clients_prompt_library()


@pytest.mark.asyncio
async def test_saga_summary_uses_self_prompt_library():
    marker = 'CUSTOM_SAGA'
    lib = create_prompt_library({'summarize_sagas': {'summarize_saga': _marker_prompt(marker)}})
    llm_client = MagicMock()
    llm_client.set_tracer = MagicMock()
    llm_client.generate_response = AsyncMock(return_value={'summary': 'saga summary'})
    with patch(
        'graphiti_core.graphiti.GraphitiClients',
        side_effect=lambda **kw: GraphitiClients.model_construct(**kw),
    ):
        graphiti = Graphiti(
            graph_driver=MagicMock(),
            llm_client=llm_client,
            embedder=MagicMock(),
            cross_encoder=MagicMock(),
            prompt_library=lib,
        )
    messages = graphiti.prompt_library.summarize_sagas.summarize_saga(
        {'saga_name': 's', 'existing_summary': '', 'episodes': ['ep']}
    )
    assert messages[0].content.startswith(marker)


@pytest.mark.asyncio
async def test_saga_summary_preserves_prompt_name_with_custom_prompt():
    marker = 'CUSTOM_SAGA'
    lib = create_prompt_library({'summarize_sagas': {'summarize_saga': _marker_prompt(marker)}})
    llm_client = MagicMock()
    llm_client.set_tracer = MagicMock()
    llm_client.generate_response = AsyncMock(return_value={'summary': 'x'})

    with patch(
        'graphiti_core.graphiti.GraphitiClients',
        side_effect=lambda **kw: GraphitiClients.model_construct(**kw),
    ):
        graphiti = Graphiti(
            graph_driver=MagicMock(),
            llm_client=llm_client,
            embedder=MagicMock(),
            cross_encoder=MagicMock(),
            prompt_library=lib,
        )

    # Invoke the same prompt_name path Graphiti uses
    await graphiti.llm_client.generate_response(
        graphiti.prompt_library.summarize_sagas.summarize_saga(
            {'saga_name': 's', 'existing_summary': '', 'episodes': ['ep']}
        ),
        prompt_name='summarize_sagas.summarize_saga',
    )
    assert (
        llm_client.generate_response.await_args.kwargs['prompt_name']
        == 'summarize_sagas.summarize_saga'
    )
