from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import BaseModel, Field

from graphiti_core.graphiti import Graphiti
from graphiti_core.graphiti_types import GraphitiClients
from graphiti_core.nodes import EpisodeType
from graphiti_core.utils.datetime_utils import utc_now
from graphiti_core.utils.maintenance.node_operations import has_extractable_entity_types


class Person(BaseModel):
    """A person mentioned in the episode."""

    role: str | None = Field(default=None, description='Role of the person')


def _make_graphiti() -> tuple[Graphiti, MagicMock]:
    llm_client = MagicMock()
    llm_client.max_tokens = 16384
    llm_client.generate_response = AsyncMock(
        return_value={'extracted_entities': [], 'edges': [], 'entity_resolutions': []}
    )
    clients = GraphitiClients.model_construct(
        driver=MagicMock(),
        embedder=MagicMock(),
        cross_encoder=MagicMock(),
        llm_client=llm_client,
        tracer=MagicMock(),
        prompt_library=None,
    )
    graphiti = Graphiti.__new__(Graphiti)
    graphiti.driver = clients.driver
    graphiti.clients = clients
    graphiti.tracer = MagicMock()
    graphiti.max_coroutines = 2
    graphiti._resolve_request_scope = MagicMock(return_value=('group', clients.driver, clients))
    graphiti.retrieve_episodes = AsyncMock(return_value=[])
    graphiti._extract_and_resolve_edges = AsyncMock(return_value=([], [], []))

    async def process_episode_data(episode, *_args, **_kwargs):
        return [], episode

    graphiti._process_episode_data = process_episode_data
    return graphiti, llm_client


def _patch_node_steps(monkeypatch):
    monkeypatch.setattr(
        'graphiti_core.graphiti.resolve_extracted_nodes',
        AsyncMock(return_value=([], {}, [])),
    )
    monkeypatch.setattr(
        'graphiti_core.graphiti.extract_attributes_from_nodes',
        AsyncMock(return_value=[]),
    )


async def _add_episode(graphiti: Graphiti, **kwargs):
    return await graphiti.add_episode(
        name='raw record',
        episode_body='Alice joined Acme as an engineer.',
        source_description='test',
        reference_time=utc_now(),
        source=EpisodeType.text,
        group_id='group',
        **kwargs,
    )


@pytest.mark.parametrize(
    ('entity_types', 'excluded_entity_types', 'expected'),
    [
        (None, None, True),
        (None, ['Entity'], False),
        ({'Person': Person}, ['Entity'], True),
        ({'Person': Person}, ['Person'], True),
        ({'Person': Person}, ['Entity', 'Person'], False),
    ],
)
def test_has_extractable_entity_types(entity_types, excluded_entity_types, expected):
    assert has_extractable_entity_types(entity_types, excluded_entity_types) is expected


@pytest.mark.asyncio
async def test_add_episode_skips_edge_extraction_when_every_entity_type_is_excluded(
    monkeypatch,
):
    graphiti, llm_client = _make_graphiti()
    _patch_node_steps(monkeypatch)

    results = await _add_episode(
        graphiti,
        entity_types={'Person': Person},
        excluded_entity_types=['Entity', 'Person'],
    )

    graphiti._extract_and_resolve_edges.assert_not_awaited()
    llm_client.generate_response.assert_not_awaited()
    assert results.nodes == []
    assert results.edges == []


@pytest.mark.asyncio
async def test_add_episode_still_extracts_edges_when_an_entity_type_remains(monkeypatch):
    graphiti, _llm_client = _make_graphiti()
    _patch_node_steps(monkeypatch)
    monkeypatch.setattr(
        'graphiti_core.graphiti.extract_nodes',
        AsyncMock(return_value=([], {})),
    )

    await _add_episode(
        graphiti,
        entity_types={'Person': Person},
        excluded_entity_types=['Entity'],
    )

    graphiti._extract_and_resolve_edges.assert_awaited_once()
