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

from unittest.mock import AsyncMock, MagicMock

import pytest

from graphiti_core.graphiti_types import GraphitiClients
from graphiti_core.nodes import EpisodeType, EpisodicNode
from graphiti_core.utils.datetime_utils import utc_now
from graphiti_core.utils.maintenance.node_operations import (
    _build_entity_types_context,
    extract_nodes,
)


def _make_clients():
    """Create mock GraphitiClients for testing."""
    driver = MagicMock()
    embedder = MagicMock()
    cross_encoder = MagicMock()
    llm_client = MagicMock()
    llm_generate = AsyncMock()
    llm_client.generate_response = llm_generate

    clients = GraphitiClients.model_construct(  # bypass validation to allow test doubles
        driver=driver,
        embedder=embedder,
        cross_encoder=cross_encoder,
        llm_client=llm_client,
    )

    return clients, llm_generate


def _make_episode(
    content: str = 'Test content',
    source: EpisodeType = EpisodeType.text,
    group_id: str = 'group',
) -> EpisodicNode:
    """Create a test episode node."""
    return EpisodicNode(
        name='test_episode',
        group_id=group_id,
        source=source,
        source_description='test',
        content=content,
        valid_at=utc_now(),
    )


class TestExtractNodesSmallInput:
    @pytest.mark.asyncio
    async def test_small_input_single_llm_call(self, monkeypatch):
        """Small inputs should use a single LLM call without chunking."""
        clients, llm_generate = _make_clients()

        # Mock LLM response
        llm_generate.return_value = {
            'extracted_entities': [
                {'name': 'Alice', 'entity_type_id': 0},
                {'name': 'Bob', 'entity_type_id': 0},
            ]
        }

        # Small content (below threshold)
        episode = _make_episode(content='Alice talked to Bob.')

        nodes = await extract_nodes(
            clients,
            episode,
            previous_episodes=[],
        )

        # Verify results
        assert len(nodes) == 2
        assert {n.name for n in nodes} == {'Alice', 'Bob'}

        # LLM should be called exactly once
        llm_generate.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_extracts_entity_types(self, monkeypatch):
        """Entity type classification should work correctly."""
        clients, llm_generate = _make_clients()

        from pydantic import BaseModel

        class Person(BaseModel):
            """A human person."""

            pass

        llm_generate.return_value = {
            'extracted_entities': [
                {'name': 'Alice', 'entity_type_id': 1},  # Person
                {'name': 'Acme Corp', 'entity_type_id': 0},  # Default Entity
            ]
        }

        episode = _make_episode(content='Alice works at Acme Corp.')

        nodes = await extract_nodes(
            clients,
            episode,
            previous_episodes=[],
            entity_types={'Person': Person},
        )

        # Alice should have Person label
        alice = next(n for n in nodes if n.name == 'Alice')
        assert 'Person' in alice.labels

        # Acme should have Entity label
        acme = next(n for n in nodes if n.name == 'Acme Corp')
        assert 'Entity' in acme.labels

    @pytest.mark.asyncio
    async def test_excludes_entity_types(self, monkeypatch):
        """Excluded entity types should not appear in results."""
        clients, llm_generate = _make_clients()

        from pydantic import BaseModel

        class User(BaseModel):
            """A user of the system."""

            pass

        llm_generate.return_value = {
            'extracted_entities': [
                {'name': 'Alice', 'entity_type_id': 1},  # User (excluded)
                {'name': 'Project X', 'entity_type_id': 0},  # Entity
            ]
        }

        episode = _make_episode(content='Alice created Project X.')

        nodes = await extract_nodes(
            clients,
            episode,
            previous_episodes=[],
            entity_types={'User': User},
            excluded_entity_types=['User'],
        )

        # Alice should be excluded
        assert len(nodes) == 1
        assert nodes[0].name == 'Project X'

    @pytest.mark.asyncio
    async def test_filters_empty_names(self, monkeypatch):
        """Entities with empty names should be filtered out."""
        clients, llm_generate = _make_clients()

        llm_generate.return_value = {
            'extracted_entities': [
                {'name': 'Alice', 'entity_type_id': 0},
                {'name': '', 'entity_type_id': 0},
                {'name': '   ', 'entity_type_id': 0},
            ]
        }

        episode = _make_episode(content='Alice is here.')

        nodes = await extract_nodes(
            clients,
            episode,
            previous_episodes=[],
        )

        assert len(nodes) == 1
        assert nodes[0].name == 'Alice'


class TestExtractNodesPromptSelection:
    @pytest.mark.asyncio
    async def test_uses_text_prompt_for_text_episodes(self, monkeypatch):
        """Text episodes should use extract_text prompt."""
        clients, llm_generate = _make_clients()
        llm_generate.return_value = {'extracted_entities': []}

        episode = _make_episode(source=EpisodeType.text)

        await extract_nodes(clients, episode, previous_episodes=[])

        # Check prompt_name parameter
        call_kwargs = llm_generate.call_args[1]
        assert call_kwargs.get('prompt_name') == 'extract_nodes.extract_text'

    @pytest.mark.asyncio
    async def test_uses_json_prompt_for_json_episodes(self, monkeypatch):
        """JSON episodes should use extract_json prompt."""
        clients, llm_generate = _make_clients()
        llm_generate.return_value = {'extracted_entities': []}

        episode = _make_episode(content='{}', source=EpisodeType.json)

        await extract_nodes(clients, episode, previous_episodes=[])

        call_kwargs = llm_generate.call_args[1]
        assert call_kwargs.get('prompt_name') == 'extract_nodes.extract_json'

    @pytest.mark.asyncio
    async def test_uses_message_prompt_for_message_episodes(self, monkeypatch):
        """Message episodes should use extract_message prompt."""
        clients, llm_generate = _make_clients()
        llm_generate.return_value = {'extracted_entities': []}

        episode = _make_episode(source=EpisodeType.message)

        await extract_nodes(clients, episode, previous_episodes=[])

        call_kwargs = llm_generate.call_args[1]
        assert call_kwargs.get('prompt_name') == 'extract_nodes.extract_message'


class TestBuildEntityTypesContext:
    def test_default_entity_type_always_included(self):
        """Default Entity type should always be at index 0."""
        context = _build_entity_types_context(None)

        assert len(context) == 1
        assert context[0]['entity_type_id'] == 0
        assert context[0]['entity_type_name'] == 'Entity'

    def test_custom_types_added_after_default(self):
        """Custom entity types should be added with sequential IDs."""
        from pydantic import BaseModel

        class Person(BaseModel):
            """A human person."""

            pass

        class Organization(BaseModel):
            """A business or organization."""

            pass

        context = _build_entity_types_context(
            {
                'Person': Person,
                'Organization': Organization,
            }
        )

        assert len(context) == 3
        assert context[0]['entity_type_name'] == 'Entity'
        assert context[1]['entity_type_name'] == 'Person'
        assert context[1]['entity_type_id'] == 1
        assert context[2]['entity_type_name'] == 'Organization'
        assert context[2]['entity_type_id'] == 2


