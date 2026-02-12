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

from graphiti_core.edges import EntityEdge
from graphiti_core.graphiti_types import GraphitiClients
from graphiti_core.nodes import EntityNode, EpisodeType, EpisodicNode
from graphiti_core.utils.datetime_utils import utc_now
from graphiti_core.utils.maintenance.node_operations import (
    _build_entity_types_context,
    _extract_entity_summaries_batch,
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


def _make_entity_node(
    name: str,
    summary: str = '',
    group_id: str = 'group',
    uuid: str | None = None,
) -> EntityNode:
    """Create a test entity node."""
    node = EntityNode(
        name=name,
        group_id=group_id,
        labels=['Entity'],
        summary=summary,
        created_at=utc_now(),
    )
    if uuid is not None:
        node.uuid = uuid
    return node


def _make_entity_edge(
    source_uuid: str,
    target_uuid: str,
    fact: str,
) -> EntityEdge:
    """Create a test entity edge."""
    return EntityEdge(
        source_node_uuid=source_uuid,
        target_node_uuid=target_uuid,
        name='TEST_RELATION',
        fact=fact,
        group_id='group',
        created_at=utc_now(),
    )


class TestExtractEntitySummariesBatch:
    @pytest.mark.asyncio
    async def test_no_nodes_needing_summarization(self):
        """When no nodes need summarization, no LLM call should be made."""
        llm_client = MagicMock()
        llm_generate = AsyncMock()
        llm_client.generate_response = llm_generate

        # Node with short summary that doesn't need LLM
        node = _make_entity_node('Alice', summary='Alice is a person.')
        nodes = [node]

        await _extract_entity_summaries_batch(
            llm_client,
            nodes,
            episode=None,
            previous_episodes=None,
            should_summarize_node=None,
            edges_by_node={},
        )

        # LLM should not be called
        llm_generate.assert_not_awaited()
        # Summary should remain unchanged
        assert nodes[0].summary == 'Alice is a person.'

    @pytest.mark.asyncio
    async def test_short_summary_with_edge_facts(self):
        """Nodes with short summaries should have edge facts appended without LLM."""
        llm_client = MagicMock()
        llm_generate = AsyncMock()
        llm_client.generate_response = llm_generate

        node = _make_entity_node('Alice', summary='Alice is a person.', uuid='alice-uuid')
        edge = _make_entity_edge('alice-uuid', 'bob-uuid', 'Alice works with Bob.')

        edges_by_node = {
            'alice-uuid': [edge],
        }

        await _extract_entity_summaries_batch(
            llm_client,
            [node],
            episode=None,
            previous_episodes=None,
            should_summarize_node=None,
            edges_by_node=edges_by_node,
        )

        # LLM should not be called
        llm_generate.assert_not_awaited()
        # Summary should include edge fact
        assert 'Alice is a person.' in node.summary
        assert 'Alice works with Bob.' in node.summary

    @pytest.mark.asyncio
    async def test_long_summary_needs_llm(self):
        """Nodes with long summaries should trigger LLM summarization."""
        llm_client = MagicMock()
        llm_generate = AsyncMock()
        llm_generate.return_value = {
            'summaries': [
                {'name': 'Alice', 'summary': 'Alice is a software engineer at Acme Corp.'}
            ]
        }
        llm_client.generate_response = llm_generate

        # Create a node with a very long summary (over MAX_SUMMARY_CHARS * 4)
        long_summary = 'Alice is a person. ' * 200  # ~3800 chars
        node = _make_entity_node('Alice', summary=long_summary)

        await _extract_entity_summaries_batch(
            llm_client,
            [node],
            episode=_make_episode(),
            previous_episodes=[],
            should_summarize_node=None,
            edges_by_node={},
        )

        # LLM should be called
        llm_generate.assert_awaited_once()
        # Summary should be updated from LLM response
        assert node.summary == 'Alice is a software engineer at Acme Corp.'

    @pytest.mark.asyncio
    async def test_should_summarize_filter(self):
        """Nodes filtered by should_summarize_node should be skipped."""
        llm_client = MagicMock()
        llm_generate = AsyncMock()
        llm_client.generate_response = llm_generate

        node = _make_entity_node('Alice', summary='')

        # Filter that rejects all nodes
        async def reject_all(n):
            return False

        await _extract_entity_summaries_batch(
            llm_client,
            [node],
            episode=_make_episode(),
            previous_episodes=[],
            should_summarize_node=reject_all,
            edges_by_node={},
        )

        # LLM should not be called
        llm_generate.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_batch_multiple_nodes(self):
        """Multiple nodes needing summarization should be batched into one call."""
        llm_client = MagicMock()
        llm_generate = AsyncMock()
        llm_generate.return_value = {
            'summaries': [
                {'name': 'Alice', 'summary': 'Alice summary.'},
                {'name': 'Bob', 'summary': 'Bob summary.'},
            ]
        }
        llm_client.generate_response = llm_generate

        # Create nodes with long summaries
        long_summary = 'X ' * 1500  # Long enough to need LLM
        alice = _make_entity_node('Alice', summary=long_summary)
        bob = _make_entity_node('Bob', summary=long_summary)

        await _extract_entity_summaries_batch(
            llm_client,
            [alice, bob],
            episode=_make_episode(),
            previous_episodes=[],
            should_summarize_node=None,
            edges_by_node={},
        )

        # LLM should be called exactly once (batch call)
        llm_generate.assert_awaited_once()
        # Both nodes should have updated summaries
        assert alice.summary == 'Alice summary.'
        assert bob.summary == 'Bob summary.'

    @pytest.mark.asyncio
    async def test_unknown_entity_in_response(self):
        """LLM returning unknown entity names should be logged but not crash."""
        llm_client = MagicMock()
        llm_generate = AsyncMock()
        llm_generate.return_value = {
            'summaries': [
                {'name': 'UnknownEntity', 'summary': 'Should be ignored.'},
                {'name': 'Alice', 'summary': 'Alice summary.'},
            ]
        }
        llm_client.generate_response = llm_generate

        long_summary = 'X ' * 1500
        alice = _make_entity_node('Alice', summary=long_summary)

        await _extract_entity_summaries_batch(
            llm_client,
            [alice],
            episode=_make_episode(),
            previous_episodes=[],
            should_summarize_node=None,
            edges_by_node={},
        )

        # Alice should have updated summary
        assert alice.summary == 'Alice summary.'

    @pytest.mark.asyncio
    async def test_no_episode_and_no_summary(self):
        """Nodes with no summary and no episode should be skipped."""
        llm_client = MagicMock()
        llm_generate = AsyncMock()
        llm_client.generate_response = llm_generate

        node = _make_entity_node('Alice', summary='')

        await _extract_entity_summaries_batch(
            llm_client,
            [node],
            episode=None,
            previous_episodes=None,
            should_summarize_node=None,
            edges_by_node={},
        )

        # LLM should not be called - no content to summarize
        llm_generate.assert_not_awaited()
        assert node.summary == ''

    @pytest.mark.asyncio
    async def test_flight_partitioning(self, monkeypatch):
        """Nodes should be partitioned into flights of MAX_NODES."""
        # Set MAX_NODES to a small value for testing
        monkeypatch.setattr(
            'graphiti_core.utils.maintenance.node_operations.MAX_NODES', 2
        )

        llm_client = MagicMock()
        call_count = 0
        call_args_list = []

        async def mock_generate(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            # Extract entity names from the context
            context = args[0][1].content if args else ''
            call_args_list.append(context)
            return {'summaries': []}

        llm_client.generate_response = mock_generate

        # Create 5 nodes with long summaries (need LLM)
        long_summary = 'X ' * 1500
        nodes = [
            _make_entity_node(f'Entity{i}', summary=long_summary)
            for i in range(5)
        ]

        await _extract_entity_summaries_batch(
            llm_client,
            nodes,
            episode=_make_episode(),
            previous_episodes=[],
            should_summarize_node=None,
            edges_by_node={},
        )

        # With MAX_NODES=2 and 5 nodes, we should have 3 flights (2+2+1)
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_case_insensitive_name_matching(self):
        """LLM response names should match case-insensitively."""
        llm_client = MagicMock()
        llm_generate = AsyncMock()
        # LLM returns name with different casing
        llm_generate.return_value = {
            'summaries': [
                {'name': 'ALICE', 'summary': 'Alice summary from LLM.'},
            ]
        }
        llm_client.generate_response = llm_generate

        # Node has lowercase name
        long_summary = 'X ' * 1500
        node = _make_entity_node('alice', summary=long_summary)

        await _extract_entity_summaries_batch(
            llm_client,
            [node],
            episode=_make_episode(),
            previous_episodes=[],
            should_summarize_node=None,
            edges_by_node={},
        )

        # Should match despite case difference
        assert node.summary == 'Alice summary from LLM.'


