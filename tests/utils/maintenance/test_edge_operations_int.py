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

import os

import pytest
from dotenv import load_dotenv

from graphiti_core.edges import EntityEdge
from graphiti_core.llm_client import LLMConfig, OpenAIClient
from graphiti_core.nodes import EpisodeType, EpisodicNode
from graphiti_core.utils.datetime_utils import utc_now
from graphiti_core.utils.maintenance.edge_operations import resolve_extracted_edge

load_dotenv()


def setup_llm_client():
    return OpenAIClient(
        LLMConfig(
            api_key=os.getenv('TEST_OPENAI_API_KEY'),
            model=os.getenv('TEST_OPENAI_MODEL', 'gpt-4.1-mini'),
            base_url='https://api.openai.com/v1',
        ),
        reasoning=None,  # Disable reasoning for non-o1/o3 models
        verbosity=None,  # Disable verbosity for non-o1/o3 models
    )


@pytest.mark.asyncio
@pytest.mark.integration
async def test_semantic_equivalence_self_referential():
    """Test that semantically equivalent self-referential facts are recognized as duplicates.

    This tests the scenario where facts like:
      - "Department of Energy is a sub-agency of Department of Energy"
      - "Department of Energy is its own sub-agency"

    should be recognized as semantically identical and deduplicated.

    This is an integration test that calls the real LLM to verify the prompt
    correctly guides the model to identify semantic duplicates.
    """
    llm_client = setup_llm_client()
    now = utc_now()

    # New fact using different phrasing
    extracted_edge = EntityEdge(
        source_node_uuid='doe_uuid',
        target_node_uuid='doe_uuid',  # Self-referential
        name='SUB_AGENCY_OF',
        group_id='group_1',
        fact='Department of Energy is its own sub-agency',
        episodes=[],
        created_at=now,
        valid_at=None,
        invalid_at=None,
    )

    episode = EpisodicNode(
        uuid='episode_uuid',
        name='Episode',
        group_id='group_1',
        source=EpisodeType.message,
        source_description='Government agency data',
        content='Agency hierarchy information',
        valid_at=now,
    )

    # Existing fact with equivalent meaning but different phrasing
    existing_edge = EntityEdge(
        uuid='existing_edge_uuid',
        source_node_uuid='doe_uuid',
        target_node_uuid='doe_uuid',  # Self-referential
        name='SUB_AGENCY_OF',
        group_id='group_1',
        fact='Department of Energy is a sub-agency of Department of Energy',
        episodes=['episode_1'],
        created_at=now,
        valid_at=None,
        invalid_at=None,
    )

    resolved_edge, invalidated, duplicates = await resolve_extracted_edge(
        llm_client,
        extracted_edge,
        [existing_edge],  # related_edges
        [],  # existing_edges for invalidation
        episode,
        edge_type_candidates=None,
        custom_edge_type_names=set(),
    )

    # The existing edge should be identified as a duplicate
    assert len(duplicates) == 1, (
        f'Expected 1 duplicate but got {len(duplicates)}. '
        f'The LLM failed to recognize semantic equivalence between: '
        f'"{extracted_edge.fact}" and "{existing_edge.fact}"'
    )
    assert existing_edge in duplicates

    # The resolved edge should be the existing one (merged)
    assert resolved_edge.uuid == existing_edge.uuid
    assert episode.uuid in resolved_edge.episodes


@pytest.mark.asyncio
@pytest.mark.integration
async def test_semantic_equivalence_active_passive_voice():
    """Test that active/passive voice variations are recognized as semantically equivalent.

    Facts like:
      - "DoD awarded a contract to ABC Corp"
      - "ABC Corp received a contract from DoD"

    should be recognized as duplicates expressing the same relationship.
    """
    llm_client = setup_llm_client()
    now = utc_now()

    # Active voice version
    extracted_edge = EntityEdge(
        source_node_uuid='dod_uuid',
        target_node_uuid='abc_uuid',
        name='AWARDED_CONTRACT',
        group_id='group_1',
        fact='DoD awarded a contract to ABC Corp',
        episodes=[],
        created_at=now,
        valid_at=None,
        invalid_at=None,
    )

    episode = EpisodicNode(
        uuid='episode_uuid',
        name='Episode',
        group_id='group_1',
        source=EpisodeType.message,
        source_description='Contract data',
        content='Contract information',
        valid_at=now,
    )

    # Passive voice version (semantically equivalent)
    existing_edge = EntityEdge(
        uuid='existing_edge_uuid',
        source_node_uuid='dod_uuid',
        target_node_uuid='abc_uuid',
        name='AWARDED_CONTRACT',
        group_id='group_1',
        fact='ABC Corp received a contract from DoD',
        episodes=['episode_1'],
        created_at=now,
        valid_at=None,
        invalid_at=None,
    )

    resolved_edge, invalidated, duplicates = await resolve_extracted_edge(
        llm_client,
        extracted_edge,
        [existing_edge],
        [],
        episode,
        edge_type_candidates=None,
        custom_edge_type_names=set(),
    )

    assert len(duplicates) == 1, (
        f'Expected 1 duplicate but got {len(duplicates)}. '
        f'The LLM failed to recognize active/passive voice equivalence between: '
        f'"{extracted_edge.fact}" and "{existing_edge.fact}"'
    )


@pytest.mark.asyncio
@pytest.mark.integration
async def test_numeric_format_equivalence():
    """Test that different numeric formats representing the same value are duplicates.

    Facts like:
      - "The contract value is $1,000,000"
      - "The contract value is $1M"

    should be recognized as duplicates (same value, different format).
    """
    llm_client = setup_llm_client()
    now = utc_now()

    extracted_edge = EntityEdge(
        source_node_uuid='contract_uuid',
        target_node_uuid='value_uuid',
        name='HAS_VALUE',
        group_id='group_1',
        fact='The contract value is $1M',
        episodes=[],
        created_at=now,
        valid_at=None,
        invalid_at=None,
    )

    episode = EpisodicNode(
        uuid='episode_uuid',
        name='Episode',
        group_id='group_1',
        source=EpisodeType.message,
        source_description='Contract data',
        content='Contract value information',
        valid_at=now,
    )

    existing_edge = EntityEdge(
        uuid='existing_edge_uuid',
        source_node_uuid='contract_uuid',
        target_node_uuid='value_uuid',
        name='HAS_VALUE',
        group_id='group_1',
        fact='The contract value is $1,000,000',
        episodes=['episode_1'],
        created_at=now,
        valid_at=None,
        invalid_at=None,
    )

    resolved_edge, invalidated, duplicates = await resolve_extracted_edge(
        llm_client,
        extracted_edge,
        [existing_edge],
        [],
        episode,
        edge_type_candidates=None,
        custom_edge_type_names=set(),
    )

    assert len(duplicates) == 1, (
        f'Expected 1 duplicate but got {len(duplicates)}. '
        f'The LLM failed to recognize numeric format equivalence between: '
        f'"{extracted_edge.fact}" and "{existing_edge.fact}"'
    )


@pytest.mark.asyncio
@pytest.mark.integration
async def test_different_numeric_values_not_duplicates():
    """Test that facts with genuinely different numeric values are NOT duplicates.

    Facts like:
      - "The contract value is $1M"
      - "The contract value is $2M"

    should NOT be duplicates - the numeric difference is meaningful.
    """
    llm_client = setup_llm_client()
    now = utc_now()

    extracted_edge = EntityEdge(
        source_node_uuid='contract_uuid',
        target_node_uuid='value_uuid',
        name='HAS_VALUE',
        group_id='group_1',
        fact='The contract value is $2M',
        episodes=[],
        created_at=now,
        valid_at=None,
        invalid_at=None,
    )

    episode = EpisodicNode(
        uuid='episode_uuid',
        name='Episode',
        group_id='group_1',
        source=EpisodeType.message,
        source_description='Contract data',
        content='Updated contract value',
        valid_at=now,
    )

    existing_edge = EntityEdge(
        uuid='existing_edge_uuid',
        source_node_uuid='contract_uuid',
        target_node_uuid='value_uuid',
        name='HAS_VALUE',
        group_id='group_1',
        fact='The contract value is $1M',
        episodes=['episode_1'],
        created_at=now,
        valid_at=None,
        invalid_at=None,
    )

    resolved_edge, invalidated, duplicates = await resolve_extracted_edge(
        llm_client,
        extracted_edge,
        [existing_edge],
        [],
        episode,
        edge_type_candidates=None,
        custom_edge_type_names=set(),
    )

    # These should NOT be duplicates - they have different values
    assert len(duplicates) == 0, (
        f'Expected 0 duplicates but got {len(duplicates)}. '
        f'The LLM incorrectly marked different values as duplicates: '
        f'"{extracted_edge.fact}" vs "{existing_edge.fact}"'
    )
    # The new edge should be the resolved edge (not merged with existing)
    assert resolved_edge.uuid == extracted_edge.uuid


# Run the tests
if __name__ == '__main__':
    pytest.main([__file__, '-v'])
