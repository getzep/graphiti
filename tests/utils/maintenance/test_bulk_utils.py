from unittest.mock import AsyncMock, MagicMock

import pytest

from graphiti_core.edges import EntityEdge
from graphiti_core.graphiti_types import GraphitiClients
from graphiti_core.nodes import EntityNode, EpisodeType, EpisodicNode
from graphiti_core.utils import bulk_utils
from graphiti_core.utils.bulk_utils import extract_nodes_and_edges_bulk
from graphiti_core.utils.datetime_utils import utc_now
from graphiti_core.utils.maintenance import combined_extraction


def _make_episode(uuid_suffix: str, group_id: str = 'group') -> EpisodicNode:
    return EpisodicNode(
        name=f'episode-{uuid_suffix}',
        group_id=group_id,
        labels=[],
        source=EpisodeType.message,
        content='content',
        source_description='test',
        created_at=utc_now(),
        valid_at=utc_now(),
    )


def _make_clients() -> GraphitiClients:
    driver = MagicMock()
    embedder = MagicMock()
    cross_encoder = MagicMock()
    llm_client = MagicMock()

    return GraphitiClients.model_construct(  # bypass validation to allow test doubles
        driver=driver,
        embedder=embedder,
        cross_encoder=cross_encoder,
        llm_client=llm_client,
    )


@pytest.mark.asyncio
async def test_dedupe_nodes_bulk_reuses_canonical_nodes(monkeypatch):
    clients = _make_clients()

    episode_one = _make_episode('1')
    episode_two = _make_episode('2')

    extracted_one = EntityNode(name='Alice Smith', group_id='group', labels=['Entity'])
    extracted_two = EntityNode(name='Alice Smith', group_id='group', labels=['Entity'])

    canonical = extracted_one

    async def fake_resolve(
        clients_arg,
        nodes_arg,
        episode_arg,
        previous_episodes_arg,
        entity_types_arg,
    ):
        if nodes_arg == [extracted_one]:
            return [canonical], {canonical.uuid: canonical.uuid}, []

        assert nodes_arg == [extracted_two]

        return [canonical], {extracted_two.uuid: canonical.uuid}, [(extracted_two, canonical)]

    monkeypatch.setattr(bulk_utils, 'resolve_extracted_nodes', fake_resolve)

    nodes_by_episode, compressed_map = await bulk_utils.dedupe_nodes_bulk(
        clients,
        [[extracted_one], [extracted_two]],
        [(episode_one, []), (episode_two, [])],
    )

    assert nodes_by_episode[episode_one.uuid] == [canonical]
    assert nodes_by_episode[episode_two.uuid] == [canonical]
    assert compressed_map.get(extracted_two.uuid) == canonical.uuid


@pytest.mark.asyncio
async def test_dedupe_nodes_bulk_handles_empty_batch(monkeypatch):
    clients = _make_clients()

    resolve_mock = AsyncMock()
    monkeypatch.setattr(bulk_utils, 'resolve_extracted_nodes', resolve_mock)

    nodes_by_episode, compressed_map = await bulk_utils.dedupe_nodes_bulk(
        clients,
        [],
        [],
    )

    assert nodes_by_episode == {}
    assert compressed_map == {}
    resolve_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_dedupe_nodes_bulk_single_episode(monkeypatch):
    clients = _make_clients()

    episode = _make_episode('solo')
    extracted = EntityNode(name='Solo', group_id='group', labels=['Entity'])

    resolve_mock = AsyncMock(return_value=([extracted], {extracted.uuid: extracted.uuid}, []))
    monkeypatch.setattr(bulk_utils, 'resolve_extracted_nodes', resolve_mock)

    nodes_by_episode, compressed_map = await bulk_utils.dedupe_nodes_bulk(
        clients,
        [[extracted]],
        [(episode, [])],
    )

    assert nodes_by_episode == {episode.uuid: [extracted]}
    assert compressed_map == {extracted.uuid: extracted.uuid}
    resolve_mock.assert_awaited_once()


@pytest.mark.asyncio
async def test_dedupe_nodes_bulk_uuid_map_respects_direction(monkeypatch):
    clients = _make_clients()

    episode_one = _make_episode('one')
    episode_two = _make_episode('two')

    extracted_one = EntityNode(uuid='b-uuid', name='Edge Case', group_id='group', labels=['Entity'])
    extracted_two = EntityNode(uuid='a-uuid', name='Edge Case', group_id='group', labels=['Entity'])

    canonical = extracted_one
    alias = extracted_two

    async def fake_resolve(
        clients_arg,
        nodes_arg,
        episode_arg,
        previous_episodes_arg,
        entity_types_arg,
    ):
        if nodes_arg == [extracted_one]:
            return [canonical], {canonical.uuid: canonical.uuid}, []
        assert nodes_arg == [extracted_two]
        return [canonical], {alias.uuid: canonical.uuid}, [(alias, canonical)]

    monkeypatch.setattr(bulk_utils, 'resolve_extracted_nodes', fake_resolve)

    nodes_by_episode, compressed_map = await bulk_utils.dedupe_nodes_bulk(
        clients,
        [[extracted_one], [extracted_two]],
        [(episode_one, []), (episode_two, [])],
    )

    assert nodes_by_episode[episode_one.uuid] == [canonical]
    assert nodes_by_episode[episode_two.uuid] == [canonical]
    assert compressed_map.get(alias.uuid) == canonical.uuid


@pytest.mark.asyncio
async def test_dedupe_nodes_bulk_missing_canonical_falls_back(monkeypatch, caplog):
    clients = _make_clients()

    episode = _make_episode('missing')
    extracted = EntityNode(name='Fallback', group_id='group', labels=['Entity'])

    resolve_mock = AsyncMock(return_value=([extracted], {extracted.uuid: 'missing-canonical'}, []))
    monkeypatch.setattr(bulk_utils, 'resolve_extracted_nodes', resolve_mock)

    with caplog.at_level('WARNING'):
        nodes_by_episode, compressed_map = await bulk_utils.dedupe_nodes_bulk(
            clients,
            [[extracted]],
            [(episode, [])],
        )

    assert nodes_by_episode[episode.uuid] == [extracted]
    assert compressed_map.get(extracted.uuid) == 'missing-canonical'
    assert any('Canonical node missing' in rec.message for rec in caplog.records)


def test_build_directed_uuid_map_empty():
    assert bulk_utils._build_directed_uuid_map([]) == {}


def test_build_directed_uuid_map_chain():
    mapping = bulk_utils._build_directed_uuid_map(
        [
            ('a', 'b'),
            ('b', 'c'),
        ]
    )

    assert mapping['a'] == 'c'
    assert mapping['b'] == 'c'
    assert mapping['c'] == 'c'


def test_build_directed_uuid_map_preserves_direction():
    mapping = bulk_utils._build_directed_uuid_map(
        [
            ('alias', 'canonical'),
        ]
    )

    assert mapping['alias'] == 'canonical'
    assert mapping['canonical'] == 'canonical'


def test_resolve_edge_pointers_updates_sources():
    created_at = utc_now()
    edge = EntityEdge(
        name='knows',
        fact='fact',
        group_id='group',
        source_node_uuid='alias',
        target_node_uuid='target',
        created_at=created_at,
    )

    surviving = bulk_utils.resolve_edge_pointers([edge], {'alias': 'canonical'})

    assert edge.source_node_uuid == 'canonical'
    assert edge.target_node_uuid == 'target'
    assert surviving == [edge]


def test_resolve_edge_pointers_drops_edges_collapsed_onto_one_node():
    """Dedup can map an endpoint onto the edge's other endpoint; that edge must go."""
    created_at = utc_now()
    collapsing = EntityEdge(
        name='pinned_at',
        fact='Postgres is pinned at Postgres 16.',
        group_id='group',
        source_node_uuid='postgres',
        target_node_uuid='postgres-16',
        created_at=created_at,
    )
    surviving_edge = EntityEdge(
        name='knows',
        fact='fact',
        group_id='group',
        source_node_uuid='alias',
        target_node_uuid='target',
        created_at=created_at,
    )

    surviving = bulk_utils.resolve_edge_pointers(
        [collapsing, surviving_edge], {'postgres-16': 'postgres', 'alias': 'canonical'}
    )

    assert surviving == [surviving_edge]


def test_resolve_edge_pointers_keeps_original_self_edges():
    """Extracted self-referencing edges stay after pointer resolution."""
    created_at = utc_now()
    self_edge = EntityEdge(
        name='feels_happy',
        fact='Alice feels happy',
        group_id='group',
        source_node_uuid='alice',
        target_node_uuid='alice',
        created_at=created_at,
    )
    remapped_self_edge = EntityEdge(
        name='jogs',
        fact='Alice goes jogging every morning',
        group_id='group',
        source_node_uuid='alice-alias',
        target_node_uuid='alice-alias',
        created_at=created_at,
    )

    surviving = bulk_utils.resolve_edge_pointers(
        [self_edge, remapped_self_edge], {'alice-alias': 'alice'}
    )

    assert surviving == [self_edge, remapped_self_edge]
    assert remapped_self_edge.source_node_uuid == 'alice'
    assert remapped_self_edge.target_node_uuid == 'alice'


@pytest.mark.asyncio
async def test_dedupe_edges_bulk_deduplicates_within_episode(monkeypatch):
    """Test that dedupe_edges_bulk correctly compares edges within the same episode.

    This test verifies the fix that removed the `if i == j: continue` check,
    which was preventing edges from the same episode from being compared against each other.
    """
    clients = _make_clients()

    # Track which edges are compared
    comparisons_made = []

    # Create mock embedder that sets embedding values
    async def mock_create_embeddings(embedder, edges):
        for edge in edges:
            edge.fact_embedding = [0.1, 0.2, 0.3]

    monkeypatch.setattr(bulk_utils, 'create_entity_edge_embeddings', mock_create_embeddings)

    # Mock resolve_extracted_edge to track comparisons and mark duplicates
    async def mock_resolve_extracted_edge(
        llm_client,
        extracted_edge,
        related_edges,
        existing_edges,
        episode,
        edge_type_candidates=None,
        custom_edge_type_names=None,
    ):
        # Track that this edge was compared against the related_edges
        comparisons_made.append((extracted_edge.uuid, [r.uuid for r in related_edges]))

        # If there are related edges with same source/target/fact, mark as duplicate
        for related in related_edges:
            if (
                related.uuid != extracted_edge.uuid  # Can't be duplicate of self
                and related.source_node_uuid == extracted_edge.source_node_uuid
                and related.target_node_uuid == extracted_edge.target_node_uuid
                and related.fact.strip().lower() == extracted_edge.fact.strip().lower()
            ):
                # Return the related edge and mark extracted_edge as duplicate
                return related, [], [related]
        # Otherwise return the extracted edge as-is
        return extracted_edge, [], []

    monkeypatch.setattr(bulk_utils, 'resolve_extracted_edge', mock_resolve_extracted_edge)

    episode = _make_episode('1')
    source_uuid = 'source-uuid'
    target_uuid = 'target-uuid'

    # Create 3 identical edges within the same episode
    edge1 = EntityEdge(
        name='recommends',
        fact='assistant recommends yoga poses',
        group_id='group',
        source_node_uuid=source_uuid,
        target_node_uuid=target_uuid,
        created_at=utc_now(),
        episodes=[episode.uuid],
    )
    edge2 = EntityEdge(
        name='recommends',
        fact='assistant recommends yoga poses',
        group_id='group',
        source_node_uuid=source_uuid,
        target_node_uuid=target_uuid,
        created_at=utc_now(),
        episodes=[episode.uuid],
    )
    edge3 = EntityEdge(
        name='recommends',
        fact='assistant recommends yoga poses',
        group_id='group',
        source_node_uuid=source_uuid,
        target_node_uuid=target_uuid,
        created_at=utc_now(),
        episodes=[episode.uuid],
    )

    await bulk_utils.dedupe_edges_bulk(
        clients,
        [[edge1, edge2, edge3]],
        [(episode, [])],
        [],
        {},
        {},
    )

    # Verify that edges were compared against each other (within same episode)
    # Each edge should have been compared against all 3 edges (including itself, which gets filtered)
    assert len(comparisons_made) == 3
    for _, compared_against in comparisons_made:
        # Each edge should have access to all 3 edges as candidates
        assert len(compared_against) >= 2  # At least 2 others (self is filtered out)


@pytest.mark.asyncio
async def test_extract_nodes_and_edges_bulk_passes_custom_instructions_to_combined_extractor(
    monkeypatch,
):
    """Test that custom_extraction_instructions is passed to combined extraction."""
    clients = _make_clients()
    episode = _make_episode('1')

    extract_combined_calls = []

    async def mock_extract_combined(
        clients,
        episode,
        previous_episodes,
        *,
        entity_types=None,
        excluded_entity_types=None,
        edge_type_map=None,
        edge_types=None,
        strict_edge_types=False,
        custom_extraction_instructions=None,
    ):
        extract_combined_calls.append(
            {
                'entity_types': entity_types,
                'excluded_entity_types': excluded_entity_types,
                'edge_type_map': edge_type_map,
                'edge_types': edge_types,
                'strict_edge_types': strict_edge_types,
                'custom_extraction_instructions': custom_extraction_instructions,
            }
        )
        return [], [], {}

    monkeypatch.setattr(combined_extraction, 'extract_nodes_and_edges', mock_extract_combined)

    custom_instructions = 'Focus on extracting person entities and their relationships.'

    await extract_nodes_and_edges_bulk(
        clients,
        [(episode, [])],
        edge_type_map={},
        custom_extraction_instructions=custom_instructions,
        use_combined_extraction=True,
    )

    assert len(extract_combined_calls) == 1
    assert extract_combined_calls[0]['custom_extraction_instructions'] == custom_instructions


@pytest.mark.asyncio
async def test_extract_nodes_and_edges_bulk_passes_edge_ontology_to_combined_extractor(
    monkeypatch,
):
    """Test that edge ontology args are passed to combined extraction."""
    clients = _make_clients()
    episode = _make_episode('1')

    extract_combined_calls = []

    async def mock_extract_combined(
        clients,
        episode,
        previous_episodes,
        *,
        entity_types=None,
        excluded_entity_types=None,
        edge_type_map=None,
        edge_types=None,
        strict_edge_types=False,
        custom_extraction_instructions=None,
    ):
        extract_combined_calls.append(
            {
                'edge_type_map': edge_type_map,
                'edge_types': edge_types,
                'strict_edge_types': strict_edge_types,
                'custom_extraction_instructions': custom_extraction_instructions,
            }
        )
        return [], [], {}

    monkeypatch.setattr(combined_extraction, 'extract_nodes_and_edges', mock_extract_combined)

    custom_instructions = 'Extract only professional relationships between people.'
    edge_type_map = {('Entity', 'Entity'): ['knows']}
    edge_types = {'knows': EntityNode}

    await extract_nodes_and_edges_bulk(
        clients,
        [(episode, [])],
        edge_type_map=edge_type_map,
        edge_types=edge_types,
        strict_edge_types=True,
        custom_extraction_instructions=custom_instructions,
        use_combined_extraction=True,
    )

    assert len(extract_combined_calls) == 1
    assert extract_combined_calls[0]['custom_extraction_instructions'] == custom_instructions
    assert extract_combined_calls[0]['edge_type_map'] == edge_type_map
    assert extract_combined_calls[0]['edge_types'] == edge_types
    assert extract_combined_calls[0]['strict_edge_types'] is True


@pytest.mark.asyncio
async def test_extract_nodes_and_edges_bulk_passes_custom_instructions_to_extract_nodes(
    monkeypatch,
):
    clients = _make_clients()
    episode = _make_episode('1')
    extract_nodes_calls = []

    async def mock_extract_nodes(
        clients,
        episode,
        previous_episodes,
        entity_types=None,
        excluded_entity_types=None,
        custom_extraction_instructions=None,
    ):
        extract_nodes_calls.append(custom_extraction_instructions)
        return [], {}

    async def mock_extract_edges(
        clients,
        episode,
        nodes,
        previous_episodes,
        edge_type_map,
        group_id='',
        edge_types=None,
        strict_edge_types=False,
        custom_extraction_instructions=None,
    ):
        return [], []

    monkeypatch.setattr(bulk_utils, 'extract_nodes', mock_extract_nodes)
    monkeypatch.setattr(bulk_utils, 'extract_edges', mock_extract_edges)

    custom_instructions = 'Focus on extracting person entities and their relationships.'
    await extract_nodes_and_edges_bulk(
        clients,
        [(episode, [])],
        edge_type_map={},
        custom_extraction_instructions=custom_instructions,
        use_combined_extraction=False,
    )

    assert extract_nodes_calls == [custom_instructions]


@pytest.mark.asyncio
async def test_extract_nodes_and_edges_bulk_passes_custom_instructions_to_extract_edges(
    monkeypatch,
):
    clients = _make_clients()
    episode = _make_episode('1')
    extracted_node = EntityNode(name='Test', group_id='group', labels=['Entity'])
    extract_edges_calls = []

    async def mock_extract_nodes(
        clients,
        episode,
        previous_episodes,
        entity_types=None,
        excluded_entity_types=None,
        custom_extraction_instructions=None,
    ):
        return [extracted_node], {}

    async def mock_extract_edges(
        clients,
        episode,
        nodes,
        previous_episodes,
        edge_type_map,
        group_id='',
        edge_types=None,
        strict_edge_types=False,
        custom_extraction_instructions=None,
    ):
        extract_edges_calls.append(
            (nodes, edge_type_map, edge_types, custom_extraction_instructions)
        )
        return [], []

    monkeypatch.setattr(bulk_utils, 'extract_nodes', mock_extract_nodes)
    monkeypatch.setattr(bulk_utils, 'extract_edges', mock_extract_edges)

    custom_instructions = 'Extract only professional relationships between people.'
    edge_type_map = {('Entity', 'Entity'): ['knows']}
    edge_types = {'knows': EntityNode}
    await extract_nodes_and_edges_bulk(
        clients,
        [(episode, [])],
        edge_type_map=edge_type_map,
        edge_types=edge_types,
        custom_extraction_instructions=custom_instructions,
        use_combined_extraction=False,
    )

    assert extract_edges_calls == [
        ([extracted_node], edge_type_map, edge_types, custom_instructions)
    ]


@pytest.mark.asyncio
async def test_extract_nodes_and_edges_bulk_defaults_to_combined_extraction(monkeypatch):
    clients = _make_clients()
    episode = _make_episode('1')

    extract_combined_calls = []

    async def mock_extract_combined(
        clients,
        episode,
        previous_episodes,
        *,
        entity_types=None,
        excluded_entity_types=None,
        edge_type_map=None,
        edge_types=None,
        strict_edge_types=False,
        custom_extraction_instructions=None,
    ):
        extract_combined_calls.append(
            {'custom_extraction_instructions': custom_extraction_instructions}
        )
        return [], [], {}

    monkeypatch.setattr(combined_extraction, 'extract_nodes_and_edges', mock_extract_combined)

    await extract_nodes_and_edges_bulk(
        clients,
        [(episode, [])],
        edge_type_map={},
    )

    assert len(extract_combined_calls) == 1
    assert extract_combined_calls[0]['custom_extraction_instructions'] is None


@pytest.mark.asyncio
async def test_extract_nodes_and_edges_bulk_custom_instructions_multiple_episodes(monkeypatch):
    """Test that custom_extraction_instructions is passed for all episodes in bulk."""
    clients = _make_clients()
    episode1 = _make_episode('1')
    episode2 = _make_episode('2')
    episode3 = _make_episode('3')

    extract_combined_calls = []

    async def mock_extract_combined(
        clients,
        episode,
        previous_episodes,
        *,
        entity_types=None,
        excluded_entity_types=None,
        edge_type_map=None,
        edge_types=None,
        strict_edge_types=False,
        custom_extraction_instructions=None,
    ):
        extract_combined_calls.append(
            {
                'episode_name': episode.name,
                'custom_extraction_instructions': custom_extraction_instructions,
            }
        )
        return [], [], {}

    monkeypatch.setattr(combined_extraction, 'extract_nodes_and_edges', mock_extract_combined)

    custom_instructions = 'Extract entities related to financial transactions.'

    await extract_nodes_and_edges_bulk(
        clients,
        [(episode1, []), (episode2, []), (episode3, [])],
        edge_type_map={},
        custom_extraction_instructions=custom_instructions,
        use_combined_extraction=True,
    )

    assert len(extract_combined_calls) == 3

    for call in extract_combined_calls:
        assert call['custom_extraction_instructions'] == custom_instructions
