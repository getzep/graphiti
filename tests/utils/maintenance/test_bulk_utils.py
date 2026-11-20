from collections import deque
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

    call_queue = deque()

    async def fake_resolve(
        clients_arg,
        nodes_arg,
        episode_arg,
        previous_episodes_arg,
        entity_types_arg,
        existing_nodes_override=None,
        require_shared_neighbor_for_name_match=True,
    ):
        call_queue.append(existing_nodes_override)

        if nodes_arg == [extracted_one]:
            return [canonical], {canonical.uuid: canonical.uuid}, []

        assert nodes_arg == [extracted_two]
        assert existing_nodes_override is None

        return [canonical], {extracted_two.uuid: canonical.uuid}, [(extracted_two, canonical)]

    monkeypatch.setattr(bulk_utils, 'resolve_extracted_nodes', fake_resolve)

    nodes_by_episode, compressed_map = await bulk_utils.dedupe_nodes_bulk(
        clients,
        [[extracted_one], [extracted_two]],
        [(episode_one, []), (episode_two, [])],
    )

    assert len(call_queue) == 2
    assert call_queue[0] is None
    assert call_queue[1] is None

    assert nodes_by_episode[episode_one.uuid] == [canonical]
    assert nodes_by_episode[episode_two.uuid] == [canonical]
    assert compressed_map.get(extracted_two.uuid) == canonical.uuid


async def _dedupe_new_nodes_bulk(
    monkeypatch,
    episodes_nodes: list[list[EntityNode]],
    require_shared_neighbor_for_name_match: bool = True,
):
    """Run dedupe_nodes_bulk where the graph pass keeps every node as it is."""

    async def fake_resolve(clients_arg, nodes_arg, *args, **kwargs):
        return list(nodes_arg), {node.uuid: node.uuid for node in nodes_arg}, []

    monkeypatch.setattr(bulk_utils, 'resolve_extracted_nodes', fake_resolve)
    episodes = [_make_episode(str(i)) for i in range(len(episodes_nodes))]
    return await bulk_utils.dedupe_nodes_bulk(
        _make_clients(),
        episodes_nodes,
        [(episode, []) for episode in episodes],
        require_shared_neighbor_for_name_match=require_shared_neighbor_for_name_match,
    )


@pytest.mark.asyncio
async def test_dedupe_nodes_bulk_generic_exact_name_without_shared_neighbor_stays_separate(
    monkeypatch,
):
    westlake = EntityNode(name='Westlake Services', group_id='group', labels=['Company'])
    capital_one = EntityNode(name='Capital One', group_id='group', labels=['Company'])
    loan_one = EntityNode(name="consumer's auto loan", group_id='group', labels=['Entity'])
    loan_two = EntityNode(name="consumer's auto loan", group_id='group', labels=['Entity'])

    _, compressed_map = await _dedupe_new_nodes_bulk(
        monkeypatch, [[westlake, loan_one], [capital_one, loan_two]]
    )

    assert compressed_map.get(loan_two.uuid, loan_two.uuid) == loan_two.uuid


@pytest.mark.asyncio
async def test_dedupe_nodes_bulk_exact_name_merges_when_shared_neighbor_is_not_required(
    monkeypatch,
):
    westlake = EntityNode(name='Westlake Services', group_id='group', labels=['Company'])
    capital_one = EntityNode(name='Capital One', group_id='group', labels=['Company'])
    loan_one = EntityNode(name="consumer's auto loan", group_id='group', labels=['Entity'])
    loan_two = EntityNode(name="consumer's auto loan", group_id='group', labels=['Entity'])

    _, compressed_map = await _dedupe_new_nodes_bulk(
        monkeypatch,
        [[westlake, loan_one], [capital_one, loan_two]],
        require_shared_neighbor_for_name_match=False,
    )

    assert compressed_map.get(loan_two.uuid) == loan_one.uuid


@pytest.mark.asyncio
async def test_dedupe_nodes_bulk_exact_name_with_shared_neighbor_merges(monkeypatch):
    westlake = EntityNode(name='Westlake Services', group_id='group', labels=['Company'])
    loan_one = EntityNode(name="consumer's auto loan", group_id='group', labels=['Entity'])
    loan_two = EntityNode(name="consumer's auto loan", group_id='group', labels=['Entity'])

    _, compressed_map = await _dedupe_new_nodes_bulk(
        monkeypatch, [[westlake, loan_one], [westlake, loan_two]]
    )

    assert compressed_map.get(loan_two.uuid) == loan_one.uuid


@pytest.mark.asyncio
async def test_dedupe_nodes_bulk_exact_name_checks_every_same_name_node(monkeypatch):
    capital_one = EntityNode(name='Capital One', group_id='group', labels=['Company'])
    westlake = EntityNode(name='Westlake Services', group_id='group', labels=['Company'])
    loan_a = EntityNode(name="consumer's auto loan", group_id='group', labels=['Entity'])
    loan_b = EntityNode(name="consumer's auto loan", group_id='group', labels=['Entity'])
    loan_c = EntityNode(name="consumer's auto loan", group_id='group', labels=['Entity'])

    _, compressed_map = await _dedupe_new_nodes_bulk(
        monkeypatch, [[capital_one, loan_a], [westlake, loan_b], [westlake, loan_c]]
    )

    assert compressed_map.get(loan_b.uuid, loan_b.uuid) == loan_b.uuid
    assert compressed_map.get(loan_c.uuid) == loan_b.uuid


@pytest.mark.asyncio
async def test_dedupe_nodes_bulk_same_episode_exact_name_merges(monkeypatch):
    loan_one = EntityNode(name="consumer's auto loan", group_id='group', labels=['Entity'])
    loan_two = EntityNode(name="consumer's auto loan", group_id='group', labels=['Entity'])

    _, compressed_map = await _dedupe_new_nodes_bulk(monkeypatch, [[loan_one, loan_two]])

    assert compressed_map.get(loan_two.uuid) == loan_one.uuid


@pytest.mark.asyncio
async def test_dedupe_nodes_bulk_merges_identity_matches_before_name_comparison(monkeypatch):
    clients = _make_clients()
    episode_one = _make_episode('identity-1')
    episode_two = _make_episode('identity-2')
    extracted_one = EntityNode(
        name='Alice Smith',
        group_id='group',
        labels=['Entity', 'Person'],
        attributes={'email': 'shared@example.com'},
    )
    extracted_two = EntityNode(
        name='Bob Jones',
        group_id='group',
        labels=['Entity', 'Person'],
        attributes={'email': ' shared@example.com ', 'title': 'engineer'},
    )

    async def fake_resolve(
        clients_arg,
        nodes_arg,
        episode_arg,
        previous_episodes_arg,
        entity_types_arg,
        **kwargs,
    ):
        assert kwargs['identity_properties'] == {'Person': ['email']}
        return nodes_arg, {node.uuid: node.uuid for node in nodes_arg}, []

    monkeypatch.setattr(bulk_utils, 'resolve_extracted_nodes', fake_resolve)

    nodes_by_episode, uuid_map = await bulk_utils.dedupe_nodes_bulk(
        clients,
        [[extracted_one], [extracted_two]],
        [(episode_one, []), (episode_two, [])],
        identity_properties={'Person': ['email']},
    )

    assert nodes_by_episode[episode_one.uuid] == [extracted_one]
    assert nodes_by_episode[episode_two.uuid] == [extracted_one]
    assert uuid_map[extracted_two.uuid] == extracted_one.uuid
    assert extracted_one.attributes['email'] == 'shared@example.com'
    assert extracted_one.attributes['title'] == 'engineer'


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
        existing_nodes_override=None,
        require_shared_neighbor_for_name_match=True,
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
async def test_dedupe_edges_bulk_merges_episode_provenance_for_duplicates(monkeypatch):
    clients = _make_clients()
    episode_one = _make_episode('1')
    episode_two = _make_episode('2')

    async def mock_create_embeddings(embedder, edges):
        for edge in edges:
            edge.fact_embedding = [0.1, 0.2, 0.3]

    async def mock_resolve_extracted_edge(
        llm_client,
        extracted_edge,
        related_edges,
        existing_edges,
        episode,
        edge_type_candidates=None,
        custom_edge_type_names=None,
    ):
        duplicates = [
            related
            for related in related_edges
            if related.uuid != extracted_edge.uuid and related.fact == extracted_edge.fact
        ]
        return extracted_edge, [], duplicates

    monkeypatch.setattr(bulk_utils, 'create_entity_edge_embeddings', mock_create_embeddings)
    monkeypatch.setattr(bulk_utils, 'resolve_extracted_edge', mock_resolve_extracted_edge)

    edge_one = EntityEdge(
        uuid='edge-a',
        name='recommends',
        fact='Assistant recommends yoga',
        group_id='group',
        source_node_uuid='assistant',
        target_node_uuid='yoga',
        created_at=utc_now(),
        episodes=[episode_one.uuid],
    )
    edge_two = EntityEdge(
        uuid='edge-b',
        name='recommends',
        fact='Assistant recommends yoga',
        group_id='group',
        source_node_uuid='assistant',
        target_node_uuid='yoga',
        created_at=utc_now(),
        episodes=[episode_two.uuid],
    )

    edges_by_episode = await bulk_utils.dedupe_edges_bulk(
        clients,
        [[edge_one], [edge_two]],
        [(episode_one, []), (episode_two, [])],
        [],
        {},
        {},
    )

    canonical_edge_one = edges_by_episode[episode_one.uuid][0]
    canonical_edge_two = edges_by_episode[episode_two.uuid][0]
    assert canonical_edge_one is canonical_edge_two
    assert canonical_edge_one.uuid == 'edge-a'
    assert canonical_edge_one.episodes == [episode_one.uuid, episode_two.uuid]


@pytest.mark.asyncio
async def test_extract_nodes_and_edges_bulk_passes_knowledge_to_combined_extractor(
    monkeypatch,
):
    """Test that knowledge is passed to combined extraction."""
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
        knowledge=None,
    ):
        extract_combined_calls.append(
            {
                'entity_types': entity_types,
                'excluded_entity_types': excluded_entity_types,
                'edge_type_map': edge_type_map,
                'edge_types': edge_types,
                'strict_edge_types': strict_edge_types,
                'knowledge': knowledge,
            }
        )
        return [], [], {}

    monkeypatch.setattr(combined_extraction, 'extract_nodes_and_edges', mock_extract_combined)

    knowledge = 'Focus on extracting person entities and their relationships.'

    await extract_nodes_and_edges_bulk(
        clients,
        [(episode, [])],
        edge_type_map={},
        knowledge=knowledge,
    )

    assert len(extract_combined_calls) == 1
    assert extract_combined_calls[0]['knowledge'] == knowledge


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
        knowledge=None,
    ):
        extract_combined_calls.append(
            {
                'edge_type_map': edge_type_map,
                'edge_types': edge_types,
                'strict_edge_types': strict_edge_types,
                'knowledge': knowledge,
            }
        )
        return [], [], {}

    monkeypatch.setattr(combined_extraction, 'extract_nodes_and_edges', mock_extract_combined)

    knowledge = 'Extract only professional relationships between people.'
    edge_type_map = {('Entity', 'Entity'): ['knows']}
    edge_types = {'knows': EntityNode}

    await extract_nodes_and_edges_bulk(
        clients,
        [(episode, [])],
        edge_type_map=edge_type_map,
        edge_types=edge_types,
        strict_edge_types=True,
        knowledge=knowledge,
    )

    assert len(extract_combined_calls) == 1
    assert extract_combined_calls[0]['knowledge'] == knowledge
    assert extract_combined_calls[0]['edge_type_map'] == edge_type_map
    assert extract_combined_calls[0]['edge_types'] == edge_types
    assert extract_combined_calls[0]['strict_edge_types'] is True


@pytest.mark.asyncio
async def test_extract_nodes_and_edges_bulk_knowledge_none_by_default(monkeypatch):
    """Test that knowledge defaults to None when not provided."""
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
        knowledge=None,
    ):
        extract_combined_calls.append({'knowledge': knowledge})
        return [], [], {}

    monkeypatch.setattr(combined_extraction, 'extract_nodes_and_edges', mock_extract_combined)

    await extract_nodes_and_edges_bulk(
        clients,
        [(episode, [])],
        edge_type_map={},
    )

    assert len(extract_combined_calls) == 1
    assert extract_combined_calls[0]['knowledge'] is None


@pytest.mark.asyncio
async def test_extract_nodes_and_edges_bulk_knowledge_multiple_episodes(monkeypatch):
    """Test that knowledge is passed for all episodes in bulk."""
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
        knowledge=None,
    ):
        extract_combined_calls.append(
            {
                'episode_name': episode.name,
                'knowledge': knowledge,
            }
        )
        return [], [], {}

    monkeypatch.setattr(combined_extraction, 'extract_nodes_and_edges', mock_extract_combined)

    knowledge = 'Extract entities related to financial transactions.'

    await extract_nodes_and_edges_bulk(
        clients,
        [(episode1, []), (episode2, []), (episode3, [])],
        edge_type_map={},
        knowledge=knowledge,
    )

    assert len(extract_combined_calls) == 3

    for call in extract_combined_calls:
        assert call['knowledge'] == knowledge


@pytest.mark.asyncio
async def test_extract_nodes_and_edges_bulk_passes_knowledge_to_extract_nodes(monkeypatch):
    clients = _make_clients()
    episode = _make_episode('1')
    knowledge = 'Extract person entities.'
    extracted_knowledge = []

    async def mock_extract_nodes(*args, knowledge=None, **kwargs):
        extracted_knowledge.append(knowledge)
        return [], {}

    monkeypatch.setattr(bulk_utils, 'extract_nodes', mock_extract_nodes)
    monkeypatch.setattr(bulk_utils, 'extract_edges', AsyncMock(return_value=([], [])))

    await extract_nodes_and_edges_bulk(
        clients,
        [(episode, [])],
        edge_type_map={},
        knowledge=knowledge,
        use_combined_extraction=False,
    )

    assert extracted_knowledge == [knowledge]


@pytest.mark.asyncio
async def test_extract_nodes_and_edges_bulk_passes_knowledge_to_extract_edges(monkeypatch):
    clients = _make_clients()
    episode = _make_episode('1')
    knowledge = 'Extract only professional relationships.'
    extracted_node = EntityNode(name='Alice', group_id='group', labels=['Entity'])
    extracted_knowledge = []

    async def mock_extract_nodes(*args, **kwargs):
        return [extracted_node], {}

    async def mock_extract_edges(*args, knowledge=None, **kwargs):
        extracted_knowledge.append(knowledge)
        return [], []

    monkeypatch.setattr(bulk_utils, 'extract_nodes', mock_extract_nodes)
    monkeypatch.setattr(bulk_utils, 'extract_edges', mock_extract_edges)

    await extract_nodes_and_edges_bulk(
        clients,
        [(episode, [])],
        edge_type_map={},
        knowledge=knowledge,
        use_combined_extraction=False,
    )

    assert extracted_knowledge == [knowledge]


@pytest.mark.asyncio
async def test_extract_nodes_and_edges_bulk_defaults_to_combined_extraction(monkeypatch):
    clients = _make_clients()
    combined_extractor = AsyncMock(return_value=([], [], {}))
    extract_nodes = AsyncMock(return_value=([], {}))
    extract_edges = AsyncMock(return_value=([], []))
    monkeypatch.setattr(combined_extraction, 'extract_nodes_and_edges', combined_extractor)
    monkeypatch.setattr(bulk_utils, 'extract_nodes', extract_nodes)
    monkeypatch.setattr(bulk_utils, 'extract_edges', extract_edges)

    await extract_nodes_and_edges_bulk(
        clients,
        [(_make_episode('1'), [])],
        edge_type_map={},
    )

    combined_extractor.assert_awaited_once()
    extract_nodes.assert_not_awaited()
    extract_edges.assert_not_awaited()
