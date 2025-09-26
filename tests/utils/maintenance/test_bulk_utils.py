from collections import deque
from unittest.mock import AsyncMock, MagicMock

import pytest

from graphiti_core.edges import EntityEdge
from graphiti_core.graphiti_types import GraphitiClients
from graphiti_core.nodes import EntityNode, EpisodeType, EpisodicNode
from graphiti_core.utils import bulk_utils
from graphiti_core.utils.datetime_utils import utc_now


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
        ensure_ascii=False,
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

    bulk_utils.resolve_edge_pointers([edge], {'alias': 'canonical'})

    assert edge.source_node_uuid == 'canonical'
    assert edge.target_node_uuid == 'target'
