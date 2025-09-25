from collections import deque
from unittest.mock import MagicMock

import pytest

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
        assert existing_nodes_override is not None
        assert existing_nodes_override[0] is canonical

        return [canonical], {extracted_two.uuid: canonical.uuid}, [(extracted_two, canonical)]

    monkeypatch.setattr(bulk_utils, 'resolve_extracted_nodes', fake_resolve)

    nodes_by_episode, compressed_map = await bulk_utils.dedupe_nodes_bulk(
        clients,
        [[extracted_one], [extracted_two]],
        [(episode_one, []), (episode_two, [])],
    )

    assert len(call_queue) == 2
    assert call_queue[0] is None
    assert list(call_queue[1]) == [canonical]

    assert nodes_by_episode[episode_one.uuid] == [canonical]
    assert nodes_by_episode[episode_two.uuid] == [canonical]
    assert compressed_map.get(extracted_two.uuid) == canonical.uuid
