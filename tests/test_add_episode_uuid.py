from datetime import datetime, timezone

from graphiti_core.errors import NodeNotFoundError
from graphiti_core.graphiti import _get_or_create_episode
from graphiti_core.nodes import EpisodeType, EpisodicNode


def _episode_kwargs() -> dict:
    now = datetime.now(timezone.utc)
    return {
        'name': 'e1',
        'group_id': 'g1',
        'source': EpisodeType.text,
        'content': 'text',
        'source_description': 'test',
        'created_at': now,
        'valid_at': now,
    }


async def test_supplied_uuid_is_created_when_it_does_not_exist(monkeypatch):
    async def missing(cls, driver, uuid):
        raise NodeNotFoundError(uuid)

    monkeypatch.setattr(EpisodicNode, 'get_by_uuid', classmethod(missing))

    episode = await _get_or_create_episode(object(), 'client-uuid-1', **_episode_kwargs())

    assert episode.uuid == 'client-uuid-1'
    assert episode.name == 'e1'
    assert episode.group_id == 'g1'
    assert episode.content == 'text'


async def test_supplied_uuid_reuses_the_stored_episode(monkeypatch):
    stored = EpisodicNode(uuid='client-uuid-2', **_episode_kwargs())

    async def found(cls, driver, uuid):
        return stored

    monkeypatch.setattr(EpisodicNode, 'get_by_uuid', classmethod(found))

    episode = await _get_or_create_episode(object(), 'client-uuid-2', **_episode_kwargs())

    assert episode is stored


async def test_missing_uuid_still_generates_one(monkeypatch):
    async def unexpected(cls, driver, uuid):
        raise AssertionError('get_by_uuid must not be called without a uuid')

    monkeypatch.setattr(EpisodicNode, 'get_by_uuid', classmethod(unexpected))

    episode = await _get_or_create_episode(object(), None, **_episode_kwargs())

    assert episode.uuid
    assert episode.name == 'e1'
