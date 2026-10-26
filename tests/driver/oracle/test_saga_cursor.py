"""
Copyright 2026, Zep Software, Inc.

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

from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

import graphiti_core.graphiti as graphiti_module
from graphiti_core.graphiti import Graphiti
from graphiti_core.nodes import SagaNode


@pytest.mark.asyncio
async def test_tied_episode_timestamps_are_summarized_exactly_once(monkeypatch) -> None:
    tied = datetime(2026, 10, 16, 12, tzinfo=UTC)
    saga = SagaNode(
        uuid='saga-a',
        name='tie-safe',
        group_id='group-a',
        created_at=tied,
    )
    cursor_calls: list[tuple[datetime | None, str | None]] = []
    saved_cursors: list[tuple[datetime | None, str | None]] = []

    async def get_saga(_cls, _driver, _uuid):
        return saga

    async def save_saga(self, _driver):
        saved_cursors.append((self.last_summarized_at, self.last_summarized_episode_uuid))

    async def page(_saga_uuid, since=None, after_uuid=None, limit=200):
        del limit
        cursor_calls.append((since, after_uuid))
        if since is None:
            return [('first', tied, tied, 'episode-a')]
        if (since, after_uuid) == (tied, 'episode-a'):
            return [('second', tied, tied, 'episode-b')]
        return []

    async def summarize(*_args, **_kwargs):
        return {'summary': 'updated'}

    monkeypatch.setattr(SagaNode, 'get_by_uuid', classmethod(get_saga))
    monkeypatch.setattr(SagaNode, 'save', save_saga)
    monkeypatch.setattr(
        'graphiti_core.graphiti.resolve_prompt_messages',
        lambda *_args, **_kwargs: [SimpleNamespace(content='small')],
    )
    monkeypatch.setattr('graphiti_core.graphiti.generate_prompt_response', summarize)

    graphiti = object.__new__(Graphiti)
    graphiti.driver = object()
    graphiti.llm_client = object()
    graphiti._saga_get_episode_contents = page

    await graphiti.summarize_saga(saga.uuid)
    await graphiti.summarize_saga(saga.uuid)
    await graphiti.summarize_saga(saga.uuid)

    assert cursor_calls == [
        (None, None),
        (tied, 'episode-a'),
        (tied, 'episode-b'),
    ]
    assert saved_cursors == [(tied, 'episode-a'), (tied, 'episode-b')]


@pytest.mark.asyncio
async def test_legacy_cursor_replays_frontier_then_persists_real_uuid(monkeypatch) -> None:
    tied = datetime(2026, 10, 16, 12, tzinfo=UTC)
    saga = SagaNode(
        uuid='saga-legacy',
        name='legacy-tie-safe',
        group_id='group-a',
        created_at=tied,
        last_summarized_at=tied,
        last_summarized_episode_uuid=None,
        summary='existing',
    )
    cursor_calls: list[tuple[datetime | None, str | None]] = []
    saved_cursors: list[tuple[datetime | None, str | None]] = []

    async def get_saga(_cls, _driver, _uuid):
        return saga

    async def save_saga(self, _driver):
        saved_cursors.append((self.last_summarized_at, self.last_summarized_episode_uuid))

    async def page(_saga_uuid, since=None, after_uuid=None, limit=200):
        del limit
        cursor_calls.append((since, after_uuid))
        if (since, after_uuid) == (tied, None):
            return [
                ('first', tied, tied, 'episode-a'),
                ('second', tied, tied, 'episode-b'),
            ]
        if (since, after_uuid) == (tied, 'episode-a'):
            return [('second', tied, tied, 'episode-b')]
        return []

    async def summarize(*_args, **_kwargs):
        return {'summary': 'updated'}

    original_selector = graphiti_module.select_saga_summary_episodes

    def select_one(episodes_data, max_episode_chars=100_000):
        del max_episode_chars
        return original_selector(episodes_data, max_episode_chars=5)

    monkeypatch.setattr(SagaNode, 'get_by_uuid', classmethod(get_saga))
    monkeypatch.setattr(SagaNode, 'save', save_saga)
    monkeypatch.setattr(graphiti_module, 'select_saga_summary_episodes', select_one)
    monkeypatch.setattr(
        graphiti_module,
        'resolve_prompt_messages',
        lambda *_args, **_kwargs: [SimpleNamespace(content='small')],
    )
    monkeypatch.setattr(graphiti_module, 'generate_prompt_response', summarize)

    graphiti = object.__new__(Graphiti)
    graphiti.driver = object()
    graphiti.llm_client = object()
    graphiti._saga_get_episode_contents = page

    await graphiti.summarize_saga(saga.uuid)
    await graphiti.summarize_saga(saga.uuid)
    await graphiti.summarize_saga(saga.uuid)

    assert cursor_calls == [
        (tied, None),
        (tied, 'episode-a'),
        (tied, 'episode-b'),
    ]
    assert saved_cursors == [(tied, 'episode-a'), (tied, 'episode-b')]
