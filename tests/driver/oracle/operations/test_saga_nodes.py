# Copyright 2026, Zep Software, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from datetime import UTC, datetime

import pytest

from graphiti_core.driver.oracle.codecs import saga_node_to_binds
from graphiti_core.driver.oracle.operations.saga_nodes import OracleSagaNodeOperations
from graphiti_core.nodes import SagaNode
from tests.driver.oracle.operations.recording import RecordingExecutor


def saga() -> SagaNode:
    return SagaNode(
        uuid='00000000-0000-0000-0000-000000000101',
        name='neutral-sequence',
        group_id='group-a',
        labels=[],
        summary='three temporal events',
        first_episode_uuid='00000000-0000-0000-0000-000000000111',
        last_episode_uuid='00000000-0000-0000-0000-000000000113',
        last_summarized_at=datetime(2026, 9, 21, 13, tzinfo=UTC),
        last_summarized_episode_uuid='00000000-0000-0000-0000-000000000112',
        last_summarized_episode_valid_at=datetime(2026, 9, 20, 18, tzinfo=UTC),
        created_at=datetime(2026, 9, 20, 12, tzinfo=UTC),
    )


@pytest.mark.asyncio
async def test_saga_round_trip_preserves_composite_and_temporal_watermarks() -> None:
    node = saga()
    executor = RecordingExecutor([saga_node_to_binds(node)])
    operations = OracleSagaNodeOperations()

    await operations.save(executor, node)
    restored = await operations.get_by_uuid(executor, node.uuid)

    assert restored.model_dump() == node.model_dump()


@pytest.mark.asyncio
async def test_saga_previous_episode_and_incremental_contents_use_distinct_times() -> None:
    valid_at = datetime(2026, 9, 20, 18, tzinfo=UTC)
    created_at = datetime(2026, 9, 21, 13, tzinfo=UTC)
    executor = RecordingExecutor([{'uuid': 'episode-previous'}])
    operations = OracleSagaNodeOperations()

    assert (
        await operations.get_previous_episode_uuid(executor, saga().uuid, 'episode-current')
        == 'episode-previous'
    )

    executor.records = [
        {
            'uuid': 'episode-previous',
            'content': 'event',
            'valid_at': valid_at,
            'created_at': created_at,
        }
    ]
    result = await operations.get_episode_contents(
        executor,
        saga().uuid,
        since=datetime(2026, 9, 21, 12, tzinfo=UTC),
        after_uuid='episode-earlier',
        limit=25,
    )

    assert result == [('event', valid_at, created_at, 'episode-previous')]
    assert executor.calls[-1][1]['since'] == datetime(2026, 9, 21, 12, tzinfo=UTC)
    assert executor.calls[-1][1]['after_uuid'] == 'episode-earlier'
    assert 'e.created_at IS NOT NULL' in executor.calls[-1][0]
    assert 'e.created_at > :since' in executor.calls[-1][0]
    assert 'e.created_at = :since' in executor.calls[-1][0]
    assert 'e.uuid > :after_uuid' in executor.calls[-1][0]
    assert 'ORDER BY e.created_at ASC, e.uuid ASC' in executor.calls[-1][0]


@pytest.mark.asyncio
async def test_saga_backfill_returns_ingestion_order_not_event_time_order() -> None:
    first_created = datetime(2026, 9, 21, 10, tzinfo=UTC)
    second_created = datetime(2026, 9, 21, 11, tzinfo=UTC)
    executor = RecordingExecutor(
        [
            {
                'uuid': 'episode-a',
                'content': 'late backfill',
                'valid_at': datetime(2025, 1, 1, tzinfo=UTC),
                'created_at': first_created,
            },
            {
                'uuid': 'episode-b',
                'content': 'current event',
                'valid_at': datetime(2026, 9, 21, tzinfo=UTC),
                'created_at': second_created,
            },
        ]
    )

    result = await OracleSagaNodeOperations().get_episode_contents(
        executor,
        saga().uuid,
        since=datetime(2026, 9, 21, 9, tzinfo=UTC),
    )

    assert [content for content, _, _, _ in result] == ['late backfill', 'current event']
    assert [created_at for _, _, created_at, _ in result] == [first_created, second_created]


@pytest.mark.asyncio
async def test_tied_saga_timestamps_advance_by_uuid_without_skipping() -> None:
    tied = datetime(2026, 9, 21, 10, tzinfo=UTC)
    executor = RecordingExecutor(
        [{'uuid': 'episode-b', 'content': 'second', 'valid_at': tied, 'created_at': tied}]
    )

    result = await OracleSagaNodeOperations().get_episode_contents(
        executor,
        saga().uuid,
        since=tied,
        after_uuid='episode-a',
        limit=1,
    )

    assert result == [('second', tied, tied, 'episode-b')]
    assert executor.calls[0][1]['after_uuid'] == 'episode-a'


@pytest.mark.asyncio
async def test_legacy_timestamp_cursor_replays_equal_timestamp_frontier() -> None:
    tied = datetime(2026, 9, 21, 10, tzinfo=UTC)
    executor = RecordingExecutor(
        [{'uuid': 'episode-a', 'content': 'replayed', 'valid_at': tied, 'created_at': tied}]
    )

    result = await OracleSagaNodeOperations().get_episode_contents(
        executor,
        saga().uuid,
        since=tied,
        after_uuid=None,
        limit=1,
    )

    assert result == [('replayed', tied, tied, 'episode-a')]
    statement, binds = executor.calls[0]
    assert binds['since'] == tied
    assert binds['after_uuid'] is None
    assert 'e.created_at = :since' in statement
    assert ':after_uuid IS NULL OR e.uuid > :after_uuid' in statement
