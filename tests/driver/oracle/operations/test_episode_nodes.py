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

from graphiti_core.driver.oracle.codecs import episode_node_to_binds
from graphiti_core.driver.oracle.operations.episode_nodes import OracleEpisodeNodeOperations
from graphiti_core.nodes import EpisodeType, EpisodicNode
from tests.driver.oracle.operations.recording import RecordingExecutor


def episode(uuid: str, valid_at: datetime) -> EpisodicNode:
    return EpisodicNode(
        uuid=uuid,
        name=f'Episode {uuid}',
        group_id='group-a',
        labels=[],
        source=EpisodeType.json,
        source_description='synthetic',
        content='{"event":"updated"}',
        entity_edges=[],
        created_at=datetime(2026, 9, 21, tzinfo=UTC),
        valid_at=valid_at,
    )


@pytest.mark.asyncio
async def test_retrieve_episodes_returns_codec_records_in_chronological_order() -> None:
    older = episode('episode-1', datetime(2026, 9, 20, tzinfo=UTC))
    newer = episode('episode-2', datetime(2026, 9, 21, tzinfo=UTC))
    executor = RecordingExecutor([episode_node_to_binds(older), episode_node_to_binds(newer)])

    result = await OracleEpisodeNodeOperations().retrieve_episodes(
        executor,
        reference_time=datetime(2026, 9, 21, 23, tzinfo=UTC),
        last_n=2,
        group_ids=['group-a'],
        source='json',
    )

    assert [item.uuid for item in result] == ['episode-1', 'episode-2']
    assert executor.calls[0][1]['group_ids_json'] == '["group-a"]'
    assert executor.calls[0][1]['source'] == 'json'


@pytest.mark.asyncio
async def test_episode_save_derives_search_text_without_changing_content() -> None:
    item = episode('episode-3', datetime(2026, 9, 21, tzinfo=UTC))
    executor = RecordingExecutor()

    await OracleEpisodeNodeOperations().save(executor, item)

    binds = executor.calls[1][1]
    assert binds['content'] == item.content
    assert binds['search_text'] == f'{item.name}\n{item.content}'
