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

from array import array
from datetime import UTC, datetime

import pytest

from graphiti_core.driver.oracle.codecs import (
    decode_oracle_datetime,
    ensure_utc_datetime,
    ensure_vector,
    entity_node_from_record,
    entity_node_to_binds,
    episode_node_from_record,
    episode_node_to_binds,
    vector_to_bind,
)
from graphiti_core.driver.oracle.errors import InvalidEmbeddingDimension, NaiveDatetimeError
from graphiti_core.nodes import EntityNode, EpisodeType, EpisodicNode


def test_entity_codec_round_trip_preserves_labels_attributes_and_vector() -> None:
    node = EntityNode(
        uuid='00000000-0000-0000-0000-000000000001',
        name='Alpha',
        group_id='group-a',
        labels=['Asset', 'Site'],
        summary='neutral fixture',
        attributes={'rank': 2, 'active': True},
        name_embedding=[0.0] * 1024,
        created_at=datetime(2026, 9, 21, tzinfo=UTC),
    )

    restored = entity_node_from_record(entity_node_to_binds(node))

    assert restored.model_dump() == node.model_dump()


def test_episode_codec_round_trip_preserves_temporal_and_metadata_fields() -> None:
    episode = EpisodicNode(
        uuid='00000000-0000-0000-0000-000000000002',
        name='Neutral event',
        group_id='group-a',
        labels=[],
        source=EpisodeType.json,
        source_description='synthetic fixture',
        content='{"event":"created"}',
        entity_edges=['edge-1'],
        episode_metadata={'synthetic': True},
        created_at=datetime(2026, 9, 21, 12, tzinfo=UTC),
        valid_at=datetime(2026, 9, 20, 18, tzinfo=UTC),
    )

    restored = episode_node_from_record(episode_node_to_binds(episode))

    assert restored.model_dump() == episode.model_dump()


def test_vector_dimension_is_rejected_before_database_access() -> None:
    with pytest.raises(InvalidEmbeddingDimension, match='expected 1024, received 3'):
        ensure_vector([0.0, 1.0, 0.0])


def test_vector_codec_returns_the_float32_bind_type_required_by_oracledb() -> None:
    vector = vector_to_bind([0.0] * 1024)

    assert isinstance(vector, array)
    assert vector.typecode == 'f'


def test_naive_datetime_is_rejected() -> None:
    with pytest.raises(NaiveDatetimeError, match='timezone-aware'):
        ensure_utc_datetime(datetime(2026, 9, 21, 12))


def test_oracle_timestamp_decode_normalizes_naive_driver_value_to_utc() -> None:
    decoded = decode_oracle_datetime(datetime(2026, 9, 21, 12))

    assert decoded == datetime(2026, 9, 21, 12, tzinfo=UTC)
    assert decoded.tzinfo is UTC
