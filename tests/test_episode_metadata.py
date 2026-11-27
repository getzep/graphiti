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

from datetime import datetime, timezone
from typing import Any

import pytest
from pydantic import ValidationError

from graphiti_core.nodes import EpisodeType, EpisodicNode, episode_metadata_from_properties


def make_episode(episode_metadata: Any) -> EpisodicNode:
    return EpisodicNode(
        name='episode',
        group_id='group',
        source=EpisodeType.text,
        source_description='test',
        content='content',
        valid_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        episode_metadata=episode_metadata,
    )


def test_episode_metadata_is_stored_as_node_properties():
    episode = make_episode({'agent': 'planner', 'step': 3, 'score': 0.5, 'approved': True})

    properties = episode.db_properties()

    assert properties['agent'] == 'planner'
    assert properties['step'] == 3
    assert properties['score'] == 0.5
    assert properties['approved'] is True
    assert properties['uuid'] == episode.uuid


@pytest.mark.parametrize('key', ['uuid', 'name', 'content', 'entity_edges', 'valid_at'])
def test_episode_metadata_rejects_reserved_keys(key):
    with pytest.raises(ValidationError, match='reserved episode properties'):
        make_episode({key: 'value'})


@pytest.mark.parametrize('key', ['source-doc', 'a b', 'x:y', 'z}', "q'uote", '1st', ''])
def test_episode_metadata_rejects_keys_that_are_not_identifiers(key):
    with pytest.raises(ValidationError, match='letters, digits, and underscores'):
        make_episode({key: 'value'})


@pytest.mark.parametrize('value', [{'nested': 'value'}, ['a', 'b'], None])
def test_episode_metadata_rejects_values_a_node_property_cannot_hold(value):
    with pytest.raises(ValidationError):
        make_episode({'key': value})


def test_episode_metadata_assignment_is_validated():
    episode = make_episode(None)

    with pytest.raises(ValidationError, match='reserved episode properties'):
        episode.episode_metadata = {'name': 'other'}


def test_empty_episode_metadata_is_none():
    assert make_episode({}).episode_metadata is None


def test_episode_metadata_read_skips_episode_fields_and_unsupported_values():
    properties = {
        'uuid': 'episode-1',
        'content': 'content',
        'entity_edges': ['edge-1'],
        'agent': 'planner',
        'tags': ['set', 'outside', 'graphiti'],
        'source-doc': 'set outside graphiti',
        'reviewed_at': datetime(2026, 1, 1, tzinfo=timezone.utc),
    }

    assert episode_metadata_from_properties(properties) == {'agent': 'planner'}
    assert episode_metadata_from_properties({'uuid': 'episode-1'}) is None
    assert episode_metadata_from_properties(None) is None
