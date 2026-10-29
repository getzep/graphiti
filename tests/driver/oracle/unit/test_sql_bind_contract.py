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

import re

import pytest

from graphiti_core.driver.oracle.sql.edges import MERGE_EDGE
from graphiti_core.driver.oracle.sql.nodes import (
    GET_EPISODES_BY_ENTITY,
    MERGE_COMMUNITY_NODE,
    MERGE_ENTITY_NODE,
    MERGE_EPISODE_NODE,
    MERGE_SAGA_NODE,
)


@pytest.mark.parametrize(
    'statement',
    (
        MERGE_ENTITY_NODE,
        MERGE_EPISODE_NODE,
        MERGE_COMMUNITY_NODE,
        MERGE_SAGA_NODE,
    ),
)
def test_node_merge_consumes_the_kind_emitted_by_its_codec(statement: str) -> None:
    bind_names = set(re.findall(r':([A-Za-z_][A-Za-z0-9_]*)', statement))

    assert 'kind' in bind_names


def test_episode_entity_lookup_deduplicates_without_comparing_clobs() -> None:
    assert 'SELECT DISTINCT n.*' not in GET_EPISODES_BY_ENTITY
    assert 'EXISTS (' in GET_EPISODES_BY_ENTITY


@pytest.mark.parametrize(
    ('statement', 'alias'),
    [
        (MERGE_ENTITY_NODE, 'n'),
        (MERGE_EPISODE_NODE, 'n'),
        (MERGE_COMMUNITY_NODE, 'n'),
        (MERGE_SAGA_NODE, 'n'),
        (MERGE_EDGE, 'e'),
    ],
)
def test_merge_identity_includes_group_and_cannot_reassign_it(statement: str, alias: str) -> None:
    on_clause, update_clause = statement.split('WHEN MATCHED THEN UPDATE SET', maxsplit=1)

    assert f'{alias}.group_id = :group_id' in on_clause
    assert (
        f'{alias}.group_id = :group_id'
        not in update_clause.split('WHEN NOT MATCHED', maxsplit=1)[0]
    )
