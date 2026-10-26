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

from graphiti_core.driver.oracle.codecs import entity_node_to_binds
from graphiti_core.driver.oracle.errors import InvalidSearchFilter
from graphiti_core.driver.oracle.operations.entity_nodes import OracleEntityNodeOperations
from graphiti_core.nodes import EntityNode
from tests.driver.oracle.operations.recording import RecordingExecutor


def identity_entity() -> EntityNode:
    return EntityNode(
        uuid='00000000-0000-0000-0000-000000000901',
        name='ONT A-42',
        group_id='tenant-a',
        labels=['NetworkAsset'],
        summary='Synthetic identity lookup fixture.',
        attributes={'serial_number': 'SN-42', 'slot': 7},
        created_at=datetime(2026, 10, 16, tzinfo=UTC),
    )


@pytest.mark.asyncio
async def test_exact_name_lookup_binds_names_and_groups() -> None:
    node = identity_entity()
    executor = RecordingExecutor([entity_node_to_binds(node)])

    restored = await OracleEntityNodeOperations().get_by_names(
        executor,
        ['', node.name, node.name],
        ['tenant-b', node.group_id, node.group_id],
    )

    assert [candidate.uuid for candidate in restored] == [node.uuid]
    statement, binds = executor.calls[-1]
    assert 'JSON_TABLE' in statement
    assert 'ONT A-42' not in statement
    assert binds['names_json'] == '["ONT A-42"]'
    assert binds['group_ids_json'] == '["tenant-a","tenant-b"]'


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('names', 'groups'),
    [([], ['tenant-a']), (['ONT A-42'], []), ([''], ['tenant-a']), (['ONT A-42'], [''])],
)
async def test_exact_name_lookup_empty_inputs_match_nothing(
    names: list[str], groups: list[str]
) -> None:
    executor = RecordingExecutor()

    assert await OracleEntityNodeOperations().get_by_names(executor, names, groups) == []
    assert executor.calls == []


@pytest.mark.asyncio
async def test_identity_attribute_lookup_matches_all_bound_values_and_label() -> None:
    node = identity_entity()
    executor = RecordingExecutor([entity_node_to_binds(node)])

    restored = await OracleEntityNodeOperations().get_by_attribute_values(
        executor,
        node.group_id,
        {'serial_number': 'SN-42', 'slot': 7},
        ['NetworkAsset'],
        limit=5,
    )

    assert [candidate.uuid for candidate in restored] == [node.uuid]
    statement, binds = executor.calls[-1]
    assert 'SN-42' not in statement
    assert 'NetworkAsset' not in statement
    assert 'serial_number' in statement
    assert 'slot' in statement
    assert binds['property_0'] == 'SN-42'
    assert binds['property_1'] == 7
    assert binds['group_ids_json'] == '["tenant-a"]'
    assert binds['node_labels_json'] == '["NetworkAsset"]'
    assert binds['result_limit'] == 5


@pytest.mark.asyncio
async def test_identity_attribute_lookup_rejects_untrusted_property_name() -> None:
    executor = RecordingExecutor()

    with pytest.raises(InvalidSearchFilter, match='property_name'):
        await OracleEntityNodeOperations().get_by_attribute_values(
            executor,
            'tenant-a',
            {"serial') OR 1=1 --": 'SN-42'},
        )

    assert executor.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('attributes', 'labels', 'limit'),
    [({}, None, 10), ({'serial_number': 'SN-42'}, [], 10), ({'serial_number': 'SN-42'}, None, 0)],
)
async def test_identity_attribute_lookup_empty_constraints_match_nothing(
    attributes: dict[str, object], labels: list[str] | None, limit: int
) -> None:
    executor = RecordingExecutor()

    assert (
        await OracleEntityNodeOperations().get_by_attribute_values(
            executor,
            'tenant-a',
            attributes,
            labels,
            limit=limit,
        )
        == []
    )
    assert executor.calls == []
