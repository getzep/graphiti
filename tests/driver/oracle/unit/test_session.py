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

from decimal import Decimal

import oracledb
import pytest

import graphiti_core.driver.oracle.session as session_module
from graphiti_core.driver.oracle.errors import OracleQueryContractError
from graphiti_core.driver.oracle.session import (
    Oracle26aiSession,
    execute_on_connection,
    validate_oracle_statement,
)
from tests.driver.oracle.unit.fakes import FakeConnection, FakeCursor, FakeLOB, FakePool

PAYLOAD_BIND_NAMES = {
    'summary',
    'content',
    'fact',
    'search_text',
    'labels',
    'properties',
    'entity_edge_uuids',
    'episode_uuids',
}


class FailingCursor(FakeCursor):
    async def execute(self, statement: str, parameters: dict[str, object]) -> None:
        await super().execute(statement, parameters)
        raise RuntimeError('statement failed')


@pytest.mark.parametrize(
    'statement',
    [
        'MATCH (n) RETURN n',
        'OPTIONAL MATCH (n) RETURN n',
        'CALL db.indexes()',
        'SELECT * FROM graphiti_nodes WHERE uuid = $uuid',
    ],
)
def test_session_rejects_cypher_and_dollar_binds(statement: str) -> None:
    with pytest.raises(OracleQueryContractError):
        validate_oracle_statement(statement)


def test_session_accepts_graph_table_match_and_json_path() -> None:
    validate_oracle_statement(
        'SELECT reached FROM GRAPH_TABLE (GRAPHITI_GRAPH '
        'MATCH (a)-[e]->(b) COLUMNS (b.uuid AS reached))'
    )
    validate_oracle_statement(
        "SELECT value FROM JSON_TABLE(:items, '$[*]' COLUMNS (value PATH '$'))"
    )


@pytest.mark.asyncio
async def test_read_maps_column_names_and_releases_connection() -> None:
    cursor = FakeCursor(rows=[('node-1', 'Alpha')], columns=('UUID', 'NAME'))
    pool = FakePool(FakeConnection(cursor))
    session = Oracle26aiSession(pool)

    records, columns, summary = await session.run(
        'select uuid, name from graphiti_nodes where uuid = :uuid',
        uuid='node-1',
    )

    assert records == [{'uuid': 'node-1', 'name': 'Alpha'}]
    assert columns == ['uuid', 'name']
    assert summary is None
    assert cursor.executions[0][1] == {'uuid': 'node-1'}
    assert pool.released == 1


@pytest.mark.asyncio
async def test_non_transactional_write_commits_once() -> None:
    connection = FakeConnection(FakeCursor())
    pool = FakePool(connection)
    session = Oracle26aiSession(pool)

    assert await session.run('delete from graphiti_edges where uuid = :uuid', uuid='edge-1') == (
        [],
        [],
        None,
    )
    assert connection.commits == 1
    assert pool.released == 1


@pytest.mark.parametrize(
    'payload',
    [
        pytest.param('a' * 3999, id='ascii-3999-bytes'),
        pytest.param('a' * 4000, id='ascii-4000-bytes'),
        pytest.param('a' * 4001, id='ascii-4001-bytes'),
        pytest.param('a' * 4095, id='ascii-4095-bytes'),
        pytest.param('a' * 4096, id='ascii-4096-bytes'),
        pytest.param('a' * 16383, id='ascii-16383-bytes'),
        pytest.param('a' * 16384, id='ascii-16384-bytes'),
        pytest.param('á' * 8192, id='utf8-16384-bytes'),
        pytest.param('a' * 32767, id='ascii-32767-bytes'),
        pytest.param('a' * 32768, id='ascii-32768-bytes'),
        pytest.param('á' * 16384, id='utf8-32768-bytes'),
    ],
)
@pytest.mark.asyncio
async def test_large_payload_binds_are_declared_as_clob_before_execute(payload: str) -> None:
    cursor = FakeCursor()
    connection = FakeConnection(cursor)
    binds: dict[str, object] = {
        'uuid': 'node-1',
        'summary': payload,
        'content': payload,
        'fact': payload,
        'search_text': payload,
        'labels': payload,
        'properties': payload,
        'entity_edge_uuids': payload,
        'episode_uuids': payload,
        'text_query': payload,
    }
    original_binds = dict(binds)

    await execute_on_connection(
        connection,
        'INSERT INTO graphiti_nodes (uuid) VALUES (:uuid)',
        binds,
        commit_write=False,
    )

    byte_length = len(payload.encode('utf-8'))
    if byte_length <= 4000:
        expected_sizes = {name: oracledb.DB_TYPE_CLOB for name in PAYLOAD_BIND_NAMES}
        assert connection.lob_creations == []
        assert cursor.input_size_calls == [expected_sizes]
        assert cursor.events == [
            ('setinputsizes', expected_sizes),
            ('execute', binds),
        ]
        assert cursor.executions[0][1] == binds
    else:
        assert cursor.input_size_calls == []
        assert len(connection.lob_creations) == len(PAYLOAD_BIND_NAMES)
        assert all(lob.lob_type is oracledb.DB_TYPE_CLOB for lob in connection.lob_creations)
        assert all(lob.data == payload for lob in connection.lob_creations)
        executed_binds = cursor.executions[0][1]
        assert all(isinstance(executed_binds[name], FakeLOB) for name in PAYLOAD_BIND_NAMES)
        assert executed_binds['uuid'] == 'node-1'
        assert executed_binds['text_query'] == payload
        assert cursor.events == [('execute', executed_binds)]
    assert binds == original_binds
    assert byte_length in {3999, 4000, 4001, 4095, 4096, 16383, 16384, 32767, 32768}


@pytest.mark.asyncio
async def test_mixed_payload_binds_use_clob_type_or_locator_before_execute() -> None:
    cursor = FakeCursor()
    connection = FakeConnection(cursor)
    binds: dict[str, object] = {
        'summary': 'small summary',
        'content': 'á' * 16384,
        'text_query': 'not a persisted payload',
    }

    await execute_on_connection(
        connection,
        'INSERT INTO graphiti_nodes (uuid) VALUES (:uuid)',
        binds,
        commit_write=False,
    )

    assert cursor.input_size_calls == [{'summary': oracledb.DB_TYPE_CLOB}]
    assert [(lob.lob_type, lob.data) for lob in connection.lob_creations] == [
        (oracledb.DB_TYPE_CLOB, binds['content'])
    ]
    executed_binds = cursor.executions[0][1]
    assert executed_binds['summary'] == binds['summary']
    assert isinstance(executed_binds['content'], FakeLOB)
    assert executed_binds['text_query'] == binds['text_query']
    assert cursor.events == [
        ('setinputsizes', {'summary': oracledb.DB_TYPE_CLOB}),
        ('execute', executed_binds),
    ]


@pytest.mark.asyncio
async def test_none_and_non_payload_binds_keep_driver_inference() -> None:
    cursor = FakeCursor()
    session = Oracle26aiSession(FakePool(FakeConnection(cursor)))

    await session.run(
        'INSERT INTO graphiti_edges (uuid) VALUES (:uuid)',
        uuid='edge-1',
        fact=None,
        text_query='not a persisted payload',
    )

    assert cursor.input_size_calls == []


@pytest.mark.asyncio
async def test_failed_session_statement_rolls_back_before_pool_release() -> None:
    connection = FakeConnection(FailingCursor())
    pool = FakePool(connection)
    session = Oracle26aiSession(pool)

    with pytest.raises(RuntimeError, match='statement failed'):
        await session.run('delete from graphiti_edges where uuid = :uuid', uuid='edge-1')

    assert connection.commits == 0
    assert connection.rollbacks == 1
    assert pool.acquired == pool.released == 1


@pytest.mark.asyncio
async def test_execute_write_commits_all_bulk_statements_as_one_transaction() -> None:
    connection = FakeConnection(FakeCursor())
    pool = FakePool(connection)

    async def bulk(tx):
        await tx.run('INSERT INTO graphiti_nodes (uuid) VALUES (:uuid)', uuid='episode-1')
        await tx.run('INSERT INTO graphiti_edges (uuid) VALUES (:uuid)', uuid='fact-1')

    await Oracle26aiSession(pool).execute_write(bulk)

    assert connection.commits == 1
    assert connection.rollbacks == 0
    assert pool.acquired == pool.released == 1


@pytest.mark.asyncio
async def test_execute_write_rolls_back_episode_when_a_later_bulk_statement_fails() -> None:
    connection = FakeConnection(FakeCursor())
    pool = FakePool(connection)

    async def bulk(tx):
        await tx.run('INSERT INTO graphiti_nodes (uuid) VALUES (:uuid)', uuid='episode-1')
        raise RuntimeError('later edge failed')

    with pytest.raises(RuntimeError, match='later edge failed'):
        await Oracle26aiSession(pool).execute_write(bulk)

    assert connection.commits == 0
    assert connection.rollbacks == 1
    assert pool.acquired == pool.released == 1


@pytest.mark.asyncio
async def test_read_materializes_async_lobs_and_normalizes_json_decimals(monkeypatch) -> None:
    class FakeAsyncLOB:
        def __init__(self, value: str) -> None:
            self.value = value

        async def read(self) -> str:
            return self.value

    monkeypatch.setattr(session_module.oracledb, 'AsyncLOB', FakeAsyncLOB)
    cursor = FakeCursor(
        rows=[(FakeAsyncLOB('Alpha summary'), {'rank': Decimal('2'), 'ratio': Decimal('1.5')})],
        columns=('SUMMARY', 'PROPERTIES'),
    )
    session = Oracle26aiSession(FakePool(FakeConnection(cursor)))

    records, _, _ = await session.run('select summary, properties from graphiti_nodes')

    assert records == [{'summary': 'Alpha summary', 'properties': {'rank': 2, 'ratio': 1.5}}]
