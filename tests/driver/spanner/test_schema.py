"""
Copyright 2024, Zep Software, Inc.

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

# Hermetic tests for the Spanner DDL generator.

import pytest

from graphiti_core.driver.spanner.schema import (
    EDGE_TABLES,
    INDEX_NAMES,
    KIND_INDEX,
    KIND_PROPERTY_GRAPH,
    KIND_SEARCH_INDEX,
    KIND_TABLE,
    NODE_TABLES,
    SEARCH_INDEX_NAMES,
    build_ddl,
    build_drop_ddl,
    ddl_object_name,
    edge_select_columns,
    node_select_columns,
    resolve_table,
)

EDGE_SPECS = {
    'RelatesTo': ('RELATES_TO', 'Entity', 'Entity'),
    'Mentions': ('MENTIONS', 'Episodic', 'Entity'),
    'HasMember': ('HAS_MEMBER', 'Community', 'Entity'),
    'HasEpisode': ('HAS_EPISODE', 'Saga', 'Episodic'),
    'NextEpisode': ('NEXT_EPISODE', 'Episodic', 'Episodic'),
}


def _create_table(statements: list[str], name: str) -> str:
    matches = [s for s in statements if ddl_object_name(s) == (KIND_TABLE, name)]
    assert len(matches) == 1, name
    return matches[0]


def _property_graph(statements: list[str]) -> str:
    matches = [s for s in statements if ddl_object_name(s)[0] == KIND_PROPERTY_GRAPH]
    assert len(matches) == 1
    return matches[0]


def test_tables_match_spec():
    assert list(NODE_TABLES) == ['Entity', 'Episodic', 'Community', 'Saga']
    assert list(EDGE_TABLES) == list(EDGE_SPECS)
    for name, (label, source, target) in EDGE_SPECS.items():
        table = EDGE_TABLES[name]
        assert (table.label, table.source_table, table.target_table) == (label, source, target)
        assert table.primary_key == ('uuid', 'edge_uuid')
        assert table.interleave_in_parent == source
        assert table.writable_columns[:4] == ('uuid', 'edge_uuid', 'target_node_uuid', 'group_id')
        assert 'created_at' in table.writable_columns
    assert EDGE_TABLES['RelatesTo'].writable_columns == (
        'uuid',
        'edge_uuid',
        'target_node_uuid',
        'group_id',
        'name',
        'fact',
        'fact_embedding',
        'episodes',
        'attributes',
        'created_at',
        'valid_at',
        'invalid_at',
        'expired_at',
        'reference_time',
    )
    assert NODE_TABLES['Entity'].writable_columns == (
        'uuid',
        'group_id',
        'name',
        'labels',
        'summary',
        'attributes',
        'name_embedding',
        'created_at',
    )


def test_edge_table_ddl_has_composite_key_and_parent():
    statements = build_ddl(1024, 'graphiti')
    for name, (_, source, _) in EDGE_SPECS.items():
        ddl = _create_table(statements, name)
        assert 'PRIMARY KEY (uuid, edge_uuid)' in ddl
        assert f'INTERLEAVE IN PARENT {source} ON DELETE CASCADE' in ddl
        assert 'uuid STRING(MAX) NOT NULL' in ddl
        assert 'target_node_uuid STRING(MAX) NOT NULL' in ddl
        assert 'created_at TIMESTAMP NOT NULL' in ddl


def test_parents_are_created_before_children():
    statements = build_ddl(1024, 'graphiti')
    position = {ddl_object_name(s): i for i, s in enumerate(statements)}
    for name, (_, source, target) in EDGE_SPECS.items():
        assert position[(KIND_TABLE, source)] < position[(KIND_TABLE, name)]
        assert position[(KIND_TABLE, target)] < position[(KIND_TABLE, name)]
    graph_position = position[(KIND_PROPERTY_GRAPH, 'graphiti')]
    assert graph_position == len(statements) - 1


def test_vector_length_only_on_embedding_columns():
    statements = build_ddl(1024, 'graphiti')
    joined = '\n'.join(statements)
    assert joined.count('vector_length=>1024') == 3
    assert 'name_embedding ARRAY<FLOAT32>(vector_length=>1024)' in _create_table(
        statements, 'Entity'
    )
    assert 'name_embedding ARRAY<FLOAT32>(vector_length=>1024)' in _create_table(
        statements, 'Community'
    )
    assert 'fact_embedding ARRAY<FLOAT32>(vector_length=>1024)' in _create_table(
        statements, 'RelatesTo'
    )


def test_no_vector_length_without_dimension():
    statements = build_ddl(None, 'graphiti')
    assert not any('vector_length' in s for s in statements)
    assert 'name_embedding ARRAY<FLOAT32>,' in _create_table(statements, 'Entity')


def test_generated_token_columns():
    statements = build_ddl(1024, 'graphiti')
    entity = _create_table(statements, 'Entity')
    assert (
        "search_tokens TOKENLIST AS (TOKENIZE_FULLTEXT(CONCAT(name, ' ', COALESCE(summary, ''))))"
        ' HIDDEN'
    ) in entity
    assert 'group_tokens TOKENLIST AS (TOKEN(group_id)) HIDDEN' in entity
    assert 'label_tokens TOKENLIST AS (TOKEN(labels)) HIDDEN' in entity
    assert 'labels ARRAY<STRING(MAX)> NOT NULL' in entity
    episodic = _create_table(statements, 'Episodic')
    assert "TOKENIZE_FULLTEXT(CONCAT(content, ' ', source, ' ', source_description))" in episodic
    assert "TOKENIZE_FULLTEXT(CONCAT(name, ' ', fact))" in _create_table(statements, 'RelatesTo')
    assert 'search_tokens' not in _create_table(statements, 'Saga')
    for table in (*NODE_TABLES.values(), *EDGE_TABLES.values()):
        for column in table.columns:
            if column.ddl_type == 'TOKENLIST':
                assert column.name not in table.writable_columns


def test_indexes_and_search_indexes():
    statements = build_ddl(1024, 'graphiti')
    assert 'CREATE INDEX EntityByGroup ON Entity(group_id, uuid DESC)' in statements
    assert 'CREATE INDEX EntityByGroupName ON Entity(group_id, name)' in statements
    assert 'CREATE INDEX EpisodicByGroupValidAt ON Episodic(group_id, valid_at DESC)' in statements
    assert 'CREATE INDEX SagaByGroupName ON Saga(group_id, name)' in statements
    for name in EDGE_SPECS:
        assert f'CREATE UNIQUE INDEX {name}ByEdgeUuid ON {name}(edge_uuid)' in statements
        assert f'CREATE INDEX {name}ByTarget ON {name}(target_node_uuid, uuid)' in statements
    assert (
        'CREATE SEARCH INDEX EntitySearch ON Entity(search_tokens, group_tokens, label_tokens)'
        in statements
    )
    for name in ('EpisodicSearch', 'CommunitySearch', 'RelatesToSearch'):
        matches = [s for s in statements if ddl_object_name(s) == (KIND_SEARCH_INDEX, name)]
        assert len(matches) == 1
        assert 'group_tokens' in matches[0]
    assert len(INDEX_NAMES) == 5 + 2 * len(EDGE_SPECS)
    assert SEARCH_INDEX_NAMES == (
        'EntitySearch',
        'EpisodicSearch',
        'CommunitySearch',
        'RelatesToSearch',
    )


def test_property_graph_statement():
    graph = _property_graph(build_ddl(1024, 'my_graph'))
    assert graph.startswith('CREATE OR REPLACE PROPERTY GRAPH my_graph')
    for node in NODE_TABLES:
        assert f'    {node}' in graph
    for name, (label, source, target) in EDGE_SPECS.items():
        assert f'LABEL {label} PROPERTIES (edge_uuid AS uuid, uuid AS source_node_uuid,' in graph
        assert f'    {name}\n      SOURCE KEY (uuid) REFERENCES {source} (uuid)' in graph
        assert f'DESTINATION KEY (target_node_uuid) REFERENCES {target} (uuid)' in graph
    # TOKENLIST columns are exposed so SEARCH() works on edges in GQL.
    assert 'reference_time, search_tokens, group_tokens)' in graph


def test_ddl_object_name_parses_every_statement():
    statements = build_ddl(1024, 'graphiti')
    keys = [ddl_object_name(s) for s in statements]
    assert len(set(keys)) == len(keys) == 9 + len(INDEX_NAMES) + len(SEARCH_INDEX_NAMES) + 1
    assert {name for kind, name in keys if kind == KIND_TABLE} == {*NODE_TABLES, *EDGE_TABLES}
    assert {name for kind, name in keys if kind == KIND_INDEX} == set(INDEX_NAMES)
    assert {name for kind, name in keys if kind == KIND_SEARCH_INDEX} == set(SEARCH_INDEX_NAMES)
    assert ('PROPERTY GRAPH', 'graphiti') in keys


@pytest.mark.parametrize(
    ('statement', 'expected'),
    [
        ('CREATE TABLE Entity (\n  uuid STRING(MAX)\n) PRIMARY KEY (uuid)', ('TABLE', 'Entity')),
        ('create unique index Foo ON Bar(x)', ('INDEX', 'Foo')),
        ('CREATE NULL_FILTERED INDEX Foo ON Bar(x)', ('INDEX', 'Foo')),
        ('CREATE SEARCH INDEX EntitySearch ON Entity(t)', ('SEARCH INDEX', 'EntitySearch')),
        ('CREATE VECTOR INDEX V ON Entity(e)', ('VECTOR INDEX', 'V')),
        ('CREATE OR REPLACE PROPERTY GRAPH g NODE TABLES (A)', ('PROPERTY GRAPH', 'g')),
        ('DROP PROPERTY GRAPH IF EXISTS g', ('PROPERTY GRAPH', 'g')),
        ('DROP SEARCH INDEX S', ('SEARCH INDEX', 'S')),
        ('CREATE TABLE IF NOT EXISTS `T` (a INT64) PRIMARY KEY (a)', ('TABLE', 'T')),
        ('ALTER TABLE RelatesTo ADD COLUMN x INT64', ('TABLE', 'RelatesTo')),
    ],
)
def test_ddl_object_name_variants(statement: str, expected: tuple[str, str]):
    assert ddl_object_name(statement) == expected


def test_ddl_object_name_rejects_other_statements():
    with pytest.raises(ValueError):
        ddl_object_name('SELECT 1')


def test_drop_ddl_reverse_order():
    drops = build_drop_ddl('graphiti')
    keys = [ddl_object_name(s) for s in drops]
    assert keys[0] == ('PROPERTY GRAPH', 'graphiti')
    kinds = [kind for kind, _ in keys]
    assert kinds == sorted(kinds, key=['PROPERTY GRAPH', 'SEARCH INDEX', 'INDEX', 'TABLE'].index)
    tables = [name for kind, name in keys if kind == KIND_TABLE]
    assert tables[: len(EDGE_TABLES)] == list(reversed(EDGE_TABLES))
    assert tables[len(EDGE_TABLES) :] == list(reversed(NODE_TABLES))
    created = {ddl_object_name(s) for s in build_ddl(1024, 'graphiti')}
    assert set(keys) == created


def test_select_columns():
    assert edge_select_columns('RelatesTo').startswith(
        'edge_uuid AS uuid, uuid AS source_node_uuid, target_node_uuid, group_id, name, fact,'
    )
    assert edge_select_columns('MENTIONS') == (
        'edge_uuid AS uuid, uuid AS source_node_uuid, target_node_uuid, group_id, created_at'
    )
    assert edge_select_columns('HasMember', alias='e') == (
        'e.edge_uuid AS uuid, e.uuid AS source_node_uuid, '
        'e.target_node_uuid AS target_node_uuid, e.group_id AS group_id, '
        'e.created_at AS created_at'
    )
    assert 'search_tokens' not in edge_select_columns('RelatesTo')
    assert node_select_columns('Entity') == (
        'uuid, group_id, name, labels, summary, attributes, name_embedding, created_at'
    )
    assert node_select_columns('Saga', alias='s').startswith('s.uuid AS uuid, s.group_id AS')
    with pytest.raises(ValueError):
        edge_select_columns('Entity')
    with pytest.raises(ValueError):
        node_select_columns('RelatesTo')
    with pytest.raises(KeyError):
        resolve_table('Nope')
    assert resolve_table('RELATES_TO') is EDGE_TABLES['RelatesTo']


@pytest.mark.parametrize('graph_name', ['', '1graph', 'my-graph', 'g; DROP TABLE Entity'])
def test_invalid_graph_name(graph_name: str):
    with pytest.raises(ValueError):
        build_ddl(1024, graph_name)


@pytest.mark.parametrize('dim', [0, -1, True])
def test_invalid_embedding_dim(dim: int):
    with pytest.raises(ValueError):
        build_ddl(dim, 'graphiti')
