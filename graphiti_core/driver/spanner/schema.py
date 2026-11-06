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

# Spanner schema for Graphiti: the single source of truth for table and column layout.
#
# Node kinds and edge kinds each get a fixed table. Edge tables are interleaved in their
# source node table, so their primary key is ``(uuid, edge_uuid)``: Spanner requires the
# child's leading key column to carry the parent key's name and type, which means the
# ``uuid`` column of an edge TABLE holds the SOURCE NODE uuid and ``edge_uuid`` holds the
# Graphiti edge uuid. Two layers hide that from callers:
#
# * SQL: ``edge_select_columns(table)`` renders ``edge_uuid AS uuid, uuid AS source_node_uuid``
#   so result rows carry Graphiti's field names.
# * GQL: the property graph exposes the same aliases through ``PROPERTIES (edge_uuid AS uuid,
#   uuid AS source_node_uuid, ...)``, so inside ``GRAPH ... MATCH`` ``e.uuid`` is the edge uuid
#   and ``e.source_node_uuid`` the source node uuid.
#
# Generated ``TOKENLIST`` columns are HIDDEN, never written and never selected. They are listed
# in the property graph so ``SEARCH()`` works on nodes and edges in GQL.

from __future__ import annotations

import re
from dataclasses import dataclass

GRAPH_NAME_PATTERN = re.compile(r'^[A-Za-z_][A-Za-z0-9_]*$')

KIND_TABLE = 'TABLE'
KIND_INDEX = 'INDEX'
KIND_SEARCH_INDEX = 'SEARCH INDEX'
KIND_VECTOR_INDEX = 'VECTOR INDEX'
KIND_PROPERTY_GRAPH = 'PROPERTY GRAPH'

_STRING = 'STRING(MAX)'
_STRING_ARRAY = 'ARRAY<STRING(MAX)>'
_TIMESTAMP = 'TIMESTAMP'
_JSON = 'JSON'
_EMBEDDING = 'ARRAY<FLOAT32>'
_TOKENLIST = 'TOKENLIST'


@dataclass(frozen=True)
class Column:
    """One column of a Spanner table.

    ``generated_as`` holds the expression of a generated column (rendered as
    ``<name> <type> AS (<expr>)``); generated columns are never written by mutations.
    ``is_embedding`` marks ``ARRAY<FLOAT32>`` embedding columns, which get
    ``(vector_length=>N)`` when the schema is built with a known embedding dimension.
    """

    name: str
    ddl_type: str
    nullable: bool = True
    generated_as: str | None = None
    hidden: bool = False
    is_embedding: bool = False

    @property
    def is_generated(self) -> bool:
        return self.generated_as is not None

    def render(self, embedding_dim: int | None) -> str:
        ddl_type = self.ddl_type
        if self.is_embedding and embedding_dim is not None:
            ddl_type = f'{ddl_type}(vector_length=>{embedding_dim})'
        parts = [self.name, ddl_type]
        if not self.nullable:
            parts.append('NOT NULL')
        if self.generated_as is not None:
            parts.append(f'AS ({self.generated_as})')
        if self.hidden:
            parts.append('HIDDEN')
        return ' '.join(parts)


@dataclass(frozen=True, kw_only=True)
class Table:
    """A node or edge table. Edge tables also set ``source_table`` and ``target_table``."""

    name: str
    columns: tuple[Column, ...]
    primary_key: tuple[str, ...]
    label: str
    interleave_in_parent: str | None = None
    source_table: str | None = None
    target_table: str | None = None

    @property
    def is_edge(self) -> bool:
        return self.source_table is not None

    @property
    def column_names(self) -> tuple[str, ...]:
        return tuple(column.name for column in self.columns)

    @property
    def writable_columns(self) -> tuple[str, ...]:
        """Columns a mutation writes, in table order (all except generated columns)."""
        return tuple(column.name for column in self.columns if not column.is_generated)

    @property
    def embedding_columns(self) -> tuple[str, ...]:
        return tuple(column.name for column in self.columns if column.is_embedding)

    def column(self, name: str) -> Column:
        for column in self.columns:
            if column.name == name:
                return column
        raise KeyError(f'Table {self.name} has no column {name}')

    def create_ddl(self, embedding_dim: int | None) -> str:
        body = ',\n'.join(f'  {column.render(embedding_dim)}' for column in self.columns)
        ddl = f'CREATE TABLE {self.name} (\n{body}\n) PRIMARY KEY ({", ".join(self.primary_key)})'
        if self.interleave_in_parent is not None:
            ddl += f',\n  INTERLEAVE IN PARENT {self.interleave_in_parent} ON DELETE CASCADE'
        return ddl


@dataclass(frozen=True)
class Index:
    """A secondary index. ``key_parts`` may carry a direction, e.g. ``'uuid DESC'``."""

    name: str
    table: str
    key_parts: tuple[str, ...]
    unique: bool = False

    def create_ddl(self) -> str:
        unique = 'UNIQUE ' if self.unique else ''
        return f'CREATE {unique}INDEX {self.name} ON {self.table}({", ".join(self.key_parts)})'

    def drop_ddl(self) -> str:
        return f'DROP INDEX {self.name}'


@dataclass(frozen=True)
class SearchIndex:
    """A full-text search index over TOKENLIST columns."""

    name: str
    table: str
    columns: tuple[str, ...]

    def create_ddl(self) -> str:
        return f'CREATE SEARCH INDEX {self.name} ON {self.table}({", ".join(self.columns)})'

    def drop_ddl(self) -> str:
        return f'DROP SEARCH INDEX {self.name}'


def _col(name: str, ddl_type: str, nullable: bool = True) -> Column:
    return Column(name, ddl_type, nullable)


def _embedding(name: str) -> Column:
    return Column(name, _EMBEDDING, nullable=True, is_embedding=True)


def _fulltext_tokens(expression: str) -> Column:
    return Column(
        'search_tokens',
        _TOKENLIST,
        generated_as=f'TOKENIZE_FULLTEXT({expression})',
        hidden=True,
    )


def _token(name: str, source_column: str) -> Column:
    return Column(name, _TOKENLIST, generated_as=f'TOKEN({source_column})', hidden=True)


_GROUP_TOKENS = _token('group_tokens', 'group_id')

ENTITY_TABLE = Table(
    name='Entity',
    label='Entity',
    columns=(
        _col('uuid', _STRING, nullable=False),
        _col('group_id', _STRING, nullable=False),
        _col('name', _STRING, nullable=False),
        _col('labels', _STRING_ARRAY, nullable=False),
        _col('summary', _STRING),
        _col('attributes', _JSON),
        _embedding('name_embedding'),
        _col('created_at', _TIMESTAMP, nullable=False),
        _fulltext_tokens("CONCAT(name, ' ', COALESCE(summary, ''))"),
        _GROUP_TOKENS,
        _token('label_tokens', 'labels'),
    ),
    primary_key=('uuid',),
)

EPISODIC_TABLE = Table(
    name='Episodic',
    label='Episodic',
    columns=(
        _col('uuid', _STRING, nullable=False),
        _col('group_id', _STRING, nullable=False),
        _col('name', _STRING, nullable=False),
        _col('source', _STRING, nullable=False),
        _col('source_description', _STRING, nullable=False),
        _col('content', _STRING, nullable=False),
        _col('valid_at', _TIMESTAMP, nullable=False),
        _col('created_at', _TIMESTAMP, nullable=False),
        _col('entity_edges', _STRING_ARRAY),
        _col('episode_metadata', _JSON),
        _fulltext_tokens("CONCAT(content, ' ', source, ' ', source_description)"),
        _GROUP_TOKENS,
    ),
    primary_key=('uuid',),
)

COMMUNITY_TABLE = Table(
    name='Community',
    label='Community',
    columns=(
        _col('uuid', _STRING, nullable=False),
        _col('group_id', _STRING, nullable=False),
        _col('name', _STRING, nullable=False),
        _col('summary', _STRING),
        _embedding('name_embedding'),
        _col('created_at', _TIMESTAMP, nullable=False),
        _fulltext_tokens("CONCAT(name, ' ', COALESCE(summary, ''))"),
        _GROUP_TOKENS,
    ),
    primary_key=('uuid',),
)

SAGA_TABLE = Table(
    name='Saga',
    label='Saga',
    columns=(
        _col('uuid', _STRING, nullable=False),
        _col('group_id', _STRING, nullable=False),
        _col('name', _STRING, nullable=False),
        _col('summary', _STRING),
        _col('first_episode_uuid', _STRING),
        _col('last_episode_uuid', _STRING),
        _col('last_summarized_at', _TIMESTAMP),
        _col('last_summarized_episode_valid_at', _TIMESTAMP),
        _col('created_at', _TIMESTAMP, nullable=False),
    ),
    primary_key=('uuid',),
)


def _edge_table(
    name: str,
    label: str,
    source_table: str,
    target_table: str,
    extra_columns: tuple[Column, ...] = (),
) -> Table:
    key_columns = (
        # Source node uuid: an interleaved child's leading key must match the parent key.
        _col('uuid', _STRING, nullable=False),
        # The Graphiti edge uuid.
        _col('edge_uuid', _STRING, nullable=False),
        _col('target_node_uuid', _STRING, nullable=False),
        _col('group_id', _STRING, nullable=False),
    )
    if extra_columns:
        columns = key_columns + extra_columns
    else:
        columns = key_columns + (_col('created_at', _TIMESTAMP, nullable=False),)
    return Table(
        name=name,
        label=label,
        columns=columns,
        primary_key=('uuid', 'edge_uuid'),
        interleave_in_parent=source_table,
        source_table=source_table,
        target_table=target_table,
    )


RELATES_TO_TABLE = _edge_table(
    'RelatesTo',
    'RELATES_TO',
    'Entity',
    'Entity',
    extra_columns=(
        _col('name', _STRING, nullable=False),
        _col('fact', _STRING, nullable=False),
        _embedding('fact_embedding'),
        _col('episodes', _STRING_ARRAY),
        _col('attributes', _JSON),
        _col('created_at', _TIMESTAMP, nullable=False),
        _col('valid_at', _TIMESTAMP),
        _col('invalid_at', _TIMESTAMP),
        _col('expired_at', _TIMESTAMP),
        _col('reference_time', _TIMESTAMP),
        _fulltext_tokens("CONCAT(name, ' ', fact)"),
        _GROUP_TOKENS,
    ),
)
MENTIONS_TABLE = _edge_table('Mentions', 'MENTIONS', 'Episodic', 'Entity')
HAS_MEMBER_TABLE = _edge_table('HasMember', 'HAS_MEMBER', 'Community', 'Entity')
HAS_EPISODE_TABLE = _edge_table('HasEpisode', 'HAS_EPISODE', 'Saga', 'Episodic')
NEXT_EPISODE_TABLE = _edge_table('NextEpisode', 'NEXT_EPISODE', 'Episodic', 'Episodic')

# Dict order is dependency order: parents before their interleaved children.
NODE_TABLES: dict[str, Table] = {
    table.name: table for table in (ENTITY_TABLE, EPISODIC_TABLE, COMMUNITY_TABLE, SAGA_TABLE)
}
EDGE_TABLES: dict[str, Table] = {
    table.name: table
    for table in (
        RELATES_TO_TABLE,
        MENTIONS_TABLE,
        HAS_MEMBER_TABLE,
        HAS_EPISODE_TABLE,
        NEXT_EPISODE_TABLE,
    )
}
TABLES: dict[str, Table] = {**NODE_TABLES, **EDGE_TABLES}

# Graph label -> table name, e.g. 'RELATES_TO' -> 'RelatesTo', 'Entity' -> 'Entity'.
TABLE_BY_LABEL: dict[str, str] = {table.label: table.name for table in TABLES.values()}

INDEXES: tuple[Index, ...] = (
    Index('EntityByGroup', 'Entity', ('group_id', 'uuid DESC')),
    # Entity lookup by name runs on every add_episode (dedup candidates).
    Index('EntityByGroupName', 'Entity', ('group_id', 'name')),
    Index('EpisodicByGroupValidAt', 'Episodic', ('group_id', 'valid_at DESC')),
    Index('CommunityByGroup', 'Community', ('group_id', 'uuid DESC')),
    Index('SagaByGroupName', 'Saga', ('group_id', 'name')),
    *(
        index
        for table in EDGE_TABLES
        for index in (
            Index(f'{table}ByEdgeUuid', table, ('edge_uuid',), unique=True),
            Index(f'{table}ByTarget', table, ('target_node_uuid', 'uuid')),
        )
    ),
)

SEARCH_INDEXES: tuple[SearchIndex, ...] = (
    SearchIndex('EntitySearch', 'Entity', ('search_tokens', 'group_tokens', 'label_tokens')),
    SearchIndex('EpisodicSearch', 'Episodic', ('search_tokens', 'group_tokens')),
    SearchIndex('CommunitySearch', 'Community', ('search_tokens', 'group_tokens')),
    SearchIndex('RelatesToSearch', 'RelatesTo', ('search_tokens', 'group_tokens')),
)

INDEX_NAMES: tuple[str, ...] = tuple(index.name for index in INDEXES)
SEARCH_INDEX_NAMES: tuple[str, ...] = tuple(index.name for index in SEARCH_INDEXES)


def resolve_table(name: str) -> Table:
    """Return a table by table name (``'RelatesTo'``) or graph label (``'RELATES_TO'``)."""
    table_name = name if name in TABLES else TABLE_BY_LABEL.get(name)
    if table_name is None:
        raise KeyError(f'Unknown Spanner table or label: {name}')
    return TABLES[table_name]


def validate_graph_name(graph_name: str) -> str:
    if not GRAPH_NAME_PATTERN.match(graph_name):
        raise ValueError(f'Invalid Spanner property graph name: {graph_name!r}')
    return graph_name


def validate_embedding_dim(embedding_dim: int | None) -> int | None:
    if embedding_dim is None:
        return None
    if isinstance(embedding_dim, bool) or not isinstance(embedding_dim, int) or embedding_dim < 1:
        raise ValueError(f'embedding_dim must be a positive integer or None, got {embedding_dim!r}')
    return embedding_dim


def edge_property_list(table: Table) -> list[str]:
    """PROPERTIES entries of an edge table in the property graph, under Graphiti's names."""
    properties = ['edge_uuid AS uuid', 'uuid AS source_node_uuid']
    properties.extend(name for name in table.column_names if name not in ('uuid', 'edge_uuid'))
    return properties


def build_property_graph_ddl(graph_name: str) -> str:
    validate_graph_name(graph_name)
    node_tables = ',\n'.join(f'    {name}' for name in NODE_TABLES)
    edge_tables = ',\n'.join(
        f'    {table.name}\n'
        f'      SOURCE KEY (uuid) REFERENCES {table.source_table} (uuid)\n'
        f'      DESTINATION KEY (target_node_uuid) REFERENCES {table.target_table} (uuid)\n'
        f'      LABEL {table.label} PROPERTIES ({", ".join(edge_property_list(table))})'
        for table in EDGE_TABLES.values()
    )
    return (
        f'CREATE OR REPLACE PROPERTY GRAPH {graph_name}\n'
        f'  NODE TABLES (\n{node_tables}\n  )\n'
        f'  EDGE TABLES (\n{edge_tables}\n  )'
    )


def build_ddl(embedding_dim: int | None, graph_name: str) -> list[str]:
    """Every DDL statement of the schema, in dependency order.

    Tables (parents before interleaved children), secondary indexes, search indexes, then the
    property graph. ``embedding_dim=None`` leaves embedding columns as plain ``ARRAY<FLOAT32>``
    (exact KNN still works; ANN vector indexes need a fixed length).
    """
    validate_embedding_dim(embedding_dim)
    statements = [table.create_ddl(embedding_dim) for table in NODE_TABLES.values()]
    statements.extend(table.create_ddl(embedding_dim) for table in EDGE_TABLES.values())
    statements.extend(index.create_ddl() for index in INDEXES)
    statements.extend(index.create_ddl() for index in SEARCH_INDEXES)
    statements.append(build_property_graph_ddl(graph_name))
    return statements


def build_drop_ddl(graph_name: str) -> list[str]:
    """DROP statements for the whole schema, in reverse dependency order."""
    validate_graph_name(graph_name)
    statements = [f'DROP PROPERTY GRAPH {graph_name}']
    statements.extend(index.drop_ddl() for index in reversed(SEARCH_INDEXES))
    statements.extend(index.drop_ddl() for index in reversed(INDEXES))
    statements.extend(f'DROP TABLE {name}' for name in reversed(EDGE_TABLES))
    statements.extend(f'DROP TABLE {name}' for name in reversed(NODE_TABLES))
    return statements


_DDL_OBJECT_PATTERN = re.compile(
    r'^\s*(?:CREATE|DROP|ALTER)\s+(?:OR\s+REPLACE\s+)?(?:UNIQUE\s+)?(?:NULL_FILTERED\s+)?'
    r'(TABLE|INDEX|SEARCH\s+INDEX|VECTOR\s+INDEX|PROPERTY\s+GRAPH)\s+'
    r'(?:IF\s+(?:NOT\s+)?EXISTS\s+)?`?([A-Za-z_][A-Za-z0-9_]*)`?',
    re.IGNORECASE,
)


def ddl_object_name(statement: str) -> tuple[str, str]:
    """Return ``(kind, name)`` for a CREATE, DROP or ALTER statement.

    ``kind`` is one of ``'TABLE'``, ``'INDEX'`` (unique or not), ``'SEARCH INDEX'``,
    ``'VECTOR INDEX'`` or ``'PROPERTY GRAPH'``.
    """
    match = _DDL_OBJECT_PATTERN.match(statement)
    if match is None:
        raise ValueError(f'Cannot parse a DDL object name from: {statement[:120]}')
    kind = ' '.join(match.group(1).upper().split())
    return kind, match.group(2)


def _qualified(column: str, alias: str | None) -> str:
    return f'{alias}.{column}' if alias else column


def node_select_columns(table: str, alias: str | None = None) -> str:
    """SELECT list for a node table: every non-generated column under its own name."""
    resolved = resolve_table(table)
    if resolved.is_edge:
        raise ValueError(f'{resolved.name} is an edge table; use edge_select_columns')
    return ', '.join(
        f'{_qualified(name, alias)} AS {name}' if alias else name
        for name in resolved.writable_columns
    )


def edge_select_columns(table: str, alias: str | None = None) -> str:
    """SELECT list for an edge table that renames columns to Graphiti's field names.

    Renders ``edge_uuid AS uuid, uuid AS source_node_uuid, target_node_uuid, group_id, ...``.
    With ``alias='e'`` every column is qualified (``e.edge_uuid AS uuid, ...``) for joins.
    """
    resolved = resolve_table(table)
    if not resolved.is_edge:
        raise ValueError(f'{resolved.name} is a node table; use node_select_columns')
    parts = [
        f'{_qualified("edge_uuid", alias)} AS uuid',
        f'{_qualified("uuid", alias)} AS source_node_uuid',
    ]
    parts.extend(
        f'{_qualified(name, alias)} AS {name}' if alias else name
        for name in resolved.writable_columns
        if name not in ('uuid', 'edge_uuid')
    )
    return ', '.join(parts)
