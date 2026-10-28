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

Spanner adapters for the two legacy driver hooks that core still consults before falling back
to inline Cypher: ``driver.graph_operations_interface`` (``nodes.py``, ``edges.py``,
``graphiti.py``, ``utils/bulk_utils.py``, ``utils/maintenance/*``) and
``driver.search_interface`` (``search/search_utils.py``). The Spanner driver rejects Cypher,
so every hook method is implemented here, mostly as a delegation to the Spanner ops classes
with the arguments reordered (the driver is the ops executor).

A few methods have no ops counterpart, or an ops counterpart whose return shape differs from
what the legacy call site expects. They run their own GoogleSQL through ``SpannerDriver.read``:
``node_get_by_names``, ``node_get_by_attribute_values``, ``node_load_embeddings_bulk``,
``edge_load_embeddings_bulk``, ``saga_get_previous_episode_uuid``,
``saga_get_episode_contents``, ``determine_entity_community`` (the ops ABC returns ``None``),
``get_embeddings_for_communities``, ``node_distance_reranker`` and
``episode_mentions_reranker`` (the ops rerankers return nodes, not ``(uuids, scores)``).

These adapters are disposable: once Phase 2 of ``spec/driver-operations-redesign.md`` routes
core through the ops classes, nothing consults the legacy hooks and this module can be deleted.
"""

from __future__ import annotations

import re
from collections.abc import Awaitable, Callable, Iterable, Iterator, Mapping, Sequence
from datetime import datetime
from typing import TYPE_CHECKING, Any, TypeVar

from pydantic import BaseModel

from graphiti_core.driver.graph_operations.graph_operations import GraphOperationsInterface
from graphiti_core.driver.search_interface.search_interface import SearchInterface
from graphiti_core.driver.spanner.codecs import STRING_ARRAY
from graphiti_core.driver.spanner.operations.base import spanner_driver
from graphiti_core.driver.spanner.operations.search_ops import SpannerSearchOperations
from graphiti_core.driver.spanner.rows import community_node_from_row, entity_node_from_row
from graphiti_core.driver.spanner.schema import node_select_columns
from graphiti_core.edges import (
    CommunityEdge,
    EntityEdge,
    EpisodicEdge,
    HasEpisodeEdge,
    NextEpisodeEdge,
)
from graphiti_core.errors import EdgeNotFoundError
from graphiti_core.helpers import parse_db_date
from graphiti_core.nodes import CommunityNode, EntityNode, EpisodeType, EpisodicNode, SagaNode

if TYPE_CHECKING:
    from graphiti_core.driver.spanner.driver import SpannerDriver

M = TypeVar('M', bound=BaseModel)

# IN UNNEST lists are chunked so a single read stays small.
_READ_CHUNK = 500

# Attribute names are interpolated into a JSON path, so they must be plain identifiers.
_JSON_KEY = re.compile(r'^[A-Za-z_][A-Za-z0-9_]*$')

_NODE_TABLES: tuple[str, ...] = ('Entity', 'Episodic', 'Community', 'Saga')
_EDGE_TABLES: tuple[str, ...] = ('RelatesTo', 'Mentions', 'HasMember', 'HasEpisode', 'NextEpisode')

# Node kinds a group-wide ``Node.delete_by_group_id`` removes; the same labels as the Cypher
# fallback in ``nodes.py`` (it never deletes sagas).
_GROUP_DELETE_NODE_TABLES: tuple[str, ...] = ('Episodic', 'Community', 'Entity')

# Core keys of the rows ``bulk_utils.add_nodes_and_edges_bulk_tx`` builds. Every other key of
# an entity-node or entity-edge row is a flattened attribute.
ENTITY_NODE_BULK_KEYS: frozenset[str] = frozenset(
    {'uuid', 'name', 'group_id', 'summary', 'created_at', 'name_embedding', 'labels'}
)
ENTITY_EDGE_BULK_KEYS: frozenset[str] = frozenset(
    {
        'uuid',
        'source_node_uuid',
        'target_node_uuid',
        'name',
        'fact',
        'group_id',
        'episodes',
        'created_at',
        'expired_at',
        'valid_at',
        'invalid_at',
        'reference_time',
        'fact_embedding',
    }
)

Deleter = Callable[..., Awaitable[None]]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _chunks(items: Sequence[str], size: int) -> Iterator[list[str]]:
    size = max(size, 1)
    for start in range(0, len(items), size):
        yield list(items[start : start + size])


def _required_datetime(row: Mapping[str, Any], key: str) -> datetime:
    value = parse_db_date(row.get(key))
    if value is None:
        raise ValueError(f'{key} cannot be None for row {row.get("uuid", "unknown")}')
    return value


def _row_attributes(row: Mapping[str, Any], core_keys: frozenset[str]) -> dict[str, Any]:
    """Rebuild ``attributes`` from a bulk row: every non-core key, plus a nested
    ``attributes`` mapping when the row is shaped like ``model_dump()``."""
    attributes: dict[str, Any] = {}
    nested = row.get('attributes')
    nested_is_mapping = isinstance(nested, Mapping)
    if nested_is_mapping:
        attributes.update(nested)
    for key, value in row.items():
        if key in core_keys or (key == 'attributes' and nested_is_mapping):
            continue
        attributes[key] = value
    return attributes


def _as_models(model: type[M], items: Iterable[Any]) -> list[M]:
    """Pass models through; validate dicts (e.g. ``model_dump()`` rows) into ``model``."""
    return [item if isinstance(item, model) else model.model_validate(dict(item)) for item in items]


def _source_value(source: Any) -> str | None:
    if source is None:
        return None
    if isinstance(source, EpisodeType):
        return source.value
    return str(source)


def _embedding(value: Any) -> list[float] | None:
    return [float(item) for item in value] if value is not None else None


async def _uuids_by_table(
    driver: SpannerDriver,
    tables: Sequence[str],
    key_column: str,
    uuids: Sequence[str],
    group_id: str | None = None,
) -> dict[str, list[str]]:
    """Map each table to the given uuids it holds (only rows of ``group_id`` when given).

    One ``UNION ALL`` read per chunk; ``key_column`` is ``uuid`` for node tables and
    ``edge_uuid`` for edge tables (both indexed).
    """
    found: dict[str, list[str]] = {table: [] for table in tables}
    unique = list(dict.fromkeys(uuids))
    if not unique:
        return found
    group_clause = ' AND group_id = @group_id' if group_id is not None else ''
    sql = ' UNION ALL '.join(
        f"SELECT '{table}' AS kind, {key_column} AS uuid FROM {table} "
        f'WHERE {key_column} IN UNNEST(@uuids){group_clause}'
        for table in tables
    )
    for chunk in _chunks(unique, _READ_CHUNK):
        params: dict[str, Any] = {'uuids': chunk}
        if group_id is not None:
            params['group_id'] = group_id
        for record in await driver.read(sql, params, {'uuids': STRING_ARRAY}):
            found[record['kind']].append(record['uuid'])
    return found


async def _scoped_uuids(
    driver: SpannerDriver, table: str, key_column: str, uuids: list[str], group_id: str | None
) -> list[str]:
    """``uuids`` unchanged without ``group_id``; otherwise only those of rows in the group."""
    if group_id is None or not uuids:
        return uuids
    return (await _uuids_by_table(driver, (table,), key_column, uuids, group_id))[table]


async def _embeddings_by_uuid(
    driver: SpannerDriver,
    table: str,
    key_column: str,
    embedding_column: str,
    uuids: Iterable[str],
    batch_size: int,
) -> dict[str, list[float]]:
    """``{uuid: embedding}`` for the rows that exist and have an embedding."""
    found: dict[str, list[float]] = {}
    unique = list(dict.fromkeys(uuids))
    for chunk in _chunks(unique, batch_size):
        records = await driver.read(
            f'SELECT {key_column} AS uuid, {embedding_column} AS embedding FROM {table} '
            f'WHERE {key_column} IN UNNEST(@uuids)',
            {'uuids': chunk},
            {'uuids': STRING_ARRAY},
        )
        for record in records:
            embedding = _embedding(record['embedding'])
            if embedding is not None:
                found[record['uuid']] = embedding
    return found


def _node_deleters(driver: SpannerDriver) -> dict[str, Deleter]:
    return {
        'Entity': driver.entity_node_ops.delete_by_uuids,
        'Episodic': driver.episode_node_ops.delete_by_uuids,
        'Community': driver.community_node_ops.delete_by_uuids,
        'Saga': driver.saga_node_ops.delete_by_uuids,
    }


def _group_deleters(driver: SpannerDriver) -> dict[str, Deleter]:
    return {
        'Entity': driver.entity_node_ops.delete_by_group_id,
        'Episodic': driver.episode_node_ops.delete_by_group_id,
        'Community': driver.community_node_ops.delete_by_group_id,
        'Saga': driver.saga_node_ops.delete_by_group_id,
    }


def _edge_deleters(driver: SpannerDriver) -> dict[str, Deleter]:
    return {
        'RelatesTo': driver.entity_edge_ops.delete_by_uuids,
        'Mentions': driver.episodic_edge_ops.delete_by_uuids,
        'HasMember': driver.community_edge_ops.delete_by_uuids,
        'HasEpisode': driver.has_episode_edge_ops.delete_by_uuids,
        'NextEpisode': driver.next_episode_edge_ops.delete_by_uuids,
    }


# ---------------------------------------------------------------------------
# Bulk rows (bulk_utils.add_nodes_and_edges_bulk_tx) -> models
# ---------------------------------------------------------------------------


def entity_node_from_bulk_row(row: Mapping[str, Any] | EntityNode) -> EntityNode:
    """Entity row: the keys in ``ENTITY_NODE_BULK_KEYS`` plus flattened attributes."""
    if isinstance(row, EntityNode):
        return row
    fields: dict[str, Any] = {
        'uuid': row['uuid'],
        'name': row['name'],
        'group_id': row['group_id'],
        'labels': list(row.get('labels') or []),
        'summary': row.get('summary') or '',
        'name_embedding': row.get('name_embedding'),
        'attributes': _row_attributes(row, ENTITY_NODE_BULK_KEYS),
    }
    created_at = parse_db_date(row.get('created_at'))
    if created_at is not None:
        fields['created_at'] = created_at
    return EntityNode(**fields)


def entity_edge_from_bulk_row(row: Mapping[str, Any] | EntityEdge) -> EntityEdge:
    """Entity-edge row: the keys in ``ENTITY_EDGE_BULK_KEYS`` plus flattened attributes."""
    if isinstance(row, EntityEdge):
        return row
    return EntityEdge(
        uuid=row['uuid'],
        source_node_uuid=row['source_node_uuid'],
        target_node_uuid=row['target_node_uuid'],
        name=row['name'],
        fact=row['fact'],
        group_id=row['group_id'],
        episodes=list(row.get('episodes') or []),
        created_at=_required_datetime(row, 'created_at'),
        expired_at=parse_db_date(row.get('expired_at')),
        valid_at=parse_db_date(row.get('valid_at')),
        invalid_at=parse_db_date(row.get('invalid_at')),
        reference_time=parse_db_date(row.get('reference_time')),
        fact_embedding=row.get('fact_embedding'),
        attributes=_row_attributes(row, ENTITY_EDGE_BULK_KEYS),
    )


def episodic_node_from_bulk_row(row: Mapping[str, Any] | EpisodicNode) -> EpisodicNode:
    """Episode row: ``dict(EpisodicNode)`` with ``source`` as its string value and ``labels``
    removed."""
    if isinstance(row, EpisodicNode):
        return row
    source = row['source']
    fields: dict[str, Any] = {
        'uuid': row['uuid'],
        'name': row['name'],
        'group_id': row['group_id'],
        'source': source if isinstance(source, EpisodeType) else EpisodeType(source),
        'source_description': row['source_description'],
        'content': row['content'],
        'valid_at': _required_datetime(row, 'valid_at'),
        'entity_edges': list(row.get('entity_edges') or []),
        'episode_metadata': row.get('episode_metadata'),
    }
    created_at = parse_db_date(row.get('created_at'))
    if created_at is not None:
        fields['created_at'] = created_at
    if row.get('labels'):
        fields['labels'] = list(row['labels'])
    return EpisodicNode(**fields)


def episodic_edge_from_bulk_row(row: Mapping[str, Any] | EpisodicEdge) -> EpisodicEdge:
    """MENTIONS row: ``EpisodicEdge.model_dump()``."""
    if isinstance(row, EpisodicEdge):
        return row
    return EpisodicEdge(
        uuid=row['uuid'],
        group_id=row['group_id'],
        source_node_uuid=row['source_node_uuid'],
        target_node_uuid=row['target_node_uuid'],
        created_at=_required_datetime(row, 'created_at'),
    )


# ---------------------------------------------------------------------------
# GraphOperationsInterface
# ---------------------------------------------------------------------------


class SpannerLegacyGraphOperations(GraphOperationsInterface):
    """Spanner adapter for the legacy ``driver.graph_operations_interface`` hook.

    Every method of ``GraphOperationsInterface`` is overridden, so none raises
    ``NotImplementedError`` itself (an ops method that is not implemented yet still does, and
    core then falls back to Cypher, which the driver rejects). Methods receive the
    ``SpannerDriver`` as ``driver`` and pass it to the ops classes as their executor.
    """

    # -----------------
    # Node: Save/Delete
    # -----------------

    async def node_save(self, node: Any, driver: Any) -> None:
        d = spanner_driver(driver)
        await d.entity_node_ops.save(d, node)

    async def node_delete(self, node: Any, driver: Any) -> None:
        """``Node.delete`` is inherited by entity, episodic and community nodes."""
        d = spanner_driver(driver)
        if isinstance(node, EntityNode):
            await d.entity_node_ops.delete(d, node)
        elif isinstance(node, EpisodicNode):
            await d.episode_node_ops.delete(d, node)
        elif isinstance(node, CommunityNode):
            await d.community_node_ops.delete(d, node)
        elif isinstance(node, SagaNode):
            await d.saga_node_ops.delete(d, node)
        else:
            await self.node_delete_by_uuids(None, d, [node.uuid])

    async def node_save_bulk(
        self,
        _cls: Any,
        driver: Any,
        transaction: Any,
        nodes: list[Any],
        batch_size: int = 100,
    ) -> None:
        d = spanner_driver(driver)
        models = [entity_node_from_bulk_row(row) for row in nodes]
        await d.entity_node_ops.save_bulk(d, models, tx=transaction, batch_size=batch_size)

    async def node_delete_by_group_id(
        self,
        _cls: Any,
        driver: Any,
        group_id: str,
        batch_size: int = 100,
    ) -> None:
        """Generic over node kinds like the Cypher fallback: deletes the group's episodic,
        community and entity nodes (and their edges), whatever ``_cls`` is."""
        d = spanner_driver(driver)
        deleters = _group_deleters(d)
        for table in _GROUP_DELETE_NODE_TABLES:
            await deleters[table](d, group_id, batch_size=batch_size)

    async def node_delete_by_uuids(
        self,
        _cls: Any,
        driver: Any,
        uuids: list[str],
        group_id: str | None = None,
        batch_size: int = 100,
    ) -> None:
        """Generic over node kinds (``Node.delete_by_uuids``): finds which node table holds
        each uuid in one read, then deletes through the matching ops class."""
        if not uuids:
            return
        d = spanner_driver(driver)
        located = await _uuids_by_table(d, _NODE_TABLES, 'uuid', uuids, group_id)
        deleters = _node_deleters(d)
        for table, found in located.items():
            if found:
                await deleters[table](d, found, batch_size=batch_size)

    # -----------------
    # Node: Read
    # -----------------

    async def node_get_by_uuid(self, _cls: Any, driver: Any, uuid: str) -> EntityNode:
        d = spanner_driver(driver)
        return await d.entity_node_ops.get_by_uuid(d, uuid)

    async def node_get_by_uuids(
        self, _cls: Any, driver: Any, uuids: list[str], group_id: str | None = None
    ) -> list[EntityNode]:
        d = spanner_driver(driver)
        nodes = await d.entity_node_ops.get_by_uuids(d, uuids)
        if group_id is not None:
            nodes = [node for node in nodes if node.group_id == group_id]
        return nodes

    async def node_get_by_group_ids(
        self,
        _cls: Any,
        driver: Any,
        group_ids: list[str],
        limit: int | None = None,
        uuid_cursor: str | None = None,
    ) -> list[EntityNode]:
        d = spanner_driver(driver)
        return await d.entity_node_ops.get_by_group_ids(d, group_ids, limit, uuid_cursor)

    async def node_get_by_names(
        self,
        _cls: Any,
        driver: Any,
        names: list[str],
        group_ids: list[str],
    ) -> list[EntityNode]:
        """Entity nodes whose exact ``name`` is in ``names``, in any of ``group_ids``."""
        unique_names = sorted({name for name in names if name})
        unique_group_ids = sorted({group_id for group_id in group_ids if group_id})
        if not unique_names or not unique_group_ids:
            return []
        d = spanner_driver(driver)
        records = await d.read(
            f'SELECT {node_select_columns("Entity")} FROM Entity '
            'WHERE group_id IN UNNEST(@group_ids) AND name IN UNNEST(@names)',
            {'group_ids': unique_group_ids, 'names': unique_names},
            {'group_ids': STRING_ARRAY, 'names': STRING_ARRAY},
        )
        return [entity_node_from_row(record) for record in records]

    async def node_get_by_attribute_values(
        self,
        _cls: Any,
        driver: Any,
        group_id: str,
        attributes: dict[str, Any],
        node_labels: list[str] | None = None,
        limit: int = 100,
    ) -> list[EntityNode]:
        """Entity nodes of ``group_id`` with every label in ``node_labels`` whose JSON
        ``attributes`` match every ``attributes`` item, oldest first.

        Strings match after trimming whitespace on both sides and numbers match as FLOAT64,
        mirroring the identity-property normalisation in ``dedup_helpers``; the caller
        re-checks candidates exactly. ``bool`` and ``None`` are supported too; other value
        types raise ``TypeError``. An empty ``attributes`` returns no nodes.
        """
        if not attributes:
            return []
        where = ['group_id = @group_id']
        params: dict[str, Any] = {'group_id': group_id}
        for index, (key, value) in enumerate(attributes.items()):
            if not _JSON_KEY.match(key):
                raise ValueError(f'Attribute name {key!r} is not a valid identifier')
            expr = f"JSON_VALUE(attributes, '$.{key}')"
            param = f'v{index}'
            if value is None:
                where.append(f'{expr} IS NULL')
            elif isinstance(value, bool):
                where.append(f'{expr} = @{param}')
                params[param] = 'true' if value else 'false'
            elif isinstance(value, int | float):
                where.append(f'SAFE_CAST({expr} AS FLOAT64) = @{param}')
                params[param] = float(value)
            elif isinstance(value, str):
                where.append(f'TRIM({expr}) = @{param}')
                params[param] = value.strip()
            else:
                raise TypeError(
                    f'Unsupported value type {type(value).__name__} for attribute {key!r}'
                )
        for index, label in enumerate(node_labels or []):
            where.append(f'@label{index} IN UNNEST(labels)')
            params[f'label{index}'] = label
        sql = (
            f'SELECT {node_select_columns("Entity")} FROM Entity WHERE '
            + ' AND '.join(where)
            + ' ORDER BY created_at, uuid'
        )
        if limit is not None:
            sql += ' LIMIT @limit'
            params['limit'] = limit
        d = spanner_driver(driver)
        records = await d.read(sql, params)
        return [entity_node_from_row(record) for record in records]

    # --------------------------
    # Node: Embeddings (load)
    # --------------------------

    async def node_load_embeddings(self, node: Any, driver: Any) -> None:
        d = spanner_driver(driver)
        await d.entity_node_ops.load_embeddings(d, node)

    async def node_load_embeddings_bulk(
        self,
        driver: Any,
        nodes: list[Any],
        batch_size: int = 100,
    ) -> dict[str, list[float]]:
        """``{uuid: name_embedding}``; the nodes themselves are not modified."""
        d = spanner_driver(driver)
        return await _embeddings_by_uuid(
            d, 'Entity', 'uuid', 'name_embedding', (node.uuid for node in nodes), batch_size
        )

    # --------------------------
    # EpisodicNode: Save/Delete
    # --------------------------

    async def episodic_node_save(self, node: Any, driver: Any) -> None:
        d = spanner_driver(driver)
        await d.episode_node_ops.save(d, node)

    async def episodic_node_delete(self, node: Any, driver: Any) -> None:
        d = spanner_driver(driver)
        await d.episode_node_ops.delete(d, node)

    async def episodic_node_save_bulk(
        self,
        _cls: Any,
        driver: Any,
        transaction: Any,
        nodes: list[Any],
        batch_size: int = 100,
    ) -> None:
        d = spanner_driver(driver)
        models = [episodic_node_from_bulk_row(row) for row in nodes]
        await d.episode_node_ops.save_bulk(d, models, tx=transaction, batch_size=batch_size)

    async def episodic_edge_save_bulk(
        self,
        _cls: Any,
        driver: Any,
        transaction: Any,
        episodic_edges: list[Any],
        batch_size: int = 100,
    ) -> None:
        d = spanner_driver(driver)
        models = [episodic_edge_from_bulk_row(row) for row in episodic_edges]
        await d.episodic_edge_ops.save_bulk(d, models, tx=transaction, batch_size=batch_size)

    async def episodic_node_delete_by_group_id(
        self,
        _cls: Any,
        driver: Any,
        group_id: str,
        batch_size: int = 100,
    ) -> None:
        d = spanner_driver(driver)
        await d.episode_node_ops.delete_by_group_id(d, group_id, batch_size=batch_size)

    async def episodic_node_delete_by_uuids(
        self,
        _cls: Any,
        driver: Any,
        uuids: list[str],
        group_id: str | None = None,
        batch_size: int = 100,
    ) -> None:
        d = spanner_driver(driver)
        uuids = await _scoped_uuids(d, 'Episodic', 'uuid', uuids, group_id)
        if uuids:
            await d.episode_node_ops.delete_by_uuids(d, uuids, batch_size=batch_size)

    # -----------------------
    # EpisodicNode: Read
    # -----------------------

    async def episodic_node_get_by_uuid(self, _cls: Any, driver: Any, uuid: str) -> EpisodicNode:
        d = spanner_driver(driver)
        return await d.episode_node_ops.get_by_uuid(d, uuid)

    async def episodic_node_get_by_uuids(
        self, _cls: Any, driver: Any, uuids: list[str]
    ) -> list[EpisodicNode]:
        d = spanner_driver(driver)
        return await d.episode_node_ops.get_by_uuids(d, uuids)

    async def episodic_node_get_by_group_ids(
        self,
        _cls: Any,
        driver: Any,
        group_ids: list[str],
        limit: int | None = None,
        uuid_cursor: str | None = None,
    ) -> list[EpisodicNode]:
        d = spanner_driver(driver)
        return await d.episode_node_ops.get_by_group_ids(d, group_ids, limit, uuid_cursor)

    async def retrieve_episodes(
        self,
        driver: Any,
        reference_time: Any,
        last_n: int = 3,
        group_ids: list[str] | None = None,
        source: Any | None = None,
        saga: str | None = None,
    ) -> list[EpisodicNode]:
        """Chronological order (oldest first), as the legacy contract and the Cypher fallback
        return; the ops method returns newest first, so the result is sorted here."""
        d = spanner_driver(driver)
        episodes = await d.episode_node_ops.retrieve_episodes(
            d, reference_time, last_n, group_ids, _source_value(source), saga
        )
        return sorted(episodes, key=lambda episode: (episode.valid_at, episode.created_at))

    # -----------------------
    # CommunityNode: Save/Delete
    # -----------------------

    async def community_node_save(self, node: Any, driver: Any) -> None:
        d = spanner_driver(driver)
        await d.community_node_ops.save(d, node)

    async def community_node_delete(self, node: Any, driver: Any) -> None:
        d = spanner_driver(driver)
        await d.community_node_ops.delete(d, node)

    async def community_node_save_bulk(
        self,
        _cls: Any,
        driver: Any,
        transaction: Any,
        nodes: list[Any],
        batch_size: int = 100,
    ) -> None:
        d = spanner_driver(driver)
        models = _as_models(CommunityNode, nodes)
        await d.community_node_ops.save_bulk(d, models, tx=transaction, batch_size=batch_size)

    async def community_node_delete_by_group_id(
        self,
        _cls: Any,
        driver: Any,
        group_id: str,
        batch_size: int = 100,
    ) -> None:
        d = spanner_driver(driver)
        await d.community_node_ops.delete_by_group_id(d, group_id, batch_size=batch_size)

    async def community_node_delete_by_uuids(
        self,
        _cls: Any,
        driver: Any,
        uuids: list[str],
        group_id: str | None = None,
        batch_size: int = 100,
    ) -> None:
        d = spanner_driver(driver)
        uuids = await _scoped_uuids(d, 'Community', 'uuid', uuids, group_id)
        if uuids:
            await d.community_node_ops.delete_by_uuids(d, uuids, batch_size=batch_size)

    # -----------------------
    # CommunityNode: Read
    # -----------------------

    async def community_node_get_by_uuid(self, _cls: Any, driver: Any, uuid: str) -> CommunityNode:
        d = spanner_driver(driver)
        return await d.community_node_ops.get_by_uuid(d, uuid)

    async def community_node_get_by_uuids(
        self, _cls: Any, driver: Any, uuids: list[str]
    ) -> list[CommunityNode]:
        d = spanner_driver(driver)
        return await d.community_node_ops.get_by_uuids(d, uuids)

    async def community_node_get_by_group_ids(
        self,
        _cls: Any,
        driver: Any,
        group_ids: list[str],
        limit: int | None = None,
        uuid_cursor: str | None = None,
    ) -> list[CommunityNode]:
        d = spanner_driver(driver)
        return await d.community_node_ops.get_by_group_ids(d, group_ids, limit, uuid_cursor)

    # -----------------------
    # SagaNode: Save/Delete
    # -----------------------

    async def saga_node_save(self, node: Any, driver: Any) -> None:
        d = spanner_driver(driver)
        await d.saga_node_ops.save(d, node)

    async def saga_node_delete(self, node: Any, driver: Any) -> None:
        d = spanner_driver(driver)
        await d.saga_node_ops.delete(d, node)

    async def saga_node_save_bulk(
        self,
        _cls: Any,
        driver: Any,
        transaction: Any,
        nodes: list[Any],
        batch_size: int = 100,
    ) -> None:
        d = spanner_driver(driver)
        models = _as_models(SagaNode, nodes)
        await d.saga_node_ops.save_bulk(d, models, tx=transaction, batch_size=batch_size)

    async def saga_node_delete_by_group_id(
        self,
        _cls: Any,
        driver: Any,
        group_id: str,
        batch_size: int = 100,
    ) -> None:
        d = spanner_driver(driver)
        await d.saga_node_ops.delete_by_group_id(d, group_id, batch_size=batch_size)

    async def saga_node_delete_by_uuids(
        self,
        _cls: Any,
        driver: Any,
        uuids: list[str],
        group_id: str | None = None,
        batch_size: int = 100,
    ) -> None:
        d = spanner_driver(driver)
        uuids = await _scoped_uuids(d, 'Saga', 'uuid', uuids, group_id)
        if uuids:
            await d.saga_node_ops.delete_by_uuids(d, uuids, batch_size=batch_size)

    # -----------------------
    # SagaNode: Read
    # -----------------------

    async def saga_node_get_by_uuid(self, _cls: Any, driver: Any, uuid: str) -> SagaNode:
        d = spanner_driver(driver)
        return await d.saga_node_ops.get_by_uuid(d, uuid)

    async def saga_node_get_by_uuids(
        self, _cls: Any, driver: Any, uuids: list[str]
    ) -> list[SagaNode]:
        d = spanner_driver(driver)
        return await d.saga_node_ops.get_by_uuids(d, uuids)

    async def saga_node_get_by_group_ids(
        self,
        _cls: Any,
        driver: Any,
        group_ids: list[str],
        limit: int | None = None,
        uuid_cursor: str | None = None,
    ) -> list[SagaNode]:
        d = spanner_driver(driver)
        return await d.saga_node_ops.get_by_group_ids(d, group_ids, limit, uuid_cursor)

    # -----------------------
    # Saga: Queries
    # -----------------------

    async def saga_get_previous_episode_uuid(
        self,
        driver: Any,
        saga_uuid: str,
        current_episode_uuid: str,
    ) -> str | None:
        """Latest episode of the saga by ``valid_at`` then ``created_at`` (as the Cypher
        fallback in ``graphiti.py``), excluding ``current_episode_uuid``."""
        d = spanner_driver(driver)
        records = await d.read(
            'SELECT e.uuid AS uuid FROM HasEpisode AS h '
            'JOIN Episodic AS e ON e.uuid = h.target_node_uuid '
            'WHERE h.uuid = @saga_uuid AND e.uuid != @current_episode_uuid '
            'ORDER BY e.valid_at DESC, e.created_at DESC, e.uuid DESC LIMIT 1',
            {'saga_uuid': saga_uuid, 'current_episode_uuid': current_episode_uuid},
        )
        return records[0]['uuid'] if records else None

    async def saga_get_episode_contents(
        self,
        driver: Any,
        saga_uuid: str,
        since: Any | None = None,
        limit: int = 200,
    ) -> list[tuple[str, Any, Any]]:
        """``(content, valid_at, created_at)`` of the saga's episodes in ingestion order
        (``created_at``, then ``valid_at``), only those ingested after ``since`` when given.
        Episodes with empty content are skipped, as in the Cypher fallback."""
        sql = (
            'SELECT e.content AS content, e.valid_at AS valid_at, e.created_at AS created_at '
            'FROM HasEpisode AS h JOIN Episodic AS e ON e.uuid = h.target_node_uuid '
            'WHERE h.uuid = @saga_uuid'
        )
        params: dict[str, Any] = {'saga_uuid': saga_uuid}
        since_at = parse_db_date(since)
        if since_at is not None:
            sql += ' AND e.created_at > @since'
            params['since'] = since_at
        sql += ' ORDER BY e.created_at ASC, e.valid_at ASC, e.uuid ASC'
        if limit is not None:
            sql += ' LIMIT @limit'
            params['limit'] = limit
        d = spanner_driver(driver)
        records = await d.read(sql, params)
        return [
            (record['content'], record['valid_at'], record['created_at'])
            for record in records
            if record['content'] and record['created_at'] is not None
        ]

    # -----------------
    # Edge: Save/Delete
    # -----------------

    async def edge_save(self, edge: Any, driver: Any) -> None:
        d = spanner_driver(driver)
        await d.entity_edge_ops.save(d, edge)

    async def edge_delete(self, edge: Any, driver: Any) -> None:
        """``Edge.delete`` is inherited by entity, episodic and community edges."""
        d = spanner_driver(driver)
        if isinstance(edge, EntityEdge):
            await d.entity_edge_ops.delete(d, edge)
        elif isinstance(edge, EpisodicEdge):
            await d.episodic_edge_ops.delete(d, edge)
        elif isinstance(edge, CommunityEdge):
            await d.community_edge_ops.delete(d, edge)
        elif isinstance(edge, HasEpisodeEdge):
            await d.has_episode_edge_ops.delete(d, edge)
        elif isinstance(edge, NextEpisodeEdge):
            await d.next_episode_edge_ops.delete(d, edge)
        else:
            await self.edge_delete_by_uuids(None, d, [edge.uuid])

    async def edge_save_bulk(
        self,
        _cls: Any,
        driver: Any,
        transaction: Any,
        edges: list[Any],
        batch_size: int = 100,
    ) -> None:
        d = spanner_driver(driver)
        models = [entity_edge_from_bulk_row(row) for row in edges]
        await d.entity_edge_ops.save_bulk(d, models, tx=transaction, batch_size=batch_size)

    async def edge_delete_by_uuids(
        self,
        _cls: Any,
        driver: Any,
        uuids: list[str],
        group_id: str | None = None,
    ) -> None:
        """Generic over edge kinds (``Edge.delete_by_uuids``): finds which edge table holds
        each uuid in one read, then deletes through the matching ops class."""
        if not uuids:
            return
        d = spanner_driver(driver)
        located = await _uuids_by_table(d, _EDGE_TABLES, 'edge_uuid', uuids, group_id)
        deleters = _edge_deleters(d)
        for table, found in located.items():
            if found:
                await deleters[table](d, found)

    # -----------------
    # Edge: Read
    # -----------------

    async def edge_get_by_uuid(self, _cls: Any, driver: Any, uuid: str) -> EntityEdge:
        d = spanner_driver(driver)
        return await d.entity_edge_ops.get_by_uuid(d, uuid)

    async def edge_get_by_uuids(self, _cls: Any, driver: Any, uuids: list[str]) -> list[EntityEdge]:
        d = spanner_driver(driver)
        return await d.entity_edge_ops.get_by_uuids(d, uuids)

    async def edge_get_by_group_ids(
        self,
        _cls: Any,
        driver: Any,
        group_ids: list[str],
        limit: int | None = None,
        uuid_cursor: str | None = None,
    ) -> list[EntityEdge]:
        d = spanner_driver(driver)
        return await d.entity_edge_ops.get_by_group_ids(d, group_ids, limit, uuid_cursor)

    # -----------------
    # Edge: Embeddings (load)
    # -----------------

    async def edge_load_embeddings(self, edge: Any, driver: Any) -> None:
        d = spanner_driver(driver)
        await d.entity_edge_ops.load_embeddings(d, edge)

    async def edge_load_embeddings_bulk(
        self,
        driver: Any,
        edges: list[Any],
        batch_size: int = 100,
    ) -> dict[str, list[float]]:
        """``{uuid: fact_embedding}``; the edges themselves are not modified."""
        d = spanner_driver(driver)
        return await _embeddings_by_uuid(
            d, 'RelatesTo', 'edge_uuid', 'fact_embedding', (edge.uuid for edge in edges), batch_size
        )

    # ---------------------------
    # EpisodicEdge: Save/Delete
    # ---------------------------

    async def episodic_edge_save(self, edge: Any, driver: Any) -> None:
        d = spanner_driver(driver)
        await d.episodic_edge_ops.save(d, edge)

    async def episodic_edge_delete(self, edge: Any, driver: Any) -> None:
        d = spanner_driver(driver)
        await d.episodic_edge_ops.delete(d, edge)

    async def episodic_edge_delete_by_uuids(
        self,
        _cls: Any,
        driver: Any,
        uuids: list[str],
        group_id: str | None = None,
    ) -> None:
        d = spanner_driver(driver)
        uuids = await _scoped_uuids(d, 'Mentions', 'edge_uuid', uuids, group_id)
        if uuids:
            await d.episodic_edge_ops.delete_by_uuids(d, uuids)

    # ---------------------------
    # EpisodicEdge: Read
    # ---------------------------

    async def episodic_edge_get_by_uuid(self, _cls: Any, driver: Any, uuid: str) -> EpisodicEdge:
        d = spanner_driver(driver)
        return await d.episodic_edge_ops.get_by_uuid(d, uuid)

    async def episodic_edge_get_by_uuids(
        self, _cls: Any, driver: Any, uuids: list[str]
    ) -> list[EpisodicEdge]:
        """Raises ``EdgeNotFoundError`` when nothing matches, like the Cypher fallback of
        ``EpisodicEdge.get_by_uuids`` (the only edge kind that does)."""
        d = spanner_driver(driver)
        edges = await d.episodic_edge_ops.get_by_uuids(d, uuids)
        if not edges and uuids:
            raise EdgeNotFoundError(uuids[0])
        return edges

    async def episodic_edge_get_by_group_ids(
        self,
        _cls: Any,
        driver: Any,
        group_ids: list[str],
        limit: int | None = None,
        uuid_cursor: str | None = None,
    ) -> list[EpisodicEdge]:
        d = spanner_driver(driver)
        return await d.episodic_edge_ops.get_by_group_ids(d, group_ids, limit, uuid_cursor)

    # ---------------------------
    # CommunityEdge: Save/Delete
    # ---------------------------

    async def community_edge_save(self, edge: Any, driver: Any) -> None:
        d = spanner_driver(driver)
        await d.community_edge_ops.save(d, edge)

    async def community_edge_delete(self, edge: Any, driver: Any) -> None:
        d = spanner_driver(driver)
        await d.community_edge_ops.delete(d, edge)

    async def community_edge_delete_by_uuids(
        self,
        _cls: Any,
        driver: Any,
        uuids: list[str],
        group_id: str | None = None,
    ) -> None:
        d = spanner_driver(driver)
        uuids = await _scoped_uuids(d, 'HasMember', 'edge_uuid', uuids, group_id)
        if uuids:
            await d.community_edge_ops.delete_by_uuids(d, uuids)

    # ---------------------------
    # CommunityEdge: Read
    # ---------------------------

    async def community_edge_get_by_uuid(self, _cls: Any, driver: Any, uuid: str) -> CommunityEdge:
        d = spanner_driver(driver)
        return await d.community_edge_ops.get_by_uuid(d, uuid)

    async def community_edge_get_by_uuids(
        self, _cls: Any, driver: Any, uuids: list[str]
    ) -> list[CommunityEdge]:
        d = spanner_driver(driver)
        return await d.community_edge_ops.get_by_uuids(d, uuids)

    async def community_edge_get_by_group_ids(
        self,
        _cls: Any,
        driver: Any,
        group_ids: list[str],
        limit: int | None = None,
        uuid_cursor: str | None = None,
    ) -> list[CommunityEdge]:
        d = spanner_driver(driver)
        return await d.community_edge_ops.get_by_group_ids(d, group_ids, limit, uuid_cursor)

    # ---------------------------
    # HasEpisodeEdge: Save/Delete
    # ---------------------------

    async def has_episode_edge_save(self, edge: Any, driver: Any) -> None:
        d = spanner_driver(driver)
        await d.has_episode_edge_ops.save(d, edge)

    async def has_episode_edge_delete(self, edge: Any, driver: Any) -> None:
        d = spanner_driver(driver)
        await d.has_episode_edge_ops.delete(d, edge)

    async def has_episode_edge_save_bulk(
        self,
        _cls: Any,
        driver: Any,
        transaction: Any,
        edges: list[Any],
        batch_size: int = 100,
    ) -> None:
        d = spanner_driver(driver)
        models = _as_models(HasEpisodeEdge, edges)
        await d.has_episode_edge_ops.save_bulk(d, models, tx=transaction, batch_size=batch_size)

    async def has_episode_edge_delete_by_uuids(
        self,
        _cls: Any,
        driver: Any,
        uuids: list[str],
        group_id: str | None = None,
    ) -> None:
        d = spanner_driver(driver)
        uuids = await _scoped_uuids(d, 'HasEpisode', 'edge_uuid', uuids, group_id)
        if uuids:
            await d.has_episode_edge_ops.delete_by_uuids(d, uuids)

    # ---------------------------
    # HasEpisodeEdge: Read
    # ---------------------------

    async def has_episode_edge_get_by_uuid(
        self, _cls: Any, driver: Any, uuid: str
    ) -> HasEpisodeEdge:
        d = spanner_driver(driver)
        return await d.has_episode_edge_ops.get_by_uuid(d, uuid)

    async def has_episode_edge_get_by_uuids(
        self, _cls: Any, driver: Any, uuids: list[str]
    ) -> list[HasEpisodeEdge]:
        d = spanner_driver(driver)
        return await d.has_episode_edge_ops.get_by_uuids(d, uuids)

    async def has_episode_edge_get_by_group_ids(
        self,
        _cls: Any,
        driver: Any,
        group_ids: list[str],
        limit: int | None = None,
        uuid_cursor: str | None = None,
    ) -> list[HasEpisodeEdge]:
        d = spanner_driver(driver)
        return await d.has_episode_edge_ops.get_by_group_ids(d, group_ids, limit, uuid_cursor)

    # ----------------------------
    # NextEpisodeEdge: Save/Delete
    # ----------------------------

    async def next_episode_edge_save(self, edge: Any, driver: Any) -> None:
        d = spanner_driver(driver)
        await d.next_episode_edge_ops.save(d, edge)

    async def next_episode_edge_delete(self, edge: Any, driver: Any) -> None:
        d = spanner_driver(driver)
        await d.next_episode_edge_ops.delete(d, edge)

    async def next_episode_edge_save_bulk(
        self,
        _cls: Any,
        driver: Any,
        transaction: Any,
        edges: list[Any],
        batch_size: int = 100,
    ) -> None:
        d = spanner_driver(driver)
        models = _as_models(NextEpisodeEdge, edges)
        await d.next_episode_edge_ops.save_bulk(d, models, tx=transaction, batch_size=batch_size)

    async def next_episode_edge_delete_by_uuids(
        self,
        _cls: Any,
        driver: Any,
        uuids: list[str],
        group_id: str | None = None,
    ) -> None:
        d = spanner_driver(driver)
        uuids = await _scoped_uuids(d, 'NextEpisode', 'edge_uuid', uuids, group_id)
        if uuids:
            await d.next_episode_edge_ops.delete_by_uuids(d, uuids)

    # ----------------------------
    # NextEpisodeEdge: Read
    # ----------------------------

    async def next_episode_edge_get_by_uuid(
        self, _cls: Any, driver: Any, uuid: str
    ) -> NextEpisodeEdge:
        d = spanner_driver(driver)
        return await d.next_episode_edge_ops.get_by_uuid(d, uuid)

    async def next_episode_edge_get_by_uuids(
        self, _cls: Any, driver: Any, uuids: list[str]
    ) -> list[NextEpisodeEdge]:
        d = spanner_driver(driver)
        return await d.next_episode_edge_ops.get_by_uuids(d, uuids)

    async def next_episode_edge_get_by_group_ids(
        self,
        _cls: Any,
        driver: Any,
        group_ids: list[str],
        limit: int | None = None,
        uuid_cursor: str | None = None,
    ) -> list[NextEpisodeEdge]:
        d = spanner_driver(driver)
        return await d.next_episode_edge_ops.get_by_group_ids(d, group_ids, limit, uuid_cursor)

    # -----------------
    # Search
    # -----------------

    async def get_mentioned_nodes(
        self,
        driver: Any,
        episodes: list[Any],
    ) -> list[EntityNode]:
        d = spanner_driver(driver)
        return await d.graph_ops.get_mentioned_nodes(d, episodes)

    async def get_communities_by_nodes(
        self,
        driver: Any,
        nodes: list[Any],
    ) -> list[CommunityNode]:
        d = spanner_driver(driver)
        return await d.graph_ops.get_communities_by_nodes(d, nodes)

    # -----------------
    # Maintenance
    # -----------------

    async def clear_data(
        self,
        driver: Any,
        group_ids: list[str] | None = None,
    ) -> None:
        d = spanner_driver(driver)
        await d.graph_ops.clear_data(d, group_ids)

    async def get_community_clusters(
        self,
        driver: Any,
        group_ids: list[str] | None,
    ) -> list[list[Any]]:
        d = spanner_driver(driver)
        return await d.graph_ops.get_community_clusters(d, group_ids)

    async def remove_communities(
        self,
        driver: Any,
        group_ids: list[str] | None = None,
    ) -> None:
        d = spanner_driver(driver)
        await d.graph_ops.remove_communities(d, group_ids=group_ids)

    async def determine_entity_community(
        self,
        driver: Any,
        entity: Any,
    ) -> tuple[CommunityNode | None, bool]:
        """``(community, is_new)`` as in ``community_operations.determine_entity_community``.

        Implemented here because ``GraphMaintenanceOperations.determine_entity_community``
        returns ``None``. An existing membership wins (``is_new=False``); otherwise the
        community with the most HAS_MEMBER links to the entity's RELATES_TO neighbours (either
        direction, one count per edge pair as in the Cypher fallback; ties broken by uuid).
        """
        d = spanner_driver(driver)
        community_columns = node_select_columns('Community', 'c')
        params = {'entity_uuid': entity.uuid}
        records = await d.read(
            f'SELECT {community_columns} FROM HasMember AS m '
            'JOIN Community AS c ON c.uuid = m.uuid '
            'WHERE m.target_node_uuid = @entity_uuid ORDER BY c.uuid LIMIT 1',
            params,
        )
        if records:
            return community_node_from_row(records[0]), False

        records = await d.read(
            f'SELECT {community_columns} FROM ('
            'SELECT m.uuid AS community_uuid, COUNT(*) AS member_count FROM ('
            'SELECT target_node_uuid AS neighbor_uuid FROM RelatesTo WHERE uuid = @entity_uuid '
            'UNION ALL '
            'SELECT uuid AS neighbor_uuid FROM RelatesTo WHERE target_node_uuid = @entity_uuid'
            ') AS n JOIN HasMember AS m ON m.target_node_uuid = n.neighbor_uuid '
            'GROUP BY m.uuid'
            ') AS counts JOIN Community AS c ON c.uuid = counts.community_uuid '
            'ORDER BY counts.member_count DESC, c.uuid LIMIT 1',
            params,
        )
        if records:
            return community_node_from_row(records[0]), True
        return None, False

    # -----------------
    # Additional Node Operations
    # -----------------

    async def episodic_node_get_by_entity_node_uuid(
        self,
        _cls: Any,
        driver: Any,
        entity_node_uuid: str,
    ) -> list[EpisodicNode]:
        d = spanner_driver(driver)
        return await d.episode_node_ops.get_by_entity_node_uuid(d, entity_node_uuid)

    async def community_node_load_name_embedding(
        self,
        node: Any,
        driver: Any,
    ) -> None:
        d = spanner_driver(driver)
        await d.community_node_ops.load_name_embedding(d, node)

    # -----------------
    # Additional Edge Operations
    # -----------------

    async def edge_get_between_nodes(
        self,
        _cls: Any,
        driver: Any,
        source_node_uuid: str,
        target_node_uuid: str,
    ) -> list[EntityEdge]:
        d = spanner_driver(driver)
        return await d.entity_edge_ops.get_between_nodes(d, source_node_uuid, target_node_uuid)

    async def edge_get_by_node_uuid(
        self,
        _cls: Any,
        driver: Any,
        node_uuid: str,
    ) -> list[EntityEdge]:
        d = spanner_driver(driver)
        return await d.entity_edge_ops.get_by_node_uuid(d, node_uuid)


# ---------------------------------------------------------------------------
# SearchInterface
# ---------------------------------------------------------------------------


class SpannerLegacySearchInterface(SearchInterface):
    """Spanner adapter for the legacy ``driver.search_interface`` hook used by
    ``search/search_utils.py``.

    The nine searches delegate to ``driver.search_ops``. The two rerankers run their own SQL
    because the ops rerankers return nodes, while the legacy contract is ``(uuids, scores)``;
    scoring mirrors the Cypher fallbacks in ``search_utils``.
    """

    async def edge_fulltext_search(
        self,
        driver: Any,
        query: str,
        search_filter: Any,
        group_ids: list[str] | None = None,
        limit: int = 100,
    ) -> list[EntityEdge]:
        d = spanner_driver(driver)
        return await d.search_ops.edge_fulltext_search(d, query, search_filter, group_ids, limit)

    async def edge_similarity_search(
        self,
        driver: Any,
        search_vector: list[float],
        source_node_uuid: str | None,
        target_node_uuid: str | None,
        search_filter: Any,
        group_ids: list[str] | None = None,
        limit: int = 100,
        min_score: float = 0.7,
    ) -> list[EntityEdge]:
        d = spanner_driver(driver)
        return await d.search_ops.edge_similarity_search(
            d,
            search_vector,
            source_node_uuid,
            target_node_uuid,
            search_filter,
            group_ids,
            limit,
            min_score,
        )

    async def node_fulltext_search(
        self,
        driver: Any,
        query: str,
        search_filter: Any,
        group_ids: list[str] | None = None,
        limit: int = 100,
    ) -> list[EntityNode]:
        d = spanner_driver(driver)
        return await d.search_ops.node_fulltext_search(d, query, search_filter, group_ids, limit)

    async def node_similarity_search(
        self,
        driver: Any,
        search_vector: list[float],
        search_filter: Any,
        group_ids: list[str] | None = None,
        limit: int = 100,
        min_score: float = 0.7,
    ) -> list[EntityNode]:
        d = spanner_driver(driver)
        return await d.search_ops.node_similarity_search(
            d, search_vector, search_filter, group_ids, limit, min_score
        )

    async def episode_fulltext_search(
        self,
        driver: Any,
        query: str,
        search_filter: Any,
        group_ids: list[str] | None = None,
        limit: int = 100,
    ) -> list[EpisodicNode]:
        d = spanner_driver(driver)
        return await d.search_ops.episode_fulltext_search(d, query, search_filter, group_ids, limit)

    async def edge_bfs_search(
        self,
        driver: Any,
        bfs_origin_node_uuids: list[str] | None,
        bfs_max_depth: int,
        search_filter: Any,
        group_ids: list[str] | None = None,
        limit: int = 100,
    ) -> list[EntityEdge]:
        if not bfs_origin_node_uuids:
            return []
        d = spanner_driver(driver)
        return await d.search_ops.edge_bfs_search(
            d, bfs_origin_node_uuids, bfs_max_depth, search_filter, group_ids, limit
        )

    async def node_bfs_search(
        self,
        driver: Any,
        bfs_origin_node_uuids: list[str] | None,
        search_filter: Any,
        bfs_max_depth: int,
        group_ids: list[str] | None = None,
        limit: int = 100,
    ) -> list[EntityNode]:
        if not bfs_origin_node_uuids or bfs_max_depth < 1:
            return []
        d = spanner_driver(driver)
        return await d.search_ops.node_bfs_search(
            d, bfs_origin_node_uuids, search_filter, bfs_max_depth, group_ids, limit
        )

    async def community_fulltext_search(
        self,
        driver: Any,
        query: str,
        group_ids: list[str] | None = None,
        limit: int = 100,
    ) -> list[CommunityNode]:
        d = spanner_driver(driver)
        return await d.search_ops.community_fulltext_search(d, query, group_ids, limit)

    async def community_similarity_search(
        self,
        driver: Any,
        search_vector: list[float],
        group_ids: list[str] | None = None,
        limit: int = 100,
        min_score: float = 0.6,
    ) -> list[CommunityNode]:
        d = spanner_driver(driver)
        return await d.search_ops.community_similarity_search(
            d, search_vector, group_ids, limit, min_score
        )

    async def get_embeddings_for_communities(
        self,
        driver: Any,
        communities: list[Any],
    ) -> dict[str, list[float]]:
        d = spanner_driver(driver)
        return await _embeddings_by_uuid(
            d,
            'Community',
            'uuid',
            'name_embedding',
            (community.uuid for community in communities),
            _READ_CHUNK,
        )

    async def node_distance_reranker(
        self,
        driver: Any,
        node_uuids: list[str],
        center_node_uuid: str,
        min_score: float = 0,
    ) -> tuple[list[str], list[float]]:
        """Score 1 for nodes with a RELATES_TO edge to the center (either direction), 0 for
        the rest, 10 for the center itself; same ordering and ``min_score`` filter as
        ``search_utils.node_distance_reranker``."""
        filtered_uuids = [uuid for uuid in node_uuids if uuid != center_node_uuid]
        scores: dict[str, float] = {center_node_uuid: 0.0}
        if filtered_uuids:
            d = spanner_driver(driver)
            records = await d.read(
                'SELECT target_node_uuid AS uuid FROM RelatesTo '
                'WHERE uuid = @center_uuid AND target_node_uuid IN UNNEST(@uuids) '
                'UNION DISTINCT '
                'SELECT uuid AS uuid FROM RelatesTo '
                'WHERE target_node_uuid = @center_uuid AND uuid IN UNNEST(@uuids)',
                {'center_uuid': center_node_uuid, 'uuids': list(dict.fromkeys(filtered_uuids))},
                {'uuids': STRING_ARRAY},
            )
            for record in records:
                scores[record['uuid']] = 1
        for uuid in filtered_uuids:
            if uuid not in scores:
                scores[uuid] = float('inf')

        filtered_uuids.sort(key=lambda cur_uuid: scores[cur_uuid])
        if center_node_uuid in node_uuids:
            scores[center_node_uuid] = 0.1
            filtered_uuids = [center_node_uuid] + filtered_uuids

        return [uuid for uuid in filtered_uuids if (1 / scores[uuid]) >= min_score], [
            1 / scores[uuid] for uuid in filtered_uuids if (1 / scores[uuid]) >= min_score
        ]

    async def episode_mentions_reranker(
        self,
        driver: Any,
        node_uuids: list[list[str]],
        min_score: float = 0,
    ) -> tuple[list[str], list[float]]:
        """RRF over the input lists, then MENTIONS edge count per node, descending; same as
        ``search_utils.episode_mentions_reranker``."""
        # Imported here: search_utils is heavy and is not needed to construct the driver.
        from graphiti_core.search.search_utils import rrf

        sorted_uuids, _ = rrf(node_uuids)
        scores: dict[str, float] = {}
        if sorted_uuids:
            d = spanner_driver(driver)
            for chunk in _chunks(sorted_uuids, _READ_CHUNK):
                records = await d.read(
                    'SELECT target_node_uuid AS uuid, COUNT(*) AS score FROM Mentions '
                    'WHERE target_node_uuid IN UNNEST(@uuids) GROUP BY target_node_uuid',
                    {'uuids': chunk},
                    {'uuids': STRING_ARRAY},
                )
                for record in records:
                    scores[record['uuid']] = record['score']
        for uuid in sorted_uuids:
            if uuid not in scores:
                scores[uuid] = 0

        sorted_uuids.sort(key=lambda cur_uuid: scores[cur_uuid], reverse=True)
        return [uuid for uuid in sorted_uuids if scores[uuid] >= min_score], [
            scores[uuid] for uuid in sorted_uuids if scores[uuid] >= min_score
        ]

    # ---------- SEARCH FILTERS (sync) ----------

    # These get no driver argument, so they use a fresh (stateless) Spanner search ops object.

    def build_node_search_filters(self, search_filters: Any) -> Any:
        return SpannerSearchOperations().build_node_search_filters(search_filters)

    def build_edge_search_filters(self, search_filters: Any) -> Any:
        return SpannerSearchOperations().build_edge_search_filters(search_filters)


__all__ = [
    'ENTITY_EDGE_BULK_KEYS',
    'ENTITY_NODE_BULK_KEYS',
    'SpannerLegacyGraphOperations',
    'SpannerLegacySearchInterface',
    'entity_edge_from_bulk_row',
    'entity_node_from_bulk_row',
    'episodic_edge_from_bulk_row',
    'episodic_node_from_bulk_row',
]
