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

from __future__ import annotations

import hashlib
import json
from array import array
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Any, cast

from graphiti_core.driver.query_executor import QueryExecutor, Transaction
from graphiti_core.edges import EntityEdge
from graphiti_core.errors import NodeNotFoundError
from graphiti_core.nodes import EntityNode, EpisodicNode

from ..codecs import edge_from_record, entity_node_from_record, episode_node_from_record
from ..errors import EpisodeRemovalConflict, EpisodeSummaryRebuildError
from ..sql.episode_removal import (
    GET_REMOVAL_EDGES,
    GET_REMOVAL_NODES,
    LOCK_REMOVAL_EDGES,
    LOCK_REMOVAL_NODES,
)
from .base import composite_transaction, resolve_group_scope

SummaryRebuilder = Callable[[EntityNode, list[EpisodicNode]], Awaitable[None]]


def _json_value(value: object) -> object:
    if isinstance(value, datetime):
        return value.isoformat(timespec='microseconds')
    if isinstance(value, array):
        return value.tolist()
    raise TypeError(f'unsupported snapshot type: {type(value).__name__}')


@dataclass(frozen=True)
class RemovalSnapshot:
    nodes: list[dict[str, Any]]
    edges: list[dict[str, Any]]

    def signature(self) -> str:
        payload = json.dumps(
            [self.nodes, self.edges],
            default=_json_value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(',', ':'),
            allow_nan=False,
        )
        return hashlib.sha256(payload.encode()).hexdigest()


async def _snapshot(
    executor: QueryExecutor,
    group_id: str,
    tx: Transaction | None = None,
) -> RemovalSnapshot:
    if tx is None:
        nodes, _, _ = await executor.execute_query(GET_REMOVAL_NODES, group_id=group_id)
        edges, _, _ = await executor.execute_query(GET_REMOVAL_EDGES, group_id=group_id)
    else:
        nodes, _, _ = await tx.run(GET_REMOVAL_NODES, group_id=group_id)
        edges, _, _ = await tx.run(GET_REMOVAL_EDGES, group_id=group_id)
    return RemovalSnapshot(nodes=nodes, edges=edges)


async def remove_episode(
    executor: Any,
    episode_uuid: str,
    rebuild_summary: SummaryRebuilder,
) -> None:
    """Stage summaries first, then atomically remove one source from a stable group snapshot."""

    episode = await executor.episode_node_ops.get_by_uuid(executor, episode_uuid)
    group_id = resolve_group_scope(executor, episode.group_id)
    assert group_id is not None
    before = await _snapshot(executor, group_id)
    episodes = {
        str(row['uuid']): episode_node_from_record(row)
        for row in before.nodes
        if row['kind'] == 'EPISODIC'
    }
    if episode_uuid not in episodes:
        raise NodeNotFoundError(episode_uuid)
    episode = episodes[episode_uuid]
    if any(
        row['kind'] in {'HAS_EPISODE', 'NEXT_EPISODE'}
        and episode_uuid in (row['source_uuid'], row['target_uuid'])
        for row in before.edges
    ):
        raise EpisodeRemovalConflict('Saga-associated removal requires an approved reconciliation')
    entities = {
        str(row['uuid']): entity_node_from_record(row)
        for row in before.nodes
        if row['kind'] == 'ENTITY'
    }
    facts = [
        cast(EntityEdge, edge_from_record(row))
        for row in before.edges
        if row['kind'] == 'RELATES_TO'
    ]
    supported = [edge for edge in facts if episode_uuid in edge.episodes]
    affected = {
        str(row['target_uuid'])
        for row in before.edges
        if row['kind'] == 'MENTIONS' and row['source_uuid'] == episode_uuid
    }
    affected.update(
        uuid for edge in supported for uuid in (edge.source_node_uuid, edge.target_node_uuid)
    )
    if not affected.issubset(entities):
        raise EpisodeRemovalConflict('episode removal encountered an absent entity endpoint')

    surviving_facts = [
        edge for edge in facts if any(uuid != episode_uuid for uuid in edge.episodes)
    ]
    orphan_uuids: set[str] = set()
    summaries: dict[str, EntityNode] = {}
    for uuid in sorted(affected):
        incident = [
            edge for edge in facts if uuid in (edge.source_node_uuid, edge.target_node_uuid)
        ]
        if any(not edge.episodes for edge in incident):
            raise EpisodeRemovalConflict('episode removal requires attributed incident facts')
        source_uuids = {
            str(row['source_uuid'])
            for row in before.edges
            if row['kind'] == 'MENTIONS'
            and row['target_uuid'] == uuid
            and row['source_uuid'] != episode_uuid
        }
        source_uuids.update(
            source_uuid
            for edge in incident
            for source_uuid in edge.episodes
            if source_uuid != episode_uuid
        )
        if not source_uuids.issubset(episodes):
            raise EpisodeRemovalConflict('episode removal encountered missing surviving provenance')
        has_surviving_fact = any(
            uuid in (edge.source_node_uuid, edge.target_node_uuid) for edge in surviving_facts
        )
        if not source_uuids and not has_surviving_fact:
            orphan_uuids.add(uuid)
            continue
        sources = sorted(
            (episodes[source_uuid] for source_uuid in source_uuids),
            key=lambda source: (source.valid_at, source.created_at, source.uuid),
        )
        if any(not source.content.strip() for source in sources):
            raise EpisodeRemovalConflict('remaining source content is unavailable for rebuilding')
        prepared = entities[uuid].model_copy(deep=True)
        try:
            await rebuild_summary(prepared, sources)
            if not isinstance(prepared.summary, str):
                raise TypeError('prepared summary must be text')
        except Exception as error:
            raise EpisodeSummaryRebuildError(
                f'summary preparation failed for entity {uuid}; episode was not removed'
            ) from error
        summaries[uuid] = entities[uuid].model_copy(update={'summary': prepared.summary})

    async with composite_transaction(executor, None) as tx:
        await tx.run(LOCK_REMOVAL_NODES, group_id=group_id)
        await tx.run(LOCK_REMOVAL_EDGES, group_id=group_id)
        current = await _snapshot(executor, group_id, tx)
        if current.signature() != before.signature():
            raise EpisodeRemovalConflict('graph group changed during summary preparation; retry')
        for edge in supported:
            edge.episodes = [uuid for uuid in edge.episodes if uuid != episode_uuid]
            if edge.episodes:
                await executor.entity_edge_ops.save(executor, edge, tx=tx)
            else:
                await executor.entity_edge_ops.delete(executor, edge, tx=tx)
        await executor.episode_node_ops.delete(executor, episode, tx=tx)
        for uuid in sorted(orphan_uuids):
            await executor.entity_node_ops.delete(executor, entities[uuid], tx=tx)
        for uuid in sorted(summaries):
            await executor.entity_node_ops.save(executor, summaries[uuid], tx=tx)
