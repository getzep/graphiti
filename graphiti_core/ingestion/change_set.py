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

import zlib
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field

from graphiti_core.edges import EntityEdge, EpisodicEdge
from graphiti_core.nodes import EntityNode, EpisodicNode, SagaNode
from graphiti_core.utils.datetime_utils import ensure_utc


class AddEpisodesToEdge(BaseModel):
    """Add episode UUIDs to the `episodes` list of a stored edge."""

    op: Literal['add_episodes_to_edge'] = 'add_episodes_to_edge'
    edge_uuid: str
    episode_uuids: list[str]


class EndEdge(BaseModel):
    """Set the end dates of a stored edge."""

    op: Literal['end_edge'] = 'end_edge'
    edge_uuid: str
    invalid_at: datetime | None = None
    expired_at: datetime
    only_if_open: bool = False


class SetEdgeAttributes(BaseModel):
    """Set the attributes of a stored edge with a merge by key.

    `base_attributes` is the map that the analysis read. `attributes` is the map that
    the analysis computed.
    """

    op: Literal['set_edge_attributes'] = 'set_edge_attributes'
    edge_uuid: str
    base_attributes: dict[str, Any] = Field(default_factory=dict)
    attributes: dict[str, Any] = Field(default_factory=dict)


class PatchNode(BaseModel):
    """Overlay the labels and the attributes of an extracted node on a stored node."""

    op: Literal['patch_node'] = 'patch_node'
    node_uuid: str
    labels: list[str] = Field(default_factory=list)
    attributes: dict[str, Any] = Field(default_factory=dict)


EdgeOperation = AddEpisodesToEdge | EndEdge | SetEdgeAttributes
Operation = Annotated[
    AddEpisodesToEdge | EndEdge | SetEdgeAttributes | PatchNode,
    Field(discriminator='op'),
]


class SagaLink(BaseModel):
    """The saga records that the commit writes for one episode."""

    saga_uuid: str
    episode_uuid: str
    previous_episode_uuid: str | None = None
    has_episode_edge_uuid: str
    next_episode_edge_uuid: str | None = None
    created_at: datetime


class ChangeSet(BaseModel):
    """The records to create and the operations on stored records of one ingestion unit.

    A change set contains no copy of a stored node or of a stored edge. A reference to
    a stored record is its UUID.
    """

    group_id: str
    unit_id: str
    seq: int
    snapshot_seq: int
    analysis_started_at: float
    episodes: list[EpisodicNode] = Field(default_factory=list)
    create_nodes: list[EntityNode] = Field(default_factory=list)
    create_edges: list[EntityEdge] = Field(default_factory=list)
    episodic_edges: list[EpisodicEdge] = Field(default_factory=list)
    operations: list[Operation] = Field(default_factory=list)
    node_uuids: list[str] = Field(default_factory=list)
    node_first_episode: dict[str, str] = Field(default_factory=dict)
    seen_node_candidates: dict[str, list[str]] = Field(default_factory=dict)
    # The labels of each node candidate at the time when the analysis read it.
    seen_node_labels: dict[str, list[str]] = Field(default_factory=dict)
    seen_duplicate_candidates: dict[str, list[str]] = Field(default_factory=dict)
    seen_invalidation_candidates: dict[str, list[str]] = Field(default_factory=dict)
    # The created edges whose end date is the start of a contradiction that the
    # analysis saw. An end date that the episode gave is not in this list.
    edges_ended_by_candidate: list[str] = Field(default_factory=list)
    uuid_map: dict[str, str] = Field(default_factory=dict)
    saga_links: list[SagaLink] = Field(default_factory=list)
    # The saga node of `saga_links`, as the commit read it or made it. With
    # `create_saga`, the node is new and the apply of the commit writes it first.
    saga_node: SagaNode | None = None
    create_saga: bool = False

    def to_bytes(self) -> bytes:
        """Return the compressed JSON of the change set."""
        return zlib.compress(self.model_dump_json().encode())

    @classmethod
    def from_bytes(cls, payload: bytes) -> 'ChangeSet':
        """Return the change set of a payload that `to_bytes` made."""
        return cls.model_validate_json(zlib.decompress(payload))

    def patches_by_node(self) -> dict[str, list[PatchNode]]:
        """Return the `PatchNode` operations for each node UUID, in list order."""
        patches: dict[str, list[PatchNode]] = {}
        for operation in self.operations:
            if isinstance(operation, PatchNode):
                patches.setdefault(operation.node_uuid, []).append(operation)
        return patches

    def edge_operations(self) -> list[EdgeOperation]:
        """Return the operations on stored edges, in list order."""
        return [operation for operation in self.operations if not isinstance(operation, PatchNode)]


@dataclass
class _EdgeState:
    edge: EntityEdge
    episodes: list[str]
    attributes: dict[str, Any]
    invalid_at: datetime | None
    expired_at: datetime | None
    ended_as_candidate: bool = False


@dataclass
class ResolutionTrace:
    """Records what node resolution and edge resolution saw and changed.

    The resolution functions change stored records in memory. The trace keeps the
    candidates that each extracted item saw, and one operation for each change to a
    stored record.
    """

    seen_node_candidates: dict[str, list[str]] = field(default_factory=dict)
    seen_node_labels: dict[str, list[str]] = field(default_factory=dict)
    seen_duplicate_candidates: dict[str, list[str]] = field(default_factory=dict)
    seen_invalidation_candidates: dict[str, list[str]] = field(default_factory=dict)
    operations: list[Any] = field(default_factory=list)
    stored_node_uuids: set[str] = field(default_factory=set)
    # The edges whose end date is the start of a contradicted candidate.
    ended_by_candidate: set[str] = field(default_factory=set)
    # The number of model calls that node resolution made for the trace.
    node_llm_calls: int = 0
    _edge_states: dict[int, _EdgeState] = field(default_factory=dict)

    def record_node_candidates(self, node_uuid: str, candidates: list[EntityNode]) -> None:
        seen = self.seen_node_candidates.setdefault(node_uuid, [])
        for candidate in candidates:
            if candidate.uuid not in seen:
                seen.append(candidate.uuid)
            # The first read of a candidate is before a promotion of its labels in memory.
            self.seen_node_labels.setdefault(candidate.uuid, list(candidate.labels))

    def record_node_resolution(self, extracted_node: EntityNode, resolved_uuid: str) -> None:
        """Record that an extracted node resolved to a stored node."""
        if resolved_uuid == extracted_node.uuid:
            return
        self.stored_node_uuids.add(resolved_uuid)
        self.operations.append(
            PatchNode(
                node_uuid=resolved_uuid,
                labels=list(extracted_node.labels),
                attributes=deepcopy(extracted_node.attributes or {}),
            )
        )

    def record_edge_candidates(
        self,
        edge_uuid: str,
        duplicate_candidates: list[EntityEdge],
        invalidation_candidates: list[EntityEdge],
    ) -> None:
        self.seen_duplicate_candidates[edge_uuid] = [edge.uuid for edge in duplicate_candidates]
        self.seen_invalidation_candidates[edge_uuid] = [
            edge.uuid for edge in invalidation_candidates
        ]

    def watch_stored_edges(self, edges: list[EntityEdge]) -> None:
        """Keep the state of stored edge objects before resolution changes them."""
        for edge in edges:
            if id(edge) in self._edge_states:
                continue
            self._edge_states[id(edge)] = _EdgeState(
                edge=edge,
                episodes=list(edge.episodes),
                attributes=deepcopy(edge.attributes or {}),
                invalid_at=edge.invalid_at,
                expired_at=edge.expired_at,
            )

    def mark_ended_as_candidate(self, edges: list[EntityEdge]) -> None:
        """Mark stored edges that a contradiction ended (`only_if_open=False`)."""
        for edge in edges:
            state = self._edge_states.get(id(edge))
            if state is not None:
                state.ended_as_candidate = True

    def collect_edge_operations(self, edges: list[EntityEdge] | None = None) -> list[EdgeOperation]:
        """Make one operation for each change to a watched edge, and stop the watch.

        With `edges`, only the watch of those edge objects stops. Calls that run at the
        same time on one trace thus do not collect the edges of each other.
        """
        keys = list(self._edge_states) if edges is None else [id(edge) for edge in edges]
        operations: list[EdgeOperation] = []
        for key in keys:
            state = self._edge_states.pop(key, None)
            if state is not None:
                operations.extend(_diff_edge_state(state))
        self.operations.extend(operations)
        return operations


def _diff_edge_state(state: _EdgeState) -> list[EdgeOperation]:
    edge = state.edge
    operations: list[EdgeOperation] = []

    added_episodes = [uuid for uuid in edge.episodes if uuid not in state.episodes]
    if added_episodes:
        operations.append(AddEpisodesToEdge(edge_uuid=edge.uuid, episode_uuids=added_episodes))

    current_attributes = edge.attributes or {}
    if current_attributes != state.attributes:
        operations.append(
            SetEdgeAttributes(
                edge_uuid=edge.uuid,
                base_attributes=deepcopy(state.attributes),
                attributes=deepcopy(current_attributes),
            )
        )

    dates_changed = edge.invalid_at != state.invalid_at or edge.expired_at != state.expired_at
    if dates_changed and edge.expired_at is not None:
        operations.append(
            EndEdge(
                edge_uuid=edge.uuid,
                invalid_at=edge.invalid_at if edge.invalid_at != state.invalid_at else None,
                expired_at=edge.expired_at,
                only_if_open=not state.ended_as_candidate,
            )
        )
    return operations


def merge_edge_attributes(
    base: dict[str, Any], new: dict[str, Any], current: dict[str, Any]
) -> dict[str, Any]:
    """Merge three attribute maps by key.

    A key with the same state in `current` and in `base` gets the state of `new`.
    A key with a different state in `current` keeps the state of `current`.
    """
    missing = object()
    merged: dict[str, Any] = {}
    for key in [*current.keys(), *[k for k in new if k not in current]]:
        current_value = current.get(key, missing)
        unchanged = current_value == base.get(key, missing)
        value = new.get(key, missing) if unchanged else current_value
        if value is not missing:
            merged[key] = value
    return merged


def apply_edge_operation(edge: EntityEdge, operation: EdgeOperation) -> bool:
    """Apply one operation to the current state of a stored edge.

    Each rule gives the same record when it runs two times. Returns True when the
    operation changed the edge.
    """
    if isinstance(operation, AddEpisodesToEdge):
        added = [uuid for uuid in operation.episode_uuids if uuid not in edge.episodes]
        edge.episodes.extend(dict.fromkeys(added))
        return bool(added)

    if isinstance(operation, EndEdge):
        if operation.only_if_open and edge.expired_at is not None:
            return False
        changed = False
        if operation.invalid_at is not None:
            current_invalid_at = ensure_utc(edge.invalid_at)
            new_invalid_at = ensure_utc(operation.invalid_at)
            if current_invalid_at is None or (
                new_invalid_at is not None and new_invalid_at < current_invalid_at
            ):
                edge.invalid_at = operation.invalid_at
                changed = True
        if edge.expired_at is None:
            edge.expired_at = operation.expired_at
            changed = True
        return changed

    current = edge.attributes or {}
    merged = merge_edge_attributes(operation.base_attributes, operation.attributes, current)
    if merged == current:
        return False
    edge.attributes = merged
    return True
