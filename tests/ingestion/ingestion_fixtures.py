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

# Test doubles for concurrent ingestion: an in-memory graph, a recording tracer, and
# functions that make units and change sets with no model call.

from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import numpy as np

from graphiti_core.edges import EntityEdge, EpisodicEdge
from graphiti_core.errors import NodeNotFoundError
from graphiti_core.ingestion import phases
from graphiti_core.ingestion.change_set import ChangeSet
from graphiti_core.ingestion.config import ConcurrentIngestionConfig
from graphiti_core.ingestion.phases import AdmittedUnit
from graphiti_core.ingestion.state_store import InMemoryIngestionStateStore
from graphiti_core.nodes import EntityNode, EpisodeType, EpisodicNode
from graphiti_core.tracer import Tracer, TracerSpan
from graphiti_core.utils.bulk_utils import update_entity_edges_tx, update_entity_nodes_tx
from graphiti_core.utils.maintenance.dedup_helpers import (
    _identity_key_for_node,
    _normalize_string_exact,
)

GROUP = 'g'
T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
T1 = T0 + timedelta(days=10)
T2 = T0 + timedelta(days=20)

# Embeddings with three dimensions. Two vectors on different axes have no similarity.
X = [1.0, 0.0, 0.0]
Y = [0.0, 1.0, 0.0]
Z = [0.0, 0.0, 1.0]
NEAR_X = [0.9, 0.1, 0.0]


def cosine(a: list[float] | None, b: list[float] | None) -> float:
    if not a or not b:
        return 0.0
    left, right = np.asarray(a), np.asarray(b)
    return float(left @ right / (np.linalg.norm(left) * np.linalg.norm(right)))


def make_node(uuid: str, name: str, embedding: list[float] | None = None, **values) -> EntityNode:
    values.setdefault('labels', ['Entity'])
    return EntityNode(
        uuid=uuid, name=name, group_id=GROUP, created_at=T0, name_embedding=embedding or X, **values
    )


def make_edge(
    uuid: str,
    source: str,
    target: str,
    fact: str,
    embedding: list[float] | None = None,
    **values,
) -> EntityEdge:
    values.setdefault('name', 'RELATES')
    values.setdefault('episodes', [])
    return EntityEdge(
        uuid=uuid,
        group_id=GROUP,
        source_node_uuid=source,
        target_node_uuid=target,
        fact=fact,
        created_at=T0,
        fact_embedding=embedding or X,
        **values,
    )


def make_episode(uuid: str = 'ep1', valid_at: datetime = T0, content: str = 'content'):
    return EpisodicNode(
        uuid=uuid,
        name=uuid,
        group_id=GROUP,
        source=EpisodeType.text,
        source_description='test',
        content=content,
        valid_at=valid_at,
        created_at=valid_at,
    )


class FakeClock:
    def __init__(self, now: float = 1000.0):
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class RecordingSpan(TracerSpan):
    def __init__(self, name: str):
        self.name = name
        self.attributes: dict[str, Any] = {}
        self.status: str | None = None

    def add_attributes(self, attributes: dict[str, Any]) -> None:
        self.attributes.update(attributes)

    def set_status(self, status: str, description: str | None = None) -> None:
        self.status = status

    def record_exception(self, exception: Exception) -> None:
        self.status = 'error'


class RecordingTracer(Tracer):
    def __init__(self):
        self.spans: list[RecordingSpan] = []

    @contextmanager
    def start_span(self, name: str):
        span = RecordingSpan(name)
        self.spans.append(span)
        yield span

    def named(self, name: str) -> list[RecordingSpan]:
        return [span for span in self.spans if span.name == name]


class FakeGraph:
    """An in-memory graph with the reads of `GraphReader` and the writes of the commit.

    A search returns a record when it is not in `hidden`. A read by UUID returns each
    record that exists. Each read returns a copy.
    """

    def __init__(self):
        self.nodes: dict[str, EntityNode] = {}
        self.edges: dict[str, EntityEdge] = {}
        self.episodes: dict[str, EpisodicNode] = {}
        self.episodic_edges: dict[tuple[str, str], EpisodicEdge] = {}
        self.sagas: list[Any] = []
        self.hidden: set[str] = set()
        self.log: list[str] = []
        self.write_timeouts: list[float | None] = []
        # The number of next writes of each kind that raise an error.
        self.fail_bulk_writes = 0
        self.fail_edge_updates = 0
        self.fail_node_updates = 0
        # When True, a bulk write that raises writes the nodes first.
        self.partial_bulk_write = False
        self.before_edge_update: Any = None

    def add(self, *records: EntityNode | EntityEdge) -> None:
        for record in records:
            if isinstance(record, EntityNode):
                self.nodes[record.uuid] = deepcopy(record)
            else:
                self.edges[record.uuid] = deepcopy(record)

    # --- reads -------------------------------------------------------------------

    async def node_candidates(self, nodes, identity_properties=None):
        self.log.append('read:node_candidates')
        results = []
        for node in nodes:
            key = _identity_key_for_node(node, identity_properties)
            candidates = []
            for stored in self.nodes.values():
                if stored.uuid in self.hidden or stored.uuid == node.uuid:
                    continue
                same_name = _normalize_string_exact(stored.name) == _normalize_string_exact(
                    node.name
                )
                same_key = (
                    key is not None and _identity_key_for_node(stored, identity_properties) == key
                )
                similar = cosine(stored.name_embedding, node.name_embedding) >= 0.6
                if same_name or same_key or similar:
                    # A search does not return the name embedding.
                    candidates.append(stored.model_copy(deep=True, update={'name_embedding': None}))
            results.append(candidates)
        return results

    async def edge_duplicate_candidates(self, edges):
        self.log.append('read:edge_duplicate_candidates')
        results = []
        for edge in edges:
            pair = {edge.source_node_uuid, edge.target_node_uuid}
            results.append(
                [
                    deepcopy(stored)
                    for stored in self.edges.values()
                    if stored.uuid not in self.hidden
                    and stored.uuid != edge.uuid
                    and {stored.source_node_uuid, stored.target_node_uuid} == pair
                ]
            )
        return results

    async def edge_invalidation_candidates(self, edges):
        self.log.append('read:edge_invalidation_candidates')
        return [
            [
                deepcopy(stored)
                for stored in self.edges.values()
                if stored.uuid not in self.hidden
                and stored.uuid != edge.uuid
                and cosine(stored.fact_embedding, edge.fact_embedding) >= 0.6
            ]
            for edge in edges
        ]

    async def nodes_by_uuids(self, uuids, group_id):
        return [
            self.nodes[uuid].model_copy(deep=True, update={'name_embedding': None})
            for uuid in uuids
            if uuid in self.nodes
        ]

    async def episodes_by_uuids(self, uuids):
        return [deepcopy(self.episodes[uuid]) for uuid in uuids if uuid in self.episodes]

    async def edges_by_uuids(self, uuids):
        return [deepcopy(self.edges[uuid]) for uuid in uuids if uuid in self.edges]

    # --- writes ------------------------------------------------------------------

    async def add_nodes_and_edges_bulk(
        self,
        driver,
        episodic_nodes,
        episodic_edges,
        entity_nodes,
        entity_edges,
        embedder,
        write_timeout=None,
    ):
        self.log.append('write:bulk')
        self.write_timeouts.append(write_timeout)
        if self.fail_bulk_writes > 0:
            self.fail_bulk_writes -= 1
            if self.partial_bulk_write:
                for node in entity_nodes:
                    self.nodes[node.uuid] = deepcopy(node)
            raise RuntimeError('bulk write failed')
        for episode in episodic_nodes:
            self.episodes[episode.uuid] = deepcopy(episode)
        for node in entity_nodes:
            self.nodes[node.uuid] = deepcopy(node)
        for edge in episodic_edges:
            self.episodic_edges[(edge.source_node_uuid, edge.target_node_uuid)] = deepcopy(edge)
        for edge in entity_edges:
            # The save query matches the two endpoint nodes.
            if edge.source_node_uuid in self.nodes and edge.target_node_uuid in self.nodes:
                self.edges[edge.uuid] = deepcopy(edge)

    async def execute_write_with_timeout(self, graph_driver, timeout, func, /, *args, **kwargs):
        self.write_timeouts.append(timeout)
        records = args[0]
        if func is update_entity_edges_tx:
            self.log.append('write:edge_update')
            if self.before_edge_update is not None:
                self.before_edge_update()
            if self.fail_edge_updates > 0:
                self.fail_edge_updates -= 1
                raise RuntimeError('edge update failed')
            for edge in records:
                # The update query creates nothing.
                if edge.uuid in self.edges:
                    self.edges[edge.uuid] = deepcopy(edge)
        elif func is update_entity_nodes_tx:
            self.log.append('write:node_update')
            if self.fail_node_updates > 0:
                self.fail_node_updates -= 1
                raise RuntimeError('node update failed')
            for node in records:
                if node.uuid in self.nodes:
                    self.nodes[node.uuid] = deepcopy(node)
        else:
            raise AssertionError(f'unexpected write function {func}')


class LoggingStore(InMemoryIngestionStateStore):
    """An in-memory store that adds each call to the log of the graph."""

    def __init__(self, graph: FakeGraph, clock=None):
        super().__init__(**({'clock': clock} if clock is not None else {}))
        self.graph = graph

    async def save_final(self, group_id, seq, fence, payload):
        self.graph.log.append('store:save_final')
        await super().save_final(group_id, seq, fence, payload)

    async def finish_commit(self, group_id, seq, fence, node_uuids, retain_seconds):
        self.graph.log.append('store:finish_commit')
        await super().finish_commit(group_id, seq, fence, node_uuids, retain_seconds)


class Harness:
    """A fake graph, a store, a fake `Graphiti`, and functions that make units."""

    def __init__(self, monkeypatch, clock: FakeClock | None = None, **config_values):
        self.graph = FakeGraph()
        self.clock = clock
        self.store = LoggingStore(self.graph, clock)
        config_values.setdefault('window', 4)
        config_values.setdefault('poll_interval_seconds', 0.001)
        self.config = ConcurrentIngestionConfig(state_store=self.store, **config_values)
        self.tracer = RecordingTracer()

        self.llm = AsyncMock(side_effect=AssertionError('unexpected model call'))
        llm_client = MagicMock()
        llm_client.max_tokens = 32768
        llm_client.generate_response = self.llm
        embedder = MagicMock()
        embedder.create_batch = AsyncMock(side_effect=lambda names: [X for _ in names])
        driver = MagicMock()
        driver.graph_operations_interface = None
        self.clients = SimpleNamespace(
            driver=driver,
            llm_client=llm_client,
            embedder=embedder,
            cross_encoder=MagicMock(),
            max_coroutines=None,
        )
        self.graphiti = SimpleNamespace(
            _resolve_request_scope=lambda group_id: (group_id, self.clients.driver, self.clients),
            tracer=self.tracer,
            store_raw_episode_content=True,
            max_coroutines=None,
            clients=self.clients,
        )

        async def extract_attributes(clients, nodes, episode=None, previous=None, *args, **kwargs):
            for node in nodes:
                node.summary = f'{node.summary}+{episode.uuid}'
            return nodes

        async def episodes_by_uuids(driver, uuids, group_id=None, *, require_all=False):
            episodes = [
                deepcopy(self.graph.episodes[uuid]) for uuid in uuids if uuid in self.graph.episodes
            ]
            if group_id is not None:
                episodes = [episode for episode in episodes if episode.group_id == group_id]
            found_uuids = {episode.uuid for episode in episodes}
            if require_all and any(uuid not in found_uuids for uuid in uuids):
                missing_uuid = next(uuid for uuid in uuids if uuid not in found_uuids)
                raise NodeNotFoundError(missing_uuid)
            return episodes

        self.graphiti.retrieve_episodes = AsyncMock(return_value=[])
        monkeypatch.setattr(EpisodicNode, 'get_by_uuids', episodes_by_uuids)
        self.extract_attributes = AsyncMock(side_effect=extract_attributes)
        monkeypatch.setattr(phases, 'add_nodes_and_edges_bulk', self.graph.add_nodes_and_edges_bulk)
        monkeypatch.setattr(
            phases, 'execute_write_with_timeout', self.graph.execute_write_with_timeout
        )
        monkeypatch.setattr(phases, 'extract_attributes_from_nodes', self.extract_attributes)

    async def admit(self, unit_id: str, episode: EpisodicNode | None = None) -> AdmittedUnit:
        """Admit a unit with store calls only."""
        admission = await self.store.begin_admit(GROUP, unit_id, self.config.lease_seconds)
        assert admission is not None
        await self.store.end_admit(GROUP, admission.seq)
        episode = episode or make_episode(f'ep-{unit_id}')
        return AdmittedUnit(
            group_id=GROUP,
            unit_id=unit_id,
            seq=admission.seq,
            created_at=T0,
            episodes=[episode],
            previous_episodes=[[]],
        )

    async def analyze(self, unit: AdmittedUnit, **fields) -> ChangeSet:
        """Record the snapshot of a unit and store a change set with the given fields."""
        analysis = await self.store.start_analysis(GROUP, unit.seq)
        create_nodes = fields.get('create_nodes', [])
        create_edges = fields.get('create_edges', [])
        episode = unit.episodes[0]
        for edge in create_edges:
            if episode.uuid not in edge.episodes:
                edge.episodes.append(episode.uuid)
        fields.setdefault('node_uuids', [node.uuid for node in create_nodes])
        fields.setdefault('node_first_episode', dict.fromkeys(fields['node_uuids'], episode.uuid))
        fields.setdefault(
            'episodic_edges',
            [
                EpisodicEdge(
                    source_node_uuid=episode.uuid,
                    target_node_uuid=uuid,
                    group_id=GROUP,
                    created_at=T0,
                )
                for uuid in fields['node_uuids']
            ],
        )
        written = episode.model_copy(deep=True)
        written.entity_edges = [edge.uuid for edge in create_edges]
        change_set = ChangeSet(
            group_id=GROUP,
            unit_id=unit.unit_id,
            seq=unit.seq,
            snapshot_seq=analysis.snapshot_seq,
            analysis_started_at=analysis.started_at,
            episodes=[written],
            **fields,
        )
        await self.store.save_change_set(GROUP, unit.seq, change_set.to_bytes())
        return change_set

    async def commit(self, unit: AdmittedUnit, change_set: ChangeSet, **kwargs):
        kwargs.setdefault('reader', self.graph)
        return await phases.commit_unit(
            self.graphiti, self.store, self.config, unit, change_set, **kwargs
        )

    async def enrich(self, unit: AdmittedUnit, commit_result, **kwargs):
        kwargs.setdefault('reader', self.graph)
        return await phases.enrich_unit(
            self.graphiti, self.store, self.config, unit, commit_result, **kwargs
        )

    async def run(self, unit_id: str, **fields):
        """Admit, analyze, and commit one unit."""
        unit = await self.admit(unit_id)
        change_set = await self.analyze(unit, **fields)
        result = await self.commit(unit, change_set)
        return unit, result
