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

import asyncio
import contextlib
import logging
import time
from collections.abc import Awaitable, Callable
from datetime import datetime
from typing import TYPE_CHECKING, Any
from uuid import NAMESPACE_URL, uuid4, uuid5

from pydantic import BaseModel, Field

from graphiti_core.edges import EntityEdge, EpisodicEdge, HasEpisodeEdge, NextEpisodeEdge
from graphiti_core.errors import (
    NodeNotFoundError,
    StaleCommitError,
    UnitExpiredError,
    UnitInProgressError,
)
from graphiti_core.graphiti_types import GraphitiClients, uses_prompt_routing
from graphiti_core.helpers import semaphore_gather
from graphiti_core.ingestion.change_set import (
    ChangeSet,
    PatchNode,
    ResolutionTrace,
    SagaLink,
    apply_edge_operation,
)
from graphiti_core.ingestion.config import ConcurrentIngestionConfig
from graphiti_core.ingestion.delta import Delta, load_delta_digests, patch_as_node
from graphiti_core.ingestion.graph_reader import GraphReader
from graphiti_core.ingestion.state_store import (
    REASON_APPLY_FAILED,
    Admission,
    CommitRecover,
    CommitTurn,
    CommitWait,
    DeltaUnit,
    IngestionStateStore,
)
from graphiti_core.nodes import (
    EntityNode,
    EpisodeType,
    EpisodicNode,
    SagaNode,
    create_entity_node_embeddings,
)
from graphiti_core.search.search_utils import RELEVANT_SCHEMA_LIMIT
from graphiti_core.utils.bulk_utils import (
    RawEpisode,
    add_nodes_and_edges_bulk,
    dedupe_edges_bulk,
    execute_write_with_timeout,
    missing_ingestion_interface_operations,
    resolve_edge_pointers,
    retrieve_previous_episodes_bulk,
    update_entity_edges_tx,
    update_entity_nodes_tx,
)
from graphiti_core.utils.datetime_utils import ensure_utc, utc_now
from graphiti_core.utils.maintenance.dedup_helpers import _promote_resolved_node
from graphiti_core.utils.maintenance.edge_operations import (
    _apply_temporal_invalidation,
    _same_pair_edges,
    build_episodic_edges,
    decide_edge_against_candidates,
    find_exact_duplicate_edge,
)
from graphiti_core.utils.maintenance.node_operations import (
    _identity_type_for_node,
    extract_attributes_from_nodes,
    extract_attributes_only_from_nodes,
    extract_nodes,
    resolve_extracted_nodes,
    resolve_nodes_with_candidates,
)

if TYPE_CHECKING:
    from graphiti_core.graphiti import Graphiti

logger = logging.getLogger(__name__)

WaitFn = Callable[[], Awaitable[None]]

REPORT_KEYS = (
    'turn_wait_ms',
    'duration_ms',
    'delta.unit_count',
    'node.checked',
    'node.collided',
    'node.merged',
    'edge.checked',
    'edge.collided',
    'edge.deduplicated',
    'edge.ended',
    'repair.llm_calls',
    'repair.duration_ms',
    'repair.c3.node_merged',
    'repair.c4a.edge_removed',
    'repair.c5.edge_merged',
    'repair.c6.edge_ended',
    'operation.applied',
    'operation.dropped',
    'apply.attempts',
    'recovered.count',
    'retry',
)


def new_report() -> dict[str, float]:
    """Return a commit report with each count at zero."""
    return dict.fromkeys(REPORT_KEYS, 0)


class AdmittedUnit(BaseModel):
    """One admitted ingestion unit: its position, its episodes, and their context.

    `episodes` have their content. `previous_episodes` has one list for each episode.
    """

    group_id: str
    unit_id: str
    seq: int
    created_at: datetime
    bulk: bool = False
    episodes: list[EpisodicNode]
    previous_episodes: list[list[EpisodicNode]]


class CommitResult(BaseModel):
    """The result of a commit.

    `change_set` is the final change set. `edges` are the created edges and the stored
    edges that the commit changed. `report` has the counts of the commit.
    """

    change_set: ChangeSet
    edges: list[EntityEdge] = Field(default_factory=list)
    report: dict[str, float] = Field(default_factory=dict)


class UnitLease:
    """The local deadline of a unit.

    The local deadline is the time of the last successful `heartbeat` or
    `begin_commit` call, read before the call, plus `lease_seconds`. It is thus never
    later than the deadline in the store.

    `fence` is the fence of the commit turn of this attempt. `commit_unit` sets it.
    With a fence, a heartbeat raises `StaleCommitError` when a different attempt has
    the turn of the unit.
    """

    def __init__(
        self,
        store: IngestionStateStore,
        config: ConcurrentIngestionConfig,
        group_id: str,
        seq: int,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.store = store
        self.config = config
        self.group_id = group_id
        self.seq = seq
        self.clock = clock
        self.fence: int | None = None
        self._deadline: float | None = None

    def renewed(self, call_started_at: float) -> None:
        """Record a successful store call that set the deadline of the unit."""
        self._deadline = call_started_at + self.config.lease_seconds

    async def heartbeat(self) -> None:
        started_at = self.clock()
        fence = self.fence
        try:
            await self.store.heartbeat(self.group_id, self.seq, self.config.lease_seconds, fence)
        except StaleCommitError:
            if fence is not None:
                raise
            # A different attempt applies the final change set of the unit. This attempt
            # waits for the turn and does not set the deadline.
            return
        self.renewed(started_at)

    def write_budget(self) -> float:
        """Return the time limit in seconds for a graph write that starts now."""
        if self._deadline is None:
            raise UnitExpiredError(self.group_id, self.seq)
        budget = self._deadline - self.clock() - self.config.write_margin_seconds
        if budget <= 0:
            raise UnitExpiredError(self.group_id, self.seq)
        return budget


def _default_wait(config: ConcurrentIngestionConfig) -> WaitFn:
    async def wait() -> None:
        await asyncio.sleep(config.poll_interval_seconds)

    return wait


async def _fail_unit(
    store: IngestionStateStore, group_id: str, seq: int, error: BaseException
) -> None:
    """Release the position of a unit that has no stored final change set."""
    try:
        await store.fail(group_id, seq, type(error).__name__)
    except Exception:
        # The unit has a stored final change set, or the store has no access. The
        # deadline of the unit then releases its position.
        logger.debug('could not mark ingestion unit %s of group %s as failed', seq, group_id)


def _default_edge_type_map(
    edge_types: dict[str, type[BaseModel]] | None,
) -> dict[tuple[str, str], list[str]]:
    if edge_types is not None:
        return {('Entity', 'Entity'): list(edge_types.keys())}
    return {('Entity', 'Entity'): []}


def _written_episode(graphiti: 'Graphiti', episode: EpisodicNode, entity_edges: list[str]):
    """Return the copy of an episode that a graph write must contain."""
    written = episode.model_copy(deep=True)
    written.entity_edges = list(dict.fromkeys(entity_edges))
    if not graphiti.store_raw_episode_content:
        written.content = ''
    return written


def _check_driver(driver: Any) -> None:
    """Raise an error before the admission when the driver cannot write a commit."""
    missing = missing_ingestion_interface_operations(driver)
    if missing:
        raise ValueError(
            'concurrent ingestion needs these operations of graph_operations_interface, '
            f'and the interface of the driver does not have them: {", ".join(missing)}'
        )


async def _begin_admit(
    store: IngestionStateStore,
    config: ConcurrentIngestionConfig,
    group_id: str,
    unit_id: str,
    wait: WaitFn,
) -> Admission:
    while True:
        # A store call in a worker thread completes after a cancellation of the caller.
        # The call is thus shielded, and a cancelled caller releases the unit.
        call = asyncio.ensure_future(store.begin_admit(group_id, unit_id, config.lease_seconds))
        try:
            admission = await asyncio.shield(call)
        except asyncio.CancelledError as error:
            with contextlib.suppress(Exception):
                admission = await call
                if admission is not None:
                    await _fail_admission(store, group_id, admission, error)
            raise
        if admission is None:
            await wait()
            continue
        if admission.existing:
            # A different attempt has the unit. This attempt writes nothing, and it calls
            # no store method that changes state.
            raise UnitInProgressError(group_id, unit_id, admission.seq)
        return admission


@contextlib.asynccontextmanager
async def _admit_heartbeat(
    store: IngestionStateStore,
    config: ConcurrentIngestionConfig,
    group_id: str,
    admission: Admission,
):
    """Renew the deadline of a unit while its admission reads and writes the graph.

    The caller of the Admit phase does not have the `seq` before the phase returns, so
    the phase renews the deadline itself.
    """

    async def beat() -> None:
        while True:
            await asyncio.sleep(config.lease_seconds / 3)
            try:
                await store.heartbeat(group_id, admission.seq, config.lease_seconds)
            except (UnitExpiredError, StaleCommitError):
                # The unit is expired or failed. `end_admit` raises the error.
                return
            except Exception:
                logger.warning(
                    'heartbeat of ingestion unit %s of group %s failed in the admission',
                    admission.seq,
                    group_id,
                    exc_info=True,
                )

    task = asyncio.create_task(beat())
    try:
        yield
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


def _episode_uuid(group_id: str, unit_id: str, index: int = 0) -> str:
    """Return the UUID of an episode that a unit writes.

    The UUID is a function of the `unit_id`. A retry of the admission thus writes the
    same episode node, and not one more node.
    """
    return str(uuid5(NAMESPACE_URL, f'graphiti:ingest:{group_id}:{unit_id}:{index}'))


async def _written_episodes(
    driver: Any, episode_uuids: list[str], group_id: str
) -> dict[str, EpisodicNode]:
    """Return the episode nodes of the `unit_id` that are in the graph.

    The read does not use the state store: the store can have no record of a unit that
    wrote its episode nodes. The admission writes only the nodes that this read does
    not return, so it does not change a stored node or its `entity_edges`.
    """
    if not episode_uuids:
        return {}
    episodes = await EpisodicNode.get_by_uuids(driver, episode_uuids, group_id=group_id)
    return {episode.uuid: episode for episode in episodes}


async def _fail_admission(
    store: IngestionStateStore, group_id: str, admission: Admission, error: BaseException
) -> None:
    # A unit that a different attempt admitted keeps its position.
    if not admission.existing:
        await _fail_unit(store, group_id, admission.seq, error)


async def admit_episode(
    graphiti: 'Graphiti',
    store: IngestionStateStore,
    config: ConcurrentIngestionConfig,
    *,
    unit_id: str,
    group_id: str | None,
    name: str,
    episode_body: str,
    source_description: str,
    reference_time: datetime,
    source: EpisodeType = EpisodeType.message,
    uuid: str | None = None,
    previous_episode_uuids: list[str] | None = None,
    wait: WaitFn | None = None,
) -> AdmittedUnit:
    """Admit one episode: get its sequence number and make the episode node visible.

    The store gives the sequence number with the admit lease. With the lease, the
    function reads the previous episodes and writes the episode node with an empty
    `entity_edges` list. With `uuid`, the function loads the episode and does not
    write it.

    When the `unit_id` has a live unit, the function raises `UnitInProgressError`. It
    then writes nothing, and it calls no store method that changes state.
    """
    group_id, driver, clients = graphiti._resolve_request_scope(group_id)
    _check_driver(driver)
    wait = wait or _default_wait(config)

    with graphiti.tracer.start_span('ingest.admit') as span:
        start = time.time()
        admission = await _begin_admit(store, config, group_id, unit_id, wait)
        seq = admission.seq
        admitted_at = time.time()
        try:
            async with _admit_heartbeat(store, config, group_id, admission):
                now = utc_now()
                previous_episodes = (
                    await graphiti.retrieve_episodes(
                        reference_time,
                        last_n=RELEVANT_SCHEMA_LIMIT,
                        group_ids=[group_id],
                        source=source,
                        driver=driver,
                    )
                    if previous_episode_uuids is None
                    else await EpisodicNode.get_by_uuids(
                        driver,
                        previous_episode_uuids,
                        group_id=group_id,
                        require_all=True,
                    )
                )
                if uuid is not None:
                    episode = await EpisodicNode.get_by_uuid(driver, uuid, group_id=group_id)
                else:
                    episode_uuid = _episode_uuid(group_id, unit_id)
                    written = await _written_episodes(driver, [episode_uuid], group_id)
                    if episode_uuid in written:
                        # The node is in the graph, and the admission does not change it.
                        # The unit keeps the content in memory.
                        episode = written[episode_uuid]
                        episode.content = episode_body
                        now = episode.created_at
                    else:
                        episode = EpisodicNode(
                            uuid=episode_uuid,
                            name=name,
                            group_id=group_id,
                            labels=[],
                            source=source,
                            content=episode_body,
                            source_description=source_description,
                            created_at=now,
                            valid_at=reference_time,
                        )
                        await add_nodes_and_edges_bulk(
                            driver,
                            [_written_episode(graphiti, episode, [])],
                            [],
                            [],
                            [],
                            clients.embedder,
                        )
                # After a retry, the node of the unit is in the graph. It is not its own context.
                previous_episodes = [
                    previous for previous in previous_episodes if previous.uuid != episode.uuid
                ]
                await store.end_admit(group_id, seq)
        except BaseException as error:
            await _fail_admission(store, group_id, admission, error)
            raise

        span.add_attributes(
            {
                'ingest.seq': seq,
                'wait_ms': (admitted_at - start) * 1000,
                'duration_ms': (time.time() - start) * 1000,
            }
        )
        return AdmittedUnit(
            group_id=group_id,
            unit_id=unit_id,
            seq=seq,
            created_at=now,
            episodes=[episode],
            previous_episodes=[list(previous_episodes)],
        )


async def admit_bulk(
    graphiti: 'Graphiti',
    store: IngestionStateStore,
    config: ConcurrentIngestionConfig,
    *,
    unit_id: str,
    group_id: str | None,
    bulk_episodes: list[RawEpisode],
    wait: WaitFn | None = None,
) -> AdmittedUnit:
    """Admit one bulk call as one unit with one sequence number.

    The function writes each new episode node and then reads the previous episodes of
    each episode, in the order of the bulk call of today.

    When the `unit_id` has a live unit, the function raises `UnitInProgressError`. It
    then writes nothing, and it calls no store method that changes state.
    """
    group_id, driver, clients = graphiti._resolve_request_scope(group_id)
    _check_driver(driver)
    wait = wait or _default_wait(config)

    with graphiti.tracer.start_span('ingest.admit') as span:
        start = time.time()
        admission = await _begin_admit(store, config, group_id, unit_id, wait)
        seq = admission.seq
        admitted_at = time.time()
        try:
            async with _admit_heartbeat(store, config, group_id, admission):
                now = utc_now()
                episodes: list[EpisodicNode] = []
                new_episodes: list[EpisodicNode] = []
                written = await _written_episodes(
                    driver,
                    [
                        _episode_uuid(group_id, unit_id, index)
                        for index, raw_episode in enumerate(bulk_episodes)
                        if raw_episode.uuid is None
                    ],
                    group_id,
                )
                for index, raw_episode in enumerate(bulk_episodes):
                    if raw_episode.uuid is not None:
                        episodes.append(
                            await EpisodicNode.get_by_uuid(
                                driver, raw_episode.uuid, group_id=group_id
                            )
                        )
                        continue
                    episode_uuid = _episode_uuid(group_id, unit_id, index)
                    if episode_uuid in written:
                        # The node is in the graph, and the admission does not change it.
                        episode = written[episode_uuid]
                        episode.content = raw_episode.content
                        episodes.append(episode)
                        continue
                    episode = EpisodicNode(
                        uuid=episode_uuid,
                        name=raw_episode.name,
                        labels=[],
                        source=raw_episode.source,
                        content=raw_episode.content,
                        source_description=raw_episode.source_description,
                        group_id=group_id,
                        created_at=now,
                        valid_at=raw_episode.reference_time,
                    )
                    episodes.append(episode)
                    new_episodes.append(episode)

                if new_episodes:
                    await add_nodes_and_edges_bulk(
                        driver,
                        [_written_episode(graphiti, episode, []) for episode in new_episodes],
                        [],
                        [],
                        [],
                        clients.embedder,
                    )
                episode_context = await retrieve_previous_episodes_bulk(
                    driver, episodes, max_coroutines=graphiti.max_coroutines
                )
                await store.end_admit(group_id, seq)
        except BaseException as error:
            await _fail_admission(store, group_id, admission, error)
            raise

        span.add_attributes(
            {
                'ingest.seq': seq,
                'episode.count': len(episodes),
                'wait_ms': (admitted_at - start) * 1000,
                'duration_ms': (time.time() - start) * 1000,
            }
        )
        return AdmittedUnit(
            group_id=group_id,
            unit_id=unit_id,
            seq=seq,
            created_at=now,
            bulk=True,
            episodes=[episode for episode, _ in episode_context],
            previous_episodes=[list(previous) for _, previous in episode_context],
        )


def _unique_by_uuid(records: list[Any]) -> list[Any]:
    unique: dict[str, Any] = {}
    for record in records:
        unique.setdefault(record.uuid, record)
    return list(unique.values())


async def _save_analysis(
    graphiti: 'Graphiti',
    store: IngestionStateStore,
    unit: AdmittedUnit,
    change_set: ChangeSet,
    clients: GraphitiClients,
    span: Any,
    start: float,
) -> None:
    # A node that the unit creates gets its name embedding here, not in the commit.
    for node in change_set.create_nodes:
        if node.name_embedding is None:
            await node.generate_name_embedding(clients.embedder)
    await store.save_change_set(unit.group_id, unit.seq, change_set.to_bytes())
    span.add_attributes(
        {
            'ingest.seq': unit.seq,
            'snapshot_seq': change_set.snapshot_seq,
            'duration_ms': (time.time() - start) * 1000,
            'create_node.count': len(change_set.create_nodes),
            'create_edge.count': len(change_set.create_edges),
            'operation.count': len(change_set.operations),
        }
    )


async def analyze_episode(
    graphiti: 'Graphiti',
    store: IngestionStateStore,
    config: ConcurrentIngestionConfig,
    unit: AdmittedUnit,
    *,
    entity_types: dict[str, type[BaseModel]] | None = None,
    excluded_entity_types: list[str] | None = None,
    edge_types: dict[str, type[BaseModel]] | None = None,
    edge_type_map: dict[tuple[str, str], list[str]] | None = None,
    custom_extraction_instructions: str | None = None,
    identity_properties: dict[str, list[str]] | None = None,
) -> ChangeSet:
    """Run extraction and resolution for one episode and store the change set.

    The function writes nothing to the graph. It records the snapshot before its first
    graph read. It extracts identity attributes when identity properties are configured.
    """
    group_id, _driver, clients = graphiti._resolve_request_scope(unit.group_id)
    episode = unit.episodes[0]
    previous_episodes = unit.previous_episodes[0]

    with graphiti.tracer.start_span('ingest.analyze') as span:
        start = time.time()
        try:
            analysis = await store.start_analysis(group_id, unit.seq)
            trace = ResolutionTrace()

            extracted_nodes, node_episode_index_map = await extract_nodes(
                clients,
                episode,
                previous_episodes,
                entity_types,
                excluded_entity_types,
                custom_extraction_instructions,
            )
            identity_nodes = (
                [
                    node
                    for node in extracted_nodes
                    if _identity_type_for_node(node, identity_properties) is not None
                ]
                if identity_properties
                else []
            )
            if identity_nodes:
                await extract_attributes_only_from_nodes(
                    clients,
                    identity_nodes,
                    episode,
                    previous_episodes,
                    entity_types,
                    validate_shape=False,
                )
            nodes, uuid_map, _ = await resolve_extracted_nodes(
                clients,
                extracted_nodes,
                episode,
                previous_episodes,
                entity_types,
                identity_properties=identity_properties,
                trace=trace,
            )
            (
                resolved_edges,
                invalidated_edges,
                new_edges,
            ) = await graphiti._extract_and_resolve_edges(
                episode,
                extracted_nodes,
                previous_episodes,
                edge_type_map or _default_edge_type_map(edge_types),
                group_id,
                edge_types,
                nodes,
                uuid_map,
                custom_extraction_instructions=custom_extraction_instructions,
                entity_types=entity_types,
                identity_properties=identity_properties,
                clients=clients,
                trace=trace,
            )

            nodes = _unique_by_uuid(nodes)
            change_set = ChangeSet(
                group_id=group_id,
                unit_id=unit.unit_id,
                seq=unit.seq,
                snapshot_seq=analysis.snapshot_seq,
                analysis_started_at=analysis.started_at,
                episodes=[
                    _written_episode(
                        graphiti,
                        episode,
                        [edge.uuid for edge in [*resolved_edges, *invalidated_edges]],
                    )
                ],
                create_nodes=[node for node in nodes if node.uuid not in trace.stored_node_uuids],
                create_edges=_unique_by_uuid(new_edges),
                episodic_edges=build_episodic_edges(
                    nodes, [episode.uuid], unit.created_at, node_episode_index_map
                ),
                operations=list(trace.operations),
                node_uuids=[node.uuid for node in nodes],
                node_first_episode={node.uuid: episode.uuid for node in nodes},
                seen_node_candidates=trace.seen_node_candidates,
                seen_node_labels=trace.seen_node_labels,
                edges_ended_by_candidate=sorted(trace.ended_by_candidate),
                seen_duplicate_candidates=trace.seen_duplicate_candidates,
                seen_invalidation_candidates=trace.seen_invalidation_candidates,
            )
            await _save_analysis(graphiti, store, unit, change_set, clients, span, start)
        except StaleCommitError:
            # The unit has a stored change set from a different attempt. It keeps its
            # position.
            raise
        except BaseException as error:
            await _fail_unit(store, group_id, unit.seq, error)
            raise
        return change_set


async def analyze_bulk(
    graphiti: 'Graphiti',
    store: IngestionStateStore,
    config: ConcurrentIngestionConfig,
    unit: AdmittedUnit,
    *,
    entity_types: dict[str, type[BaseModel]] | None = None,
    excluded_entity_types: list[str] | None = None,
    edge_types: dict[str, type[BaseModel]] | None = None,
    edge_type_map: dict[tuple[str, str], list[str]] | None = None,
    custom_extraction_instructions: str | None = None,
    identity_properties: dict[str, list[str]] | None = None,
    strict_edge_types: bool = False,
    use_combined_extraction: bool = True,
) -> ChangeSet:
    """Run the batch pipeline for one bulk unit and store the change set.

    The items of the unit are compared with each other here, as today. The function
    writes nothing to the graph and extracts configured identity attributes.
    """
    group_id, _driver, clients = graphiti._resolve_request_scope(unit.group_id)
    episode_context = list(zip(unit.episodes, unit.previous_episodes, strict=True))
    resolved_edge_type_map = edge_type_map or _default_edge_type_map(edge_types)

    with graphiti.tracer.start_span('ingest.analyze') as span:
        start = time.time()
        try:
            analysis = await store.start_analysis(group_id, unit.seq)
            trace = ResolutionTrace()

            (
                nodes_by_episode,
                uuid_map,
                extracted_edges_bulk,
            ) = await graphiti._extract_and_dedupe_nodes_bulk(
                episode_context,
                resolved_edge_type_map,
                edge_types,
                entity_types,
                excluded_entity_types,
                strict_edge_types=strict_edge_types,
                custom_extraction_instructions=custom_extraction_instructions,
                use_combined_extraction=use_combined_extraction,
                clients=clients,
                # The first pass also resolves extracted nodes to stored nodes.
                trace=trace,
                identity_properties=identity_properties,
            )

            episodic_edges: list[EpisodicEdge] = []
            for episode_uuid, episode_nodes in nodes_by_episode.items():
                episodic_edges.extend(
                    build_episodic_edges(episode_nodes, episode_uuid, unit.created_at)
                )

            edges_by_episode = await dedupe_edges_bulk(
                clients,
                [resolve_edge_pointers(edges, uuid_map) for edges in extracted_edges_bulk],
                episode_context,
                [],
                edge_types or {},
                resolved_edge_type_map,
            )

            (
                nodes,
                resolved_edges,
                invalidated_edges,
                final_uuid_map,
            ) = await graphiti._resolve_nodes_and_edges_bulk(
                nodes_by_episode,
                edges_by_episode,
                episode_context,
                entity_types,
                edge_types,
                resolved_edge_type_map,
                unit.episodes,
                clients=clients,
                hydrate=False,
                trace=trace,
                identity_properties=identity_properties,
            )

            nodes = _unique_by_uuid(nodes)
            final_node_uuids = {node.uuid for node in nodes}

            def final_uuid(node_uuid: str) -> str:
                # The two maps of the batch pipeline can make a chain of UUIDs.
                for _ in range(len(uuid_map) + len(final_uuid_map) + 1):
                    next_uuid = final_uuid_map.get(node_uuid, uuid_map.get(node_uuid, node_uuid))
                    if next_uuid == node_uuid:
                        break
                    node_uuid = next_uuid
                return node_uuid

            node_first_episode: dict[str, str] = {}
            for episode in unit.episodes:
                for node in nodes_by_episode.get(episode.uuid, []):
                    node_uuid = final_uuid(node.uuid)
                    if node_uuid in final_node_uuids:
                        node_first_episode.setdefault(node_uuid, episode.uuid)
            for node in nodes:
                node_first_episode.setdefault(node.uuid, unit.episodes[0].uuid)

            edges_to_save = [*resolved_edges, *invalidated_edges]
            # An extracted edge is a key of the candidate record of the trace.
            created_edge_uuids = set(trace.seen_duplicate_candidates)
            change_set = ChangeSet(
                group_id=group_id,
                unit_id=unit.unit_id,
                seq=unit.seq,
                snapshot_seq=analysis.snapshot_seq,
                analysis_started_at=analysis.started_at,
                episodes=[
                    _written_episode(
                        graphiti,
                        episode,
                        [edge.uuid for edge in edges_to_save if episode.uuid in edge.episodes],
                    )
                    for episode in unit.episodes
                ],
                create_nodes=[node for node in nodes if node.uuid not in trace.stored_node_uuids],
                create_edges=_unique_by_uuid(
                    [edge for edge in resolved_edges if edge.uuid in created_edge_uuids]
                ),
                episodic_edges=_unique_episodic_edges(
                    resolve_edge_pointers(episodic_edges, final_uuid_map)
                ),
                operations=list(trace.operations),
                node_uuids=[node.uuid for node in nodes],
                node_first_episode=node_first_episode,
                seen_node_candidates=trace.seen_node_candidates,
                seen_node_labels=trace.seen_node_labels,
                edges_ended_by_candidate=sorted(trace.ended_by_candidate),
                seen_duplicate_candidates=trace.seen_duplicate_candidates,
                seen_invalidation_candidates=trace.seen_invalidation_candidates,
            )
            await _save_analysis(graphiti, store, unit, change_set, clients, span, start)
        except StaleCommitError:
            # The unit has a stored change set from a different attempt. It keeps its
            # position.
            raise
        except BaseException as error:
            await _fail_unit(store, group_id, unit.seq, error)
            raise
        return change_set


def _unique_episodic_edges(edges: list[EpisodicEdge]) -> list[EpisodicEdge]:
    unique: dict[tuple[str, str], EpisodicEdge] = {}
    for edge in edges:
        unique.setdefault((edge.source_node_uuid, edge.target_node_uuid), edge)
    return list(unique.values())


def _replace_entity_edges(
    change_set: ChangeSet,
    replaced: dict[str, str],
    dropped: set[str] | None = None,
    added: list[str] | None = None,
) -> None:
    """Change the edge UUIDs in `entity_edges` of each episode of a change set."""
    dropped = dropped or set()
    for episode in change_set.episodes:
        uuids = [replaced.get(uuid, uuid) for uuid in episode.entity_edges if uuid not in dropped]
        episode.entity_edges = list(dict.fromkeys([*uuids, *(added or [])]))


def _episode_context(unit: AdmittedUnit) -> dict[str, tuple[EpisodicNode, list[EpisodicNode]]]:
    return {
        episode.uuid: (episode, previous_episodes)
        for episode, previous_episodes in zip(unit.episodes, unit.previous_episodes, strict=True)
    }


async def _repair_nodes(
    clients: GraphitiClients,
    reader: GraphReader,
    final: ChangeSet,
    delta: Delta,
    unit: AdmittedUnit,
    entity_types: dict[str, type[BaseModel]] | None,
    identity_properties: dict[str, list[str]] | None,
    report: dict[str, float],
) -> None:
    """Step C3: resolve each created node that has a candidate that the analysis did not see."""
    created = list(final.create_nodes)
    report['node.checked'] = len(created)
    if not created:
        return
    own_uuids = {node.uuid for node in created}

    # The candidates of the graph at the time of the commit, with the stored embeddings.
    graph_candidates = await reader.node_candidates(created, identity_properties)
    current: dict[str, EntityNode] = {}
    for candidates in graph_candidates:
        for candidate in candidates:
            current.setdefault(candidate.uuid, candidate)

    # The nodes that the delta created or patched. A search can miss them (index lag).
    delta_uuids: list[list[str]] = []
    for node, matches, identity_matches in zip(
        created,
        delta.match_nodes(created, identity_properties),
        delta.patched_identity_matches(created, identity_properties),
        strict=True,
    ):
        patched_seen = [
            uuid for uuid in final.seen_node_candidates.get(node.uuid, []) if uuid in delta.patches
        ]
        delta_uuids.append(list(dict.fromkeys([*matches, *identity_matches, *patched_seen])))

    # A delta record is a candidate only if it is in the graph (step C2a). The read
    # also gives the current state of the record.
    missing = {uuid for uuids in delta_uuids for uuid in uuids} - set(current) - own_uuids
    for node in await reader.nodes_by_uuids(sorted(missing), final.group_id):
        current[node.uuid] = node

    # The Enrich phase of an earlier unit possibly did not write its patch at this
    # time. The comparison thus uses a copy with the patches of the delta.
    patch_changed: dict[str, bool] = {}
    for node_uuid, node in current.items():
        delta_patches = delta.patches.get(node_uuid)
        if not delta_patches:
            continue
        # The labels that the analysis saw, when the analysis read the node. The Enrich
        # phase of an earlier unit possibly wrote a promoted label after that read.
        labels_before = set(final.seen_node_labels.get(node_uuid, node.labels))
        for patch in delta_patches:
            _promote_resolved_node(patch_as_node(node, patch), node)
        patch_changed[node_uuid] = (
            any(patch.attributes for patch in delta_patches) or set(node.labels) != labels_before
        )

    collided: list[tuple[EntityNode, list[EntityNode]]] = []
    for node, graph_list, extra_uuids in zip(created, graph_candidates, delta_uuids, strict=True):
        seen = set(final.seen_node_candidates.get(node.uuid, []))
        candidates: list[EntityNode] = []
        for uuid in dict.fromkeys([*[candidate.uuid for candidate in graph_list], *extra_uuids]):
            candidate = current.get(uuid)
            if candidate is None or uuid in own_uuids:
                continue
            # The analysis saw the candidate. It stays a candidate only if a patch of
            # the delta gave it attributes or labels that the analysis did not see.
            if uuid in seen and not patch_changed.get(uuid, False):
                continue
            candidates.append(candidate)
        if candidates:
            collided.append((node, candidates))

    report['node.collided'] = len(collided)
    if not collided:
        return

    # A bulk unit resolves the nodes of each episode with the context of that episode.
    context = _episode_context(unit)
    default_episode_uuid = unit.episodes[0].uuid
    groups: dict[str, list[tuple[EntityNode, list[EntityNode]]]] = {}
    for node, candidates in collided:
        episode_uuid = final.node_first_episode.get(node.uuid, default_episode_uuid)
        if episode_uuid not in context:
            episode_uuid = default_episode_uuid
        groups.setdefault(episode_uuid, []).append((node, candidates))

    merged: dict[str, str] = {}
    patches: list[PatchNode] = []
    for episode_uuid, items in groups.items():
        episode, previous_episodes = context[episode_uuid]
        trace = ResolutionTrace()
        _, uuid_map, _ = await resolve_nodes_with_candidates(
            clients,
            [node for node, _ in items],
            [candidates for _, candidates in items],
            episode,
            previous_episodes,
            entity_types,
            identity_properties,
            trace=trace,
        )
        merged.update({uuid: target for uuid, target in uuid_map.items() if uuid != target})
        patches.extend(
            operation for operation in trace.operations if isinstance(operation, PatchNode)
        )
        report['repair.llm_calls'] += trace.node_llm_calls

    if not merged:
        return
    final.create_nodes = [node for node in final.create_nodes if node.uuid not in merged]
    final.uuid_map.update(merged)
    final.operations.extend(patches)
    final.node_uuids = list(dict.fromkeys(merged.get(uuid, uuid) for uuid in final.node_uuids))
    node_first_episode: dict[str, str] = {}
    for node_uuid, episode_uuid in final.node_first_episode.items():
        node_first_episode.setdefault(merged.get(node_uuid, node_uuid), episode_uuid)
    final.node_first_episode = node_first_episode
    report['repair.c3.node_merged'] = len(merged)


def _remap_edges(final: ChangeSet, report: dict[str, float]) -> None:
    """Steps C4 and C4a: apply the UUID map to all edges, then remove exact duplicates."""
    if not final.uuid_map:
        return
    uuids_before = {edge.uuid for edge in final.create_edges}
    final.create_edges = resolve_edge_pointers(final.create_edges, final.uuid_map)
    dropped = uuids_before - {edge.uuid for edge in final.create_edges}
    final.episodic_edges = _unique_episodic_edges(
        resolve_edge_pointers(final.episodic_edges, final.uuid_map)
    )

    # A node merge can give two created edges the same endpoints. This is the rule of
    # serial ingestion for the edges of one episode, with no model call.
    seen: dict[tuple[str, str, str], EntityEdge] = {}
    survivors: list[EntityEdge] = []
    replaced: dict[str, str] = {}
    for edge in final.create_edges:
        earlier = find_exact_duplicate_edge(edge, seen)
        if earlier is None:
            survivors.append(edge)
            continue
        earlier.episodes.extend(uuid for uuid in edge.episodes if uuid not in earlier.episodes)
        replaced[edge.uuid] = earlier.uuid
    final.create_edges = survivors
    _replace_entity_edges(final, replaced, dropped)
    report['repair.c4a.edge_removed'] = len(replaced)


def _move_end_date_to_earliest_contradiction(
    edge: EntityEdge, contradicted: list[EntityEdge]
) -> None:
    valid_at = ensure_utc(edge.valid_at)
    invalid_at = ensure_utc(edge.invalid_at)
    if valid_at is None or invalid_at is None:
        return
    for candidate in contradicted:
        candidate_valid_at = ensure_utc(candidate.valid_at)
        if candidate_valid_at is not None and valid_at < candidate_valid_at < invalid_at:
            edge.invalid_at = candidate.valid_at
            invalid_at = candidate_valid_at


async def _repair_edges(
    clients: GraphitiClients,
    reader: GraphReader,
    final: ChangeSet,
    delta: Delta,
    unit: AdmittedUnit,
    report: dict[str, float],
) -> None:
    """Steps C5 to C7: find and decide the edge duplicates and the edge contradictions."""
    edges = list(final.create_edges)
    report['edge.checked'] = len(edges)
    if not edges:
        return
    own_uuids = {edge.uuid for edge in edges}

    # C5 has the endpoint filter. C6 is a separate pass with no endpoint filter.
    duplicate_lists, invalidation_lists = await asyncio.gather(
        reader.edge_duplicate_candidates(edges), reader.edge_invalidation_candidates(edges)
    )
    delta_pair_uuids = delta.edges_between(edges)
    delta_similar_uuids = delta.similar_edges(edges)

    current: dict[str, EntityEdge] = {}
    for candidate_lists in (duplicate_lists, invalidation_lists):
        for candidates in candidate_lists:
            for candidate in candidates:
                current.setdefault(candidate.uuid, candidate)
    # A delta edge is a candidate only if it is in the graph (step C2a). The read also
    # gives its current end dates.
    missing = (
        {uuid for uuids in [*delta_pair_uuids, *delta_similar_uuids] for uuid in uuids}
        - set(current)
        - own_uuids
    )
    for edge in await reader.edges_by_uuids(sorted(missing)):
        current[edge.uuid] = edge

    collided: list[tuple[EntityEdge, list[EntityEdge], list[EntityEdge]]] = []
    for index, edge in enumerate(edges):
        seen_duplicates = set(final.seen_duplicate_candidates.get(edge.uuid, []))
        seen_invalidations = set(final.seen_invalidation_candidates.get(edge.uuid, []))
        pair_uuids = list(
            dict.fromkeys(
                [
                    *[candidate.uuid for candidate in duplicate_lists[index]],
                    *delta_pair_uuids[index],
                ]
            )
        )
        duplicates = _same_pair_edges(
            edge,
            [
                current[uuid]
                for uuid in pair_uuids
                if uuid in current and uuid not in own_uuids and uuid not in seen_duplicates
            ],
        )
        excluded = {*pair_uuids, *seen_duplicates, *seen_invalidations, *own_uuids}
        invalidations = [
            current[uuid]
            for uuid in dict.fromkeys(
                [
                    *[candidate.uuid for candidate in invalidation_lists[index]],
                    *delta_similar_uuids[index],
                ]
            )
            if uuid in current and uuid not in excluded
        ]
        if duplicates or invalidations:
            collided.append((edge, duplicates, invalidations))

    report['edge.collided'] = len(collided)
    if not collided:
        return

    resolver_kwargs: dict[str, Any] = {'clients': clients} if uses_prompt_routing(clients) else {}
    decisions = await semaphore_gather(
        *[
            decide_edge_against_candidates(
                clients.llm_client, edge, duplicates, invalidations, **resolver_kwargs
            )
            for edge, duplicates, invalidations in collided
        ],
        max_coroutines=getattr(clients, 'max_coroutines', None),
    )

    # Each candidate is a stored record. The trace makes one operation for each change.
    trace = ResolutionTrace()
    trace.watch_stored_edges(
        [
            candidate
            for _, duplicates, invalidations in collided
            for candidate in [*duplicates, *invalidations]
        ]
    )
    replaced: dict[str, str] = {}
    ended_by_candidate = set(final.edges_ended_by_candidate)
    ended_uuids: list[str] = []
    ended_count = 0
    for (edge, _, _), decision in zip(collided, decisions, strict=True):
        if decision.exact_match is None:
            report['repair.llm_calls'] += 1
        target = decision.exact_match or (decision.duplicates[0] if decision.duplicates else None)
        resolved_edge = edge
        if target is not None:
            target.episodes.extend(uuid for uuid in edge.episodes if uuid not in target.episodes)
            replaced[edge.uuid] = target.uuid
            resolved_edge = target
        if decision.exact_match is not None:
            continue
        # The date rules apply in the two directions: a candidate that starts later
        # ends the edge, and the edge ends a candidate that starts earlier.
        was_open = resolved_edge.expired_at is None
        if resolved_edge is edge and edge.uuid in ended_by_candidate:
            # The analysis ended the edge at the start of a contradiction that it saw.
            # Serial ingestion ends an edge at the earliest contradiction that starts
            # later, so an earlier one of this commit moves the end date.
            _move_end_date_to_earliest_contradiction(edge, decision.contradicted)
        invalidated = _apply_temporal_invalidation(resolved_edge, list(decision.contradicted))
        trace.mark_ended_as_candidate(invalidated)
        ended_uuids.extend(candidate.uuid for candidate in invalidated)
        ended_count += len(invalidated) + int(was_open and resolved_edge.expired_at is not None)

    final.operations.extend(trace.collect_edge_operations())
    final.create_edges = [edge for edge in final.create_edges if edge.uuid not in replaced]
    # Serial ingestion of one episode adds the edges that the episode ended to
    # `entity_edges`. A bulk call adds only the edges that have the episode.
    _replace_entity_edges(final, replaced, added=None if unit.bulk else ended_uuids)
    report['repair.c5.edge_merged'] = len(replaced)
    report['repair.c6.edge_ended'] = ended_count


async def rebase_change_set(
    clients: GraphitiClients,
    change_set: ChangeSet,
    delta_units: list[DeltaUnit],
    *,
    unit: AdmittedUnit,
    reader: GraphReader | None = None,
    entity_types: dict[str, type[BaseModel]] | None = None,
    identity_properties: dict[str, list[str]] | None = None,
    report: dict[str, float] | None = None,
) -> ChangeSet:
    """Repair a change set against the graph and the delta (steps C2a to C7).

    The function reads the candidates of each created record again. It decides only a
    record that has a candidate that the analysis did not see. It writes nothing, and
    it does not change `change_set`. The result is the change set to make final.
    """
    report = report if report is not None else new_report()
    reader = reader or GraphReader(clients)
    start = time.time()

    final = change_set.model_copy(deep=True)
    delta = load_delta_digests(
        final.group_id, [delta_unit for delta_unit in delta_units if delta_unit.seq != final.seq]
    )
    report['delta.unit_count'] = delta.unit_count

    await _repair_nodes(
        clients, reader, final, delta, unit, entity_types, identity_properties, report
    )
    _remap_edges(final, report)
    await _repair_edges(clients, reader, final, delta, unit, report)

    report['node.merged'] = report['repair.c3.node_merged']
    report['edge.deduplicated'] = (
        report['repair.c4a.edge_removed'] + report['repair.c5.edge_merged']
    )
    report['edge.ended'] = report['repair.c6.edge_ended']
    report['repair.duration_ms'] = (time.time() - start) * 1000
    return final


async def _episodes_with_stored_links(
    reader: GraphReader, episodes: list[EpisodicNode]
) -> list[EpisodicNode]:
    """Return copies of the episodes that keep the `entity_edges` of the stored nodes.

    The `entity_edges` of each copy is the stored list and then the new UUIDs of the
    change set. A commit thus removes no link that an earlier commit wrote, and a
    second apply writes the same list.
    """
    stored = {
        episode.uuid: episode
        for episode in await reader.episodes_by_uuids([episode.uuid for episode in episodes])
    }
    written: list[EpisodicNode] = []
    for episode in episodes:
        stored_episode = stored.get(episode.uuid)
        if stored_episode is None or not stored_episode.entity_edges:
            written.append(episode)
            continue
        links = list(dict.fromkeys([*stored_episode.entity_edges, *episode.entity_edges]))
        written.append(episode.model_copy(update={'entity_edges': links}))
    return written


class ApplyResult(BaseModel):
    """The result of `apply_change_set`: the stored edges that changed, and the counts."""

    changed_edges: list[EntityEdge] = Field(default_factory=list)
    applied: int = 0
    dropped: int = 0


async def _write_saga_links(
    clients: GraphitiClients,
    change_set: ChangeSet,
    write_budget: Callable[[], Awaitable[float]],
) -> None:
    links = change_set.saga_links
    if not links or change_set.saga_node is None:
        return
    driver = clients.driver
    try:
        # The saga is in the graph. The apply changes only the episode fields of the
        # stored node, so the summary and its dates stay.
        saga_node = await SagaNode.get_by_uuid(driver, change_set.saga_node.uuid)
    except NodeNotFoundError:
        saga_node = change_set.saga_node.model_copy()
        if change_set.create_saga:
            # The saga is new. Its node is the first saga write. A second apply reads
            # the node that the first apply wrote.
            await asyncio.wait_for(saga_node.save(driver), await write_budget())
    else:
        if saga_node.group_id != change_set.group_id:
            raise NodeNotFoundError(change_set.saga_node.uuid)
    for link in links:
        if link.previous_episode_uuid is not None and link.next_episode_edge_uuid is not None:
            next_episode_edge = NextEpisodeEdge(
                uuid=link.next_episode_edge_uuid,
                source_node_uuid=link.previous_episode_uuid,
                target_node_uuid=link.episode_uuid,
                group_id=change_set.group_id,
                created_at=link.created_at,
            )
            await asyncio.wait_for(next_episode_edge.save(driver), await write_budget())
        has_episode_edge = HasEpisodeEdge(
            uuid=link.has_episode_edge_uuid,
            source_node_uuid=link.saga_uuid,
            target_node_uuid=link.episode_uuid,
            group_id=change_set.group_id,
            created_at=link.created_at,
        )
        await asyncio.wait_for(has_episode_edge.save(driver), await write_budget())

    # Track the first and the last episode on the saga node, as serial ingestion does.
    if saga_node.first_episode_uuid is None:
        saga_node.first_episode_uuid = links[0].episode_uuid
    saga_node.last_episode_uuid = links[-1].episode_uuid
    await asyncio.wait_for(saga_node.save(driver), await write_budget())


async def apply_change_set(
    clients: GraphitiClients,
    change_set: ChangeSet,
    *,
    write_budget: Callable[[], float],
    reader: GraphReader | None = None,
    check_turn: Callable[[], Awaitable[None]] | None = None,
) -> ApplyResult:
    """Apply a final change set to the graph (steps C8 and C9).

    Each operation applies to the current state of its stored edge. An operation on an
    edge that does not exist is dropped. `write_budget` returns the time limit in
    seconds of a write that starts now, or raises `UnitExpiredError`. `check_turn`
    runs before each graph write. It raises `StaleCommitError` when a different
    attempt has the turn of the unit. The function mints no UUID and no date, so a
    second call writes the same records.
    """
    driver = clients.driver
    reader = reader or GraphReader(clients)

    async def checked_budget() -> float:
        if check_turn is not None:
            try:
                await check_turn()
            except (StaleCommitError, UnitExpiredError):
                raise
            except Exception:
                # The store gave no answer. The local deadline then limits the write.
                logger.warning(
                    'the turn of ingestion unit %s of group %s was not checked before a write',
                    change_set.seq,
                    change_set.group_id,
                    exc_info=True,
                )
        return write_budget()

    operations = change_set.edge_operations()
    edge_uuids = list(dict.fromkeys(operation.edge_uuid for operation in operations))
    current = {edge.uuid: edge for edge in await reader.edges_by_uuids(edge_uuids)}
    result = ApplyResult()
    changed: dict[str, EntityEdge] = {}
    for operation in operations:
        edge = current.get(operation.edge_uuid)
        if edge is None:
            # A writer outside the sequence deleted the edge. It is not created again.
            result.dropped += 1
            continue
        result.applied += 1
        if apply_edge_operation(edge, operation):
            changed[edge.uuid] = edge
    for edge in changed.values():
        if edge.fact_embedding is None:
            await edge.load_fact_embedding(driver)
    result.changed_edges = list(changed.values())

    await add_nodes_and_edges_bulk(
        driver,
        await _episodes_with_stored_links(reader, change_set.episodes),
        change_set.episodic_edges,
        change_set.create_nodes,
        change_set.create_edges,
        clients.embedder,
        write_timeout=await checked_budget(),
    )
    if changed:
        await execute_write_with_timeout(
            driver,
            await checked_budget(),
            update_entity_edges_tx,
            result.changed_edges,
            clients.embedder,
            driver=driver,
        )
    await _write_saga_links(clients, change_set, checked_budget)
    return result


async def _apply_with_retries(
    clients: GraphitiClients,
    final: ChangeSet,
    write_budget: Callable[[], float],
    max_attempts: int,
    report: dict[str, float],
    reader: GraphReader,
    check_turn: Callable[[], Awaitable[None]],
) -> list[EntityEdge]:
    """Apply the same final change set until one attempt succeeds. No model call is made."""
    changed: dict[str, EntityEdge] = {}
    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        report['apply.attempts'] = attempt
        try:
            result = await apply_change_set(
                clients, final, write_budget=write_budget, reader=reader, check_turn=check_turn
            )
        except (UnitExpiredError, StaleCommitError):
            raise
        except Exception as error:
            last_error = error
            logger.warning(
                'apply attempt %s of ingestion unit %s of group %s failed: %s',
                attempt,
                final.seq,
                final.group_id,
                type(error).__name__,
            )
            continue
        # A retry after a partial write finds some edges already changed.
        changed.update({edge.uuid: edge for edge in result.changed_edges})
        report['operation.applied'] = result.applied
        report['operation.dropped'] = result.dropped
        return list(changed.values())
    assert last_error is not None
    raise last_error


async def _mark_apply_failed(
    store: IngestionStateStore, group_id: str, seq: int, fence: int
) -> None:
    logger.error(
        'each apply attempt of ingestion unit %s of group %s failed; a part of its '
        'change set is possibly in the graph',
        seq,
        group_id,
    )
    # With one of these errors, a different attempt has the turn of the unit.
    with contextlib.suppress(StaleCommitError, UnitExpiredError):
        await store.fail(group_id, seq, REASON_APPLY_FAILED, fence)


async def _recover_unit(
    clients: GraphitiClients,
    store: IngestionStateStore,
    config: ConcurrentIngestionConfig,
    group_id: str,
    recover: CommitRecover,
    call_started_at: float,
    clock: Callable[[], float],
    turns: dict[int, int],
    report: dict[str, float],
    reader: GraphReader,
) -> None:
    """Apply the stored final change set of an earlier unit that stopped, and record its commit.

    The nodes of the recovered unit get no enrichment from that unit.
    """

    # The recovery has the turn of the earlier unit, with the fence of `recover`.
    lease = UnitLease(store, config, group_id, recover.seq, clock)
    lease.fence = recover.fence
    lease.renewed(call_started_at)

    turns[recover.seq] = turns.get(recover.seq, 0) + 1
    try:
        await _apply_with_retries(
            clients,
            ChangeSet.from_bytes(recover.final_payload),
            lease.write_budget,
            config.max_apply_attempts,
            new_report(),
            reader,
            lease.heartbeat,
        )
        await store.finish_commit(
            group_id, recover.seq, recover.fence, [], config.index_lag_seconds
        )
        report['recovered.count'] += 1
        logger.warning('recovered ingestion unit %s of group %s', recover.seq, group_id)
    except StaleCommitError:
        # A different attempt has the turn of that unit.
        return
    except UnitExpiredError:
        # The time of this turn ended. The next `begin_commit` call gives a new turn.
        if turns[recover.seq] >= config.max_apply_attempts:
            await _mark_apply_failed(store, group_id, recover.seq, recover.fence)
    except Exception:
        await _mark_apply_failed(store, group_id, recover.seq, recover.fence)


async def _build_saga_links(
    graphiti: 'Graphiti',
    clients: GraphitiClients,
    unit: AdmittedUnit,
    saga: 'str | SagaNode | None',
    saga_previous_episode_uuid: str | None,
) -> tuple[list[SagaLink], SagaNode | None, bool]:
    """Step C7a: find or make the saga node, read its last episode, and mint the edge UUIDs.

    The function writes nothing. It returns the links, the saga node, and True when the
    saga node is new and the apply of the commit must create it.
    """
    if saga is None:
        return [], None, False
    driver = clients.driver
    now = unit.created_at
    if unit.bulk:
        episodes = sorted(unit.episodes, key=lambda episode: episode.valid_at)
        valid_ats = [episode.valid_at for episode in episodes if episode.valid_at is not None]
        saga_created_at = min(valid_ats) if valid_ats else now
        current_episode_uuid = ''
        previous_episode_uuid = None
    else:
        episodes = [unit.episodes[0]]
        saga_created_at = episodes[0].valid_at or now
        current_episode_uuid = episodes[0].uuid
        previous_episode_uuid = saga_previous_episode_uuid

    create_saga = False
    if isinstance(saga, str):
        found = await graphiti._find_saga(saga, unit.group_id, driver=driver)
        create_saga = found is None
        saga_node = found or SagaNode(name=saga, group_id=unit.group_id, created_at=saga_created_at)
    else:
        # Serial ingestion uses the object of the caller and saves it after the edges.
        if saga.group_id != unit.group_id:
            raise ValueError(f'saga {saga.uuid} does not belong to group {unit.group_id}')
        try:
            stored_saga = await SagaNode.get_by_uuid(driver, saga.uuid)
        except NodeNotFoundError:
            pass
        else:
            if stored_saga.group_id != unit.group_id:
                raise NodeNotFoundError(saga.uuid)
        saga_node = saga.model_copy()
    if previous_episode_uuid is None and not create_saga:
        previous_episode_uuid = await graphiti._saga_get_previous_episode_uuid(
            saga_node.uuid, current_episode_uuid, unit.group_id, driver=driver
        )

    links: list[SagaLink] = []
    for episode in episodes:
        links.append(
            SagaLink(
                saga_uuid=saga_node.uuid,
                episode_uuid=episode.uuid,
                previous_episode_uuid=previous_episode_uuid,
                has_episode_edge_uuid=str(uuid4()),
                next_episode_edge_uuid=str(uuid4()) if previous_episode_uuid is not None else None,
                created_at=now,
            )
        )
        previous_episode_uuid = episode.uuid
    return links, saga_node, create_saga


async def commit_unit(
    graphiti: 'Graphiti',
    store: IngestionStateStore,
    config: ConcurrentIngestionConfig,
    unit: AdmittedUnit,
    change_set: ChangeSet,
    *,
    entity_types: dict[str, type[BaseModel]] | None = None,
    identity_properties: dict[str, list[str]] | None = None,
    saga: 'str | SagaNode | None' = None,
    saga_previous_episode_uuid: str | None = None,
    wait: WaitFn | None = None,
    lease: UnitLease | None = None,
    reader: GraphReader | None = None,
) -> CommitResult:
    """Commit a unit in sequence order (steps C1 to C10).

    The function waits for the commit turn. With the turn, it repairs the change set
    against the graph and the delta, makes the final change set durable, applies it to
    the current records, and records the commit. When an earlier unit stopped after it
    stored its final change set, the function applies that change set first.

    The caller renews the deadline of the unit with `lease.heartbeat()` while the
    function runs. With no `lease`, the function renews the deadline while it waits.
    """
    group_id, _driver, clients = graphiti._resolve_request_scope(unit.group_id)
    wait = wait or _default_wait(config)
    reader = reader or GraphReader(clients)
    own_lease = lease is None
    lease = lease or UnitLease(store, config, group_id, unit.seq)
    report = new_report()

    with graphiti.tracer.start_span('ingest.commit') as span:
        start = time.time()
        final_saved = False
        try:
            recover_turns: dict[int, int] = {}
            while True:
                call_started_at = lease.clock()
                turn = await store.begin_commit(group_id, unit.seq, config.lease_seconds)
                if isinstance(turn, CommitTurn):
                    lease.fence = turn.fence
                    lease.renewed(call_started_at)
                    break
                if isinstance(turn, CommitRecover):
                    await _recover_unit(
                        clients,
                        store,
                        config,
                        group_id,
                        turn,
                        call_started_at,
                        lease.clock,
                        recover_turns,
                        report,
                        reader,
                    )
                    continue
                assert isinstance(turn, CommitWait)
                if own_lease:
                    await lease.heartbeat()
                await wait()
            report['turn_wait_ms'] = (time.time() - start) * 1000

            if turn.final_payload is not None:
                # C1: a retry applies the stored final change set and makes no model call.
                final = ChangeSet.from_bytes(turn.final_payload)
                final_saved = True
                report['retry'] = 1
            else:
                delta_units = await store.load_delta(
                    group_id,
                    change_set.snapshot_seq,
                    unit.seq,
                    change_set.analysis_started_at - config.index_lag_seconds,
                )
                final = await rebase_change_set(
                    clients,
                    change_set,
                    delta_units,
                    unit=unit,
                    reader=reader,
                    entity_types=entity_types,
                    identity_properties=identity_properties,
                    report=report,
                )
                (
                    final.saga_links,
                    final.saga_node,
                    final.create_saga,
                ) = await _build_saga_links(
                    graphiti, clients, unit, saga, saga_previous_episode_uuid
                )
                if config.before_commit_write is not None:
                    final = await config.before_commit_write(final)
                # C7c: the final change set is durable before the first graph write.
                await store.save_final(group_id, unit.seq, turn.fence, final.to_bytes())
                final_saved = True
        except BaseException as error:
            if not final_saved:
                await _fail_unit(store, group_id, unit.seq, error)
            raise

        try:
            changed_edges = await _apply_with_retries(
                clients,
                final,
                lease.write_budget,
                config.max_apply_attempts,
                report,
                reader,
                lease.heartbeat,
            )
        except (UnitExpiredError, StaleCommitError):
            raise
        except Exception:
            await _mark_apply_failed(store, group_id, unit.seq, turn.fence)
            raise
        await store.finish_commit(
            group_id, unit.seq, turn.fence, final.node_uuids, config.index_lag_seconds
        )

        report['duration_ms'] = (time.time() - start) * 1000
        span.add_attributes({'ingest.seq': unit.seq, **report})
        logger.info(
            'committed ingestion unit %s of group %s',
            unit.seq,
            group_id,
            extra={'ingest_group_id': group_id, 'ingest_seq': unit.seq, 'ingest_report': report},
        )
        return CommitResult(
            change_set=final, edges=[*final.create_edges, *changed_edges], report=report
        )


async def _hydrate_nodes(
    clients: GraphitiClients,
    unit: AdmittedUnit,
    final: ChangeSet,
    nodes: list[EntityNode],
    entity_types: dict[str, type[BaseModel]] | None,
) -> int:
    """Extract the attributes and the summaries of nodes. Return the number of failed nodes."""
    context = _episode_context(unit)
    default_episode_uuid = unit.episodes[0].uuid
    groups: dict[str, list[EntityNode]] = {}
    for node in nodes:
        episode_uuid = final.node_first_episode.get(node.uuid, default_episode_uuid)
        if episode_uuid not in context:
            episode_uuid = default_episode_uuid
        groups.setdefault(episode_uuid, []).append(node)

    async def hydrate(episode_uuid: str, group_nodes: list[EntityNode]) -> int:
        episode, previous_episodes = context[episode_uuid]
        try:
            await extract_attributes_from_nodes(
                clients,
                group_nodes,
                episode,
                previous_episodes,
                entity_types,
                # A bulk call of today passes no edges to the summary.
                edges=None if unit.bulk else final.create_edges,
            )
        except Exception:
            # The graph write of the commit is complete. The nodes keep the summary
            # and the attributes that they have.
            logger.error(
                'attribute extraction failed for nodes %s of ingestion unit %s of group %s',
                [node.uuid for node in group_nodes],
                unit.seq,
                unit.group_id,
                exc_info=True,
            )
            return len(group_nodes)
        return 0

    failed = await semaphore_gather(
        *[hydrate(episode_uuid, group_nodes) for episode_uuid, group_nodes in groups.items()],
        max_coroutines=getattr(clients, 'max_coroutines', None),
    )
    return sum(failed)


async def enrich_unit(
    graphiti: 'Graphiti',
    store: IngestionStateStore,
    config: ConcurrentIngestionConfig,
    unit: AdmittedUnit,
    commit_result: CommitResult,
    *,
    entity_types: dict[str, type[BaseModel]] | None = None,
    wait: WaitFn | None = None,
    lease: UnitLease | None = None,
    reader: GraphReader | None = None,
) -> list[EntityNode]:
    """Extract the attributes and the summaries of the nodes of a committed unit.

    For one node, the enrichment of two units runs in sequence order and not at the
    same time. The function reads the current node, applies the patches of the unit,
    and writes the node with a query that creates nothing. It returns the nodes that
    it wrote. An extraction error does not raise.
    """
    group_id, driver, clients = graphiti._resolve_request_scope(unit.group_id)
    wait = wait or _default_wait(config)
    reader = reader or GraphReader(clients)
    own_lease = lease is None
    lease = lease or UnitLease(store, config, group_id, unit.seq)

    final = commit_result.change_set
    patches = final.patches_by_node()
    created_embeddings = {node.uuid: node.name_embedding for node in final.create_nodes}
    remaining = list(dict.fromkeys(final.node_uuids))
    enriched: dict[str, EntityNode] = {}

    with graphiti.tracer.start_span('ingest.enrich') as span:
        start = time.time()
        waited = 0.0
        batch_count = 0
        failed_count = 0
        try:
            while remaining:
                if own_lease:
                    await lease.heartbeat()
                ready_uuids = set(await store.ready_enrich(group_id, unit.seq))
                ready = [uuid for uuid in remaining if uuid in ready_uuids]
                if not ready:
                    wait_started_at = time.time()
                    await wait()
                    waited += time.time() - wait_started_at
                    continue

                nodes = await reader.nodes_by_uuids(ready, group_id)
                for node in nodes:
                    for patch in patches.get(node.uuid, []):
                        _promote_resolved_node(patch_as_node(node, patch), node)
                failed_count += await _hydrate_nodes(clients, unit, final, nodes, entity_types)
                if config.before_enrichment_write is not None:
                    nodes = await config.before_enrichment_write(nodes, unit)

                if nodes:
                    for node in nodes:
                        if node.name_embedding is None:
                            node.name_embedding = created_embeddings.get(node.uuid)
                    await create_entity_node_embeddings(
                        clients.embedder, [node for node in nodes if node.name_embedding is None]
                    )
                    await execute_write_with_timeout(
                        driver,
                        lease.write_budget(),
                        update_entity_nodes_tx,
                        nodes,
                        clients.embedder,
                        driver=driver,
                    )
                await store.finish_enrich(group_id, unit.seq, ready)
                batch_count += 1
                remaining = [uuid for uuid in remaining if uuid not in ready_uuids]
                enriched.update({node.uuid: node for node in nodes})
        except BaseException:
            # Release the enrichment order of the nodes for the later units.
            try:
                await store.finish_enrich(group_id, unit.seq, remaining)
            except Exception:
                logger.debug(
                    'could not release the enrichment entries of unit %s of group %s',
                    unit.seq,
                    group_id,
                )
            raise

        span.add_attributes(
            {
                'ingest.seq': unit.seq,
                'wait_ms': waited * 1000,
                'duration_ms': (time.time() - start) * 1000,
                'node.count': len(enriched),
                'batch.count': batch_count,
                'node.failed': failed_count,
            }
        )
    return [enriched[uuid] for uuid in final.node_uuids if uuid in enriched]
