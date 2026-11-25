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
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from pydantic import BaseModel

from graphiti_core.errors import StaleCommitError, UnitExpiredError
from graphiti_core.ingestion.config import ConcurrentIngestionConfig
from graphiti_core.ingestion.phases import (
    AdmittedUnit,
    CommitResult,
    UnitLease,
    _fail_unit,
    admit_bulk,
    admit_episode,
    analyze_bulk,
    analyze_episode,
    commit_unit,
    enrich_unit,
)
from graphiti_core.ingestion.state_store import IngestionStateStore
from graphiti_core.nodes import EntityNode, EpisodeType, SagaNode
from graphiti_core.utils.bulk_utils import RawEpisode

if TYPE_CHECKING:
    from graphiti_core.graphiti import AddBulkEpisodeResults, AddEpisodeResults, Graphiti

logger = logging.getLogger(__name__)


@dataclass
class _GroupState:
    """The window, the admit lock, and the wake-up event of one graph."""

    window: asyncio.Semaphore
    admit_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    changed: asyncio.Event = field(default_factory=asyncio.Event)

    def notify(self) -> None:
        """Wake up each unit of the graph that waits for a different unit."""
        self.changed.set()
        self.changed = asyncio.Event()


class ConcurrentIngestor:
    """The in-process scheduler of concurrent ingestion.

    For each graph, at most `window` units run at one time. A unit keeps its window
    slot from admission to the end of enrichment. The window slot and the admit lock
    are the first two awaits of a call. Calls that one task starts in order thus get
    their sequence numbers in that order.
    """

    def __init__(self, graphiti: 'Graphiti', config: ConcurrentIngestionConfig):
        self.graphiti = graphiti
        self.config = config
        self.store: IngestionStateStore = config.get_state_store()
        self._groups: dict[str, _GroupState] = {}

    def _group(self, group_id: str) -> _GroupState:
        state = self._groups.get(group_id)
        if state is None:
            state = _GroupState(window=asyncio.Semaphore(self.config.window))
            self._groups[group_id] = state
        return state

    def _wait_local(self, state: _GroupState):
        async def wait() -> None:
            # A unit of this process wakes up the waiters. The poll interval applies
            # to a change that a different process made.
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(
                    state.changed.wait(), timeout=self.config.poll_interval_seconds
                )

        return wait

    async def _heartbeat(self, lease: UnitLease) -> None:
        interval = self.config.lease_seconds / 3
        while True:
            await asyncio.sleep(interval)
            try:
                await lease.heartbeat()
            except (UnitExpiredError, StaleCommitError):
                # The unit is expired or failed, or a different attempt has its turn.
                # The next store call of the unit raises the error.
                return
            except Exception:
                logger.warning(
                    'heartbeat of ingestion unit %s of group %s failed',
                    lease.seq,
                    lease.group_id,
                    exc_info=True,
                )

    async def _run_unit(
        self,
        state: _GroupState,
        unit: AdmittedUnit,
        span: Any,
        analyze: Any,
        commit_kwargs: dict[str, Any],
        entity_types: dict[str, type[BaseModel]] | None,
    ) -> tuple[CommitResult, list[EntityNode]]:
        """Run the Analyze, Commit, and Enrich phases of an admitted unit."""
        graphiti, store, config = self.graphiti, self.store, self.config
        span.add_attributes(
            {
                'ingest.mode': 'concurrent',
                'ingest.seq': unit.seq,
                'ingest.window': config.window,
                'ingest.unit_id': unit.unit_id,
                'group_id': unit.group_id,
            }
        )
        wait = self._wait_local(state)
        lease = UnitLease(store, config, unit.group_id, unit.seq)
        heartbeat = asyncio.create_task(self._heartbeat(lease))
        commit_result: CommitResult | None = None
        try:
            await lease.heartbeat()
            change_set = await analyze(unit)
            commit_result = await commit_unit(
                graphiti,
                store,
                config,
                unit,
                change_set,
                entity_types=entity_types,
                wait=wait,
                lease=lease,
                **commit_kwargs,
            )
            state.notify()
            nodes = await enrich_unit(
                graphiti,
                store,
                config,
                unit,
                commit_result,
                entity_types=entity_types,
                wait=wait,
                lease=lease,
            )
            return commit_result, nodes
        except BaseException as error:
            if commit_result is None:
                # The call has no effect on a unit that has a stored final change set.
                await _fail_unit(store, unit.group_id, unit.seq, error)
            raise
        finally:
            heartbeat.cancel()
            await asyncio.gather(heartbeat, return_exceptions=True)
            state.notify()

    async def add_episode(
        self,
        *,
        name: str,
        episode_body: str,
        source_description: str,
        reference_time: datetime,
        source: EpisodeType,
        group_id: str,
        uuid: str | None,
        entity_types: dict[str, type[BaseModel]] | None,
        excluded_entity_types: list[str] | None,
        previous_episode_uuids: list[str] | None,
        edge_types: dict[str, type[BaseModel]] | None,
        edge_type_map: dict[tuple[str, str], list[str]] | None,
        custom_extraction_instructions: str | None,
        saga: str | SagaNode | None,
        saga_previous_episode_uuid: str | None,
        identity_properties: dict[str, list[str]] | None = None,
    ) -> 'AddEpisodeResults':
        from graphiti_core.graphiti import AddEpisodeResults

        graphiti, store, config = self.graphiti, self.store, self.config
        state = self._group(group_id)

        async with state.window:
            with graphiti.tracer.start_span('add_episode') as span:
                try:
                    async with state.admit_lock:
                        unit = await admit_episode(
                            graphiti,
                            store,
                            config,
                            unit_id=str(uuid4()),
                            group_id=group_id,
                            name=name,
                            episode_body=episode_body,
                            source_description=source_description,
                            reference_time=reference_time,
                            source=source,
                            uuid=uuid,
                            previous_episode_uuids=previous_episode_uuids,
                        )

                    async def analyze(unit: AdmittedUnit):
                        return await analyze_episode(
                            graphiti,
                            store,
                            config,
                            unit,
                            entity_types=entity_types,
                            excluded_entity_types=excluded_entity_types,
                            edge_types=edge_types,
                            edge_type_map=edge_type_map,
                            custom_extraction_instructions=custom_extraction_instructions,
                            identity_properties=identity_properties,
                        )

                    commit_result, nodes = await self._run_unit(
                        state,
                        unit,
                        span,
                        analyze,
                        {
                            'saga': saga,
                            'saga_previous_episode_uuid': saga_previous_episode_uuid,
                            'identity_properties': identity_properties,
                        },
                        entity_types,
                    )
                except Exception as error:
                    span.set_status('error', str(error))
                    span.record_exception(error)
                    raise

                final = commit_result.change_set
                span.add_attributes(
                    {
                        'episode.uuid': final.episodes[0].uuid,
                        'node.count': len(nodes),
                        'edge.count': len(commit_result.edges),
                    }
                )
                return AddEpisodeResults(
                    episode=final.episodes[0],
                    episodic_edges=final.episodic_edges,
                    nodes=nodes,
                    edges=commit_result.edges,
                    communities=[],
                    community_edges=[],
                )

    async def add_episode_bulk(
        self,
        *,
        bulk_episodes: list[RawEpisode],
        group_id: str,
        entity_types: dict[str, type[BaseModel]] | None,
        excluded_entity_types: list[str] | None,
        edge_types: dict[str, type[BaseModel]] | None,
        edge_type_map: dict[tuple[str, str], list[str]] | None,
        strict_edge_types: bool,
        custom_extraction_instructions: str | None,
        saga: str | SagaNode | None,
        use_combined_extraction: bool,
        identity_properties: dict[str, list[str]] | None = None,
    ) -> 'AddBulkEpisodeResults':
        from graphiti_core.graphiti import AddBulkEpisodeResults

        graphiti, store, config = self.graphiti, self.store, self.config
        state = self._group(group_id)

        async with state.window:
            with graphiti.tracer.start_span('add_episode_bulk') as span:
                span.add_attributes({'episode.count': len(bulk_episodes)})
                try:
                    async with state.admit_lock:
                        unit = await admit_bulk(
                            graphiti,
                            store,
                            config,
                            unit_id=str(uuid4()),
                            group_id=group_id,
                            bulk_episodes=bulk_episodes,
                        )

                    async def analyze(unit: AdmittedUnit):
                        return await analyze_bulk(
                            graphiti,
                            store,
                            config,
                            unit,
                            entity_types=entity_types,
                            excluded_entity_types=excluded_entity_types,
                            edge_types=edge_types,
                            edge_type_map=edge_type_map,
                            custom_extraction_instructions=custom_extraction_instructions,
                            identity_properties=identity_properties,
                            strict_edge_types=strict_edge_types,
                            use_combined_extraction=use_combined_extraction,
                        )

                    commit_result, nodes = await self._run_unit(
                        state,
                        unit,
                        span,
                        analyze,
                        {'saga': saga, 'identity_properties': identity_properties},
                        entity_types,
                    )
                except Exception as error:
                    span.set_status('error', str(error))
                    span.record_exception(error)
                    raise

                final = commit_result.change_set
                span.add_attributes(
                    {'node.count': len(nodes), 'edge.count': len(commit_result.edges)}
                )
                return AddBulkEpisodeResults(
                    episodes=final.episodes,
                    episodic_edges=final.episodic_edges,
                    nodes=nodes,
                    edges=commit_result.edges,
                    communities=[],
                    community_edges=[],
                )

    async def close(self) -> None:
        """Close the state store."""
        await self.store.close()
