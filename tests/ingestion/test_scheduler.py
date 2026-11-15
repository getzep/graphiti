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

# Tests for the in-process scheduler and for the dispatch in `Graphiti`.

import asyncio
import subprocess
import sys
from unittest.mock import AsyncMock, Mock

import pytest

from graphiti_core.cross_encoder.client import CrossEncoderClient
from graphiti_core.embedder import EmbedderClient
from graphiti_core.graphiti import Graphiti
from graphiti_core.graphiti_types import GraphitiClients
from graphiti_core.ingestion import ConcurrentIngestionConfig, scheduler
from graphiti_core.ingestion.scheduler import ConcurrentIngestor
from graphiti_core.ingestion.state_store import STATUS_FAILED, InMemoryIngestionStateStore
from graphiti_core.llm_client import LLMClient
from graphiti_core.nodes import EpisodeType
from graphiti_core.tracer import NoOpTracer
from tests.ingestion.ingestion_fixtures import GROUP, T0, Harness, make_node


def one_hot(index: int) -> list[float]:
    """Return an embedding that has no similarity with the embedding of a different index."""
    return [1.0 if position == index else 0.0 for position in range(64)]


class SchedulerHarness(Harness):
    """A scheduler on the fake graph, with an analysis that makes one node for each unit."""

    def __init__(self, monkeypatch, analysis_seconds: float = 0.01, **config_values):
        super().__init__(monkeypatch, **config_values)
        self.ingestor = ConcurrentIngestor(self.graphiti, self.config)
        self.seq_by_name: dict[str, int] = {}
        self.running = 0
        self.max_running = 0
        self.fail_names: set[str] = set()

        async def admit(graphiti, store, config, *, unit_id, name, **kwargs):
            unit = await self.admit(unit_id)
            unit.episodes[0].name = name
            self.seq_by_name[name] = unit.seq
            return unit

        async def analyze(graphiti, store, config, unit, **kwargs):
            self.running += 1
            self.max_running = max(self.max_running, self.running)
            try:
                await asyncio.sleep(analysis_seconds)
                if unit.episodes[0].name in self.fail_names:
                    await store.fail(unit.group_id, unit.seq, 'analysis')
                    raise RuntimeError('analysis failed')
                return await self.analyze(
                    unit,
                    create_nodes=[make_node(f'n{unit.seq}', f'Node {unit.seq}', one_hot(unit.seq))],
                )
            finally:
                self.running -= 1

        async def commit(graphiti, store, config, unit, change_set, **kwargs):
            kwargs['reader'] = self.graph
            return await real_commit(graphiti, store, config, unit, change_set, **kwargs)

        async def enrich(graphiti, store, config, unit, commit_result, **kwargs):
            kwargs['reader'] = self.graph
            return await real_enrich(graphiti, store, config, unit, commit_result, **kwargs)

        real_commit, real_enrich = scheduler.commit_unit, scheduler.enrich_unit
        monkeypatch.setattr(scheduler, 'admit_episode', admit)
        monkeypatch.setattr(scheduler, 'analyze_episode', analyze)
        monkeypatch.setattr(scheduler, 'commit_unit', commit)
        monkeypatch.setattr(scheduler, 'enrich_unit', enrich)

    async def add(self, name: str):
        return await self.ingestor.add_episode(
            name=name,
            episode_body='content',
            source_description='test',
            reference_time=T0,
            source=EpisodeType.text,
            group_id=GROUP,
            uuid=None,
            entity_types=None,
            excluded_entity_types=None,
            previous_episode_uuids=None,
            edge_types=None,
            edge_type_map=None,
            custom_extraction_instructions=None,
            saga=None,
            saga_previous_episode_uuid=None,
        )


async def test_window_limits_the_units_and_sequence_order_is_call_order(monkeypatch):
    h = SchedulerHarness(monkeypatch, window=3)
    names = [f'episode-{index}' for index in range(10)]

    # All calls are in flight at one time. This is more than the window.
    results = await asyncio.gather(*[h.add(name) for name in names])

    assert [h.seq_by_name[name] for name in names] == list(range(1, 11))
    assert h.max_running == 3
    assert h.store._groups[GROUP].watermark == 10
    assert set(h.graph.nodes) == {f'n{seq}' for seq in range(1, 11)}
    for seq, result in enumerate(results, start=1):
        assert [node.uuid for node in result.nodes] == [f'n{seq}']
        assert result.nodes[0].summary == f'+ep-{result.episode.uuid[3:]}'
        assert result.communities == [] and result.community_edges == []
    # Each commit and each enrichment is complete when the calls return.
    assert h.store._groups[GROUP].enrich == {}
    assert h.graph.log.count('store:finish_commit') == 10

    root_spans = h.tracer.named('add_episode')
    assert len(root_spans) == 10
    assert {span.attributes['ingest.seq'] for span in root_spans} == set(range(1, 11))
    for span in root_spans:
        assert span.attributes['ingest.mode'] == 'concurrent'
        assert span.attributes['ingest.window'] == 3
        assert span.attributes['ingest.unit_id']
    assert len(h.tracer.named('ingest.commit')) == 10
    assert len(h.tracer.named('ingest.enrich')) == 10


async def test_window_of_one_runs_one_unit_at_a_time(monkeypatch):
    h = SchedulerHarness(monkeypatch, window=1)
    await asyncio.gather(*[h.add(f'episode-{index}') for index in range(4)])

    assert h.max_running == 1
    assert h.graph.log.count('store:finish_commit') == 4


async def test_failed_unit_releases_its_slot_and_its_position(monkeypatch):
    h = SchedulerHarness(monkeypatch, window=2)
    h.fail_names = {'episode-1'}
    results = await asyncio.gather(
        *[h.add(f'episode-{index}') for index in range(4)], return_exceptions=True
    )

    assert isinstance(results[1], RuntimeError)
    assert [isinstance(result, Exception) for result in results] == [False, True, False, False]
    assert h.store._groups[GROUP].units[2].status == STATUS_FAILED
    assert set(h.graph.nodes) == {'n1', 'n3', 'n4'}
    assert h.tracer.named('add_episode')[1].status == 'error'
    # No heartbeat task stays after the calls.
    await asyncio.sleep(0)
    assert [task for task in asyncio.all_tasks() if task is not asyncio.current_task()] == []


async def test_heartbeat_renews_the_deadline_of_a_slow_unit(monkeypatch):
    h = SchedulerHarness(
        monkeypatch, window=1, analysis_seconds=0.25, lease_seconds=0.15, write_margin_seconds=0.01
    )
    result = await h.add('episode-0')

    # The analysis takes more time than the lease. The heartbeat keeps the unit live.
    assert [node.uuid for node in result.nodes] == ['n1']


# --- dispatch in Graphiti ----------------------------------------------------------


class SerialPathRan(Exception):
    pass


def make_graphiti(config: ConcurrentIngestionConfig | None) -> Graphiti:
    driver = Mock()
    driver._database = GROUP
    driver.close = AsyncMock()
    graphiti = Graphiti.__new__(Graphiti)
    graphiti.driver = driver
    graphiti.tracer = NoOpTracer()
    graphiti.clients = GraphitiClients.model_construct(  # no validation, for test doubles
        driver=driver,
        llm_client=Mock(spec=LLMClient),
        embedder=Mock(spec=EmbedderClient),
        cross_encoder=Mock(spec=CrossEncoderClient),
        tracer=graphiti.tracer,
    )
    graphiti.concurrent_ingestion = config
    graphiti._concurrent_ingestor = None
    # The first call of the serial path raises, so a test can see that the path ran.
    graphiti.retrieve_episodes = AsyncMock(side_effect=SerialPathRan())  # type: ignore[method-assign]
    return graphiti


async def add_episode(graphiti: Graphiti, **kwargs):
    return await graphiti.add_episode(
        name='episode',
        episode_body='content',
        source_description='test',
        reference_time=T0,
        group_id=GROUP,
        **kwargs,
    )


async def test_mode_off_runs_serial_ingestion_and_makes_no_scheduler(monkeypatch):
    concurrent_call = AsyncMock()
    monkeypatch.setattr(ConcurrentIngestor, 'add_episode', concurrent_call)

    graphiti = make_graphiti(None)
    with pytest.raises(SerialPathRan):
        await add_episode(graphiti)
    assert graphiti._concurrent_ingestor is None

    # An object with no call of `__init__` has no configuration attribute.
    del graphiti.concurrent_ingestion, graphiti._concurrent_ingestor
    with pytest.raises(SerialPathRan):
        await add_episode(graphiti)
    concurrent_call.assert_not_awaited()


async def test_enabled_for_group_selects_the_path(monkeypatch):
    concurrent_call = AsyncMock(return_value='concurrent result')
    monkeypatch.setattr(ConcurrentIngestor, 'add_episode', concurrent_call)
    config = ConcurrentIngestionConfig(
        window=2, enabled_for_group=lambda group_id: group_id == GROUP
    )
    graphiti = make_graphiti(config)

    assert await add_episode(graphiti, saga='thread') == 'concurrent result'
    assert concurrent_call.await_args.kwargs['group_id'] == GROUP
    assert concurrent_call.await_args.kwargs['saga'] == 'thread'

    config.enabled_for_group = lambda group_id: False
    disabled = make_graphiti(config)
    with pytest.raises(SerialPathRan):
        await add_episode(disabled)
    # The scheduler and the state store are not made for a graph that does not use the mode.
    assert disabled._concurrent_ingestor is None
    assert concurrent_call.await_count == 1


async def test_update_communities_is_rejected_on_the_concurrent_path(monkeypatch):
    monkeypatch.setattr(ConcurrentIngestor, 'add_episode', AsyncMock())
    graphiti = make_graphiti(ConcurrentIngestionConfig(window=2))

    with pytest.raises(ValueError, match='update_communities'):
        await add_episode(graphiti, update_communities=True)


async def test_bulk_call_uses_the_scheduler_only_when_the_mode_is_on(monkeypatch):
    concurrent_call = AsyncMock(return_value='concurrent result')
    monkeypatch.setattr(ConcurrentIngestor, 'add_episode_bulk', concurrent_call)

    graphiti = make_graphiti(ConcurrentIngestionConfig(window=2))
    assert await graphiti.add_episode_bulk([], group_id=GROUP) == 'concurrent result'
    assert concurrent_call.await_args.kwargs['group_id'] == GROUP
    assert concurrent_call.await_args.kwargs['use_combined_extraction'] is True


async def test_close_closes_the_state_store():
    store = InMemoryIngestionStateStore()
    store.close = AsyncMock()  # type: ignore[method-assign]
    graphiti = make_graphiti(ConcurrentIngestionConfig(window=2, state_store=store))

    await graphiti.close()

    store.close.assert_awaited_once()
    graphiti.driver.close.assert_awaited_once()


def test_serial_ingestion_does_not_import_the_ingestion_package():
    # The test runs in a new process. In this process, a different test imported the package.
    code = (
        'import sys\n'
        'import graphiti_core\n'
        'from graphiti_core import Graphiti\n'
        'import graphiti_core.utils.bulk_utils\n'
        'import graphiti_core.utils.maintenance.edge_operations\n'
        'import graphiti_core.utils.maintenance.node_operations\n'
        "loaded = sorted(name for name in sys.modules if name.startswith('graphiti_core.ingestion'))\n"
        'assert loaded == [], loaded\n'
        "print('ok')\n"
    )
    result = subprocess.run(
        [sys.executable, '-c', code], capture_output=True, text=True, timeout=120, check=False
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == 'ok'
