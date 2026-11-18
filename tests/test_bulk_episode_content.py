"""`store_raw_episode_content` must hold for the bulk write path too.

`add_episode` clears the content before the graph write (`_process_episode_data`) and
concurrent ingestion does the same (`_written_episode`). `add_episode_bulk` writes the
episodes twice - before and after the extraction - so both writes have to apply the
flag while the model stages keep the text.
"""

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from graphiti_core import graphiti as graphiti_module
from graphiti_core.driver.driver import GraphProvider
from graphiti_core.graphiti import Graphiti
from graphiti_core.graphiti_types import GraphitiClients
from graphiti_core.nodes import EpisodeType
from graphiti_core.tracer import NoOpTracer
from graphiti_core.utils.bulk_utils import RawEpisode

GROUP_ID = 'bulk-content-group'
RAW_CONTENT = 'Alice works at Acme since 2020. Bob met Alice in Berlin.'


class _FakeDriver:
    """Only the members that `_resolve_request_scope` reads."""

    provider = GraphProvider.NEO4J
    _database = GROUP_ID

    def clone(self, database: str) -> '_FakeDriver':
        return self


def _make_graphiti(store_raw_episode_content: bool) -> Graphiti:
    driver = _FakeDriver()
    graphiti = Graphiti.__new__(Graphiti)
    graphiti.driver = driver
    graphiti.tracer = NoOpTracer()
    graphiti.max_coroutines = None
    graphiti.concurrent_ingestion = None
    graphiti.store_raw_episode_content = store_raw_episode_content
    graphiti.clients = GraphitiClients.model_construct(
        driver=driver,
        llm_client=SimpleNamespace(),
        embedder=SimpleNamespace(),
        cross_encoder=SimpleNamespace(),
        tracer=graphiti.tracer,
        prompt_library=None,
        llm_runtime=None,
        max_coroutines=None,
    )
    return graphiti


def _raw_episode(name: str) -> RawEpisode:
    return RawEpisode(
        name=name,
        content=RAW_CONTENT,
        source_description='bulk-content-test',
        source=EpisodeType.text,
        reference_time=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )


def _stub_pipeline(monkeypatch) -> tuple[list[str], list[str]]:
    """Stub the model stages and record the episode records of each graph write."""
    written_contents: list[str] = []
    extracted_contents: list[str] = []

    async def fake_add_nodes_and_edges_bulk(*args, **kwargs):
        episodes = kwargs.get('episodic_nodes')
        if episodes is None:
            episodes = args[1]
        written_contents.extend(episode.content for episode in episodes)

    async def fake_retrieve_previous_episodes_bulk(driver, episodes, max_coroutines=None):
        return [(episode, []) for episode in episodes]

    async def fake_extract_and_dedupe(self, episode_context, *args, **kwargs):
        extracted_contents.extend(episode.content for episode, _ in episode_context)
        return (
            {episode.uuid: [] for episode, _ in episode_context},
            {},
            [[] for _ in episode_context],
        )

    async def fake_dedupe_edges_bulk(*args, **kwargs):
        return {}

    async def fake_resolve_nodes_and_edges_bulk(self, *args, **kwargs):
        return ([], [], [], {})

    monkeypatch.setattr(graphiti_module, 'add_nodes_and_edges_bulk', fake_add_nodes_and_edges_bulk)
    monkeypatch.setattr(
        graphiti_module, 'retrieve_previous_episodes_bulk', fake_retrieve_previous_episodes_bulk
    )
    monkeypatch.setattr(
        Graphiti, '_extract_and_dedupe_nodes_bulk', fake_extract_and_dedupe, raising=True
    )
    monkeypatch.setattr(graphiti_module, 'dedupe_edges_bulk', fake_dedupe_edges_bulk)
    monkeypatch.setattr(
        Graphiti, '_resolve_nodes_and_edges_bulk', fake_resolve_nodes_and_edges_bulk, raising=True
    )
    return written_contents, extracted_contents


@pytest.mark.asyncio
async def test_add_episode_bulk_does_not_store_raw_content_when_disabled(monkeypatch):
    written_contents, extracted_contents = _stub_pipeline(monkeypatch)
    graphiti = _make_graphiti(store_raw_episode_content=False)

    results = await graphiti.add_episode_bulk(
        [_raw_episode('first'), _raw_episode('second')], group_id=GROUP_ID
    )

    # Each episode is written twice: before and after the extraction.
    assert len(written_contents) == 4
    assert written_contents == ['', '', '', '']
    # The model stages still receive the text, otherwise extraction sees nothing.
    assert extracted_contents == [RAW_CONTENT, RAW_CONTENT]
    # The records of the result match what the graph holds, as in `add_episode`.
    assert [episode.content for episode in results.episodes] == ['', '']


@pytest.mark.asyncio
async def test_add_episode_bulk_stores_raw_content_by_default(monkeypatch):
    written_contents, extracted_contents = _stub_pipeline(monkeypatch)
    graphiti = _make_graphiti(store_raw_episode_content=True)

    results = await graphiti.add_episode_bulk([_raw_episode('first')], group_id=GROUP_ID)

    assert written_contents == [RAW_CONTENT, RAW_CONTENT]
    assert extracted_contents == [RAW_CONTENT]
    assert [episode.content for episode in results.episodes] == [RAW_CONTENT]
