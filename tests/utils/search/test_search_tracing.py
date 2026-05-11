from collections.abc import Generator
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from graphiti_core.search.search import edge_search, search
from graphiti_core.search.search_config import EdgeSearchConfig, EdgeSearchMethod, SearchConfig
from graphiti_core.search.search_filters import SearchFilters


class RecordingSpan:
    def __init__(self, name: str):
        self.name = name
        self.attributes: dict[str, object] = {}
        self.status: tuple[str, str | None] | None = None
        self.exception: Exception | None = None

    def add_attributes(self, attributes: dict[str, object]) -> None:
        self.attributes.update(attributes)

    def set_status(self, status: str, description: str | None = None) -> None:
        self.status = (status, description)

    def record_exception(self, exception: Exception) -> None:
        self.exception = exception


class RecordingTracer:
    def __init__(self):
        self.spans: list[RecordingSpan] = []

    @contextmanager
    def start_span(self, name: str) -> Generator[RecordingSpan, None, None]:
        span = RecordingSpan(name)
        self.spans.append(span)
        yield span


@pytest.mark.asyncio
async def test_search_emits_trace_spans_for_edge_similarity(monkeypatch):
    async def fake_edge_similarity_search(*args, **kwargs):
        return []

    async def fake_embedder_create(*, input_data):
        return [0.1, 0.2, 0.3]

    monkeypatch.setattr(
        'graphiti_core.search.search.edge_similarity_search',
        fake_edge_similarity_search,
    )

    embedder = SimpleNamespace(create=fake_embedder_create)
    tracer = RecordingTracer()
    clients = SimpleNamespace(
        driver=SimpleNamespace(),
        embedder=embedder,
        cross_encoder=SimpleNamespace(),
        tracer=tracer,
    )

    results = await search(
        clients,
        query='where is the edge',
        group_ids=None,
        config=SearchConfig(
            edge_config=EdgeSearchConfig(search_methods=[EdgeSearchMethod.cosine_similarity]),
        ),
        search_filter=SearchFilters(),
    )

    assert results.edges == []
    assert results.nodes == []
    assert results.episodes == []
    assert results.communities == []

    span_names = [span.name for span in tracer.spans]
    assert 'search.embed_query_vector' in span_names
    assert 'search.execute_scopes' in span_names
    assert 'search.edge_search' in span_names
    assert 'search.edge_search.execute_methods' in span_names
    assert 'search.edge_search.rerank' in span_names

    execute_scopes_span = next(
        span for span in tracer.spans if span.name == 'search.execute_scopes'
    )
    assert execute_scopes_span.attributes['scope.edges'] is True
    assert execute_scopes_span.attributes['result.edges'] == 0

    embed_span = next(span for span in tracer.spans if span.name == 'search.embed_query_vector')
    assert embed_span.attributes['query.length'] == len('where is the edge')
    assert embed_span.attributes['query_vector.dimension'] == 3


@pytest.mark.asyncio
async def test_search_uses_noop_tracer_when_client_has_no_tracer(monkeypatch):
    async def fake_edge_similarity_search(*args, **kwargs):
        return []

    async def fake_embedder_create(*, input_data):
        return [0.1, 0.2, 0.3]

    monkeypatch.setattr(
        'graphiti_core.search.search.edge_similarity_search',
        fake_edge_similarity_search,
    )

    clients = SimpleNamespace(
        driver=SimpleNamespace(),
        embedder=SimpleNamespace(create=fake_embedder_create),
        cross_encoder=SimpleNamespace(),
    )

    results = await search(
        clients,
        query='where is the edge',
        group_ids=None,
        config=SearchConfig(
            edge_config=EdgeSearchConfig(search_methods=[EdgeSearchMethod.cosine_similarity]),
        ),
        search_filter=SearchFilters(),
    )

    assert results.edges == []


@pytest.mark.asyncio
async def test_edge_search_uses_noop_tracer_when_none_is_passed(monkeypatch):
    async def fake_edge_similarity_search(*args, **kwargs):
        return []

    monkeypatch.setattr(
        'graphiti_core.search.search.edge_similarity_search',
        fake_edge_similarity_search,
    )

    edges, scores = await edge_search(
        driver=SimpleNamespace(),
        cross_encoder=SimpleNamespace(),
        query='where is the edge',
        query_vector=[0.1, 0.2, 0.3],
        group_ids=None,
        config=EdgeSearchConfig(search_methods=[EdgeSearchMethod.cosine_similarity]),
        search_filter=SearchFilters(),
        search_tracer=None,
    )

    assert edges == []
    assert scores == []
