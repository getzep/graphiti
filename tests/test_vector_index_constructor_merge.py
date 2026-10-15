"""Guard vector-driver wiring alongside the upstream prompt/runtime constructor."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from graphiti_core.graphiti import Graphiti
from graphiti_core.prompts import create_prompt_library


@pytest.mark.parametrize('use_runtime', [False, True], ids=['prompt-library', 'llm-runtime'])
@pytest.mark.parametrize('use_vector_index', [False, True])
def test_vector_driver_keeps_prompt_runtime_and_concurrency_wiring(
    monkeypatch, use_runtime, use_vector_index
):
    driver = SimpleNamespace()
    driver_factory = Mock(return_value=driver)
    embedder = SimpleNamespace(config=SimpleNamespace(embedding_dim=768))
    embedder_factory = Mock(return_value=embedder)
    llm = SimpleNamespace(set_tracer=Mock())
    library = create_prompt_library()
    runtime = SimpleNamespace(client=llm, library=library, set_tracer=Mock())
    clients_factory = Mock(side_effect=lambda **kwargs: SimpleNamespace(**kwargs))
    monkeypatch.setattr('graphiti_core.graphiti.Neo4jDriver', driver_factory)
    monkeypatch.setattr('graphiti_core.graphiti.OpenAIEmbedder', embedder_factory)
    monkeypatch.setattr('graphiti_core.graphiti.OpenAIClient', Mock(side_effect=AssertionError))
    monkeypatch.setattr('graphiti_core.graphiti.GraphitiClients', clients_factory)
    monkeypatch.setattr('graphiti_core.graphiti.NodeNamespace', Mock())
    monkeypatch.setattr('graphiti_core.graphiti.EdgeNamespace', Mock())
    options = (
        {'llm_runtime': runtime} if use_runtime else {'llm_client': llm, 'prompt_library': library}
    )

    graphiti = Graphiti(
        uri='bolt://unused',
        user='neo4j',
        password='unused',
        cross_encoder=SimpleNamespace(),
        max_coroutines=3,
        use_vector_index=use_vector_index,
        **options,
    )

    embedder_factory.assert_called_once_with()
    driver_factory.assert_called_once_with(
        'bolt://unused', 'neo4j', 'unused', embedding_dim=768, use_vector_index=use_vector_index
    )
    assert graphiti.embedder is embedder
    assert graphiti.llm_client is llm
    assert graphiti.prompt_library is library
    assert graphiti.llm_runtime is (runtime if use_runtime else None)
    assert graphiti.clients.driver is driver
    assert graphiti.clients.embedder is embedder
    assert graphiti.clients.prompt_library is library
    assert graphiti.clients.llm_runtime is graphiti.llm_runtime
    assert graphiti.clients.max_coroutines == 3
    (runtime.set_tracer if use_runtime else llm.set_tracer).assert_called_once_with(graphiti.tracer)
    (llm.set_tracer if use_runtime else runtime.set_tracer).assert_not_called()


@pytest.mark.parametrize('extra', ['llm_client', 'prompt_library'])
def test_runtime_exclusivity_precedes_vector_driver_initialization(monkeypatch, extra):
    driver_factory = Mock(side_effect=AssertionError)
    embedder_factory = Mock(side_effect=AssertionError)
    monkeypatch.setattr('graphiti_core.graphiti.Neo4jDriver', driver_factory)
    monkeypatch.setattr('graphiti_core.graphiti.OpenAIEmbedder', embedder_factory)

    with pytest.raises(ValueError, match=f'llm_runtime cannot be combined with: {extra}'):
        Graphiti(
            uri='bolt://unused',
            llm_runtime=SimpleNamespace(),
            use_vector_index=True,
            **{extra: SimpleNamespace()},
        )

    driver_factory.assert_not_called()
    embedder_factory.assert_not_called()


def _patch_graphiti_dependencies(monkeypatch):
    driver = SimpleNamespace()
    embedder = SimpleNamespace(config=SimpleNamespace(embedding_dim=768))
    llm = SimpleNamespace(set_tracer=Mock())
    monkeypatch.setattr('graphiti_core.graphiti.OpenAIEmbedder', Mock(return_value=embedder))
    monkeypatch.setattr(
        'graphiti_core.graphiti.OpenAIRerankerClient', Mock(side_effect=AssertionError)
    )
    monkeypatch.setattr('graphiti_core.graphiti.OpenAIClient', Mock(side_effect=AssertionError))
    monkeypatch.setattr(
        'graphiti_core.graphiti.GraphitiClients',
        Mock(side_effect=lambda **kwargs: SimpleNamespace(**kwargs)),
    )
    monkeypatch.setattr('graphiti_core.graphiti.NodeNamespace', Mock())
    monkeypatch.setattr('graphiti_core.graphiti.EdgeNamespace', Mock())
    return driver, embedder, llm


def test_upstream_positional_prompt_library_compatibility(monkeypatch):
    driver, embedder, llm = _patch_graphiti_dependencies(monkeypatch)
    library = create_prompt_library()
    cross_encoder = SimpleNamespace()

    graphiti = Graphiti(
        None,
        None,
        None,
        llm,
        embedder,
        cross_encoder,
        True,
        driver,
        None,
        None,
        'graphiti',
        library,
    )

    assert graphiti.prompt_library is library
    assert graphiti.llm_runtime is None


def test_upstream_positional_runtime_compatibility(monkeypatch):
    driver, embedder, llm = _patch_graphiti_dependencies(monkeypatch)
    library = create_prompt_library()
    runtime = SimpleNamespace(client=llm, library=library, set_tracer=Mock())
    cross_encoder = SimpleNamespace()

    graphiti = Graphiti(
        None,
        None,
        None,
        None,
        embedder,
        cross_encoder,
        True,
        driver,
        None,
        None,
        'graphiti',
        None,
        runtime,
    )

    assert graphiti.prompt_library is library
    assert graphiti.llm_runtime is runtime
