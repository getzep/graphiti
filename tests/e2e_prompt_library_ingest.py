"""Local e2e: Graphiti add_episode against Neo4j.

Modes:
  mock (default) — stub LLM/embedder; no OpenAI calls
  openai         — real OpenAI LLM + embedder (Graphiti defaults)

Run mock:
  GRAPHITI_TELEMETRY_ENABLED=false NEO4J_PASSWORD=testpass \\
    uv run python tests/e2e_prompt_library_ingest.py

Run real OpenAI:
  set -a && source .env && set +a
  GRAPHITI_TELEMETRY_ENABLED=false NEO4J_PASSWORD=testpass \\
    uv run python tests/e2e_prompt_library_ingest.py --openai
"""

from __future__ import annotations

import argparse
import asyncio
import os
import re
import sys
import uuid
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel

from graphiti_core.cross_encoder.client import CrossEncoderClient
from graphiti_core.driver.neo4j_driver import Neo4jDriver
from graphiti_core.embedder.client import EmbedderClient
from graphiti_core.graphiti import Graphiti
from graphiti_core.llm_client import OpenAIClient
from graphiti_core.llm_client.client import LLMClient
from graphiti_core.llm_client.config import LLMConfig, ModelSize
from graphiti_core.llm_client.llm_runtime import (
    LLMModel,
    LLMPromptOverrides,
    LLMRuntime,
    PromptRoutes,
)
from graphiti_core.nodes import EpisodeType
from graphiti_core.prompts import (
    ChatPrompt,
    SystemMessage,
    UserMessage,
    create_prompt_library,
)
from graphiti_core.prompts import (
    prompt_library as builtin_prompt_library,
)
from graphiti_core.prompts.models import Message

EPISODE_BODY = 'Alice works at Acme Corp as an engineer.'
CUSTOM_MARKER = 'CUSTOM E2E EXTRACT'


def _redact(text: str) -> str:
    """Redact secrets / key-shaped strings from error output."""
    text = re.sub(r'sk-[A-Za-z0-9_-]{10,}', 'sk-[REDACTED]', text)
    text = re.sub(
        r'(?i)(api[_-]?key|authorization|bearer)\s*[:=]\s*\S+',
        r'\1=[REDACTED]',
        text,
    )
    return text


def _validate_openai_key() -> str:
    key = os.environ.get('OPENAI_API_KEY', '')
    if not key:
        return 'missing'
    if (
        key.strip() in ('sk-...', 'sk-xxx', 'your-key-here')
        or key.startswith('sk-...')
        or len(key) <= 20
    ):
        return 'placeholder'
    return 'ok'


class StubLLM(LLMClient):
    def __init__(self) -> None:
        super().__init__(config=LLMConfig(model='gpt-4.1-mini', small_model='gpt-4.1-nano'))
        self.calls: list[dict[str, Any]] = []

    async def _generate_response(
        self,
        messages: list[Message],
        response_model: type[BaseModel] | None = None,
        max_tokens: int | None = None,
        model_size: ModelSize = ModelSize.medium,
        *,
        model: str | None = None,
        small_model: str | None = None,
    ) -> dict[str, Any]:
        return {}

    async def generate_response(
        self,
        messages: list[Message],
        response_model: type[BaseModel] | None = None,
        max_tokens: int | None = None,
        model_size: ModelSize = ModelSize.medium,
        group_id: str | None = None,
        prompt_name: str | None = None,
        *,
        attribute_extraction: bool = False,
        model: str | None = None,
        small_model: str | None = None,
    ) -> dict[str, Any]:
        self.calls.append({'prompt_name': prompt_name, 'messages': messages})
        name = prompt_name or ''
        if name.startswith('extract_nodes.extract_'):
            return {
                'extracted_entities': [
                    {'name': 'Alice', 'entity_type_id': 0, 'episode_indices': [0]},
                    {'name': 'Acme', 'entity_type_id': 0, 'episode_indices': [0]},
                ]
            }
        if name == 'extract_edges.edge':
            return {
                'edges': [
                    {
                        'source_entity_name': 'Alice',
                        'target_entity_name': 'Acme',
                        'relation_type': 'WORKS_AT',
                        'fact': 'Alice works at Acme',
                        'episode_indices': [0],
                    }
                ]
            }
        if name == 'dedupe_nodes.nodes':
            return {'entity_resolutions': []}
        if name == 'dedupe_edges.resolve_edge':
            return {'duplicate_facts': [], 'contradicted_facts': []}
        if 'summar' in name:
            return {'summaries': []}
        if name.endswith('extract_attributes'):
            return {}
        if 'timestamp' in name:
            return {'valid_at': None, 'invalid_at': None, 'timestamps': []}
        return {}


class RecordingLLM(LLMClient):
    """Delegates to a real LLMClient while recording prompt_name + first message."""

    def __init__(self, inner: LLMClient) -> None:
        config = getattr(inner, 'config', None) or LLMConfig()
        super().__init__(config=config)
        self.inner = inner
        self.calls: list[dict[str, Any]] = []
        # Mirror model attrs used when constructing LLMRuntime
        if hasattr(inner, 'model'):
            self.model = inner.model  # type: ignore[attr-defined]
        if hasattr(inner, 'small_model'):
            self.small_model = inner.small_model  # type: ignore[attr-defined]

    async def _generate_response(
        self,
        messages: list[Message],
        response_model: type[BaseModel] | None = None,
        max_tokens: int | None = None,
        model_size: ModelSize = ModelSize.medium,
        *,
        model: str | None = None,
        small_model: str | None = None,
    ) -> dict[str, Any]:
        return await self.inner._generate_response(  # noqa: SLF001
            messages,
            response_model=response_model,
            max_tokens=max_tokens,
            model_size=model_size,
            model=model,
            small_model=small_model,
        )

    async def generate_response(
        self,
        messages: list[Message],
        response_model: type[BaseModel] | None = None,
        max_tokens: int | None = None,
        model_size: ModelSize = ModelSize.medium,
        group_id: str | None = None,
        prompt_name: str | None = None,
        *,
        attribute_extraction: bool = False,
        model: str | None = None,
        small_model: str | None = None,
    ) -> dict[str, Any]:
        self.calls.append(
            {
                'prompt_name': prompt_name,
                'first_content': messages[0].content if messages else '',
            }
        )
        return await self.inner.generate_response(
            messages,
            response_model=response_model,
            max_tokens=max_tokens,
            model_size=model_size,
            group_id=group_id,
            prompt_name=prompt_name,
            attribute_extraction=attribute_extraction,
            model=model,
            small_model=small_model,
        )


class StubEmbedder(EmbedderClient):
    async def create(self, input_data: str | list[str] | Any = None) -> list[float]:
        return [0.1] * 8

    async def create_batch(self, input_data_list: list[str]) -> list[list[float]]:
        return [[0.1] * 8 for _ in input_data_list]


class StubCrossEncoder(CrossEncoderClient):
    async def rank(self, query: str, passages: list[str]) -> list[tuple[str, float]]:
        return [(p, 1.0 - (i * 0.01)) for i, p in enumerate(passages)]


def _custom_extract_message(ctx: dict[str, Any]) -> ChatPrompt:
    """Override that keeps default extract quality but marks system content."""
    base = builtin_prompt_library.extract_nodes.extract_message(ctx)
    return ChatPrompt(
        system=SystemMessage(content=f'{CUSTOM_MARKER}\n{base.system.content}'),
        user=base.user,
    )


async def _ingest(graphiti: Graphiti, group_id: str) -> Any:
    await graphiti.build_indices_and_constraints()
    result = await graphiti.add_episode(
        name='e2e-ep',
        episode_body=EPISODE_BODY,
        source_description='e2e test',
        reference_time=datetime.now(timezone.utc),
        source=EpisodeType.message,
        group_id=group_id,
    )
    assert result.episode is not None
    assert len(result.nodes) >= 1
    print(
        f'OK ingest group={group_id}: episode={result.episode.uuid} '
        f'nodes={len(result.nodes)} edges={len(result.edges)}'
    )
    return result


async def _search_check(graphiti: Graphiti, group_id: str) -> None:
    edges = await graphiti.search('Alice Acme', group_ids=[group_id], num_results=5)
    # Soft check: ingest already asserted nodes; search may be empty if embeddings lag,
    # but with real embedder we expect at least a fact edge for this episode.
    print(f'OK search group={group_id}: edges={len(edges)}')
    if edges:
        print(f'  sample fact={edges[0].fact!r}')


async def _run_mock(driver: Neo4jDriver) -> int:
    # Path 1: default prompt library
    g1 = Graphiti(
        graph_driver=driver,
        llm_client=StubLLM(),
        embedder=StubEmbedder(),
        cross_encoder=StubCrossEncoder(),
    )
    await _ingest(g1, 'e2e-default')

    # Path 2: custom ChatPrompt override library
    custom = create_prompt_library(
        {
            'extract_nodes': {
                'extract_message': lambda ctx: ChatPrompt(
                    system=SystemMessage(content=CUSTOM_MARKER),
                    user=UserMessage(content=str(ctx.get('episode_content', ''))),
                )
            }
        }
    )
    llm2 = StubLLM()
    g2 = Graphiti(
        graph_driver=driver,
        llm_client=llm2,
        embedder=StubEmbedder(),
        cross_encoder=StubCrossEncoder(),
        prompt_library=custom,
    )
    await _ingest(g2, 'e2e-custom-prompts')
    extract_calls = [c for c in llm2.calls if c['prompt_name'] == 'extract_nodes.extract_message']
    assert extract_calls, 'expected extract_nodes.extract_message call'
    assert extract_calls[0]['messages'][0].content.startswith(CUSTOM_MARKER)
    print('OK custom ChatPrompt override observed in LLM messages')

    # Path 3: LLMRuntime
    transport = StubLLM()
    nano = LLMModel(id='gpt-4.1-nano')
    runtime = LLMRuntime(
        transport,
        model=LLMModel(id='gpt-4.1-mini'),
        routes=PromptRoutes(
            extract_nodes=PromptRoutes.ExtractNodes(extract_message=nano),
        ),
    )
    g3 = Graphiti(
        graph_driver=driver,
        embedder=StubEmbedder(),
        cross_encoder=StubCrossEncoder(),
        llm_runtime=runtime,
    )
    await _ingest(g3, 'e2e-llm-runtime')
    print('OK LLMRuntime path ingest')

    print('ALL E2E PATHS PASSED')
    return 0


async def _run_openai_path(
    name: str,
    factory,
) -> tuple[str, bool, str]:
    """Run one path; return (name, ok, detail). Never includes secrets."""
    try:
        await factory()
        return name, True, 'pass'
    except Exception as exc:  # noqa: BLE001 — report per-path failures
        detail = _redact(f'{type(exc).__name__}: {exc}')
        return name, False, detail


async def _run_openai(driver: Neo4jDriver) -> int:
    key_status = _validate_openai_key()
    print(f'OPENAI_API_KEY status: {key_status}')
    if key_status != 'ok':
        print(
            'STOP: put a real OpenAI key in .env (must start with sk- and length > 20).',
            file=sys.stderr,
        )
        return 3

    run_id = uuid.uuid4().hex[:12]
    results: list[tuple[str, bool, str]] = []

    async def path_default() -> None:
        group_id = f'e2e-oa-default-{run_id}'
        g = Graphiti(graph_driver=driver)  # default OpenAI LLM + embedder + reranker
        await _ingest(g, group_id)
        await _search_check(g, group_id)

    async def path_custom() -> None:
        group_id = f'e2e-oa-custom-{run_id}'
        custom = create_prompt_library(
            {'extract_nodes': {'extract_message': _custom_extract_message}}
        )
        llm = RecordingLLM(OpenAIClient())
        g = Graphiti(
            graph_driver=driver,
            llm_client=llm,
            prompt_library=custom,
        )
        await _ingest(g, group_id)
        extract_calls = [
            c for c in llm.calls if c['prompt_name'] == 'extract_nodes.extract_message'
        ]
        assert extract_calls, 'expected extract_nodes.extract_message call'
        assert CUSTOM_MARKER in extract_calls[0]['first_content'], (
            'custom ChatPrompt marker missing from LLM system message'
        )
        print('OK custom ChatPrompt override observed in LLM messages')
        await _search_check(g, group_id)

    async def path_bound() -> None:
        group_id = f'e2e-oa-bound-{run_id}'
        transport = OpenAIClient()
        nano = LLMModel(id='gpt-4.1-nano')
        runtime = LLMRuntime(
            transport,
            model=LLMModel(id='gpt-4.1-mini'),
            routes=PromptRoutes(
                extract_nodes=PromptRoutes.ExtractNodes(extract_message=nano),
            ),
            prompt_overrides=LLMPromptOverrides(
                extract_nodes=LLMPromptOverrides.ExtractNodes(
                    extract_message=_custom_extract_message,
                ),
            ),
        )
        g = Graphiti(graph_driver=driver, llm_runtime=runtime)
        await _ingest(g, group_id)
        print('OK LLMRuntime path ingest')
        await _search_check(g, group_id)

    for name, factory in (
        ('default_prompt_library', path_default),
        ('chatprompt_override', path_custom),
        ('llm_runtime', path_bound),
    ):
        print(f'\n=== PATH: {name} ===')
        results.append(await _run_openai_path(name, factory))

    print('\n=== SUMMARY ===')
    all_ok = True
    for name, ok, detail in results:
        status = 'PASS' if ok else 'FAIL'
        print(f'{status}  {name}: {detail}')
        all_ok = all_ok and ok

    if all_ok:
        print('ALL OPENAI E2E PATHS PASSED')
        return 0
    print('SOME OPENAI E2E PATHS FAILED', file=sys.stderr)
    return 1


async def main(openai: bool = False) -> int:
    os.environ.setdefault('GRAPHITI_TELEMETRY_ENABLED', 'false')

    uri = os.environ.get('NEO4J_URI', 'bolt://localhost:7687')
    user = os.environ.get('NEO4J_USER', 'neo4j')
    password = os.environ.get('NEO4J_PASSWORD', 'testpass')

    driver = Neo4jDriver(uri, user, password)
    try:
        await driver.execute_query('RETURN 1 AS n')
    except Exception as exc:
        print(f'SKIP e2e: Neo4j not reachable at {uri}: {_redact(str(exc))}', file=sys.stderr)
        await driver.close()
        return 2

    try:
        if openai:
            return await _run_openai(driver)
        return await _run_mock(driver)
    finally:
        await driver.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Graphiti prompt-library e2e ingest')
    parser.add_argument(
        '--openai',
        action='store_true',
        help='Use real OpenAI LLM + embedder (requires OPENAI_API_KEY)',
    )
    args = parser.parse_args()
    raise SystemExit(asyncio.run(main(openai=args.openai)))
