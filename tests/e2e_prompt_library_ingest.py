"""Local e2e: Graphiti add_episode with stub LLM against Neo4j.

Exercises complete_prompt + prompt_library / PromptBoundLLM wiring without OpenAI.
Run:
  GRAPHITI_TELEMETRY_ENABLED=false NEO4J_PASSWORD=testpass \\
    uv run python tests/e2e_prompt_library_ingest.py
"""

from __future__ import annotations

import asyncio
import os
import sys
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel

from graphiti_core.cross_encoder.client import CrossEncoderClient
from graphiti_core.driver.neo4j_driver import Neo4jDriver
from graphiti_core.embedder.client import EmbedderClient
from graphiti_core.graphiti import Graphiti
from graphiti_core.llm_client.client import LLMClient
from graphiti_core.llm_client.config import LLMConfig, ModelSize
from graphiti_core.llm_client.prompt_bound import LLMModelConfig, create_prompt_bound_llm
from graphiti_core.nodes import EpisodeType
from graphiti_core.prompts import ChatPrompt, SystemMessage, UserMessage, create_prompt_library
from graphiti_core.prompts.models import Message


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


class StubEmbedder(EmbedderClient):
    async def create(self, input_data: str | list[str] | Any = None) -> list[float]:
        return [0.1] * 8

    async def create_batch(self, input_data_list: list[str]) -> list[list[float]]:
        return [[0.1] * 8 for _ in input_data_list]


class StubCrossEncoder(CrossEncoderClient):
    async def rank(self, query: str, passages: list[str]) -> list[tuple[str, float]]:
        return [(p, 1.0 - (i * 0.01)) for i, p in enumerate(passages)]


async def _ingest(graphiti: Graphiti, group_id: str) -> None:
    await graphiti.build_indices_and_constraints()
    result = await graphiti.add_episode(
        name='e2e-ep',
        episode_body='Alice works at Acme Corp as an engineer.',
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


async def main() -> int:
    uri = os.environ.get('NEO4J_URI', 'bolt://localhost:7687')
    user = os.environ.get('NEO4J_USER', 'neo4j')
    password = os.environ.get('NEO4J_PASSWORD', 'testpass')

    driver = Neo4jDriver(uri, user, password)
    try:
        await driver.execute_query('RETURN 1 AS n')
    except Exception as exc:
        print(f'SKIP e2e: Neo4j not reachable at {uri}: {exc}', file=sys.stderr)
        await driver.close()
        return 2

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
                    system=SystemMessage(content='CUSTOM E2E EXTRACT'),
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
    assert extract_calls[0]['messages'][0].content.startswith('CUSTOM E2E EXTRACT')
    print('OK custom ChatPrompt override observed in LLM messages')

    # Path 3: PromptBoundLLM bundle
    transport = StubLLM()
    bundle = create_prompt_bound_llm(
        transport,
        models={
            'default': LLMModelConfig(model='gpt-4.1-mini'),
            'fast': LLMModelConfig(model='gpt-4.1-nano'),
        },
        prompt_models={'extract_nodes.extract_message': 'fast'},
    )
    g3 = Graphiti(
        graph_driver=driver,
        embedder=StubEmbedder(),
        cross_encoder=StubCrossEncoder(),
        prompt_bound_llm=bundle,
    )
    await _ingest(g3, 'e2e-prompt-bound')
    print('OK PromptBoundLLM path ingest')

    await driver.close()
    print('ALL E2E PATHS PASSED')
    return 0


if __name__ == '__main__':
    raise SystemExit(asyncio.run(main()))
