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
import os

from dotenv import load_dotenv

from graphiti_core import Graphiti
from graphiti_core.cross_encoder.bge_reranker_client import BGERerankerClient
from graphiti_core.driver.falkordb_driver import FalkorDriver
from graphiti_core.embedder.huggingface import HuggingFaceEmbedder, HuggingFaceEmbedderConfig
from graphiti_core.llm_client.anthropic_client import AnthropicClient
from graphiti_core.llm_client.config import LLMConfig

load_dotenv()

falkor_host = os.environ.get('FALKORDB_HOST', 'localhost')
falkor_port = os.environ.get('FALKORDB_PORT', '6379')
falkor_username = os.environ.get('FALKORDB_USERNAME', None)
falkor_password = os.environ.get('FALKORDB_PASSWORD', None)


def build_client() -> Graphiti:
    return Graphiti(
        graph_driver=FalkorDriver(
            host=falkor_host, port=falkor_port, username=falkor_username, password=falkor_password
        ),
        llm_client=AnthropicClient(
            LLMConfig(api_key=os.environ.get('ANTHROPIC_API_KEY'), model='claude-haiku-4-5-20251001')
        ),
        embedder=HuggingFaceEmbedder(HuggingFaceEmbedderConfig(embedding_dim=384)),
        cross_encoder=BGERerankerClient(),
    )


async def search(query: str, num_results: int = 10) -> list[str]:
    client = build_client()
    results = await client.search(query, num_results=num_results)
    return [r.fact for r in results]


async def main(query: str):
    facts = await search(query)
    for fact in facts:
        print(fact)
        print()


if __name__ == '__main__':
    import sys

    query = ' '.join(sys.argv[1:]) if len(sys.argv) > 1 else 'Who is Dorothy?'
    asyncio.run(main(query))
