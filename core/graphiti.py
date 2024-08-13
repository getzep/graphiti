import asyncio
from datetime import datetime
import logging

from neo4j import AsyncGraphDatabase

from core.nodes import SemanticNode, EpisodicNode
from core.edges import SemanticEdge

logger = logging.getLogger(__name__)


class Graphiti:
    def __init__(self, uri, user, password):
        self.driver = AsyncGraphDatabase.driver(uri, auth=(user, password))
        self.database = "neo4j"

    def close(self):
        self.driver.close()

    async def retrieve_episodes(self, last_n: int) -> list[EpisodicNode]: ...
    async def clear_data(self): ...
    async def retrieve_relevant_schema(self, query: str = None) -> dict[str, any]: ...
    async def generate_llm_response(self, prompt: str) -> dict[str, any]: ...

    async def extract_new_edges(
        self,
        episode: EpisodicNode,
        new_nodes: list[SemanticNode],
        relevant_schema: dict[str, any],
        previous_episodes: list[EpisodicNode],
    ) -> list[SemanticEdge]: ...

    async def extract_new_nodes(
        self,
        episode: EpisodicNode,
        relevant_schema: dict[str, any],
        previous_episodes: list[EpisodicNode],
    ) -> list[SemanticNode]: ...

    async def invalidate_edges(
        self,
        episode: EpisodicNode,
        new_nodes: list[SemanticNode],
        new_edges: list[SemanticEdge],
        relevant_schema: dict[str, any],
        previous_episodes: list[EpisodicNode],
    ) -> list[SemanticEdge]: ...

    async def process_episode(
        self,
        source: str,
        content: str,
        source_description: str,
        reference_time: datetime = None,
        content_type="string",
    ): ...
