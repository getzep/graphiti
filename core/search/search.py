import logging
from datetime import datetime
from time import time

from neo4j import AsyncDriver
from pydantic import BaseModel

from core.edges import Edge
from core.llm_client.config import EMBEDDING_DIM
from core.nodes import Node
from core.search.search_utils import (
    edge_fulltext_search,
    edge_similarity_search,
    get_mentioned_nodes,
    rrf,
)
from core.utils import retrieve_episodes
from core.utils.maintenance.graph_data_operations import EPISODE_WINDOW_LEN

logger = logging.getLogger(__name__)


class SearchConfig(BaseModel):
    num_results: int = 10
    num_episodes: int = EPISODE_WINDOW_LEN
    similarity_search: str = "cosine"
    text_search: str = "BM25"
    reranker: str = "rrf"


async def hybrid_search(
    driver: AsyncDriver, embedder, query: str, timestamp: datetime, config: SearchConfig
) -> dict[str, [Node | Edge]]:
    start = time()

    episodes = []
    nodes = []
    edges = []

    search_results = []

    if config.num_episodes > 0:
        episodes.extend(await retrieve_episodes(driver, timestamp))
        nodes.extend(await get_mentioned_nodes(driver, episodes))

    if config.text_search == "BM25":
        text_search = await edge_fulltext_search(query, driver)
        search_results.append(text_search)

    if config.similarity_search == "cosine":
        query_text = query.replace("\n", " ")
        search_vector = (
            (await embedder.create(input=[query_text], model="text-embedding-3-small"))
            .data[0]
            .embedding[:EMBEDDING_DIM]
        )

        similarity_search = await edge_similarity_search(search_vector, driver)
        search_results.append(similarity_search)

    if len(search_results) == 1:
        edges = search_results[0]

    elif len(search_results) > 1 and not config.reranker == "rrf":
        logger.exception("Multiple searches enabled without a reranker")
        raise Exception("Multiple searches enabled without a reranker")

    elif config.reranker == "rrf":
        edge_uuid_map = {}
        search_result_uuids = []

        logger.info([[edge.fact for edge in result] for result in search_results])

        for result in search_results:
            result_uuids = []
            for edge in result:
                result_uuids.append(edge.uuid)
                edge_uuid_map[edge.uuid] = edge

            search_result_uuids.append(result_uuids)

        search_result_uuids = [
            [edge.uuid for edge in result] for result in search_results
        ]

        reranked_uuids = rrf(search_result_uuids)

        reranked_edges = [edge_uuid_map[uuid] for uuid in reranked_uuids]
        edges.extend(reranked_edges)

    context = {
        "episodes": episodes,
        "nodes": nodes,
        "edges": edges,
    }

    end = time()

    logger.info(
        f"search returned context for query {query} in {(end - start) * 1000} ms"
    )

    return context
