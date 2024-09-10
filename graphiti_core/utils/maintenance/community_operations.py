import asyncio

from neo4j import AsyncDriver

from graphiti_core.edges import CommunityEdge
from graphiti_core.llm_client import LLMClient
from graphiti_core.nodes import EntityNode, CommunityNode
from graphiti_core.prompts import prompt_library
from graphiti_core.utils.maintenance.edge_operations import build_community_edges


async def build_community_projection(driver: AsyncDriver) -> str:
    result, _, _ = await driver.execute_query("""
    CALL gds.graph.project("communities", "Entity", "RELATES_TO")
    YIELD graphName AS graph, nodeProjection AS nodes, relationshipProjection AS edges
    """)

    return result['graph']


async def destroy_projection(driver: AsyncDriver, projection_name: str):
    await driver.execute_query("""
    CALL gds.graph.drop($projection_name)
    """, projection_name=projection_name)


async def get_community_clusters(driver: AsyncDriver, projection_name: str) -> list[list[EntityNode]]: ...


async def summarize_pair(llm_client: LLMClient, summary_pair: tuple[str]) -> str:
    # Prepare context for LLM
    context = {'node_summaries': [{'summary': summary} for summary in summary_pair]}

    llm_response = await llm_client.generate_response(prompt_library.summarize_nodes.summarize_pair(context))

    pair_summary = llm_response.get('summary', '')

    return pair_summary


async def build_community(llm_client: LLMClient, community_cluster: list[EntityNode]) -> tuple[
    CommunityNode, list[CommunityEdge]]:
    build_community_edges()


async def build_communities(driver: AsyncDriver, llm_client: LLMClient) -> tuple[
    list[CommunityNode], list[CommunityEdge]]:
    projection = await build_community_projection(driver)
    community_clusters = await get_community_clusters(driver, projection)

    communities: list[tuple[CommunityNode, list[CommunityEdge]]] = list(
        await asyncio.gather(*[build_community(llm_client, cluster) for cluster in community_clusters]))

    await destroy_projection(driver, projection)
    return communities
