from datetime import datetime, timezone

from core.nodes import EpisodicNode
from neo4j import AsyncDriver
import logging


logger = logging.getLogger(__name__)


async def clear_data(driver: AsyncDriver):
    async with driver.session() as session:

        async def delete_all(tx):
            await tx.run("MATCH (n) DETACH DELETE n")

        await session.execute_write(delete_all)


async def retrieve_relevant_schema(
    driver: AsyncDriver, query: str = None
) -> dict[str, any]:
    async with driver.session() as session:
        summary_query = """
        MATCH (n)
        OPTIONAL MATCH (n)-[r]->(m)
        RETURN DISTINCT labels(n) AS node_labels, n.uuid AS node_uuid, n.name AS node_name, 
               type(r) AS relationship_type, r.name AS relationship_name, m.name AS related_node_name
        """
        result = await session.run(summary_query)
        records = [record async for record in result]

        schema = {"nodes": {}, "relationships": []}

        for record in records:
            node_label = record["node_labels"][0]  # Assuming one label per node
            node_uuid = record["node_uuid"]
            node_name = record["node_name"]
            rel_type = record["relationship_type"]
            rel_name = record["relationship_name"]
            related_node = record["related_node_name"]

            if node_name not in schema["nodes"]:
                schema["nodes"][node_name] = {
                    "uuid": node_uuid,
                    "label": node_label,
                    "relationships": [],
                }

            if rel_type and related_node:
                schema["nodes"][node_name]["relationships"].append(
                    {"type": rel_type, "name": rel_name, "target": related_node}
                )
                schema["relationships"].append(
                    {
                        "source": node_name,
                        "type": rel_type,
                        "name": rel_name,
                        "target": related_node,
                    }
                )

        return schema


async def retrieve_episodes(
    driver: AsyncDriver, last_n: int, sources: list[str] | None = "messages"
) -> list[EpisodicNode]:
    """Retrieve the last n episodic nodes from the graph"""
    async with driver.session() as session:
        query = """
            MATCH (e:EpisodicNode)
            RETURN e.content as text, e.timestamp as timestamp, e.reference_timestamp as reference_timestamp
            ORDER BY e.timestamp DESC
            LIMIT $num_episodes
            """
        result = await session.run(query, num_episodes=last_n)
        episodes = [
            EpisodicNode(
                content=record["text"],
                transaction_from=datetime.fromtimestamp(
                    record["timestamp"].to_native().timestamp(), timezone.utc
                ),
                valid_at=(
                    datetime.fromtimestamp(
                        record["reference_timestamp"].to_native().timestamp(),
                        timezone.utc,
                    )
                    if record["reference_timestamp"] is not None
                    else None
                ),
            )
            async for record in result
        ]
        return list(reversed(episodes))  # Return in chronological order
