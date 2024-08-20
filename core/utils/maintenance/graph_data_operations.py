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


async def retrieve_episodes(
    driver: AsyncDriver, last_n: int, sources: list[str] | None = "messages"
) -> list[EpisodicNode]:
    """Retrieve the last n episodic nodes from the graph"""
    query = """
        MATCH (e:Episodic)
        RETURN e.content as content,
            e.created_at as created_at,
            e.valid_at as valid_at,
            e.uuid as uuid,
            e.name as name,
            e.source_description as source_description,
            e.source as source
        ORDER BY e.created_at DESC
        LIMIT $num_episodes
        """
    result = await driver.execute_query(query, num_episodes=last_n)
    episodes = [
        EpisodicNode(
            content=record["content"],
            created_at=datetime.fromtimestamp(
                record["created_at"].to_native().timestamp(), timezone.utc
            ),
            valid_at=(
                datetime.fromtimestamp(
                    record["valid_at"].to_native().timestamp(),
                    timezone.utc,
                )
                if record["valid_at"] is not None
                else None
            ),
            uuid=record["uuid"],
            source=record["source"],
            name=record["name"],
            source_description=record["source_description"],
        )
        for record in result.records
    ]
    return list(reversed(episodes))  # Return in chronological order
