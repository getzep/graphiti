from neo4j import AsyncDriver


async def build_community_projection(driver: AsyncDriver, group_ids: list[str | None] | None):