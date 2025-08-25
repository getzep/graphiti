from graphiti_core.driver.driver import GraphDriver
from graphiti_core.helpers import validate_group_id
from graphiti_core.utils.maintenance.graph_data_operations import build_dynamic_indexes


async def neo4j_node_group_labels(driver: GraphDriver, group_id: str, batch_size: int = 100):
    validate_group_id(group_id)
    await build_dynamic_indexes(driver, group_id)

    episode_query = """
                        MATCH (n:Episodic {group_id: $group_id})
                        CALL {
                            WITH n
                            SET n:$group_label
                        } IN TRANSACTIONS OF $batch_size ROWS"""

    entity_query = """
                        MATCH (n:Entity {group_id: $group_id})
                        CALL {
                            WITH n
                            SET n:$group_label
                        } IN TRANSACTIONS OF $batch_size ROWS"""

    community_query = """
                        MATCH (n:Community {group_id: $group_id})
                        CALL {
                            WITH n
                            SET n:$group_label
                        } IN TRANSACTIONS OF $batch_size ROWS"""

    async with driver.session() as session:
        await session.run(
            episode_query,
            group_id=group_id,
            group_label='Episodic_' + group_id.replace('-', ''),
            batch_size=batch_size,
        )

    async with driver.session() as session:
        await session.run(
            entity_query,
            group_id=group_id,
            group_label='Entity_' + group_id.replace('-', ''),
            batch_size=batch_size,
        )

    async with driver.session() as session:
        await session.run(
            community_query,
            group_id=group_id,
            group_label='Community_' + group_id.replace('-', ''),
            batch_size=batch_size,
        )
