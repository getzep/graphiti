import asyncio
import csv
import os

from graphiti_core.driver.driver import GraphDriver
from graphiti_core.driver.neo4j_driver import Neo4jDriver
from graphiti_core.helpers import validate_group_id
from graphiti_core.utils.maintenance.graph_data_operations import build_dynamic_indexes


async def neo4j_node_group_labels(driver: GraphDriver, group_id: str, batch_size: int = 100):
    validate_group_id(group_id)
    await build_dynamic_indexes(driver, group_id)

    episode_query = """
                        MATCH (n:Episodic {group_id: $group_id})
                        CALL {
                            WITH n
                            SET n:$($group_label)
                        } IN TRANSACTIONS OF $batch_size ROWS"""

    entity_query = """
                        MATCH (n:Entity {group_id: $group_id})
                        CALL {
                            WITH n
                            SET n:$($group_label)
                        } IN TRANSACTIONS OF $batch_size ROWS"""

    community_query = """
                        MATCH (n:Community {group_id: $group_id})
                        CALL {
                            WITH n
                            SET n:$($group_label)
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


def pop_last_n_group_ids(csv_file: str = 'group_ids.csv', count: int = 10):
    with open(csv_file) as file:
        reader = csv.reader(file)
        group_ids = [row[0] for row in reader]

    total_count = len(group_ids)
    popped = group_ids[-count:]
    remaining = group_ids[:-count]

    with open(csv_file, 'w', newline='') as file:
        writer = csv.writer(file)
        for gid in remaining:
            writer.writerow([gid])

    return popped, total_count


async def get_group_ids(driver: GraphDriver):
    query = """MATCH (n:Episodic)
                RETURN DISTINCT n.group_id AS group_id"""

    results, _, _ = await driver.execute_query(query)
    group_ids = [result['group_id'] for result in results]

    with open('group_ids.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        for gid in group_ids:
            writer.writerow([gid])


async def neo4j_node_label_migration(driver: GraphDriver, batch_size: int = 10):
    group_ids, total = pop_last_n_group_ids(csv_file='group_ids.csv', count=batch_size)
    while len(group_ids) > 0:
        await asyncio.gather(*[neo4j_node_group_labels(driver, group_id) for group_id in group_ids])
        group_ids, _ = pop_last_n_group_ids(csv_file='group_ids.csv', count=batch_size)


async def main():
    neo4j_uri = os.environ.get('NEO4J_URI') or 'bolt://localhost:7687'
    neo4j_user = os.environ.get('NEO4J_USER') or 'neo4j'
    neo4j_password = os.environ.get('NEO4J_PASSWORD') or 'password'

    driver = Neo4jDriver(
        uri=neo4j_uri,
        user=neo4j_user,
        password=neo4j_password,
    )
    await get_group_ids(driver)
    await neo4j_node_label_migration(driver)
    await driver.close()


if __name__ == '__main__':
    asyncio.run(main())
