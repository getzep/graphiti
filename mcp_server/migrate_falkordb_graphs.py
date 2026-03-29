#!/usr/bin/env python3
"""
Migrate FalkorDB data from multiple graphs to a single unified graph.

This script consolidates data from multiple graphs (created by the old FalkorDB driver)
into a single GRAPHITI graph with logical isolation via group_id filtering.

Usage:
    python migrate_falkordb_graphs.py [--target-graph GRAPHITI]
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

# Add parent directory to path to import graphiti_core
sys.path.insert(0, str(Path(__file__).parent.parent))

from falkordb.asyncio import FalkorDB as AsyncFalkorDB

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)


class FalkorDBGraphMigrator:
    """Migrate data from multiple FalkorDB graphs to a single unified graph."""

    def __init__(
        self,
        host: str = 'localhost',
        port: int = 6379,
        password: str | None = None,
        target_graph: str = 'GRAPHITI',
    ):
        self.host = host
        self.port = port
        self.password = password
        self.target_graph = target_graph
        self.client: AsyncFalkorDB | None = None

    async def connect(self):
        """Connect to FalkorDB."""
        self.client = AsyncFalkorDB(
            host=self.host,
            port=self.port,
            password=self.password,
        )
        logger.info(f'Connected to FalkorDB at {self.host}:{self.port}')

    async def close(self):
        """Close the connection."""
        if self.client and hasattr(self.client, 'aclose'):
            await self.client.aclose()
        elif self.client and hasattr(self.client.connection, 'aclose'):
            await self.client.connection.aclose()

    async def list_graphs(self) -> list[str]:
        """List all graphs in FalkorDB."""
        try:
            # Use redis-cli to list graphs
            import subprocess

            result = subprocess.run(
                ['redis-cli', '-p', str(self.port), 'GRAPH.LIST'],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                graphs = result.stdout.strip().split('\n')
                logger.info(f'Found {len(graphs)} graphs')
                return graphs
            else:
                logger.error(f'Failed to list graphs: {result.stderr}')
                return []
        except Exception as e:
            logger.error(f'Error listing graphs: {e}')
            return []

    async def count_nodes(self, graph_name: str, label: str = '') -> int:
        """Count nodes in a graph."""
        try:
            graph = self.client.select_graph(graph_name)
            label_filter = f':{label}' if label else ''
            result = await graph.query(f'MATCH (n{label_filter}) RETURN count(n) as count')
            if result.result_set:
                return result.result_set[0][0]
            return 0
        except Exception as e:
            logger.warning(f'Error counting nodes in {graph_name}: {e}')
            return 0

    async def migrate_graph(self, source_graph: str, dry_run: bool = False):
        """Migrate data from source graph to target graph."""
        logger.info(f'\n{"=" * 60}')
        logger.info(f'Migrating graph: {source_graph}')
        logger.info(f'{"=" * 60}')

        source_db = self.client.select_graph(source_graph)
        target_db = self.client.select_graph(self.target_graph)

        # Count nodes before migration
        episodic_count = await self.count_nodes(source_graph, 'Episodic')
        entity_count = await self.count_nodes(source_graph, 'Entity')
        logger.info(f'  Episodic nodes: {episodic_count}')
        logger.info(f'  Entity nodes: {entity_count}')

        if episodic_count == 0 and entity_count == 0:
            logger.info(f'  Skipping {source_graph} (no data)')
            return

        if dry_run:
            logger.info(f'  [DRY RUN] Would migrate {source_graph}')
            return

        # Migrate Episodic nodes
        logger.info('  Migrating Episodic nodes...')
        await self._migrate_episodic_nodes(source_db, target_db)

        # Migrate Entity nodes
        logger.info('  Migrating Entity nodes...')
        await self._migrate_entity_nodes(source_db, target_db)

        # Migrate edges
        logger.info('  Migrating edges...')
        await self._migrate_edges(source_db, target_db)

        logger.info(f'  ✓ Migration complete for {source_graph}')

    async def _migrate_episodic_nodes(self, source_db, target_db):
        """Migrate Episodic nodes from source to target graph."""
        # Query all episodic nodes from source
        result = await source_db.query(
            """
            MATCH (e:Episodic)
            RETURN e.uuid, e.name, e.group_id, e.source, e.source_description,
                   e.content, e.created_at, e.valid_at, e.entity_edges
            """
        )

        for row in result.result_set:
            (
                uuid,
                name,
                group_id,
                source,
                source_description,
                content,
                created_at,
                valid_at,
                entity_edges,
            ) = row

            # Check if node already exists in target
            existing = await target_db.query(
                'MATCH (e:Episodic {uuid: $uuid}) RETURN count(e) as count', {'uuid': uuid}
            )

            if existing.result_set and existing.result_set[0][0] > 0:
                logger.debug(f'    Episodic node {uuid} already exists, skipping')
                continue

            # Insert into target graph
            await target_db.query(
                """
                CREATE (e:Episodic {
                    uuid: $uuid,
                    name: $name,
                    group_id: $group_id,
                    source: $source,
                    source_description: $source_description,
                    content: $content,
                    created_at: $created_at,
                    valid_at: $valid_at,
                    entity_edges: $entity_edges
                })
                """,
                {
                    'uuid': uuid,
                    'name': name,
                    'group_id': group_id,
                    'source': source,
                    'source_description': source_description,
                    'content': content,
                    'created_at': created_at,
                    'valid_at': valid_at,
                    'entity_edges': entity_edges,
                },
            )
            logger.debug(f'    Migrated Episodic node: {name}')

    async def _migrate_entity_nodes(self, source_db, target_db):
        """Migrate Entity nodes from source to target graph."""
        # Query all entity nodes from source
        result = await source_db.query(
            """
            MATCH (n:Entity)
            RETURN n.uuid as uuid, n.name as name, n.group_id as group_id,
                   n.summary as summary, n.name_embedding as name_embedding,
                   n.created_at as created_at, n.labels as labels
            """
        )

        for row in result.result_set:
            uuid = row[0]
            name = row[1]
            group_id = row[2]
            summary = row[3]
            name_embedding = row[4]
            created_at = row[5]
            labels = row[6] if len(row) > 6 else []

            # Check if node already exists in target
            existing = await target_db.query(
                'MATCH (n:Entity {uuid: $uuid}) RETURN count(n) as count', {'uuid': uuid}
            )

            if existing.result_set and existing.result_set[0][0] > 0:
                logger.debug(f'    Entity node {uuid} already exists, skipping')
                continue

            # Build label string
            labels_list = labels if isinstance(labels, list) else []
            labels_str = ':Entity' + ''.join(f':{l}' for l in labels_list if l != 'Entity')

            # Build basic node creation
            params = {
                'uuid': uuid,
                'name': name,
                'group_id': group_id,
                'summary': summary,
                'created_at': created_at,
            }

            # Add embedding if it exists and is not None
            if name_embedding is not None:
                params['name_embedding'] = name_embedding

            # Insert into target graph
            # Use vecf32() to convert embedding list to Vectorf32 format
            query = f'CREATE (n{labels_str} {{uuid: $uuid, name: $name, group_id: $group_id, summary: $summary, created_at: $created_at'
            if name_embedding is not None:
                query += ', name_embedding: vecf32($name_embedding)'
            query += '})'

            await target_db.query(query, params)
            logger.debug(f'    Migrated Entity node: {name}')

    async def _migrate_edges(self, source_db, target_db):
        """Migrate edges from source to target graph."""
        # Query all edges from source
        result = await source_db.query(
            """
            MATCH (a)-[r]->(b)
            RETURN r.uuid, r.source_node_uuid, r.target_node_uuid, r.name,
                   r.fact, r.group_id, r.created_at, r.valid_at, r.invalid_at,
                   r.expired_at, r.fact_embedding, r.episodes, r.last_updated_at
            """
        )

        for row in result.result_set:
            (
                uuid,
                source_uuid,
                target_uuid,
                name,
                fact,
                group_id,
                created_at,
                valid_at,
                invalid_at,
                expired_at,
                fact_embedding,
                episodes,
                last_updated_at,
            ) = row

            # Check if edge already exists in target
            existing = await target_db.query(
                'MATCH ()-[r:RELATES_TO {uuid: $uuid}]->() RETURN count(r) as count', {'uuid': uuid}
            )

            if existing.result_set and existing.result_set[0][0] > 0:
                logger.debug(f'    Edge {uuid} already exists, skipping')
                continue

            # Insert into target graph
            await target_db.query(
                """
                MATCH (a:Entity {uuid: $source_uuid})
                MATCH (b:Entity {uuid: $target_uuid})
                CREATE (a)-[r:RELATES_TO {
                    uuid: $uuid,
                    source_node_uuid: $source_uuid,
                    target_node_uuid: $target_uuid,
                    name: $name,
                    fact: $fact,
                    group_id: $group_id,
                    created_at: $created_at,
                    valid_at: $valid_at,
                    invalid_at: $invalid_at,
                    expired_at: $expired_at,
                    episodes: $episodes,
                    last_updated_at: $last_updated_at
                }]->(b)
                SET r.fact_embedding = vecf32($fact_embedding)
                """,
                {
                    'uuid': uuid,
                    'source_uuid': source_uuid,
                    'target_uuid': target_uuid,
                    'name': name,
                    'fact': fact,
                    'group_id': group_id,
                    'created_at': created_at,
                    'valid_at': valid_at,
                    'invalid_at': invalid_at,
                    'expired_at': expired_at,
                    'fact_embedding': fact_embedding,
                    'episodes': episodes,
                    'last_updated_at': last_updated_at,
                },
            )
            logger.debug(f'    Migrated edge: {name}')

    async def run(self, graphs: list[str] | None = None, dry_run: bool = False):
        """Run the migration process."""
        await self.connect()

        try:
            # List all graphs if not specified
            if graphs is None:
                graphs = await self.list_graphs()

            # Filter out target graph
            graphs = [g for g in graphs if g != self.target_graph]

            if not graphs:
                logger.warning('No source graphs found to migrate')
                return

            logger.info('\nMigration Plan:')
            logger.info(f'  Target graph: {self.target_graph}')
            logger.info(f'  Source graphs: {graphs}')
            if dry_run:
                logger.info('  Mode: DRY RUN (no changes will be made)')

            # Migrate each graph
            for graph in graphs:
                await self.migrate_graph(graph, dry_run=dry_run)

            # Verify migration
            if not dry_run:
                logger.info(f'\n{"=" * 60}')
                logger.info('Verification:')
                logger.info(f'{"=" * 60}')
                await self._verify_migration(graphs)

        finally:
            await self.close()

    async def _verify_migration(self, source_graphs: list[str]):
        """Verify the migration was successful."""
        # Count nodes in target graph
        target_episodic = await self.count_nodes(self.target_graph, 'Episodic')
        target_entity = await self.count_nodes(self.target_graph, 'Entity')

        logger.info(f'  Target graph ({self.target_graph}):')
        logger.info(f'    Episodic nodes: {target_episodic}')
        logger.info(f'    Entity nodes: {target_entity}')

        # Count by group_id
        result = await self.client.select_graph(self.target_graph).query(
            """
            MATCH (e:Episodic)
            RETURN e.group_id, count(e) as count
            ORDER BY count DESC
            """
        )

        if result.result_set:
            logger.info('  Episodes by group_id:')
            for row in result.result_set:
                logger.info(f'    {row[0]}: {row[1]} episodes')


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Migrate FalkorDB data from multiple graphs to a single unified graph.'
    )
    parser.add_argument(
        '--target-graph',
        default='GRAPHITI',
        help='Target graph name (default: GRAPHITI)',
    )
    parser.add_argument(
        '--graphs',
        nargs='+',
        help='Specific source graphs to migrate (default: all except target)',
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be migrated without making changes',
    )
    parser.add_argument(
        '--host',
        default='localhost',
        help='FalkorDB host (default: localhost)',
    )
    parser.add_argument(
        '--port',
        type=int,
        default=6379,
        help='FalkorDB port (default: 6379)',
    )
    parser.add_argument(
        '--password',
        help='FalkorDB password',
    )

    args = parser.parse_args()

    migrator = FalkorDBGraphMigrator(
        host=args.host,
        port=args.port,
        password=args.password,
        target_graph=args.target_graph,
    )

    await migrator.run(graphs=args.graphs, dry_run=args.dry_run)


if __name__ == '__main__':
    asyncio.run(main())
