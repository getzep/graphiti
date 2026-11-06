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
import os
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
        target_graph: str | None = None,
    ):
        self.host = host
        self.port = port
        self.password = password
        # Default to FALKORDB_DATABASE so the tool migrates into the graph the
        # server actually reads (README documents this variable).
        self.target_graph = target_graph or os.environ.get('FALKORDB_DATABASE', 'GRAPHITI')
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
        """List all graphs in FalkorDB via the client connection.

        Uses the authenticated connection (honors --host/--password) instead
        of shelling out to redis-cli, which ignored those options and silently
        returned an empty list when the binary was missing.
        """
        try:
            return list(await self.client.list_graphs())
        except Exception as e:
            logger.error(f'Error listing graphs: {e}')
            raise

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

        # Migrate Community and Saga nodes (Saga summaries are NOT rebuildable)
        logger.info('  Migrating Community nodes...')
        await self._migrate_labeled_nodes(source_db, target_db, 'Community')
        logger.info('  Migrating Saga nodes...')
        await self._migrate_labeled_nodes(source_db, target_db, 'Saga')

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
                   e.content, e.created_at, e.valid_at, e.entity_edges,
                   e.entity_metadata
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
                entity_metadata,
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
                    entity_edges: $entity_edges,
                    entity_metadata: $entity_metadata
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
                    'entity_metadata': entity_metadata,
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
                   n.created_at as created_at, labels(n) as labels,
                   n.attributes as attributes
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
            attributes = row[7] if len(row) > 7 else None

            # Check if node already exists in target
            existing = await target_db.query(
                'MATCH (n:Entity {uuid: $uuid}) RETURN count(n) as count', {'uuid': uuid}
            )

            if existing.result_set and existing.result_set[0][0] > 0:
                logger.debug(f'    Entity node {uuid} already exists, skipping')
                continue

            # Build label string
            labels_list = labels if isinstance(labels, list) else []
            labels_str = ':Entity' + ''.join(
                f':{label}' for label in labels_list if label != 'Entity'
            )

            # Build basic node creation
            params = {
                'uuid': uuid,
                'name': name,
                'group_id': group_id,
                'summary': summary,
                'created_at': created_at,
                'attributes': attributes if attributes is not None else {},
            }

            # Add embedding if it exists and is not None
            if name_embedding is not None:
                params['name_embedding'] = name_embedding

            # Insert into target graph
            # Use vecf32() to convert embedding list to Vectorf32 format
            query = f'CREATE (n{labels_str} {{uuid: $uuid, name: $name, group_id: $group_id, summary: $summary, created_at: $created_at, attributes: $attributes'
            if name_embedding is not None:
                query += ', name_embedding: vecf32($name_embedding)'
            query += '})'

            await target_db.query(query, params)
            logger.debug(f'    Migrated Entity node: {name}')

    async def _migrate_labeled_nodes(self, source_db, target_db, label: str):
        """Migrate nodes of a simple label (Community, Saga) with their labels."""
        result = await source_db.query(
            f'MATCH (n:{label}) '
            'RETURN n.uuid as uuid, n.name as name, n.group_id as group_id, '
            'n.created_at as created_at, labels(n) as labels, '
            'n.summary as summary'
        )
        migrated = skipped = 0
        for row in result.result_set:
            uuid, name, group_id, created_at, labels, summary = row[:6]
            existing = await target_db.query(
                f'MATCH (n:{label} {{uuid: $uuid}}) RETURN count(n) as count', {'uuid': uuid}
            )
            if existing.result_set and existing.result_set[0][0] > 0:
                skipped += 1
                continue
            extra_labels = [x for x in (labels or []) if x != label]
            labels_str = f':{label}' + ''.join(f':{x}' for x in extra_labels)
            await target_db.query(
                f'CREATE (n{labels_str} {{uuid: $uuid, name: $name, group_id: $group_id, '
                'created_at: $created_at, summary: $summary})',
                {
                    'uuid': uuid,
                    'name': name,
                    'group_id': group_id,
                    'created_at': created_at,
                    'summary': summary,
                },
            )
            migrated += 1
        logger.info(f'    {label} nodes: migrated={migrated} skipped(existing)={skipped}')

    KNOWN_REL_TYPES = {'RELATES_TO', 'MENTIONS', 'HAS_MEMBER', 'HAS_EPISODE', 'NEXT_EPISODE'}

    async def _migrate_edges(self, source_db, target_db):
        """Migrate edges by relationship type, preserving every class.

        RELATES_TO joins Entity pairs; MENTIONS joins Episodic->Entity;
        HAS_MEMBER joins Community->Entity; HAS_EPISODE joins Saga->Episodic;
        NEXT_EPISODE chains Episodic pairs. Endpoints are matched by uuid, so
        HAS_MEMBER/HAS_EPISODE resolve only when Community/Saga nodes were
        migrated (see _migrate_labeled_nodes); otherwise they land in
        unmatched with a warning. Unknown relationship types are counted.
        """
        result = await source_db.query(
            """
            MATCH (a)-[r]->(b)
            RETURN type(r) AS rel_type,
                   r.uuid AS uuid, r.source_node_uuid AS source_uuid,
                   r.target_node_uuid AS target_uuid, r.name AS name,
                   r.fact AS fact, r.group_id AS group_id,
                   r.created_at AS created_at, r.valid_at AS valid_at,
                   r.invalid_at AS invalid_at, r.expired_at AS expired_at,
                   r.fact_embedding AS fact_embedding, r.episodes AS episodes,
                   r.last_updated_at AS last_updated_at, r.attributes AS attributes
            """
        )

        migrated = skipped = unmatched = unknown_type = 0
        for row in result.result_set:
            (
                rel_type,
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
                attributes,
            ) = row

            if rel_type not in self.KNOWN_REL_TYPES:
                unknown_type += 1
                logger.warning(f'    Unknown relationship type {rel_type!r} (edge {uuid}) skipped')
                continue

            existing = await target_db.query(
                f'MATCH ()-[r:{rel_type} {{uuid: $uuid}}]->() RETURN count(r) as count',
                {'uuid': uuid},
            )
            if existing.result_set and existing.result_set[0][0] > 0:
                skipped += 1
                continue

            props = (
                'uuid: $uuid, source_node_uuid: $source_uuid, target_node_uuid: $target_uuid, '
                'name: $name, fact: $fact, group_id: $group_id, created_at: $created_at, '
                'valid_at: $valid_at, invalid_at: $invalid_at, expired_at: $expired_at, '
                'episodes: $episodes, last_updated_at: $last_updated_at, attributes: $attributes'
            )
            params = {
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
                'episodes': episodes,
                'last_updated_at': last_updated_at,
                'attributes': attributes,
            }
            await target_db.query(
                f'MATCH (a {{uuid: $source_uuid}})\n'
                f'MATCH (b {{uuid: $target_uuid}})\n'
                f'CREATE (a)-[r:{rel_type} {{ {props} }}]->(b)\n'
                'WITH r\nRETURN count(r) AS created',
                params,
            )
            verified = await target_db.query(
                f'MATCH ()-[r:{rel_type} {{uuid: $uuid}}]->() RETURN count(r) as count',
                {'uuid': uuid},
            )
            if verified.result_set and verified.result_set[0][0] > 0:
                migrated += 1
                # fact_embedding is nullable: SET only when present, so an
                # embedding-less edge cannot abort the whole migration.
                if fact_embedding is not None:
                    await target_db.query(
                        f'MATCH ()-[r:{rel_type} {{uuid: $uuid}}]->() '
                        'SET r.fact_embedding = vecf32($fact_embedding)',
                        {'uuid': uuid, 'fact_embedding': fact_embedding},
                    )
            else:
                unmatched += 1
                logger.warning(
                    f'    Edge {uuid} ({rel_type}) endpoints missing in target: '
                    f'source={source_uuid} target={target_uuid}'
                )
        logger.info(
            f'    Edges: migrated={migrated} skipped(existing)={skipped} '
            f'unmatched={unmatched} unknown_type={unknown_type}'
        )
        return migrated, skipped, unmatched

    async def run(
        self,
        graphs: list[str] | None = None,
        dry_run: bool = False,
        explicit_graphs: bool = False,
        migrate_all: bool = False,
    ):
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

            # Safety boundary: only migrate graphs that actually carry
            # graphiti data. A shared Redis instance can host a second
            # deployment's unified graph - blindly merging deployments
            # co-mingles their group_ids.
            candidates = []
            for g in graphs:
                counts = {
                    label: await self.count_nodes(g, label)
                    for label in ('Episodic', 'Entity', 'Community', 'Saga')
                }
                if any(counts.values()):
                    candidates.append((g, sum(counts.values())))
            if not candidates:
                logger.warning('No source graphs contain graphiti data')
                return
            if not explicit_graphs and len(candidates) > 1 and not migrate_all:
                # discovery mode with multiple data-bearing graphs: require
                # explicit selection or --all
                logger.error(
                    f'Multiple graphs contain graphiti data: '
                    f'{[g for g, _ in candidates]}. Refusing to migrate all of '
                    f'them - a second graphiti deployment may share this Redis '
                    f'instance. Re-run with explicit --graphs or --all.'
                )
                raise SystemExit(1)
            graphs = [g for g, _ in candidates]

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

        # Edge counts per relationship type: catches dropped MENTIONS and
        # friends, which node counts alone never surface.
        edge_result = await self.client.select_graph(self.target_graph).query(
            'MATCH ()-[r]->() RETURN type(r) AS t, count(r) AS c ORDER BY c DESC'
        )
        for row in edge_result.result_set:
            logger.info(f'    Edge [{row[0]}]: {row[1]}')

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
        default=None,
        help='Target graph name (default: $FALKORDB_DATABASE or GRAPHITI)',
    )
    parser.add_argument(
        '--graphs',
        nargs='+',
        help='Specific source graphs to migrate (default: all except target)',
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Migrate every data-bearing graph even when several match '
        '(second-deployment safety otherwise refuses)',
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

    await migrator.run(
        graphs=args.graphs,
        dry_run=args.dry_run,
        explicit_graphs=args.graphs is not None,
        migrate_all=args.all,
    )


if __name__ == '__main__':
    asyncio.run(main())
